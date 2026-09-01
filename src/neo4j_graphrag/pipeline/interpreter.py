#  Copyright (c) "Neo4j"
#  Neo4j Sweden AB [https://neo4j.com]
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Interpreters evaluate pipeline definitions.

A pipeline built with :class:`~neo4j_graphrag.pipeline.pipeline.Pipeline`
is pure data — a linked list of
:class:`~neo4j_graphrag.pipeline.operators.Operator` nodes.  An
:class:`Interpreter` walks that list and executes it.

:class:`LocalInterpreter` evaluates the pipeline in the current process
using lazy generators: nothing executes until the returned stream is
consumed (``collect()``, ``to_sink()``, or direct iteration).

Async stages are evaluated **blocking**: each chunk of ``map_batch_size``
items is dispatched with ``asyncio.gather`` inside its own
``asyncio.run()`` call, and its results are yielded before the next chunk
is fetched.  This bounds memory usage and keeps the interpreter
synchronous, at the cost of two restrictions:

* async operators must not be evaluated from within a running event loop
  (use ``asyncio.to_thread`` at the call site if needed), and
* async clients that bind to an event loop (``httpx.AsyncClient``,
  ``aiohttp.ClientSession``) must be created *inside* the async function
  rather than shared across chunks, because each chunk runs on a fresh
  event loop.

:class:`AsyncInterpreter` lifts both restrictions by evaluating the whole
chain on a single event loop, with async operators applied concurrently
under a semaphore.  It is the building block for running several
:class:`SimpleKGPipeline`-scale pipelines concurrently (``asyncio.gather``
of their ``evaluate`` streams) so a keep-alive async client can be shared
across them without the per-chunk loop churn the local interpreter pays.
"""

from __future__ import annotations

import asyncio
import itertools
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Iterable,
    Iterator,
)
from concurrent.futures import ThreadPoolExecutor
from functools import reduce as _reduce
from typing import TYPE_CHECKING, Any, TypeVar

from neo4j_graphrag.pipeline.operators import (
    Filter,
    FilterOk,
    FlattenOk,
    FlatMap,
    Grouped,
    Map,
    MapAsyncChunked,
    MapOk,
    OnError,
    ReduceOp,
    SinkOp,
    Skip,
    SourceOp,
    Take,
    TakeWhile,
    TryMap,
    TryMapAsyncChunked,
    TryMapOk,
    TryMapOkAsyncChunked,
)
from neo4j_graphrag.pipeline.result import Err, Ok
from neo4j_graphrag.pipeline.sink import Sink
from neo4j_graphrag.pipeline.source import Source

if TYPE_CHECKING:
    from neo4j_graphrag.pipeline.pipeline import Pipeline, ResultPipeline

__all__ = ["AsyncInterpreter", "Interpreter", "LocalInterpreter"]

_T = TypeVar("_T")
_U = TypeVar("_U")


class Interpreter(ABC):
    """Base class for pipeline interpreters.

    An interpreter executes a pipeline definition and returns the resulting
    stream.  Whether evaluation is lazy or eager, local or distributed, is
    up to the implementation.
    """

    @abstractmethod
    def evaluate(
        self, pipeline: Pipeline[Any] | ResultPipeline[Any]
    ) -> Iterator[Any] | AsyncIterator[Any]:
        """Evaluate *pipeline* and return its output stream.

        Implementations should not execute any user code before the
        returned stream is consumed.  A pipeline ending in a sink returns a
        stream that yields nothing but writes to the sink as it is drained,
        so callers must exhaust it — :meth:`Pipeline.to_sink` does this.

        Synchronous interpreters (e.g. :class:`LocalInterpreter`) return an
        ``Iterator``; interpreters sharing a single event loop (e.g.
        :class:`AsyncInterpreter`) return an ``AsyncIterator``.
        """

    async def run(self, pipeline: Pipeline[Any] | ResultPipeline[Any]) -> None:
        """Evaluate *pipeline* and drain it, awaitable regardless of which
        kind of stream :meth:`evaluate` returns.

        A synchronous stream is drained in a thread (since a
        generator-based interpreter, e.g. :class:`LocalInterpreter`, may
        block the event loop — ``LocalInterpreter`` calls ``asyncio.run``
        internally); an async stream is drained with ``async for`` on the
        caller's own loop. Callers that only need a caller-supplied
        interpreter to finish a sink-terminated pipeline (e.g.
        ``SimpleKGPipeline``) can use this instead of branching on the
        interpreter type themselves.
        """
        stream = self.evaluate(pipeline)
        if isinstance(stream, AsyncIterator):
            async for _ in stream:
                pass
        else:
            await asyncio.to_thread(deque, stream, 0)


# ---------------------------------------------------------------------------
# Evaluation helpers (shared generator building blocks)
# ---------------------------------------------------------------------------


def _batched(iterable: Iterable[_T], n: int) -> Iterator[list[_T]]:
    """Yield successive non-overlapping lists of length *n* from *iterable*."""
    it = iter(iterable)
    while batch := list(itertools.islice(it, n)):
        yield batch


def _iter_chunked_async(
    stream: Iterable[_T],
    map_batch_size: int,
    make_chunk_coro: Callable[[list[_T]], Coroutine[None, None, list[_U]]],
) -> Iterator[_U]:
    """Shared engine for chunked async operators.

    Iterates *stream* in chunks of *map_batch_size* and, for each chunk,
    calls ``asyncio.run(make_chunk_coro(chunk))``, yielding all results
    before moving to the next chunk.

    Raises:
        ValueError: If *map_batch_size* < 1.  (Builders validate eagerly;
            this is a backstop for hand-built operator graphs.)
    """
    if map_batch_size < 1:
        raise ValueError(f"map_batch_size must be >= 1, got {map_batch_size!r}")
    for chunk in _batched(stream, map_batch_size):
        yield from asyncio.run(make_chunk_coro(chunk))


def _make_map_chunk_coro(
    func: Callable[[_T], Awaitable[_U]],
) -> Callable[[list[_T]], Coroutine[None, None, list[_U]]]:
    async def _run_chunk(chunk: list[_T]) -> list[_U]:
        return list(await asyncio.gather(*[func(item) for item in chunk]))

    return _run_chunk


def _make_try_chunk_coro(
    func: Callable[[_T], Awaitable[_U]],
) -> Callable[[list[_T]], Coroutine[None, None, list[Ok[_U] | Err]]]:
    async def _run_chunk(chunk: list[_T]) -> list[Ok[_U] | Err]:
        raw = await asyncio.gather(
            *[func(item) for item in chunk], return_exceptions=True
        )
        results: list[Ok[_U] | Err] = []
        # ``asyncio.gather`` preserves input order, so the failed item is
        # ``chunk[i]``; carry it as the retry handle on the ``Err``.
        for item, r in zip(chunk, raw):
            if isinstance(r, Exception):
                results.append(Err(exception=r, context=item))
            elif isinstance(r, BaseException):
                # SystemExit, KeyboardInterrupt, CancelledError: fatal, re-raise.
                raise r
            else:
                results.append(Ok(value=r))
        return results

    return _run_chunk


def _make_try_ok_chunk_coro(
    func: Callable[[Any], Awaitable[Any]],
) -> Callable[[list[Any]], Coroutine[None, None, list[Any]]]:
    async def _process(item: Ok[Any] | Err) -> Ok[Any] | Err:
        if isinstance(item, Err):
            return item
        try:
            return Ok(value=await func(item.value))
        except Exception as e:
            return Err(exception=e, context=item.value)

    async def _run_chunk(chunk: list[Any]) -> list[Any]:
        return list(await asyncio.gather(*[_process(item) for item in chunk]))

    return _run_chunk


def _try_map(stream: Iterator[Any], func: Callable[[Any], Any]) -> Iterator[Any]:
    for item in stream:
        try:
            yield Ok(value=func(item))
        except Exception as e:
            yield Err(exception=e, context=item)


def _map_ok(stream: Iterator[Any], func: Callable[[Any], Any]) -> Iterator[Any]:
    for item in stream:
        if isinstance(item, Err):
            yield item
        else:
            yield Ok(value=func(item.value))


def _try_map_ok(stream: Iterator[Any], func: Callable[[Any], Any]) -> Iterator[Any]:
    for item in stream:
        if isinstance(item, Err):
            yield item
        else:
            try:
                yield Ok(value=func(item.value))
            except Exception as e:
                yield Err(exception=e, context=item.value)


def _read_source(source: Source[Any]) -> Iterator[Any]:
    """Read *source*, deferred until the stream is consumed.

    ``read()`` may open a file or a connection, so it must not run while
    the generator chain is merely being built.
    """
    yield from source.read()


def _reduce_lazy(
    stream: Iterator[Any], zero: Any, combine: Callable[[Any, Any], Any]
) -> Iterator[Any]:
    """Fold *stream* into a single element, deferred until consumed.

    The fold itself is not incremental — it drains the upstream — but it
    does so on first ``next()`` rather than when the chain is built, so
    ``evaluate()`` stays free of side effects.
    """
    yield _reduce(combine, stream, zero)


def _to_sink(
    stream: Iterator[Any], sink: Sink[Any], max_concurrency: int = 1
) -> Iterator[Any]:
    """Write every element of *stream* to *sink*, yielding nothing.

    Writes are dispatched *max_concurrency* at a time via a thread pool —
    ``Sink.write`` is a blocking call (``KGWriterSink.write`` runs its own
    ``asyncio.run()`` per call), so concurrency here comes from threads, not
    an event loop.
    """
    if max_concurrency <= 1:
        for item in stream:
            sink.write(item)
        yield from ()
        return
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        for batch in _batched(stream, max_concurrency):
            for future in [executor.submit(sink.write, item) for item in batch]:
                future.result()
    yield from ()


def _flatten_ok(stream: Iterator[Any]) -> Iterator[Any]:
    for item in stream:
        if isinstance(item, Err):
            yield item
        else:
            yield from (Ok(value=v) for v in item.value)


def _on_error(stream: Iterator[Any], handler: Callable[[Err], None]) -> Iterator[Any]:
    for item in stream:
        if isinstance(item, Err):
            handler(item)
        else:
            yield item.value


# ---------------------------------------------------------------------------
# Async evaluation helpers
# ---------------------------------------------------------------------------
# The async mirror of the local helpers above.  Synchronous operators apply
# their function inline in an ``async`` generator (``map``, ``filter``, …);
# the three ``*_async_chunked`` operators await their function concurrently.
# Order and error semantics are identical to :class:`LocalInterpreter`.


async def _aempty() -> AsyncIterator[Any]:
    """An async stream that yields nothing (mirrors the sync ``iter(())``)."""
    if False:
        yield  # pragma: no cover


async def _asource(source: Source[Any]) -> AsyncIterator[Any]:
    """Yield the elements of a sync *source* from inside an async stream.

    The generator body is deferred until first ``__anext__``, so ``read()``
    does not run while the stream is merely being built.
    """
    for item in source.read():
        yield item


async def _amap(
    stream: AsyncIterator[Any], func: Callable[[Any], Any]
) -> AsyncIterator[Any]:
    async for item in stream:
        yield func(item)


async def _aflat_map(
    stream: AsyncIterator[Any], func: Callable[[Any], Iterable[Any]]
) -> AsyncIterator[Any]:
    async for item in stream:
        for sub in func(item):
            yield sub


async def _atry_map(
    stream: AsyncIterator[Any], func: Callable[[Any], Any]
) -> AsyncIterator[Any]:
    async for item in stream:
        try:
            yield Ok(value=func(item))
        except Exception as e:
            yield Err(exception=e, context=item)


async def _afilter(
    stream: AsyncIterator[Any], predicate: Callable[[Any], bool]
) -> AsyncIterator[Any]:
    async for item in stream:
        if predicate(item):
            yield item


async def _atake(stream: AsyncIterator[Any], n: int) -> AsyncIterator[Any]:
    i = 0
    async for item in stream:
        if i >= n:
            break
        yield item
        i += 1


async def _atake_while(
    stream: AsyncIterator[Any], predicate: Callable[[Any], bool]
) -> AsyncIterator[Any]:
    async for item in stream:
        if not predicate(item):
            break
        yield item


async def _askip(stream: AsyncIterator[Any], n: int) -> AsyncIterator[Any]:
    i = 0
    async for item in stream:
        if i < n:
            i += 1
            continue
        yield item


async def _agrouped(stream: AsyncIterator[Any], size: int) -> AsyncIterator[Any]:
    batch: list[Any] = []
    async for item in stream:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


async def _guarded(
    sem: asyncio.Semaphore, func: Callable[[Any], Awaitable[Any]], item: Any
) -> Any:
    """Apply async *func* to *item* under *sem* (bounds concurrent calls)."""
    async with sem:
        return await func(item)


async def _amap_chunked(
    stream: AsyncIterator[Any],
    func: Callable[[Any], Awaitable[Any]],
    batch_size: int,
    sem: asyncio.Semaphore,
) -> AsyncIterator[Any]:
    async for chunk in _agrouped(stream, batch_size):
        results = await asyncio.gather(*(_guarded(sem, func, item) for item in chunk))
        for r in results:
            yield r


async def _atry_map_chunked(
    stream: AsyncIterator[Any],
    func: Callable[[Any], Awaitable[Any]],
    batch_size: int,
    sem: asyncio.Semaphore,
) -> AsyncIterator[Any]:
    async for chunk in _agrouped(stream, batch_size):
        raw = await asyncio.gather(
            *(_guarded(sem, func, item) for item in chunk), return_exceptions=True
        )
        for item, r in zip(chunk, raw):
            if isinstance(r, Exception):
                yield Err(exception=r, context=item)
            elif isinstance(r, BaseException):
                raise r
            else:
                yield Ok(value=r)


async def _atry_map_ok_chunked(
    stream: AsyncIterator[Any],
    func: Callable[[Any], Awaitable[Any]],
    batch_size: int,
    sem: asyncio.Semaphore,
) -> AsyncIterator[Any]:
    async for chunk in _agrouped(stream, batch_size):

        async def _process(item: Any) -> Ok[Any] | Err:
            if isinstance(item, Err):
                return item
            try:
                return Ok(value=await _guarded(sem, func, item.value))
            except Exception as e:
                return Err(exception=e, context=item.value)

        results = await asyncio.gather(*(_process(item) for item in chunk))
        for r in results:
            yield r


async def _amap_ok(
    stream: AsyncIterator[Any], func: Callable[[Any], Any]
) -> AsyncIterator[Any]:
    async for item in stream:
        if isinstance(item, Err):
            yield item
        else:
            yield Ok(value=func(item.value))


async def _atry_map_ok(
    stream: AsyncIterator[Any], func: Callable[[Any], Any]
) -> AsyncIterator[Any]:
    async for item in stream:
        if isinstance(item, Err):
            yield item
        else:
            try:
                yield Ok(value=func(item.value))
            except Exception as e:
                yield Err(exception=e, context=item.value)


async def _aflatten_ok(stream: AsyncIterator[Any]) -> AsyncIterator[Any]:
    async for item in stream:
        if isinstance(item, Err):
            yield item
        else:
            for v in item.value:
                yield Ok(value=v)


async def _afilter_ok(stream: AsyncIterator[Any]) -> AsyncIterator[Any]:
    async for item in stream:
        if isinstance(item, Ok):
            yield item.value


async def _aon_error(
    stream: AsyncIterator[Any], handler: Callable[[Err], None]
) -> AsyncIterator[Any]:
    async for item in stream:
        if isinstance(item, Err):
            handler(item)
        else:
            yield item.value


async def _areduce(
    stream: AsyncIterator[Any], zero: Any, combine: Callable[[Any, Any], Any]
) -> AsyncIterator[Any]:
    acc = zero
    async for item in stream:
        acc = combine(acc, item)
    yield acc


async def _ato_sink(
    stream: AsyncIterator[Any], sink: Sink[Any], max_concurrency: int = 1
) -> AsyncIterator[Any]:
    """Write elements to *sink* off-loop, in threads, *max_concurrency* at a time.

    ``Sink.write`` is a synchronous contract that some implementations
    (e.g. ``KGWriterSink``) satisfy with their own ``asyncio.run()`` call.
    Calling it directly here — inside the interpreter's own running loop —
    would raise ``RuntimeError: asyncio.run() cannot be called from a
    running event loop``.  Running it in a thread sidesteps that; batching
    lets several writes overlap instead of completing strictly one at a time.
    """
    async for batch in _agrouped(stream, max_concurrency):
        await asyncio.gather(*(asyncio.to_thread(sink.write, item) for item in batch))
    if False:
        yield  # pragma: no cover


# ---------------------------------------------------------------------------
# Async interpreter
# ---------------------------------------------------------------------------


class AsyncInterpreter(Interpreter):
    """Evaluate a pipeline on a single event loop with bounded concurrency.

    :meth:`evaluate` returns an async iterator over the pipeline's output;
    the caller drains it inside a running event loop (``asyncio.run`` at the
    top level).  Unlike :class:`LocalInterpreter`, the whole chain shares
    one event loop, so:

    * async clients that bind to a loop (``httpx.AsyncClient``, …) can be
      created *outside* the operator functions and shared across the run,
      and
    * several pipelines (e.g. one :class:`SimpleKGPipeline` per shard) can
      be evaluated concurrently with ``asyncio.gather`` over their streams.

    The ``*_async_chunked`` operators run their calls concurrently, capped
    by *concurrency*; every synchronous operator applies element by element
    in traversal order with the same semantics as the local interpreter.

    Args:
        concurrency: Maximum number of items any async operator may process
            at once; the cap is shared by every async operator in the graph.
            Defaults to 100 (the local interpreter's default
            ``map_batch_size``).  Lower it to bound in-flight LLM calls.
    """

    def __init__(self, concurrency: int = 100) -> None:
        if concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {concurrency!r}")
        self.concurrency = concurrency

    async def collect(self, pipeline: Pipeline[Any] | ResultPipeline[Any]) -> list[Any]:
        """Evaluate *pipeline* and materialise its output into a list.

        Convenience for ``[item async for item in evaluate(pipeline)]``.
        A pipeline ending in a sink returns ``[]``; the writes happen as the
        stream drains.
        """
        return [item async for item in self.evaluate(pipeline)]

    def evaluate(
        self, pipeline: Pipeline[Any] | ResultPipeline[Any]
    ) -> AsyncIterator[Any]:
        """Build the async stream for *pipeline*; no user code runs yet.

        Returns an async iterator.  A pipeline ending in a sink yields
        nothing but writes to the sink as it is drained.
        """
        sem = asyncio.Semaphore(self.concurrency)
        stream: AsyncIterator[Any] = _aempty()
        for op in pipeline.pipeline_operators:
            match op:
                case SourceOp(source=source):
                    stream = _asource(source)
                case Map(func=func):
                    stream = _amap(stream, func)
                case FlatMap(func=func):
                    stream = _aflat_map(stream, func)
                case Filter(predicate=predicate):
                    stream = _afilter(stream, predicate)
                case Take(n=n):
                    stream = _atake(stream, n)
                case TakeWhile(predicate=predicate):
                    stream = _atake_while(stream, predicate)
                case Skip(n=n):
                    stream = _askip(stream, n)
                case Grouped(size=size):
                    stream = _agrouped(stream, size)
                case ReduceOp(zero=zero, combine=combine):
                    stream = _areduce(stream, zero, combine)
                case MapAsyncChunked(func=func, map_batch_size=batch_size):
                    stream = _amap_chunked(stream, func, batch_size, sem)
                case TryMap(func=func):
                    stream = _atry_map(stream, func)
                case TryMapAsyncChunked(func=func, map_batch_size=batch_size):
                    stream = _atry_map_chunked(stream, func, batch_size, sem)
                case MapOk(func=func):
                    stream = _amap_ok(stream, func)
                case FlattenOk():
                    stream = _aflatten_ok(stream)
                case TryMapOk(func=func):
                    stream = _atry_map_ok(stream, func)
                case TryMapOkAsyncChunked(func=func, map_batch_size=batch_size):
                    stream = _atry_map_ok_chunked(stream, func, batch_size, sem)
                case FilterOk():
                    stream = _afilter_ok(stream)
                case OnError(handler=handler):
                    stream = _aon_error(stream, handler)
                case SinkOp(sink=sink, max_concurrency=max_concurrency):
                    stream = _ato_sink(stream, sink, max_concurrency)
                case _:  # pragma: no cover
                    raise TypeError(f"Unknown operator: {op!r}")
        return stream


# ---------------------------------------------------------------------------
# Local interpreter
# ---------------------------------------------------------------------------


class LocalInterpreter(Interpreter):
    """Evaluate a pipeline in the current process with lazy generators.

    Folds the operator list into a single generator chain.  Building the
    chain runs no user code: every operator — including ``ReduceOp`` and
    ``SinkOp``, which both drain their upstream — defers its work until the
    returned stream is consumed.  The chunked-async operators are the one
    exception to incremental evaluation: each chunk blocks in
    ``asyncio.run`` while it is being produced.
    """

    def evaluate(self, pipeline: Pipeline[Any] | ResultPipeline[Any]) -> Iterator[Any]:
        stream: Iterator[Any] = iter(())
        for op in pipeline.pipeline_operators:
            match op:
                case SourceOp(source=source):
                    stream = _read_source(source)
                case Map(func=func):
                    stream = map(func, stream)
                case FlatMap(func=func):
                    stream = itertools.chain.from_iterable(map(func, stream))
                case Filter(predicate=predicate):
                    stream = filter(predicate, stream)
                case Take(n=n):
                    stream = itertools.islice(stream, n)
                case TakeWhile(predicate=predicate):
                    stream = itertools.takewhile(predicate, stream)
                case Skip(n=n):
                    stream = itertools.islice(stream, n, None)
                case Grouped(size=size):
                    stream = _batched(stream, size)
                case ReduceOp(zero=zero, combine=combine):
                    stream = _reduce_lazy(stream, zero, combine)
                case MapAsyncChunked(func=func, map_batch_size=batch_size):
                    stream = _iter_chunked_async(
                        stream, batch_size, _make_map_chunk_coro(func)
                    )
                case TryMap(func=func):
                    stream = _try_map(stream, func)
                case TryMapAsyncChunked(func=func, map_batch_size=batch_size):
                    stream = _iter_chunked_async(
                        stream, batch_size, _make_try_chunk_coro(func)
                    )
                case MapOk(func=func):
                    stream = _map_ok(stream, func)
                case FlattenOk():
                    stream = _flatten_ok(stream)
                case TryMapOk(func=func):
                    stream = _try_map_ok(stream, func)
                case TryMapOkAsyncChunked(func=func, map_batch_size=batch_size):
                    stream = _iter_chunked_async(
                        stream, batch_size, _make_try_ok_chunk_coro(func)
                    )
                case FilterOk():
                    stream = (item.value for item in stream if isinstance(item, Ok))
                case OnError(handler=handler):
                    stream = _on_error(stream, handler)
                case SinkOp(sink=sink, max_concurrency=max_concurrency):
                    stream = _to_sink(stream, sink, max_concurrency)
                case _:  # pragma: no cover
                    raise TypeError(f"Unknown operator: {op!r}")
        return stream

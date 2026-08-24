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

A future ``AsyncInterpreter`` (whole chain on a single event loop) would
lift both restrictions; the operator-graph representation already supports
it.
"""

from __future__ import annotations

import asyncio
import itertools
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Coroutine, Iterable, Iterator
from functools import reduce as _reduce
from typing import TYPE_CHECKING, Any, TypeVar

from neo4j_graphrag.pipeline.operators import (
    Filter,
    FilterOk,
    Operator,
    FlatMap,
    FlatMapOk,
    Grouped,
    Map,
    MapAsyncChunked,
    MapOk,
    OnError,
    PartitionBranch,
    PartitionState,
    ReduceOp,
    SinkOp,
    Skip,
    SourceOp,
    Take,
    TakeWhile,
    TeeBranch,
    TryFlatMap,
    TryFlatMapAsyncChunked,
    TryFlatMapOk,
    TryFlatMapOkAsyncChunked,
    TryMap,
    TryMapAsyncChunked,
    TryMapOk,
    TryMapOkAsyncChunked,
)
from neo4j_graphrag.pipeline.result import Err, Ok

if TYPE_CHECKING:
    from neo4j_graphrag.pipeline.pipeline import Pipeline, ResultPipeline

__all__ = ["Interpreter", "LocalInterpreter"]

_T = TypeVar("_T")
_U = TypeVar("_U")


class Interpreter(ABC):
    """Base class for pipeline interpreters.

    An interpreter executes a pipeline definition and returns the resulting
    stream.  Whether evaluation is lazy or eager, local or distributed, is
    up to the implementation.
    """

    @abstractmethod
    def evaluate(self, pipeline: Pipeline[Any] | ResultPipeline[Any]) -> Iterator[Any]:
        """Evaluate *pipeline* and return its output stream.

        If the pipeline ends in a sink, the stream is consumed, written to
        the sink, and an empty iterator is returned.
        """


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
        for r in raw:
            if isinstance(r, Exception):
                results.append(Err(exception=r))
            elif isinstance(r, BaseException):
                # SystemExit, KeyboardInterrupt, CancelledError: fatal, re-raise.
                raise r
            else:
                results.append(Ok(value=r))
        return results

    return _run_chunk


def _make_try_flat_chunk_coro(
    func: Callable[[_T], Awaitable[Iterable[_U]]],
) -> Callable[[list[_T]], Coroutine[None, None, list[Ok[_U] | Err]]]:
    async def _run_chunk(chunk: list[_T]) -> list[Ok[_U] | Err]:
        raw = await asyncio.gather(
            *[func(item) for item in chunk], return_exceptions=True
        )
        results: list[Ok[_U] | Err] = []
        for r in raw:
            if isinstance(r, Exception):
                results.append(Err(exception=r))
            elif isinstance(r, BaseException):
                raise r
            else:
                results.extend(Ok(value=v) for v in r)
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
            return Err(exception=e)

    async def _run_chunk(chunk: list[Any]) -> list[Any]:
        return list(await asyncio.gather(*[_process(item) for item in chunk]))

    return _run_chunk


def _make_try_flat_ok_chunk_coro(
    func: Callable[[Any], Awaitable[Iterable[Any]]],
) -> Callable[[list[Any]], Coroutine[None, None, list[Any]]]:
    async def _process(item: Ok[Any] | Err) -> list[Ok[Any] | Err]:
        if isinstance(item, Err):
            return [item]
        try:
            return [Ok(value=v) for v in await func(item.value)]
        except Exception as e:
            return [Err(exception=e)]

    async def _run_chunk(chunk: list[Any]) -> list[Any]:
        nested = await asyncio.gather(*[_process(item) for item in chunk])
        return [result for sublist in nested for result in sublist]

    return _run_chunk


def _try_map(stream: Iterator[Any], func: Callable[[Any], Any]) -> Iterator[Any]:
    for item in stream:
        try:
            yield Ok(value=func(item))
        except Exception as e:
            yield Err(exception=e)


def _try_flat_map(stream: Iterator[Any], func: Callable[[Any], Any]) -> Iterator[Any]:
    for item in stream:
        try:
            yield from (Ok(value=v) for v in func(item))
        except Exception as e:
            yield Err(exception=e)


def _map_ok(stream: Iterator[Any], func: Callable[[Any], Any]) -> Iterator[Any]:
    for item in stream:
        if isinstance(item, Err):
            yield item
        else:
            yield Ok(value=func(item.value))


def _flat_map_ok(stream: Iterator[Any], func: Callable[[Any], Any]) -> Iterator[Any]:
    for item in stream:
        if isinstance(item, Err):
            yield item
        else:
            yield from (Ok(value=v) for v in func(item.value))


def _try_map_ok(stream: Iterator[Any], func: Callable[[Any], Any]) -> Iterator[Any]:
    for item in stream:
        if isinstance(item, Err):
            yield item
        else:
            try:
                yield Ok(value=func(item.value))
            except Exception as e:
                yield Err(exception=e)


def _try_flat_map_ok(
    stream: Iterator[Any], func: Callable[[Any], Any]
) -> Iterator[Any]:
    for item in stream:
        if isinstance(item, Err):
            yield item
        else:
            try:
                yield from (Ok(value=v) for v in func(item.value))
            except Exception as e:
                yield Err(exception=e)


def _on_error(stream: Iterator[Any], handler: Callable[[Err], None]) -> Iterator[Any]:
    for item in stream:
        if isinstance(item, Err):
            handler(item)
        else:
            yield item.value


def _partition(
    stream: Iterator[Any], state: PartitionState, want_ok: bool
) -> Iterator[Any]:
    """Return the success or failure branch of a partition.

    Materialises the upstream on first use so both branches can be consumed
    independently without double-iteration.
    """
    if state.ok_values is None or state.err_values is None:
        ok_values: list[Any] = []
        err_values: list[Err] = []
        for item in stream:
            if isinstance(item, Ok):
                ok_values.append(item.value)
            else:
                err_values.append(item)
        state.ok_values = ok_values
        state.err_values = err_values
        return iter(ok_values if want_ok else err_values)
    assert state.ok_values is not None and state.err_values is not None
    return iter(state.ok_values if want_ok else state.err_values)


# ---------------------------------------------------------------------------
# Local interpreter
# ---------------------------------------------------------------------------


class LocalInterpreter(Interpreter):
    """Evaluate a pipeline in the current process with lazy generators.

    Folds the operator list into a single generator chain.  Apart from
    ``ReduceOp`` (which folds its upstream eagerly), ``PartitionBranch``
    (which materialises its upstream) and the blocking chunked-async
    operators, no work happens until the returned stream is consumed.
    """

    def evaluate(self, pipeline: Pipeline[Any] | ResultPipeline[Any]) -> Iterator[Any]:
        operators = pipeline.pipeline_operators
        start = self._resume_point(operators)
        if start is None:
            return iter(())
        stream, start_index = start
        for op in operators[start_index:]:
            match op:
                case SourceOp(source=source):
                    stream = iter(source.read())
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
                    stream = iter([_reduce(combine, stream, zero)])
                case MapAsyncChunked(func=func, map_batch_size=batch_size):
                    stream = _iter_chunked_async(
                        stream, batch_size, _make_map_chunk_coro(func)
                    )
                case TryMap(func=func):
                    stream = _try_map(stream, func)
                case TryFlatMap(func=func):
                    stream = _try_flat_map(stream, func)
                case TryMapAsyncChunked(func=func, map_batch_size=batch_size):
                    stream = _iter_chunked_async(
                        stream, batch_size, _make_try_chunk_coro(func)
                    )
                case TryFlatMapAsyncChunked(func=func, map_batch_size=batch_size):
                    stream = _iter_chunked_async(
                        stream, batch_size, _make_try_flat_chunk_coro(func)
                    )
                case MapOk(func=func):
                    stream = _map_ok(stream, func)
                case FlatMapOk(func=func):
                    stream = _flat_map_ok(stream, func)
                case TryMapOk(func=func):
                    stream = _try_map_ok(stream, func)
                case TryFlatMapOk(func=func):
                    stream = _try_flat_map_ok(stream, func)
                case TryMapOkAsyncChunked(func=func, map_batch_size=batch_size):
                    stream = _iter_chunked_async(
                        stream, batch_size, _make_try_ok_chunk_coro(func)
                    )
                case TryFlatMapOkAsyncChunked(func=func, map_batch_size=batch_size):
                    stream = _iter_chunked_async(
                        stream, batch_size, _make_try_flat_ok_chunk_coro(func)
                    )
                case FilterOk():
                    stream = (item.value for item in stream if isinstance(item, Ok))
                case OnError(handler=handler):
                    stream = _on_error(stream, handler)
                case TeeBranch(state=state, index=index):
                    if state.iterators is None:
                        state.iterators = itertools.tee(stream, state.n)
                    stream = state.iterators[index]
                case PartitionBranch(state=state, want_ok=want_ok):
                    stream = _partition(stream, state, want_ok)
                case SinkOp(sink=sink):
                    for item in stream:
                        sink.write(item)
                    return iter(())
                case _:  # pragma: no cover
                    raise TypeError(f"Unknown operator: {op!r}")
        return stream

    @staticmethod
    def _resume_point(
        operators: list[Operator],
    ) -> tuple[Iterator[Any], int] | None:
        """Find the latest already-initialised branch point, if any.

        When evaluating one branch of a ``tee``/``partition`` *after*
        another branch has already been (partially) evaluated, the shared
        state token already holds the realised upstream.  Re-folding the
        prefix would call ``source.read()`` again; instead, resume directly
        from the shared state.

        Returns ``(stream, start_index)`` — the stream to continue from and
        the index of the first operator still to apply — or ``None`` if the
        operator list is empty.
        """
        if not operators:
            return None
        stream: Iterator[Any] = iter(())
        start_index = 0
        for i, op in enumerate(operators):
            if isinstance(op, TeeBranch) and op.state.iterators is not None:
                stream = op.state.iterators[op.index]
                start_index = i + 1
            elif (
                isinstance(op, PartitionBranch)
                and op.state.ok_values is not None
                and op.state.err_values is not None
            ):
                stream = iter(op.state.ok_values if op.want_ok else op.state.err_values)
                start_index = i + 1
        return stream, start_index

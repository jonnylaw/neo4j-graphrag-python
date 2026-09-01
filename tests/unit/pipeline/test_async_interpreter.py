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
"""Parity tests for :class:`AsyncInterpreter` against :class:`LocalInterpreter`.

Both interpreters evaluate the same operator graph; they only differ in how
async stages are driven.  These tests pin that: every pipeline shape produces
identical data (and identical ``Err`` positions) regardless of interpreter.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Any

import pytest

from neo4j_graphrag.pipeline import (
    AsyncInterpreter,
    Pipeline,
    ResultPipeline,
    Sink,
    Source,
)
from neo4j_graphrag.pipeline import operators as ops
from neo4j_graphrag.pipeline.result import Err, Ok


class _InMemorySource(Source[int]):
    def __init__(self, items: list[int]) -> None:
        self._items = items

    def read(self) -> Iterable[int]:
        return iter(self._items)


class _CaptureSink(Sink[int]):
    def __init__(self) -> None:
        self.received: list[int] = []

    def write(self, element: int) -> None:
        self.received.append(element)


async def _double(x: int) -> int:
    return x * 2


async def _fail_on_even(x: int) -> int:
    if x % 2 == 0:
        raise ValueError(f"even: {x}")
    return x


async def _duplicate(x: int) -> list[int]:
    return [x, x * 10]


def _fail_on_two_sync(x: int) -> int:
    if x == 2:
        raise ValueError("two")
    return x


def _sync(result: Any) -> Any:
    """Run a coroutine to completion (no running loop in the test thread)."""
    return asyncio.run(result)


def _normalise(items: list[Any]) -> list[Any]:
    """Structural form of a result stream, comparable across interpreters.

    ``Ok``/``Err`` dataclasses compare equal only when their exception does;
    exceptions compare by identity, so two interpreters' ``Err`` values never
    ``==``.  Normalise to ``("ok", value)`` / ``("err", name, context)``.
    """

    def one(item: Any) -> Any:
        if isinstance(item, Ok):
            return ("ok", item.value)
        if isinstance(item, Err):
            return ("err", type(item.exception).__name__, item.context)
        return ("raw", item)

    return [one(item) for item in items]


def _assert_parity(pipe: Pipeline[Any] | ResultPipeline[Any]) -> None:
    """Local and async interpreters produce the same stream for *pipe*."""
    local = _normalise(pipe.collect())
    async_ = _normalise(_sync(AsyncInterpreter().collect(pipe)))
    assert async_ == local


def _sink_graph(pipe: Pipeline[Any], sink: Sink[Any]) -> Pipeline[Any]:
    """Attach a sink node to *pipe* without draining (unlike ``to_sink``)."""
    tail = pipe.pipeline_operators[-1]
    return Pipeline._wrap(ops.SinkOp(prev=tail, sink=sink))


# ---------------------------------------------------------------------------
# Sync operators
# ---------------------------------------------------------------------------


class TestAsyncSyncOperators:
    def test_map(self) -> None:
        _assert_parity(Pipeline([1, 2, 3]).map(lambda x: x * 2))

    def test_flat_map(self) -> None:
        _assert_parity(Pipeline([1, 2, 3]).flat_map(lambda x: [x, x * 10]))

    def test_filter(self) -> None:
        _assert_parity(Pipeline([1, 2, 3, 4]).filter(lambda x: x % 2 == 0))

    def test_take_skip_grouped_reduce(self) -> None:
        pipe = (
            Pipeline([1, 2, 3, 4, 5, 6, 7, 8])
            .skip(1)
            .take(6)
            .grouped(3)
            .flat_map(lambda batch: batch)
            .reduce(0, lambda a, x: a + x)
        )
        _assert_parity(pipe)

    def test_take_while(self) -> None:
        _assert_parity(Pipeline([1, 2, 3, 1]).take_while(lambda x: x < 3))


# ---------------------------------------------------------------------------
# Async operators
# ---------------------------------------------------------------------------


class TestAsyncChunkedOperators:
    def test_map_async_chunked(self) -> None:
        _assert_parity(
            Pipeline([1, 2, 3, 4, 5]).map_async_chunked(_double, map_batch_size=2)
        )

    def test_map_async_chunked_safe_partial_failure(self) -> None:
        _assert_parity(
            Pipeline([1, 2, 3, 4]).map_async_chunked_safe(
                _fail_on_even, map_batch_size=1
            )
        )


class TestAsyncResultCombinators:
    def test_map_ok_and_flatten_ok(self) -> None:
        pipe = (
            Pipeline([1, 2, 3])
            .map_safe(_fail_on_two_sync)
            .map_ok(lambda x: [x, x])
            .flatten_ok()
        )
        _assert_parity(pipe)

    def test_map_safe_chain(self) -> None:
        _assert_parity(
            Pipeline([1, 2, 3, 4])
            .map_safe(_fail_on_two_sync)
            .map_safe(lambda x: x * 10)
        )

    def test_map_async_chunked_safe_on_values(self) -> None:
        pipe = (
            Pipeline([1, 2, 3])
            .map_safe(lambda x: x)
            .map_async_chunked_safe(_fail_on_even, map_batch_size=2)
        )
        _assert_parity(pipe)

    def test_on_error(self) -> None:
        errors_local: list[Err] = []
        errors_async: list[Err] = []
        local = (
            Pipeline([1, 2, 3])
            .map_safe(_fail_on_two_sync)
            .on_error(errors_local.append)
            .collect()
        )
        async_ = _sync(
            AsyncInterpreter().collect(
                Pipeline([1, 2, 3])
                .map_safe(_fail_on_two_sync)
                .on_error(errors_async.append)
            )
        )
        assert async_ == local == [1, 3]
        assert len(errors_local) == len(errors_async) == 1


# ---------------------------------------------------------------------------
# Sink
# ---------------------------------------------------------------------------


class TestAsyncSink:
    def test_sink_writes_as_it_drains(self) -> None:
        sink_local = _CaptureSink()
        sink_async = _CaptureSink()

        _sink_graph(Pipeline([1, 2, 3]).map(lambda x: x * 2), sink_local).collect()

        async def _run() -> None:
            async for _ in AsyncInterpreter().evaluate(
                _sink_graph(Pipeline([1, 2, 3]).map(lambda x: x * 2), sink_async)
            ):
                pass  # drain

        asyncio.run(_run())
        assert sink_async.received == sink_local.received == [2, 4, 6]


# ---------------------------------------------------------------------------
# Concurrency semantics
# ---------------------------------------------------------------------------


class TestConcurrencySemantics:
    def test_semaphore_bounds_in_flight(self) -> None:
        """The shared semaphore caps concurrent async calls at concurrency."""
        running = 0
        peak = 0

        async def _work(x: int) -> int:
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.01)
            running -= 1
            return x

        pipe = Pipeline(range(10)).map_async_chunked(_work, map_batch_size=10)
        result = _sync(AsyncInterpreter(concurrency=3).collect(pipe))
        assert result == list(range(10))
        assert peak <= 3

    def test_concurrency_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="concurrency must be >= 1"):
            AsyncInterpreter(concurrency=0)


# ---------------------------------------------------------------------------
# Laziness / loop sharing
# ---------------------------------------------------------------------------


class TestAsyncLoopSemantics:
    def test_builds_no_user_code(self) -> None:
        """evaluate() builds the stream; nothing runs until it is drained."""
        calls: list[int] = []

        def _record(x: int) -> int:
            calls.append(x)
            return x

        async def _probe() -> None:
            AsyncInterpreter().evaluate(Pipeline([1, 2, 3]).map(_record))
            assert calls == []

        asyncio.run(_probe())

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
"""Source protocol for the pipeline DSL.

A ``Source`` is the entry point for pipeline data (e.g. a Neo4j query, a
file glob, or an in-memory sequence for testing).  Any object implementing
``read()`` satisfies the protocol — no inheritance is required.

Variance note
-------------
``Source`` is covariant in ``T_co``: a ``Source[Path]`` can be used wherever
a ``Source[object]`` is expected because a source only *produces* values —
it never consumes them.  This mirrors ``collections.abc.Iterable[T_co]``
and is the dual of the contravariant
:class:`~neo4j_graphrag.pipeline.sink.Sink`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, TypeVar, runtime_checkable

__all__ = ["Source"]

T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class Source(Protocol[T_co]):
    """Protocol satisfied by any object that can emit pipeline elements.

    Implementors must provide a ``read`` method that returns an iterable of
    elements.  ``read`` is called each time the pipeline is evaluated, so
    sources that can only be consumed once (e.g. wrapping a generator)
    produce single-use pipelines.

    The type parameter ``T_co`` is covariant: a ``Source[Path]`` is a valid
    ``Source[object]``.

    Example::

        class InMemorySource:
            def __init__(self, items: list[int]) -> None:
                self._items = items

            def read(self) -> list[int]:
                return list(self._items)
    """

    def read(self) -> Iterable[T_co]:
        """Return an iterable of elements to feed into the pipeline."""
        ...

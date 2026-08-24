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
"""Sink protocol for the pipeline DSL.

A ``Sink`` is the terminal destination for pipeline output (e.g. a file or
a Neo4j database).  Any object implementing ``write(element)`` satisfies
the protocol — no inheritance is required.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

__all__ = ["Sink"]

T_contra = TypeVar("T_contra", contravariant=True)


@runtime_checkable
class Sink(Protocol[T_contra]):
    """Protocol satisfied by any object that can receive pipeline elements.

    Implementors must provide a ``write`` method that accepts a single
    element of type ``T_contra`` and persists it to the backing store.

    Example::

        class InMemorySink:
            def __init__(self) -> None:
                self.received: list[Any] = []

            def write(self, element: Any) -> None:
                self.received.append(element)
    """

    def write(self, element: T_contra) -> None:
        """Write a single *element* to the backing store."""
        ...

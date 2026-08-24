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
"""Concrete :class:`~neo4j_graphrag.pipeline.sink.Sink` implementations for
graph-shaped pipeline output.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from neo4j_graphrag.components.kg_writer import KGWriter, KGWriterModel
from neo4j_graphrag.components.types import LexicalGraphConfig, Neo4jGraph
from neo4j_graphrag.pipeline.sink import Sink

__all__ = ["InMemoryGraphSink", "KGWriterSink"]


class KGWriterSink(Sink[Neo4jGraph]):
    """Upsert every graph written to this sink through a :class:`KGWriter`.

    Each element is written as it arrives. Write batching — splitting a
    large graph into several database round-trips — is the writer's
    concern, not the pipeline's: :class:`~neo4j_graphrag.components.kg_writer.Neo4jWriter`
    chunks nodes and relationships internally (see its ``batch_size``).

    Args:
        writer: The writer performing the upsert.
        lexical_graph_config: Node labels and relationship types identifying
            lexical nodes, forwarded to the writer.
    """

    def __init__(
        self,
        writer: KGWriter,
        lexical_graph_config: Optional[LexicalGraphConfig] = None,
    ) -> None:
        self.writer = writer
        self.lexical_graph_config = lexical_graph_config or LexicalGraphConfig()
        self.results: list[KGWriterModel] = []

    def write(self, element: Neo4jGraph) -> None:
        """Upsert *element* and record the writer's result."""
        self.results.append(
            asyncio.run(
                self.writer.run(
                    graph=element,
                    lexical_graph_config=self.lexical_graph_config,
                )
            )
        )


class InMemoryGraphSink(Sink[Neo4jGraph]):
    """Accumulate every written graph into a single in-memory graph.

    Useful for inspecting or testing a pipeline without a database.
    """

    def __init__(self) -> None:
        self.graph = Neo4jGraph()

    def write(self, element: Neo4jGraph) -> None:
        self.graph.nodes.extend(element.nodes)
        self.graph.relationships.extend(element.relationships)

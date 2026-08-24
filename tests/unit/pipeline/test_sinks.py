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
"""Unit tests for :mod:`neo4j_graphrag.pipeline.sinks`."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from neo4j_graphrag.components.kg_writer import KGWriter, KGWriterModel
from neo4j_graphrag.components.types import (
    LexicalGraphConfig,
    Neo4jGraph,
    Neo4jNode,
    Neo4jRelationship,
)
from neo4j_graphrag.pipeline import Pipeline
from neo4j_graphrag.pipeline.sinks import InMemoryGraphSink, KGWriterSink


def _graph(node_id: str) -> Neo4jGraph:
    return Neo4jGraph(
        nodes=[Neo4jNode(id=node_id, label="Entity")],
        relationships=[
            Neo4jRelationship(start_node_id=node_id, end_node_id=node_id, type="SELF")
        ],
    )


def test_in_memory_sink_accumulates_across_writes() -> None:
    sink = InMemoryGraphSink()

    Pipeline([_graph("a"), _graph("b")]).to_sink(sink)

    assert [n.id for n in sink.graph.nodes] == ["a", "b"]
    assert len(sink.graph.relationships) == 2


def test_kg_writer_sink_writes_each_element_and_records_results() -> None:
    writer = MagicMock(spec=KGWriter)
    writer.run = AsyncMock(return_value=KGWriterModel(status="SUCCESS", metadata=None))
    config = LexicalGraphConfig(chunk_node_label="MyChunk")
    sink = KGWriterSink(writer, config)

    Pipeline([_graph("a"), _graph("b")]).to_sink(sink)

    assert writer.run.await_count == 2
    assert [w.status for w in sink.results] == ["SUCCESS", "SUCCESS"]
    written = [call.kwargs["graph"].nodes[0].id for call in writer.run.await_args_list]
    assert written == ["a", "b"]
    assert writer.run.await_args.kwargs["lexical_graph_config"] is config


def test_kg_writer_sink_defaults_to_standard_lexical_graph_config() -> None:
    writer = MagicMock(spec=KGWriter)
    writer.run = AsyncMock(return_value=KGWriterModel(status="SUCCESS", metadata=None))
    sink = KGWriterSink(writer)

    sink.write(_graph("a"))

    assert writer.run.await_args.kwargs["lexical_graph_config"] == LexicalGraphConfig()

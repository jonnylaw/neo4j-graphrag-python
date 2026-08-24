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
"""Unit tests for the DSL-based SimpleKGPipeline.

Components are replaced with mocks after construction; tests assert on the
dataflow wiring (which stages ran, with what data) and on error handling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import neo4j
import pytest
from fsspec import AbstractFileSystem

from neo4j_graphrag.components.graph_pruning import GraphPruningResult, PruningStats
from neo4j_graphrag.components.kg_writer import KGWriter, KGWriterModel
from neo4j_graphrag.components.schema import GraphSchema
from neo4j_graphrag.components.types import (
    DocumentInfo,
    LexicalGraphConfig,
    LoadedDocument,
    Neo4jGraph,
    Neo4jNode,
    TextChunk,
    TextChunks,
)
from neo4j_graphrag.embeddings import Embedder
from neo4j_graphrag.llm.base import LLMInterface
from neo4j_graphrag.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.pipeline.sinks import InMemoryGraphSink

CHUNKS = [TextChunk(text="chunk-0", index=0), TextChunk(text="chunk-1", index=1)]


def _async_mock(method: Any) -> AsyncMock:
    """Narrow a mocked async callable for assertions."""
    return cast(AsyncMock, method)


def _set_async_mock(obj: Any, name: str, **kwargs: Any) -> AsyncMock:
    """Replace *name* on *obj* with an ``AsyncMock`` (avoids method-assign)."""
    mock = AsyncMock(**kwargs)
    setattr(obj, name, mock)
    return mock


def _make_pipe(**kwargs: Any) -> SimpleKGPipeline:
    llm = MagicMock(spec=LLMInterface)
    llm.supports_structured_output = False
    driver = MagicMock(spec=neo4j.Driver)
    embedder = MagicMock(spec=Embedder)
    kwargs.setdefault("schema", GraphSchema.create_empty())
    # avoid the real Neo4jWriter constructor (it queries the driver for the
    # server version at init); tests re-wire the sink anyway
    kwargs.setdefault("kg_writer", MagicMock(spec=KGWriter))
    pipe = SimpleKGPipeline(llm=llm, driver=driver, embedder=embedder, **kwargs)

    pipe.file_loader = MagicMock()
    _set_async_mock(
        pipe.file_loader,
        "run",
        return_value=LoadedDocument(
            text="Some text.",
            document_info=DocumentInfo(path="doc.md"),
        ),
    )
    pipe.text_splitter = MagicMock()
    _set_async_mock(
        pipe.text_splitter,
        "run",
        return_value=TextChunks(chunks=[c.model_copy() for c in CHUNKS]),
    )
    pipe.chunk_embedder = MagicMock()
    _set_async_mock(pipe.chunk_embedder, "run", side_effect=lambda tc: tc)
    pipe.lexical_graph_builder = MagicMock()
    _set_async_mock(
        pipe.lexical_graph_builder,
        "run",
        side_effect=lambda text_chunks, document_info: MagicMock(
            graph=Neo4jGraph(
                nodes=[
                    Neo4jNode(id=text_chunks.chunks[0].uid, label="Chunk"),
                    Neo4jNode(id=document_info.uid, label="Document"),
                ]
            )
        ),
    )
    pipe.extractor = MagicMock()
    _set_async_mock(
        pipe.extractor,
        "extract_chunk",
        side_effect=lambda chunk, schema, examples, builder: Neo4jGraph(
            nodes=[Neo4jNode(id=f"e{chunk.index}", label="Entity")]
        ),
    )
    pipe.extractor.combine_chunk_graphs = MagicMock(side_effect=_combine)
    pipe.pruner = MagicMock()
    _set_async_mock(
        pipe.pruner,
        "run",
        side_effect=lambda graph, schema=None, lexical_graph_config=None: (
            GraphPruningResult(graph=graph, pruning_stats=PruningStats())
        ),
    )
    if pipe.resolver is not None:
        _set_async_mock(pipe.resolver, "run", return_value=None)
    return pipe


def _combine(lexical_graph: Any, chunk_graphs: list[Neo4jGraph]) -> Neo4jGraph:
    """The real ``combine_chunk_graphs`` behaviour, for the mocked extractor."""
    graph = Neo4jGraph()
    if lexical_graph is not None:
        graph.nodes.extend(lexical_graph.nodes)
        graph.relationships.extend(lexical_graph.relationships)
    for chunk_graph in chunk_graphs:
        graph.nodes.extend(chunk_graph.nodes)
        graph.relationships.extend(chunk_graph.relationships)
    return graph


def _with_memory_sink(pipe: SimpleKGPipeline) -> InMemoryGraphSink:
    sink = InMemoryGraphSink()
    pipe.sink = sink
    return sink


def _write_md(tmp_path: Path, name: str = "doc.md") -> str:
    """Create a markdown file under *tmp_path* and return its path."""
    path = tmp_path / name
    path.write_text("Some text.")
    return str(path)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_async_full_flow(tmp_path: Path) -> None:
    pipe = _make_pipe(perform_entity_resolution=True)
    sink = _with_memory_sink(pipe)

    result = await pipe.run_async(file_path=_write_md(tmp_path))

    # load and split once, then every later stage runs per chunk
    assert _async_mock(pipe.file_loader.run).await_count == 1
    assert _async_mock(pipe.text_splitter.run).await_count == 1
    assert _async_mock(pipe.chunk_embedder.run).await_count == 2
    assert _async_mock(pipe.lexical_graph_builder.run).await_count == 2
    assert _async_mock(pipe.extractor.extract_chunk).await_count == 2
    assert _async_mock(pipe.pruner.run).await_count == 2
    assert {n.id for n in sink.graph.nodes} >= {"e0", "e1"}
    assert pipe.resolver is not None
    assert _async_mock(pipe.resolver.run).await_count == 1
    assert result.errors == []


@pytest.mark.asyncio
async def test_lexical_graph_is_built_one_chunk_at_a_time(tmp_path: Path) -> None:
    pipe = _make_pipe(perform_entity_resolution=False)
    _with_memory_sink(pipe)

    await pipe.run_async(file_path=_write_md(tmp_path))

    for call in _async_mock(pipe.lexical_graph_builder.run).await_args_list:
        assert len(call.kwargs["text_chunks"].chunks) == 1


@pytest.mark.asyncio
async def test_next_chunk_relationship_links_consecutive_chunks(
    tmp_path: Path,
) -> None:
    pipe = _make_pipe(perform_entity_resolution=False)
    sink = _with_memory_sink(pipe)

    await pipe.run_async(file_path=_write_md(tmp_path))

    next_chunk_type = LexicalGraphConfig().next_chunk_relationship_type
    linked = [r for r in sink.graph.relationships if r.type == next_chunk_type]
    # two chunks -> exactly one NEXT_CHUNK edge, from the first to the second
    assert len(linked) == 1
    chunk_ids = [c.uid for c in _async_mock(pipe.text_splitter.run).return_value.chunks]
    assert (linked[0].start_node_id, linked[0].end_node_id) == tuple(chunk_ids)


@pytest.mark.asyncio
async def test_duplicate_document_nodes_collapse_on_merge(tmp_path: Path) -> None:
    pipe = _make_pipe(perform_entity_resolution=False)
    sink = _with_memory_sink(pipe)

    await pipe.run_async(file_path=_write_md(tmp_path))

    # each chunk graph repeats the Document node; the merge deduplicates it
    document_nodes = [n for n in sink.graph.nodes if n.label == "Document"]
    assert len(document_nodes) == 1


@pytest.mark.asyncio
async def test_run_async_no_entity_resolution(tmp_path: Path) -> None:
    pipe = _make_pipe(perform_entity_resolution=False)
    _with_memory_sink(pipe)

    result = await pipe.run_async(file_path=_write_md(tmp_path))

    assert pipe.resolver is None
    assert result.resolver is None


@pytest.mark.asyncio
async def test_run_async_loads_file_with_metadata(tmp_path: Path) -> None:
    pipe = _make_pipe(perform_entity_resolution=False)
    _with_memory_sink(pipe)
    file_path = _write_md(tmp_path)

    await pipe.run_async(file_path=file_path, document_metadata={"team": "docs"})

    load = _async_mock(pipe.file_loader.run)
    load.assert_awaited_once()
    assert load.await_args is not None
    call_kwargs = load.await_args.kwargs
    assert call_kwargs["filepath"] == file_path
    assert isinstance(call_kwargs["fs"], AbstractFileSystem)
    assert call_kwargs["metadata"] == {"team": "docs"}
    _async_mock(pipe.text_splitter.run).assert_awaited_once_with(text="Some text.")


@pytest.mark.asyncio
async def test_run_async_from_directory_loads_each_file(tmp_path: Path) -> None:
    pipe = _make_pipe(perform_entity_resolution=False)
    _with_memory_sink(pipe)
    _write_md(tmp_path, "a.md")
    _write_md(tmp_path, "b.md")

    await pipe.run_async(file_path=str(tmp_path))

    assert _async_mock(pipe.file_loader.run).await_count == 2
    assert _async_mock(pipe.text_splitter.run).await_count == 2


@pytest.mark.asyncio
async def test_default_sink_records_writer_results(tmp_path: Path) -> None:
    writer = MagicMock(spec=KGWriter)
    writer.run = AsyncMock(return_value=KGWriterModel(status="SUCCESS", metadata=None))
    pipe = _make_pipe(kg_writer=writer, perform_entity_resolution=False)

    result = await pipe.run_async(file_path=_write_md(tmp_path))

    assert [w.status for w in result.writer] == ["SUCCESS"]
    assert writer.run.await_count == 1


@pytest.mark.asyncio
async def test_all_chunks_are_written_in_a_single_call(tmp_path: Path) -> None:
    """The reduce merges every chunk graph; the sink sees one graph per run."""
    writer = MagicMock(spec=KGWriter)
    writer.run = AsyncMock(return_value=KGWriterModel(status="SUCCESS", metadata=None))
    pipe = _make_pipe(kg_writer=writer, perform_entity_resolution=False)

    result = await pipe.run_async(file_path=_write_md(tmp_path))

    writer.run.assert_awaited_once()
    written = writer.run.await_args.kwargs["graph"]
    assert {n.id for n in written.nodes} >= {"e0", "e1"}
    assert len(result.writer) == 1


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_invalid_map_batch_size_raises() -> None:
    with pytest.raises(ValueError, match="map_batch_size must be >= 1"):
        _make_pipe(map_batch_size=0)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_error_ignore_captures_chunk_failures(tmp_path: Path) -> None:
    pipe = _make_pipe(on_error="IGNORE", perform_entity_resolution=False)
    sink = _with_memory_sink(pipe)

    async def _extract(chunk: TextChunk, *args: Any) -> Neo4jGraph:
        if chunk.index == 1:
            raise ValueError("extraction failed")
        return Neo4jGraph(nodes=[Neo4jNode(id=f"e{chunk.index}", label="Entity")])

    _set_async_mock(pipe.extractor, "extract_chunk", side_effect=_extract)

    result = await pipe.run_async(file_path=_write_md(tmp_path))

    node_ids = {n.id for n in sink.graph.nodes}
    assert "e0" in node_ids
    assert "e1" not in node_ids
    assert len(result.errors) == 1
    assert isinstance(result.errors[0].exception, ValueError)


@pytest.mark.asyncio
async def test_on_error_ignore_captures_lexical_graph_failures(
    tmp_path: Path,
) -> None:
    pipe = _make_pipe(on_error="IGNORE", perform_entity_resolution=False)
    sink = _with_memory_sink(pipe)

    async def _build(text_chunks: TextChunks, document_info: Any) -> Any:
        if text_chunks.chunks[0].index == 1:
            raise ValueError("lexical graph failed")
        return MagicMock(
            graph=Neo4jGraph(
                nodes=[Neo4jNode(id=text_chunks.chunks[0].uid, label="Chunk")]
            )
        )

    _set_async_mock(pipe.lexical_graph_builder, "run", side_effect=_build)

    result = await pipe.run_async(file_path=_write_md(tmp_path))

    node_ids = {n.id for n in sink.graph.nodes}
    assert "e0" in node_ids
    assert "e1" not in node_ids
    # the failing chunk never reaches the extraction stage
    assert _async_mock(pipe.extractor.extract_chunk).await_count == 1
    assert len(result.errors) == 1
    assert isinstance(result.errors[0].exception, ValueError)


@pytest.mark.asyncio
async def test_errors_do_not_leak_between_runs(tmp_path: Path) -> None:
    pipe = _make_pipe(on_error="IGNORE", perform_entity_resolution=False)
    _with_memory_sink(pipe)
    _set_async_mock(pipe.extractor, "extract_chunk", side_effect=ValueError("boom"))

    first = await pipe.run_async(file_path=_write_md(tmp_path))
    second = await pipe.run_async(file_path=_write_md(tmp_path))

    assert len(first.errors) == 2
    # the second run reports only its own errors, not the first run's
    assert len(second.errors) == 2


@pytest.mark.asyncio
async def test_on_error_raise_aborts_run(tmp_path: Path) -> None:
    pipe = _make_pipe(on_error="RAISE", perform_entity_resolution=False)
    _with_memory_sink(pipe)
    _set_async_mock(pipe.extractor, "extract_chunk", side_effect=ValueError("boom"))

    with pytest.raises(ValueError, match="boom"):
        await pipe.run_async(file_path=_write_md(tmp_path))


@pytest.mark.asyncio
async def test_split_failure_is_fatal_even_with_on_error_ignore(
    tmp_path: Path,
) -> None:
    pipe = _make_pipe(on_error="IGNORE", perform_entity_resolution=False)
    _with_memory_sink(pipe)
    _set_async_mock(pipe.text_splitter, "run", side_effect=ValueError("split failed"))

    with pytest.raises(ValueError, match="split failed"):
        await pipe.run_async(file_path=_write_md(tmp_path))


@pytest.mark.asyncio
async def test_load_failure_is_fatal_even_with_on_error_ignore(
    tmp_path: Path,
) -> None:
    pipe = _make_pipe(on_error="IGNORE", perform_entity_resolution=False)
    _with_memory_sink(pipe)
    _set_async_mock(pipe.file_loader, "run", side_effect=ValueError("load failed"))

    with pytest.raises(ValueError, match="load failed"):
        await pipe.run_async(file_path=_write_md(tmp_path))


# ---------------------------------------------------------------------------
# Schema handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_user_provided_schema_is_used_directly(tmp_path: Path) -> None:
    schema = GraphSchema.create_empty()
    pipe = _make_pipe(schema=schema, perform_entity_resolution=False)
    _with_memory_sink(pipe)

    await pipe.run_async(file_path=_write_md(tmp_path))

    for call in _async_mock(pipe.extractor.extract_chunk).await_args_list:
        assert call.args[1] is schema


@pytest.mark.asyncio
async def test_schema_dict_goes_through_schema_builder(tmp_path: Path) -> None:
    pipe = _make_pipe(
        schema={"node_types": [{"label": "Person"}]},
        perform_entity_resolution=False,
    )
    _with_memory_sink(pipe)

    await pipe.run_async(file_path=_write_md(tmp_path))

    for call in _async_mock(pipe.extractor.extract_chunk).await_args_list:
        schema = call.args[1]
        assert isinstance(schema, GraphSchema)
        assert schema.node_types[0].label == "Person"

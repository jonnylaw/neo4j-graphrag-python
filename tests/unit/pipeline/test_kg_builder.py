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
    Neo4jRelationship,
    TextChunk,
    TextChunks,
)
from neo4j_graphrag.embeddings import Embedder
from neo4j_graphrag.llm.base import LLMInterface
from neo4j_graphrag.pipeline.interpreter import AsyncInterpreter
from neo4j_graphrag.pipeline.kg_builder import SimpleKGPipeline, _ChunkPart
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


def _lexical_graph(chunks: list[TextChunk], doc_id: str = "doc-uid") -> Neo4jGraph:
    """The lexical graph built up front: Document + Chunk nodes, FROM_DOCUMENT
    edges per chunk, and an unbroken NEXT_CHUNK chain."""
    nodes: list[Neo4jNode] = [
        Neo4jNode(id=doc_id, label="Document"),
        *(Neo4jNode(id=c.uid, label="Chunk") for c in chunks),
    ]
    relationships: list[Neo4jRelationship] = [
        *(
            Neo4jRelationship(
                start_node_id=c.uid, end_node_id=doc_id, type="FROM_DOCUMENT"
            )
            for c in chunks
        ),
        *(
            Neo4jRelationship(start_node_id=p.uid, end_node_id=n.uid, type="NEXT_CHUNK")
            for p, n in zip(chunks, chunks[1:])
        ),
    ]
    return Neo4jGraph(nodes=nodes, relationships=relationships)


def _entity_graph(chunk: TextChunk) -> Neo4jGraph:
    """One entity per chunk, with ids scoped to the chunk."""
    return Neo4jGraph(nodes=[Neo4jNode(id=f"e{chunk.index}", label="Entity")])


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
        # a fresh DocumentInfo per call, so each loaded file gets its own uid
        # (matching what the real loaders produce)
        side_effect=lambda **kwargs: LoadedDocument(
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
            graph=_lexical_graph(text_chunks.chunks)
        ),
    )
    _set_async_mock(pipe.lexical_graph_builder, "process_chunk_extracted_entities")
    pipe.extractor = MagicMock()
    _set_async_mock(
        pipe.extractor,
        "extract_chunk",
        side_effect=lambda chunk, schema, examples="": _entity_graph(chunk),
    )
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

    assert _async_mock(pipe.file_loader.run).await_count == 1
    assert _async_mock(pipe.text_splitter.run).await_count == 1
    assert _async_mock(pipe.chunk_embedder.run).await_count == 1
    # lexical graph built once (per document) ...
    assert _async_mock(pipe.lexical_graph_builder.run).await_count == 1
    # ... extraction runs once per chunk
    assert _async_mock(pipe.extractor.extract_chunk).await_count == 2
    # prune runs once per document
    assert _async_mock(pipe.pruner.run).await_count == 1
    assert {n.id for n in sink.graph.nodes} >= {"e0", "e1"}
    assert pipe.resolver is not None
    assert _async_mock(pipe.resolver.run).await_count == 1
    assert result.errors == []


@pytest.mark.asyncio
async def test_run_async_with_async_interpreter(tmp_path: Path) -> None:
    """A caller-supplied AsyncInterpreter drives the whole run on the
    caller's event loop, including through a KGWriterSink whose ``write``
    makes its own (thread-offloaded) ``asyncio.run`` call."""
    writer = MagicMock(spec=KGWriter)
    writer.run = AsyncMock(return_value=KGWriterModel(status="SUCCESS", metadata=None))
    pipe = _make_pipe(
        kg_writer=writer,
        perform_entity_resolution=False,
        interpreter=AsyncInterpreter(),
    )

    result = await pipe.run_async(file_path=_write_md(tmp_path))

    assert _async_mock(pipe.extractor.extract_chunk).await_count == 2
    assert writer.run.await_count == 1
    assert [w.status for w in result.writer] == ["SUCCESS"]
    assert result.errors == []


@pytest.mark.asyncio
async def test_next_chunk_relationship_is_unbroken(tmp_path: Path) -> None:
    pipe = _make_pipe(perform_entity_resolution=False)
    sink = _with_memory_sink(pipe)

    await pipe.run_async(file_path=_write_md(tmp_path))

    config = LexicalGraphConfig()
    linked = [
        r
        for r in sink.graph.relationships
        if r.type == config.next_chunk_relationship_type
    ]
    # two chunks -> exactly one NEXT_CHUNK edge, from the first to the second
    assert [(r.start_node_id, r.end_node_id) for r in linked] == [
        tuple(c.uid for c in CHUNKS)
    ]


@pytest.mark.asyncio
async def test_lexical_graph_survives_failing_chunk(tmp_path: Path) -> None:
    """The regression this pipeline exists to prevent: a chunk that fails
    extraction must not take its Chunk node or the NEXT_CHUNK chain with it."""
    pipe = _make_pipe(on_error="IGNORE", perform_entity_resolution=False)
    sink = _with_memory_sink(pipe)

    async def _extract(
        chunk: TextChunk, schema: GraphSchema, examples: str = ""
    ) -> Neo4jGraph:
        if chunk.index == 1:
            raise ValueError("extraction failed")
        return _entity_graph(chunk)

    _set_async_mock(pipe.extractor, "extract_chunk", side_effect=_extract)

    result = await pipe.run_async(file_path=_write_md(tmp_path))

    node_ids = {n.id for n in sink.graph.nodes}
    # chunk 1's entity is gone ...
    assert "e1" not in node_ids
    assert "e0" in node_ids
    # ... but every Chunk node survives and the NEXT_CHUNK chain is intact
    chunk_ids = [c.uid for c in CHUNKS]
    assert set(chunk_ids) <= node_ids
    linked = [r for r in sink.graph.relationships if r.type == "NEXT_CHUNK"]
    assert [(r.start_node_id, r.end_node_id) for r in linked] == [tuple(chunk_ids)]
    assert len(result.errors) == 1
    assert isinstance(result.errors[0].exception, ValueError)
    # the retry handle is the failed chunk, so a retry driver can re-run it
    err = result.errors[0]
    assert isinstance(err.context, _ChunkPart)
    assert err.context.chunk.index == 1


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
async def test_run_async_from_directory_writes_each_file_separately(
    tmp_path: Path,
) -> None:
    writer = MagicMock(spec=KGWriter)
    writer.run = AsyncMock(return_value=KGWriterModel(status="SUCCESS", metadata=None))
    pipe = _make_pipe(kg_writer=writer, perform_entity_resolution=False)
    _write_md(tmp_path, "a.md")
    _write_md(tmp_path, "b.md")

    result = await pipe.run_async(file_path=str(tmp_path))

    assert _async_mock(pipe.file_loader.run).await_count == 2
    assert _async_mock(pipe.lexical_graph_builder.run).await_count == 2
    # each document is written as it completes
    assert writer.run.await_count == 2
    assert len(result.writer) == 2


@pytest.mark.asyncio
async def test_default_sink_records_writer_results(tmp_path: Path) -> None:
    writer = MagicMock(spec=KGWriter)
    writer.run = AsyncMock(return_value=KGWriterModel(status="SUCCESS", metadata=None))
    pipe = _make_pipe(kg_writer=writer, perform_entity_resolution=False)

    result = await pipe.run_async(file_path=_write_md(tmp_path))

    assert [w.status for w in result.writer] == ["SUCCESS"]
    assert writer.run.await_count == 1


@pytest.mark.asyncio
async def test_writer_results_do_not_leak_between_runs(tmp_path: Path) -> None:
    writer = MagicMock(spec=KGWriter)
    writer.run = AsyncMock(return_value=KGWriterModel(status="SUCCESS", metadata=None))
    pipe = _make_pipe(kg_writer=writer, perform_entity_resolution=False)

    first = await pipe.run_async(file_path=_write_md(tmp_path))
    second = await pipe.run_async(file_path=_write_md(tmp_path))

    assert len(first.writer) == 1
    assert len(second.writer) == 1


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_invalid_map_batch_size_raises() -> None:
    with pytest.raises(ValueError, match="map_batch_size must be >= 1"):
        _make_pipe(map_batch_size=0)


def test_unknown_schema_literal_raises() -> None:
    with pytest.raises(ValueError, match="Unknown schema value"):
        _make_pipe(schema="NOT_A_SCHEMA")


def test_defaults_are_not_shared_between_instances() -> None:
    pipe_a = _make_pipe(perform_entity_resolution=False)
    pipe_b = _make_pipe(perform_entity_resolution=False)
    assert pipe_a.text_splitter is not pipe_b.text_splitter
    assert pipe_a.file_loader is not pipe_b.file_loader


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_error_ignore_captures_extract_failures(tmp_path: Path) -> None:
    pipe = _make_pipe(on_error="IGNORE", perform_entity_resolution=False)
    sink = _with_memory_sink(pipe)
    _set_async_mock(pipe.extractor, "extract_chunk", side_effect=ValueError("boom"))

    result = await pipe.run_async(file_path=_write_md(tmp_path))

    # both chunks fail extraction, so no entities, but the lexical graph survives
    assert {n.id for n in sink.graph.nodes} == {"doc-uid", *[c.uid for c in CHUNKS]}
    assert len(result.errors) == 2


@pytest.mark.asyncio
async def test_errors_do_not_leak_between_runs(tmp_path: Path) -> None:
    pipe = _make_pipe(on_error="IGNORE", perform_entity_resolution=False)
    _with_memory_sink(pipe)
    _set_async_mock(pipe.extractor, "extract_chunk", side_effect=ValueError("boom"))

    first = await pipe.run_async(file_path=_write_md(tmp_path))
    second = await pipe.run_async(file_path=_write_md(tmp_path))

    assert len(first.errors) == 2
    assert len(second.errors) == 2


@pytest.mark.asyncio
async def test_on_error_raise_aborts_run(tmp_path: Path) -> None:
    pipe = _make_pipe(on_error="RAISE", perform_entity_resolution=False)
    _with_memory_sink(pipe)
    _set_async_mock(pipe.extractor, "extract_chunk", side_effect=ValueError("boom"))

    with pytest.raises(ValueError, match="boom"):
        await pipe.run_async(file_path=_write_md(tmp_path))


@pytest.mark.asyncio
async def test_embed_failure_is_captured_with_on_error_ignore(tmp_path: Path) -> None:
    pipe = _make_pipe(on_error="IGNORE", perform_entity_resolution=False)
    _with_memory_sink(pipe)
    _set_async_mock(pipe.chunk_embedder, "run", side_effect=ValueError("embed failed"))

    result = await pipe.run_async(file_path=_write_md(tmp_path))

    assert len(result.errors) == 1
    assert _async_mock(pipe.lexical_graph_builder.run).await_count == 0


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

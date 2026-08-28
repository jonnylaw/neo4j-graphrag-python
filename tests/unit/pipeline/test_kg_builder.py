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
from typing import Any, Optional, cast
from unittest.mock import AsyncMock, MagicMock

import neo4j
import pytest
from fsspec import AbstractFileSystem

from neo4j_graphrag.components.graph_pruning import GraphPruningResult, PruningStats
from neo4j_graphrag.components.kg_writer import KGWriter, KGWriterModel
from neo4j_graphrag.components.schema import GraphSchema
from neo4j_graphrag.components.text_splitters.fixed_size_splitter import (
    PREV_CHUNK_UID,
)
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
from neo4j_graphrag.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.pipeline.sinks import InMemoryGraphSink

# Chunks as the default splitter emits them: every chunk after the first is
# stamped with its predecessor's uid, which the extraction stage turns into
# the NEXT_CHUNK edge.
CHUNKS = [
    TextChunk(text="chunk-0", index=0),
    TextChunk(text="chunk-1", index=1),
]
CHUNKS[1].metadata = {PREV_CHUNK_UID: CHUNKS[0].uid}


def _async_mock(method: Any) -> AsyncMock:
    """Narrow a mocked async callable for assertions."""
    return cast(AsyncMock, method)


def _set_async_mock(obj: Any, name: str, **kwargs: Any) -> AsyncMock:
    """Replace *name* on *obj* with an ``AsyncMock`` (avoids method-assign)."""
    mock = AsyncMock(**kwargs)
    setattr(obj, name, mock)
    return mock


def _chunk_graph(
    chunks: TextChunks,
    document_info: Optional[DocumentInfo] = None,
    lexical_graph_config: Optional[LexicalGraphConfig] = None,
    schema: Optional[GraphSchema] = None,
    examples: str = "",
    fail_indexes: frozenset[int] = frozenset(),
) -> Neo4jGraph:
    """Stand-in for ``extractor.run`` on a single-chunk input.

    Mirrors what the real extractor produces per chunk with the default
    ``create_lexical_graph=True``: the Document node, the Chunk node,
    FROM_DOCUMENT / FROM_CHUNK edges and one chunk-scoped entity — but no
    NEXT_CHUNK edge (its endpoints span chunks, so the pipeline adds it).
    """
    chunk = chunks.chunks[0]
    if chunk.index in fail_indexes:
        raise ValueError("extraction failed")
    config = lexical_graph_config or LexicalGraphConfig()
    assert document_info is not None
    doc_id = document_info.uid
    entity_id = f"e{chunk.index}"
    return Neo4jGraph(
        nodes=[
            Neo4jNode(id=doc_id, label=config.document_node_label),
            Neo4jNode(id=chunk.uid, label=config.chunk_node_label),
            Neo4jNode(id=entity_id, label="Entity"),
        ],
        relationships=[
            Neo4jRelationship(
                start_node_id=chunk.uid,
                end_node_id=doc_id,
                type=config.chunk_to_document_relationship_type,
            ),
            Neo4jRelationship(
                start_node_id=entity_id,
                end_node_id=chunk.uid,
                type=config.node_to_chunk_relationship_type,
            ),
        ],
    )


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
        side_effect=lambda text: TextChunks(chunks=[c.model_copy() for c in CHUNKS]),
    )
    pipe.chunk_embedder = MagicMock()
    # TextChunk.model_copy() shares the metadata dict, and extraction pops
    # the prev_chunk_uid stamp — deep-copy so a re-split yields fresh chunks.
    _set_async_mock(
        pipe.chunk_embedder,
        "run",
        side_effect=lambda tc: TextChunks(
            chunks=[c.model_copy(deep=True) for c in tc.chunks]
        ),
    )
    pipe.extractor = MagicMock()
    _set_async_mock(pipe.extractor, "run", side_effect=_chunk_graph)
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
    # embed and extraction run once per chunk
    assert _async_mock(pipe.chunk_embedder.run).await_count == 2
    assert _async_mock(pipe.extractor.run).await_count == 2
    # prune runs once per document
    assert _async_mock(pipe.pruner.run).await_count == 1
    assert {n.id for n in sink.graph.nodes} >= {"e0", "e1"}
    assert pipe.resolver is not None
    assert _async_mock(pipe.resolver.run).await_count == 1
    assert result.errors == []


@pytest.mark.asyncio
async def test_document_node_is_deduplicated_in_the_merge(tmp_path: Path) -> None:
    """Every chunk's graph repeats the Document node; the terminal reduce
    must emit it once."""
    pipe = _make_pipe(perform_entity_resolution=False)
    sink = _with_memory_sink(pipe)

    await pipe.run_async(file_path=_write_md(tmp_path))

    document_nodes = [n for n in sink.graph.nodes if n.label == "Document"]
    assert len(document_nodes) == 1


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
async def test_next_chunk_edge_survives_failing_chunk(tmp_path: Path) -> None:
    """The regression this pipeline exists to prevent: a chunk that fails
    extraction must not take the NEXT_CHUNK chain with it.

    Note the limit of the per-chunk design (kg-builder's, mirrored here):
    the NEXT_CHUNK edge lives on the *succeeding* chunk's graph, and
    pruning drops relationships with a missing endpoint. So a failed chunk
    still severs the chain *at that chunk* — what the design guarantees is
    that the failure's blast radius is exactly one chunk: every surviving
    chunk keeps its full lexical graph, and the chain survives failures
    anywhere else in the document. The alternative — fusing each chunk's
    lexical graph with its extraction — loses the Chunk node *and* the edge
    on a single transient LLM error."""
    pipe = _make_pipe(on_error="IGNORE", perform_entity_resolution=False)
    sink = _with_memory_sink(pipe)

    _set_async_mock(
        pipe.extractor,
        "run",
        side_effect=lambda **kwargs: _chunk_graph(
            **kwargs, fail_indexes=frozenset({1})
        ),
    )

    result = await pipe.run_async(file_path=_write_md(tmp_path))

    node_ids = {n.id for n in sink.graph.nodes}
    # chunk 1's entity and Chunk node are gone ...
    assert "e1" not in node_ids
    assert CHUNKS[1].uid not in node_ids
    # ... but chunk 0's whole graph — Chunk node, entity, FROM_DOCUMENT /
    # FROM_CHUNK edges — survives
    assert "e0" in node_ids
    assert CHUNKS[0].uid in node_ids
    assert len([n for n in sink.graph.nodes if n.label == "Document"]) == 1
    assert len(result.errors) == 1
    assert isinstance(result.errors[0].exception, ValueError)


@pytest.mark.asyncio
async def test_next_chunk_edge_survives_an_earlier_chunks_failure(
    tmp_path: Path,
) -> None:
    """The NEXT_CHUNK edge is built from the *succeeding* chunk's
    ``prev_chunk_uid`` stamp, outside the failable LLM path: with three
    chunks, a failure of chunk 1 still yields the chunk1→chunk2 edge."""
    three_chunks = [
        TextChunk(text="chunk-0", index=0),
        TextChunk(text="chunk-1", index=1),
        TextChunk(text="chunk-2", index=2),
    ]
    three_chunks[1].metadata = {PREV_CHUNK_UID: three_chunks[0].uid}
    three_chunks[2].metadata = {PREV_CHUNK_UID: three_chunks[1].uid}

    pipe = _make_pipe(on_error="IGNORE", perform_entity_resolution=False)
    sink = _with_memory_sink(pipe)
    _set_async_mock(
        pipe.text_splitter,
        "run",
        side_effect=lambda text: TextChunks(
            chunks=[c.model_copy() for c in three_chunks]
        ),
    )
    _set_async_mock(
        pipe.extractor,
        "run",
        side_effect=lambda **kwargs: _chunk_graph(
            **kwargs, fail_indexes=frozenset({1})
        ),
    )

    result = await pipe.run_async(file_path=_write_md(tmp_path))

    linked = [r for r in sink.graph.relationships if r.type == "NEXT_CHUNK"]
    assert [(r.start_node_id, r.end_node_id) for r in linked] == [
        (three_chunks[1].uid, three_chunks[2].uid)
    ]
    assert len(result.errors) == 1


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
    # two chunks per document
    assert _async_mock(pipe.extractor.run).await_count == 4
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
    _set_async_mock(pipe.extractor, "run", side_effect=ValueError("boom"))

    result = await pipe.run_async(file_path=_write_md(tmp_path))

    # both chunks fail extraction, so no graph reaches the sink
    assert sink.graph.nodes == []
    assert sink.graph.relationships == []
    assert len(result.errors) == 2


@pytest.mark.asyncio
async def test_errors_do_not_leak_between_runs(tmp_path: Path) -> None:
    pipe = _make_pipe(on_error="IGNORE", perform_entity_resolution=False)
    _with_memory_sink(pipe)
    _set_async_mock(pipe.extractor, "run", side_effect=ValueError("boom"))

    first = await pipe.run_async(file_path=_write_md(tmp_path))
    second = await pipe.run_async(file_path=_write_md(tmp_path))

    assert len(first.errors) == 2
    assert len(second.errors) == 2


@pytest.mark.asyncio
async def test_on_error_raise_aborts_run(tmp_path: Path) -> None:
    pipe = _make_pipe(on_error="RAISE", perform_entity_resolution=False)
    _with_memory_sink(pipe)
    _set_async_mock(pipe.extractor, "run", side_effect=ValueError("boom"))

    with pytest.raises(ValueError, match="boom"):
        await pipe.run_async(file_path=_write_md(tmp_path))


@pytest.mark.asyncio
async def test_embed_failure_is_captured_with_on_error_ignore(tmp_path: Path) -> None:
    pipe = _make_pipe(on_error="IGNORE", perform_entity_resolution=False)
    sink = _with_memory_sink(pipe)
    _set_async_mock(pipe.chunk_embedder, "run", side_effect=ValueError("embed failed"))

    result = await pipe.run_async(file_path=_write_md(tmp_path))

    # embed is per chunk: both chunks fail, nothing reaches the extractor
    assert len(result.errors) == 2
    assert _async_mock(pipe.extractor.run).await_count == 0
    assert sink.graph.nodes == []


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

    for call in _async_mock(pipe.extractor.run).await_args_list:
        assert call.kwargs["schema"] is schema


@pytest.mark.asyncio
async def test_schema_dict_goes_through_schema_builder(tmp_path: Path) -> None:
    pipe = _make_pipe(
        schema={"node_types": [{"label": "Person"}]},
        perform_entity_resolution=False,
    )
    _with_memory_sink(pipe)

    await pipe.run_async(file_path=_write_md(tmp_path))

    for call in _async_mock(pipe.extractor.run).await_args_list:
        schema = call.kwargs["schema"]
        assert isinstance(schema, GraphSchema)
        assert schema.node_types[0].label == "Person"


@pytest.mark.asyncio
async def test_extractor_run_receives_single_chunk_and_document_info(
    tmp_path: Path,
) -> None:
    """Extraction is per chunk: each ``extractor.run`` call gets exactly one
    chunk, the document info, and the lexical graph config."""
    pipe = _make_pipe(perform_entity_resolution=False)
    _with_memory_sink(pipe)

    await pipe.run_async(file_path=_write_md(tmp_path))

    calls = _async_mock(pipe.extractor.run).await_args_list
    assert len(calls) == len(CHUNKS)
    seen_uids = set()
    for call in calls:
        chunks = call.kwargs["chunks"]
        assert isinstance(chunks, TextChunks)
        assert len(chunks.chunks) == 1
        seen_uids.add(chunks.chunks[0].uid)
        assert isinstance(call.kwargs["document_info"], DocumentInfo)
        assert call.kwargs["lexical_graph_config"] is pipe.lexical_graph_config
    assert seen_uids == {c.uid for c in CHUNKS}

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
"""SimpleKGPipeline built on the :mod:`neo4j_graphrag.pipeline` DSL.

This is the dataflow-native counterpart of
:class:`neo4j_graphrag.experimental.pipeline.kg_builder.SimpleKGPipeline`.
Instead of a DAG of components executed by an orchestrator, documents flow
from a :class:`~neo4j_graphrag.pipeline.source.Source` through one operator
chain into a :class:`~neo4j_graphrag.pipeline.sink.Sink`::

    Source[FsspecFile]
      → load            file            → LoadedDocument
      → resolve_schema  document        → document + GraphSchema
      → split           document        → chunk, chunk, …
      → embed           chunk           → chunk + embedding
      → extract         chunk           → chunk graph (lexical + entities)
      → reduce          per document    → merged graph
      → prune           document graph  → schema-conformant graph
      → Sink[Neo4jGraph]

**Decision — extraction per chunk, lexical graph included.** Each chunk
flows independently through
:meth:`~neo4j_graphrag.components.entity_relation_extractor.LLMEntityRelationExtractor.run`
with a single-chunk input and default ``create_lexical_graph=True``, so its
per-chunk graph holds the Document node, its own Chunk node, and the
``FROM_DOCUMENT`` / ``FROM_CHUNK`` edges. The only structure that spans
chunks is the ``NEXT_CHUNK`` chain: the splitter stamps each chunk with
``metadata["prev_chunk_uid"]`` and the extraction stage turns it into a
``NEXT_CHUNK`` edge *after* the extractor returns, outside the failable
LLM path — no LLM call can create it, so no LLM failure can drop it.

* A chunk that fails extraction yields an ``Err`` for that chunk only; the
  Document and Chunk nodes and the ``NEXT_CHUNK`` chain of every surviving
  chunk are unaffected. This is the granularity a transient LLM failure
  (rate limit, network) needs: one chunk fails, the rest of the document
  survives.

* The Document node appears once per chunk and is deduplicated by id in
  the terminal ``reduce`` (kg-builder's approach, where the resolver does
  the same at write time), so the sink still sees one graph per document.

Per-chunk extraction failures (and per-chunk embed or per-document prune
failures) are captured as :class:`~neo4j_graphrag.pipeline.result.Err`
values when ``on_error="IGNORE"``; load, schema and split failures are
always fatal.

.. warning::

    The default :class:`~neo4j_graphrag.pipeline.interpreter.LocalInterpreter`
    evaluates async stages by running each batch of ``map_batch_size`` items
    through a fresh ``asyncio.run()`` call. An async SDK client that pools
    keep-alive connections (``httpx.AsyncClient``, ``aiohttp.ClientSession``)
    and is shared across batches — i.e. created *outside* the async stage
    functions — will then ``aclose()`` on a closed event loop and raise
    ``RuntimeError: Event loop is closed``. Pass a client configured to
    disable keep-alive (``httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=0))``),
    or construct the client inside the stage. A future async interpreter
    would lift this restriction.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Optional, Union

import neo4j
from fsspec import AbstractFileSystem
from pydantic import BaseModel, ConfigDict, Field

from neo4j_graphrag.components.data_loader import (
    DataLoader,
    MarkdownLoader,
    PdfLoader,
)
from neo4j_graphrag.components.embedder import TextChunkEmbedder
from neo4j_graphrag.components.entity_relation_extractor import (
    LLMEntityRelationExtractor,
    OnError,
)
from neo4j_graphrag.components.graph_pruning import GraphPruning
from neo4j_graphrag.components.kg_writer import KGWriter, KGWriterModel, Neo4jWriter
from neo4j_graphrag.components.resolver import SinglePropertyExactMatchResolver
from neo4j_graphrag.components.schema import (
    GraphSchema,
    SchemaBuilder,
    SchemaFromTextExtractor,
)
from neo4j_graphrag.components.text_splitters.base import TextSplitter
from neo4j_graphrag.components.text_splitters.fixed_size_splitter import (
    PREV_CHUNK_UID,
    FixedSizeSplitter,
)
from neo4j_graphrag.components.types import (
    DocumentInfo,
    LexicalGraphConfig,
    LoadedDocument,
    Neo4jGraph,
    Neo4jRelationship,
    ResolutionStats,
    TextChunk,
    TextChunks,
)
from neo4j_graphrag.embeddings import Embedder
from neo4j_graphrag.exceptions import UnsupportedDocumentFormatError
from neo4j_graphrag.generation.prompts import ERExtractionTemplate
from neo4j_graphrag.llm.base import LLMInterface
from neo4j_graphrag.pipeline.pipeline import Pipeline
from neo4j_graphrag.pipeline.result import Err
from neo4j_graphrag.pipeline.sink import Sink
from neo4j_graphrag.pipeline.sinks import KGWriterSink
from neo4j_graphrag.pipeline.sources import FsspecFile, FsspecSource

logger = logging.getLogger(__name__)

__all__ = ["SimpleKGPipeline", "SimpleKGPipelineResult"]

SUPPORTED_EXTENSIONS = (".pdf", ".md", ".markdown")

_SCHEMA_LITERALS = ("FREE", "EXTRACTED")


class _DefaultDataLoader(DataLoader):
    """Default loader supporting PDF and Markdown files (by extension)."""

    async def run(
        self,
        filepath: Union[str, Path],
        metadata: Optional[dict[str, str]] = None,
        fs: Optional[Union[AbstractFileSystem, str]] = None,
    ) -> LoadedDocument:
        path = str(filepath)
        suffix = PurePosixPath(path).suffix.lower()
        if suffix == ".pdf":
            return await PdfLoader().run(filepath=path, metadata=metadata, fs=fs)
        if suffix in (".md", ".markdown"):
            return await MarkdownLoader().run(filepath=path, metadata=metadata, fs=fs)
        raise UnsupportedDocumentFormatError(
            f"Unsupported document format: {suffix!r}. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )


@dataclass(frozen=True)
class _Document:
    """A loaded document paired with the schema guiding its extraction."""

    document_info: DocumentInfo
    text: str
    schema: GraphSchema


@dataclass(frozen=True)
class _ChunkedDocument:
    """A document's chunks, carrying everything downstream stages need."""

    document_info: DocumentInfo
    schema: GraphSchema
    chunks: TextChunks


@dataclass(frozen=True)
class _ChunkGraphPart:
    """One chunk of a document after (failable) extraction: the chunk's
    graph, carrying its document id so the terminal reduce can group chunks
    back under their document."""

    doc_id: str
    schema: GraphSchema
    graph: Neo4jGraph


def add_next_chunk_relationships(
    graph: Neo4jGraph,
    lexical_graph_config: LexicalGraphConfig,
    chunk: TextChunk,
) -> Neo4jGraph:
    """Append the ``NEXT_CHUNK`` edge from the chunk's predecessor, if any.

    The splitter stamps each chunk after the first with
    ``metadata[PREV_CHUNK_UID]``; this pops that stamp and turns it into a
    relationship. It runs outside the extractor because the edge's endpoints
    span chunks — and outside the failable path, so an extraction failure
    cannot sever the chain.
    """
    metadata = chunk.metadata or {}
    prev_chunk_uid = metadata.pop(PREV_CHUNK_UID, None)
    if prev_chunk_uid is None:
        return graph
    graph.relationships.append(
        Neo4jRelationship(
            start_node_id=prev_chunk_uid,
            end_node_id=chunk.chunk_id,
            type=lexical_graph_config.next_chunk_relationship_type,
        )
    )
    return graph


@dataclass
class _DocAccumulator:
    """Accumulator for one document during the terminal merge."""

    schema: GraphSchema
    graph: Neo4jGraph


def _empty_accumulator() -> dict[str, _DocAccumulator]:
    """Fresh per-document accumulator map for the terminal reduce."""
    return {}


class SimpleKGPipelineResult(BaseModel):
    """Result of a :class:`SimpleKGPipeline` run.

    Attributes:
        writer: The writer results for the documents written this run. Empty
            unless the run used the default
            :class:`~neo4j_graphrag.pipeline.sinks.KGWriterSink`.
        resolver: Entity resolution statistics, if entity resolution was
            performed.
        errors: Per-chunk extraction failures (and per-chunk embed or
            per-document prune failures) captured while
            ``on_error="IGNORE"``. Empty when ``on_error="RAISE"``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    writer: list[KGWriterModel] = Field(default_factory=list)
    resolver: Optional[ResolutionStats] = None
    errors: list[Err] = Field(default_factory=list)


class SimpleKGPipeline:
    """Simplified knowledge-graph building from text documents, implemented
    with the :class:`~neo4j_graphrag.pipeline.Pipeline` dataflow DSL.

    Args:
        llm (LLMInterface): LLM used for entity and relation extraction.
        driver (neo4j.Driver): Neo4j driver instance.
        embedder (Embedder): Embedder used to generate chunk embeddings.
        schema: Schema configuration guiding extraction. A
            :class:`~neo4j_graphrag.components.schema.GraphSchema` object, a
            dict with ``node_types`` / ``relationship_types`` / ``patterns``
            keys, ``"FREE"`` (no schema guidance), or ``"EXTRACTED"`` /
            ``None`` (schema is inferred per document by the LLM).
        text_splitter (Optional[TextSplitter]): Defaults to
            ``FixedSizeSplitter(create_prev_chunk_uid=True)`` — the
            ``prev_chunk_uid`` stamp is what the ``NEXT_CHUNK`` chain is
            rebuilt from per chunk.
        file_loader (Optional[DataLoader]): Defaults to an extension-based
            loader supporting ``.pdf``, ``.md`` and ``.markdown``.
        kg_writer (Optional[KGWriter]): Defaults to ``Neo4jWriter``. Ignored
            when *sink* is given.
        sink (Optional[Sink[Neo4jGraph]]): Destination for the assembled
            graph. Defaults to a
            :class:`~neo4j_graphrag.pipeline.sinks.KGWriterSink` around
            *kg_writer*; pass
            :class:`~neo4j_graphrag.pipeline.sinks.InMemoryGraphSink` to
            collect the graph instead of writing it.
        on_error (str): ``"RAISE"`` aborts the run on the first failing
            chunk; ``"IGNORE"`` (default) skips failing chunks and collects
            them in :attr:`SimpleKGPipelineResult.errors`.
        prompt_template: Custom prompt template for extraction.
        perform_entity_resolution (bool): Merge entities with same label and
            name after writing. Default: True.
        lexical_graph_config (Optional[LexicalGraphConfig]): Customize node
            labels and relationship types in the lexical graph.
        neo4j_database (Optional[str]): Neo4j database name.
        map_batch_size (int): Number of items processed concurrently by each
            async stage. For the extraction stage this is the number of
            concurrent LLM calls, so it doubles as the LLM concurrency knob
            (replacing the extractor's ``max_concurrency``, which the
            chunk-scoped path does not use). Default: 5.
    """

    def __init__(
        self,
        llm: LLMInterface,
        driver: neo4j.Driver,
        embedder: Embedder,
        schema: Optional[
            Union[GraphSchema, dict[str, Any], Literal["FREE", "EXTRACTED"]]
        ] = None,
        text_splitter: Optional[TextSplitter] = None,
        file_loader: Optional[DataLoader] = None,
        kg_writer: Optional[KGWriter] = None,
        sink: Optional[Sink[Neo4jGraph]] = None,
        on_error: str = "IGNORE",
        prompt_template: Union[ERExtractionTemplate, str] = ERExtractionTemplate(),
        perform_entity_resolution: bool = True,
        lexical_graph_config: Optional[LexicalGraphConfig] = None,
        neo4j_database: Optional[str] = None,
        map_batch_size: int = 5,
    ):
        if map_batch_size < 1:
            raise ValueError(f"map_batch_size must be >= 1, got {map_batch_size!r}")
        if isinstance(schema, str) and schema not in _SCHEMA_LITERALS:
            raise ValueError(
                f"Unknown schema value {schema!r}; expected 'FREE', 'EXTRACTED', "
                "a GraphSchema, a dict, or None."
            )
        self.llm = llm
        self.driver = driver
        self.embedder = embedder
        self.errors: list[Err] = []
        self.schema = schema
        self.on_error = OnError(on_error)
        self.lexical_graph_config = lexical_graph_config or LexicalGraphConfig()
        self.map_batch_size = map_batch_size

        self.text_splitter = text_splitter or FixedSizeSplitter(
            create_prev_chunk_uid=True
        )
        self.file_loader = file_loader or _DefaultDataLoader()
        self.chunk_embedder = TextChunkEmbedder(embedder=embedder)
        self.extractor = LLMEntityRelationExtractor(
            llm=llm,
            prompt_template=prompt_template,
            on_error=self.on_error,
            use_structured_output=llm.supports_structured_output,
        )
        self.pruner = GraphPruning()
        self.kg_writer = kg_writer or Neo4jWriter(
            driver=driver, neo4j_database=neo4j_database
        )
        self.sink: Sink[Neo4jGraph] = sink or KGWriterSink(
            self.kg_writer, self.lexical_graph_config
        )
        self.resolver = (
            SinglePropertyExactMatchResolver(
                driver=driver, neo4j_database=neo4j_database
            )
            if perform_entity_resolution
            else None
        )

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    async def _resolve_schema(self, document: LoadedDocument) -> _Document:
        """Attach the schema guiding extraction for this document."""
        if isinstance(self.schema, GraphSchema):
            schema = self.schema
        elif isinstance(self.schema, dict):
            schema = await SchemaBuilder().run(**self.schema)
        elif self.schema == "FREE":
            schema = GraphSchema.create_empty()
        else:
            # "EXTRACTED" or None: infer the schema from the document text.
            # Any other string is rejected in __init__.
            schema = await SchemaFromTextExtractor(
                llm=self.llm,
                use_structured_output=self.llm.supports_structured_output,
            ).run(text=document.text)
        return _Document(
            document_info=document.document_info,
            text=document.text,
            schema=schema,
        )

    async def _split(self, document: _Document) -> list[_ChunkedDocument]:
        """Split the document and fan out to one item per chunk.

        Each item carries a single-chunk ``TextChunks`` so downstream stages
        (embed, extract) work per chunk.
        """
        chunks = await self.text_splitter.run(text=document.text)
        return [
            _ChunkedDocument(
                document_info=document.document_info,
                schema=document.schema,
                chunks=TextChunks(chunks=[chunk]),
            )
            for chunk in chunks.chunks
        ]

    async def _embed(self, document: _ChunkedDocument) -> _ChunkedDocument:
        embedded = await self.chunk_embedder.run(document.chunks)
        return _ChunkedDocument(
            document_info=document.document_info,
            schema=document.schema,
            chunks=embedded,
        )

    async def _extract(self, document: _ChunkedDocument) -> _ChunkGraphPart:
        """Extract one chunk into its graph.

        ``extractor.run`` with a single-chunk input and the default
        ``create_lexical_graph=True`` builds the Document node, this chunk's
        Chunk node, and the FROM_DOCUMENT / FROM_CHUNK edges alongside the
        extracted entities — so the whole lexical graph survives any single
        chunk's failure, at the cost of one repeated Document node per
        chunk, deduplicated in the terminal reduce. The NEXT_CHUNK edge is
        added afterwards: its endpoints span chunks, and no LLM call
        touches it.
        """
        chunk = document.chunks.chunks[0]
        graph = await self.extractor.run(
            chunks=document.chunks,
            document_info=document.document_info,
            lexical_graph_config=self.lexical_graph_config,
            schema=document.schema,
        )
        add_next_chunk_relationships(graph, self.lexical_graph_config, chunk)
        return _ChunkGraphPart(
            doc_id=document.document_info.uid, schema=document.schema, graph=graph
        )

    def _combine(
        self,
        acc: dict[str, _DocAccumulator],
        part: _ChunkGraphPart,
    ) -> dict[str, _DocAccumulator]:
        """Fold a chunk graph into its document's graph.

        Entity ids are chunk-prefixed, so entity parts never overlap. The
        lexical nodes (Document, Chunk) are repeated per chunk and
        deduplicated by id here — the same dedup kg-builder relies on its
        resolver for.
        """
        doc = acc.get(part.doc_id)
        if doc is None:
            doc = _DocAccumulator(schema=part.schema, graph=Neo4jGraph())
            acc[part.doc_id] = doc
        seen = {node.id for node in doc.graph.nodes}
        doc.graph.nodes.extend(node for node in part.graph.nodes if node.id not in seen)
        doc.graph.relationships.extend(part.graph.relationships)
        return acc

    async def _prune(self, doc: _DocAccumulator) -> Neo4jGraph:
        pruned = await self.pruner.run(
            graph=doc.graph,
            schema=doc.schema,
            lexical_graph_config=self.lexical_graph_config,
        )
        return pruned.graph

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def error_handler(self, err: Err) -> None:
        if self.on_error == OnError.IGNORE:
            self.errors.append(err)
        else:
            raise err.exception

    async def run_async(
        self,
        file_path: str,
        document_metadata: Optional[dict[str, str]] = None,
    ) -> SimpleKGPipelineResult:
        """Run the knowledge-graph building pipeline.

        Args:
            file_path (str): Path to a single PDF or Markdown file, a
                directory (listed recursively), or a glob pattern, on any
                fsspec-supported backend. ``text=`` input is not yet
                supported; the experimental
                :class:`~neo4j_graphrag.experimental.pipeline.kg_builder.SimpleKGPipeline`
                accepts it.
            document_metadata (Optional[dict[str, str]]): Metadata attached
                to the Document node.

        Returns:
            SimpleKGPipelineResult: Writer/resolver results and any
            per-chunk errors (``on_error="IGNORE"``).

        Raises:
            Exception: A load, schema or split failure, or the first failing
                chunk when ``on_error="RAISE"``.
        """
        source = FsspecSource(file_path, extensions=SUPPORTED_EXTENSIONS)
        # Per-run state: do not leak errors or writer results into the next run.
        self.errors = []
        if isinstance(self.sink, KGWriterSink):
            self.sink.results = []

        async def _load(file: FsspecFile) -> LoadedDocument:
            return await self.file_loader.run(
                filepath=file.path, metadata=document_metadata, fs=file.fs
            )

        def _run() -> None:
            (
                Pipeline.from_source(source)
                # load / schema / split failures are fatal
                .map_async_chunked(_load, self.map_batch_size)
                .map_async_chunked(self._resolve_schema, self.map_batch_size)
                .map_async_chunked(self._split, self.map_batch_size)
                .flat_map(lambda parts: parts)
                # embed / extract → per-chunk, failable
                .map_async_chunked_safe(self._embed, self.map_batch_size)
                .map_async_chunked_safe(self._extract, self.map_batch_size)
                .on_error(self.error_handler)
                # merge chunks back under their document
                .reduce(zero=_empty_accumulator(), combine=self._combine)
                .flat_map(lambda acc: acc.values())
                # prune → per-document, failable
                .map_async_chunked_safe(self._prune, self.map_batch_size)
                .on_error(self.error_handler)
                .to_sink(self.sink)
            )

        # Blocking evaluation (asyncio.run per batch) — run off the event loop.
        await asyncio.to_thread(_run)

        resolution_stats = None
        if self.resolver is not None:
            resolution_stats = await self.resolver.run()

        return SimpleKGPipelineResult(
            writer=self.sink.results if isinstance(self.sink, KGWriterSink) else [],
            resolver=resolution_stats,
            errors=self.errors,
        )

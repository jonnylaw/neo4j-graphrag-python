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
from a :class:`~neo4j_graphrag.pipeline.source.Source` through one unbroken
operator chain into a :class:`~neo4j_graphrag.pipeline.sink.Sink`::

    Source[FsspecFile]
      → load            file            → LoadedDocument
      → resolve_schema  document        → document + GraphSchema
      → split           document        → chunks                 (1-to-many)
      → embed           chunk           → chunk + embedding
      → lexical graph   chunk           → lexical graph
      → extract         chunk           → lexical + entity graph
      → prune           chunk graph     → schema-conformant graph
      → link            chunk graph     → graph with NEXT_CHUNK
      → reduce          chunk graphs    → one merged graph
      → Sink[Neo4jGraph]

Every stage is chunk-scoped, including lexical graph construction: each
chunk carries the uid of its predecessor, so the ``NEXT_CHUNK`` relationship
is reconstructed per chunk instead of requiring the whole document to be
materialised first. The only aggregation is the terminal ``reduce``, which
merges every chunk graph into a single graph before one write — so peak
memory is bounded by the size of the resulting knowledge graph.

Per-chunk failures (embed, lexical graph, extract, prune) are captured as
:class:`~neo4j_graphrag.pipeline.result.Err` values when ``on_error="IGNORE"``
instead of aborting the run; load, schema and split failures are always fatal.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
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
from neo4j_graphrag.components.lexical_graph import LexicalGraphBuilder
from neo4j_graphrag.components.resolver import SinglePropertyExactMatchResolver
from neo4j_graphrag.components.schema import (
    GraphSchema,
    SchemaBuilder,
    SchemaFromTextExtractor,
)
from neo4j_graphrag.components.text_splitters.base import TextSplitter
from neo4j_graphrag.components.text_splitters.fixed_size_splitter import (
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
class _Chunk:
    """One chunk, carrying everything its downstream stages need.

    ``previous_chunk_uid`` is what lets the lexical graph be built one chunk
    at a time: the ``NEXT_CHUNK`` relationship is rebuilt from it rather than
    from a materialised list of sibling chunks.
    """

    chunk: TextChunk
    document_info: DocumentInfo
    schema: GraphSchema
    previous_chunk_uid: Optional[str]


@dataclass(frozen=True)
class _LexicalChunkGraph:
    """A chunk's lexical graph, before entity extraction.

    Carries the chunk itself so the extraction stage can run on it.
    """

    lexical_graph: Neo4jGraph
    chunk: TextChunk
    schema: GraphSchema
    previous_chunk_uid: Optional[str]


@dataclass(frozen=True)
class _ChunkGraph:
    """The graph extracted from a single chunk, before it is linked up."""

    graph: Neo4jGraph
    chunk_uid: str
    schema: GraphSchema
    previous_chunk_uid: Optional[str]


@dataclass
class _GraphAccumulator:
    """Accumulator for the terminal reduce: the merged graph plus the set of
    node ids already in it.

    Every chunk graph repeats its Document node so that its ``FROM_DOCUMENT``
    edge survives pruning; ``seen`` is what collapses those duplicates while
    folding.
    """

    graph: Neo4jGraph
    seen: set[str] = field(default_factory=set)

    def merge(self, graph: Neo4jGraph) -> _GraphAccumulator:
        """Fold one chunk graph into the accumulator, deduplicating nodes by id."""
        for node in graph.nodes:
            if node.id not in self.seen:
                self.seen.add(node.id)
                self.graph.nodes.append(node)
        self.graph.relationships.extend(graph.relationships)
        return self


class SimpleKGPipelineResult(BaseModel):
    """Result of a :class:`SimpleKGPipeline` run.

    Attributes:
        writer: The writer result for the single merged-graph write. Empty
            unless the run used the default
            :class:`~neo4j_graphrag.pipeline.sinks.KGWriterSink`.
        resolver: Entity resolution statistics, if entity resolution was
            performed.
        errors: Per-chunk failures captured while ``on_error="IGNORE"``.
            Empty when ``on_error="RAISE"`` (the first failure aborts the
            run instead).
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
        text_splitter (Optional[TextSplitter]): Defaults to ``FixedSizeSplitter()``.
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
            async stage. Default: 100.

    All chunk graphs are merged into a single graph before writing, so the
    sink receives exactly one element per run. Database write batching is
    the writer's concern — pass ``kg_writer=Neo4jWriter(..., batch_size=...)``
    to control it.
    """

    def __init__(
        self,
        llm: LLMInterface,
        driver: neo4j.Driver,
        embedder: Embedder,
        schema: Optional[
            Union[GraphSchema, dict[str, Any], Literal["FREE", "EXTRACTED"]]
        ] = None,
        text_splitter: TextSplitter = FixedSizeSplitter(),
        file_loader: DataLoader = _DefaultDataLoader(),
        kg_writer: Optional[KGWriter] = None,
        sink: Optional[Sink[Neo4jGraph]] = None,
        on_error: str = "IGNORE",
        prompt_template: Union[ERExtractionTemplate, str] = ERExtractionTemplate(),
        perform_entity_resolution: bool = True,
        lexical_graph_config: Optional[LexicalGraphConfig] = None,
        neo4j_database: Optional[str] = None,
        map_batch_size: int = 100,
    ):
        if map_batch_size < 1:
            raise ValueError(f"map_batch_size must be >= 1, got {map_batch_size!r}")
        self.llm = llm
        self.driver = driver
        self.embedder = embedder
        self.errors: list[Err] = []
        self.schema = schema
        self.on_error = OnError(on_error)
        self.lexical_graph_config = lexical_graph_config or LexicalGraphConfig()
        self.map_batch_size = map_batch_size

        self.text_splitter = text_splitter
        self.file_loader = file_loader
        self.chunk_embedder = TextChunkEmbedder(embedder=embedder)
        self.lexical_graph_builder = LexicalGraphBuilder(
            config=self.lexical_graph_config
        )
        # on_error=RAISE: per-chunk failures surface as exceptions and the
        # pipeline decides how to handle them (captured as Err values when
        # self.on_error is IGNORE).
        self.extractor = LLMEntityRelationExtractor(
            llm=llm,
            prompt_template=prompt_template,
            on_error=OnError.RAISE,
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
            # "EXTRACTED" or None: infer the schema from the document text
            schema = await SchemaFromTextExtractor(
                llm=self.llm,
                use_structured_output=self.llm.supports_structured_output,
            ).run(text=document.text)
        return _Document(
            document_info=document.document_info,
            text=document.text,
            schema=schema,
        )

    async def _split(self, document: _Document) -> list[_Chunk]:
        """Split into chunks, each linked to its predecessor by uid."""
        chunks = (await self.text_splitter.run(text=document.text)).chunks
        previous_uids: list[Optional[str]] = [None, *(c.uid for c in chunks[:-1])]
        return [
            _Chunk(
                chunk=chunk,
                document_info=document.document_info,
                schema=document.schema,
                previous_chunk_uid=previous_uid,
            )
            for chunk, previous_uid in zip(chunks, previous_uids)
        ]

    async def _embed(self, item: _Chunk) -> _Chunk:
        embedded = await self.chunk_embedder.run(TextChunks(chunks=[item.chunk]))
        return _Chunk(
            chunk=embedded.chunks[0],
            document_info=item.document_info,
            schema=item.schema,
            previous_chunk_uid=item.previous_chunk_uid,
        )

    async def _build_lexical_graph(self, item: _Chunk) -> _LexicalChunkGraph:
        """Build this chunk's lexical graph.

        Runs before extraction: it moves the chunk's embedding out of
        ``metadata`` and onto the Chunk node, so extraction never sees it.
        """
        lexical_graph = (
            await self.lexical_graph_builder.run(
                text_chunks=TextChunks(chunks=[item.chunk]),
                document_info=item.document_info,
            )
        ).graph
        return _LexicalChunkGraph(
            lexical_graph=lexical_graph,
            chunk=item.chunk,
            schema=item.schema,
            previous_chunk_uid=item.previous_chunk_uid,
        )

    async def _extract(self, item: _LexicalChunkGraph) -> _ChunkGraph:
        """Extract this chunk's entities and merge with its lexical graph.

        Concurrency is bounded by the pipeline's ``map_batch_size``, so no
        semaphore is needed (or safe: each batch runs on its own event loop).
        """
        entity_graph = await self.extractor.extract_chunk(
            item.chunk,
            item.schema,
            "",
            self.lexical_graph_builder,
        )
        return _ChunkGraph(
            graph=self.extractor.combine_chunk_graphs(
                item.lexical_graph, [entity_graph]
            ),
            chunk_uid=item.chunk.uid,
            schema=item.schema,
            previous_chunk_uid=item.previous_chunk_uid,
        )

    async def _prune(self, item: _ChunkGraph) -> _ChunkGraph:
        pruned = await self.pruner.run(
            graph=item.graph,
            schema=item.schema,
            lexical_graph_config=self.lexical_graph_config,
        )
        return _ChunkGraph(
            graph=pruned.graph,
            chunk_uid=item.chunk_uid,
            schema=item.schema,
            previous_chunk_uid=item.previous_chunk_uid,
        )

    def _link_previous_chunk(self, item: _ChunkGraph) -> Neo4jGraph:
        """Add the NEXT_CHUNK edge from the predecessor chunk.

        Runs after pruning: the predecessor's Chunk node lives in a different
        chunk graph, so the pruner would drop the edge as dangling.
        """
        if item.previous_chunk_uid is not None:
            item.graph.relationships.append(
                Neo4jRelationship(
                    start_node_id=item.previous_chunk_uid,
                    end_node_id=item.chunk_uid,
                    type=self.lexical_graph_config.next_chunk_relationship_type,
                )
            )
        return item.graph

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
            file_path (str): Path to a single PDF or Markdown file on any
                fsspec-supported backend.
            document_metadata (Optional[dict[str, str]]): Metadata attached
                to the Document node.

        Returns:
            SimpleKGPipelineResult: Writer/resolver results and any per-chunk
            errors (``on_error="IGNORE"``).

        Raises:
            Exception: A load, schema or split failure, or the first failing
                chunk when ``on_error="RAISE"``.
        """
        source = FsspecSource(file_path, extensions=SUPPORTED_EXTENSIONS)
        self.errors = []  # per-run: do not leak errors into the next run

        async def _load(file: FsspecFile) -> LoadedDocument:
            return await self.file_loader.run(
                filepath=file.path, metadata=document_metadata, fs=file.fs
            )

        def _raise(err: Err) -> None:
            raise err.exception

        def _run() -> None:
            (
                Pipeline.from_source(source)
                .map_async_chunked(_load, self.map_batch_size)
                .map_async_chunked(self._resolve_schema, self.map_batch_size)
                .map_async_chunked_safe(self._split, self.map_batch_size)
                .flatten_ok()
                .on_error(_raise)  # split failures are fatal, not per-chunk
                .map_async_chunked_safe(self._embed, self.map_batch_size)
                .map_async_chunked_safe(self._build_lexical_graph, self.map_batch_size)
                .map_async_chunked_safe(self._extract, self.map_batch_size)
                .map_async_chunked_safe(self._prune, self.map_batch_size)
                .map_ok(self._link_previous_chunk)
                .on_error(self.error_handler)
                # The zero accumulator is created here, inside _run, so each
                # run starts from an empty graph.
                .reduce(
                    zero=_GraphAccumulator(graph=Neo4jGraph()),
                    combine=_GraphAccumulator.merge,
                )
                .map(lambda acc: acc.graph)
                # an empty stream (e.g. a document with no chunks) writes nothing
                .filter(lambda graph: bool(graph.nodes or graph.relationships))
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

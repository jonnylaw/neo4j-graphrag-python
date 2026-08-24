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

"""End-to-end PDF to Knowledge Graph with the new dataflow pipeline DSL.

This example uses :class:`neo4j_graphrag.pipeline.kg_builder.SimpleKGPipeline`,
the successor of the experimental pipeline shown in ``examples/kg_builder.py``.
Instead of wiring components into a DAG by hand, documents flow from a
``Source`` through a single operator chain (load → split → embed → extract →
prune → link → write) into a ``Sink``, with bounded memory and per-chunk
error handling.

This example assumes a Neo4j db is up and running. Update the credentials
below if needed.

OPENAI_API_KEY needs to be in the env vars.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import httpx
import neo4j
from dotenv import load_dotenv

from neo4j_graphrag.components.schema import (
    NodeType,
    PropertyType,
    RelationshipType,
)
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.llm import LLMInterface, OpenAILLM
from neo4j_graphrag.pipeline.kg_builder import (
    SimpleKGPipeline,
    SimpleKGPipelineResult,
)

logging.basicConfig(level=logging.INFO)


load_dotenv()


# Neo4j db infos
URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
AUTH = (USER, PASSWORD)
DATABASE = "neo4j"

# Resolve the PDF relative to this file so the example runs from any
# working directory, as the other examples do.
root_dir = Path(__file__).parent
file_path = root_dir / "data" / "Harry Potter and the Death Hallows Summary.pdf"

# The schema guides the LLM during entity and relation extraction.
NODE_TYPES = [
    NodeType(
        label="PERSON",
        description="An individual human being.",
        properties=[
            PropertyType(name="name", type="STRING"),
        ],
    ),
    NodeType(
        label="ORGANIZATION",
        description="A structured group of people with a common purpose.",
        properties=[
            PropertyType(name="name", type="STRING"),
        ],
    ),
    NodeType(
        label="LOCATION",
        description="A location or place.",
        properties=[
            PropertyType(name="name", type="STRING"),
        ],
    ),
    NodeType(
        label="HORCRUX",
        description="A magical item in the Harry Potter universe.",
        properties=[
            PropertyType(name="name", type="STRING"),
        ],
    ),
]
RELATIONSHIP_TYPES = [
    RelationshipType(
        label="SITUATED_AT", description="Indicates the location of a person."
    ),
    RelationshipType(
        label="LED_BY",
        description="Indicates the leader of an organization.",
    ),
    RelationshipType(
        label="OWNS",
        description="Indicates the ownership of an item such as a Horcrux.",
    ),
    RelationshipType(
        label="INTERACTS", description="The interaction between two people."
    ),
]
PATTERNS = [
    ("PERSON", "SITUATED_AT", "LOCATION"),
    ("PERSON", "INTERACTS", "PERSON"),
    ("PERSON", "OWNS", "HORCRUX"),
    ("ORGANIZATION", "LED_BY", "PERSON"),
]


async def define_and_run_pipeline(
    neo4j_driver: neo4j.Driver, llm: LLMInterface
) -> SimpleKGPipelineResult:
    # The new SimpleKGPipeline builds the whole operator chain for you:
    # load -> resolve schema -> split -> embed -> extract -> prune ->
    # link chunks -> merge write batches -> write to Neo4j.
    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=neo4j_driver,
        embedder=OpenAIEmbeddings(),
        schema={
            "node_types": NODE_TYPES,
            "relationship_types": RELATIONSHIP_TYPES,
            "patterns": PATTERNS,
        },
        # on_error="IGNORE" (default): failing chunks are skipped and
        # collected in SimpleKGPipelineResult.errors. Use "RAISE" to abort
        # the run on the first failing chunk.
        on_error="IGNORE",
        neo4j_database=DATABASE,
    )

    return await kg_builder.run_async(
        file_path=str(file_path),
    )


async def main() -> SimpleKGPipelineResult:
    # For this example, gpt-5 needs max_completion_tokens (not max_tokens) and counts reasoning tokens
    # against it, so too small a budget returns empty content. reasoning_effort="low"
    # halves the cost (~$0.24 -> ~$0.12 per run, 2026-07-30) with no measurable
    # difference in extraction quality.
    async with OpenAILLM(
        model_name="gpt-5",
        model_params={
            "max_completion_tokens": 16000,
            "reasoning_effort": "low",
            "response_format": {"type": "json_object"},
        },
        # The dataflow pipeline's blocking interpreter runs each batch of
        # chunks on its own event loop (asyncio.run per batch). With the
        # default httpx pooling, keep-alive connections would stay bound to
        # those short-lived loops and crash aclose() with "Event loop is
        # closed". Disabling keep-alive closes each connection on the loop
        # that created it.
        http_client=httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=0)),
    ) as llm:
        with neo4j.GraphDatabase.driver(URI, auth=AUTH) as driver:
            res = await define_and_run_pipeline(driver, llm)
    return res


if __name__ == "__main__":
    res = asyncio.run(main())
    print(res)
    # Per-chunk failures captured with on_error="IGNORE":
    for err in res.errors:
        logging.warning("chunk failed: %s", err.exception)

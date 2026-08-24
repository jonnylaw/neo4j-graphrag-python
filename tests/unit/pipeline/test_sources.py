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
"""Unit tests for :mod:`neo4j_graphrag.pipeline.sources`.

Exercised against fsspec's in-memory filesystem so the tests cover the
non-local code path that makes the source portable across backends.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fsspec.implementations.memory import MemoryFileSystem

from neo4j_graphrag.pipeline import Pipeline
from neo4j_graphrag.pipeline.sources import FsspecFile, FsspecSource


@pytest.fixture
def memory_fs() -> MemoryFileSystem:
    fs = MemoryFileSystem()
    fs.store.clear()
    fs.pseudo_dirs.clear()
    for path, content in {
        "/kb/a.md": b"alpha",
        "/kb/b.pdf": b"bravo",
        "/kb/notes.txt": b"charlie",
        "/kb/nested/c.md": b"delta",
    }.items():
        with fs.open(path, "wb") as handle:
            handle.write(content)
    return fs


def test_read_single_file(tmp_path: Path) -> None:
    target = tmp_path / "doc.md"
    target.write_text("hello")

    files = list(FsspecSource(str(target)).read())

    assert [f.path for f in files] == [str(target)]
    assert files[0].suffix == ".md"


def test_read_directory_is_recursive(memory_fs: MemoryFileSystem) -> None:
    files = list(FsspecSource("memory://kb").read())

    assert [f.path for f in files] == [
        "/kb/a.md",
        "/kb/b.pdf",
        "/kb/nested/c.md",
        "/kb/notes.txt",
    ]


def test_read_glob(memory_fs: MemoryFileSystem) -> None:
    files = list(FsspecSource("memory://kb/*.md").read())

    assert [f.path for f in files] == ["/kb/a.md"]


def test_extensions_filter(memory_fs: MemoryFileSystem) -> None:
    files = list(FsspecSource("memory://kb", extensions=["md", ".PDF"]).read())

    assert [f.path for f in files] == ["/kb/a.md", "/kb/b.pdf", "/kb/nested/c.md"]


def test_file_open_reads_from_its_own_filesystem(
    memory_fs: MemoryFileSystem,
) -> None:
    file = next(iter(FsspecSource("memory://kb/a.md").read()))

    with file.open() as handle:
        assert handle.read() == b"alpha"


def test_suffix_of_extensionless_file_is_empty() -> None:
    assert FsspecFile(path="/kb/LICENSE", fs=MemoryFileSystem()).suffix == ""


def test_source_feeds_a_pipeline(memory_fs: MemoryFileSystem) -> None:
    texts = (
        Pipeline.from_source(FsspecSource("memory://kb", extensions=["md"]))
        .map(lambda f: f.open().read().decode())
        .collect()
    )

    assert texts == ["alpha", "delta"]

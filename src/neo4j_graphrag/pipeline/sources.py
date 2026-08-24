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
"""Concrete :class:`~neo4j_graphrag.pipeline.source.Source` implementations.

:class:`FsspecSource` reads from any storage backend `fsspec
<https://filesystem-spec.readthedocs.io>`_ supports — the local filesystem,
``s3://``, ``gs://``, ``abfs://``, HTTP, ZIP archives, … — so a pipeline can
be pointed at a new backend by changing only its URL.
"""

from __future__ import annotations

import posixpath
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import IO, Any, Optional, cast

import fsspec
from fsspec import AbstractFileSystem

from neo4j_graphrag.pipeline.source import Source

__all__ = ["FsspecFile", "FsspecSource"]

_GLOB_CHARACTERS = ("*", "?", "[")


def _normalise_extension(extension: str) -> str:
    lowered = extension.lower()
    return lowered if lowered.startswith(".") else f".{lowered}"


@dataclass(frozen=True)
class FsspecFile:
    """A single file, paired with the filesystem it lives on.

    Carrying the filesystem alongside the path keeps downstream stages
    backend-agnostic: they never need to know which protocol or credentials
    produced the file.

    Attributes:
        path: Path to the file as the filesystem addresses it.
        fs: The fsspec filesystem the file can be read from.
    """

    path: str
    fs: AbstractFileSystem

    @property
    def suffix(self) -> str:
        """The lowercased file extension, including the leading dot."""
        return posixpath.splitext(self.path)[1].lower()

    def open(self, mode: str = "rb") -> IO[Any]:
        """Open the file on its filesystem."""
        return cast("IO[Any]", self.fs.open(self.path, mode))


class FsspecSource(Source[FsspecFile]):
    """Emit every file under *urlpath* as an :class:`FsspecFile`.

    *urlpath* may name a single file, a directory (listed recursively), or a
    glob pattern, on any protocol fsspec understands::

        FsspecSource("docs/")                          # local directory
        FsspecSource("s3://bucket/reports/*.pdf")      # glob on S3
        FsspecSource("gs://bucket/kb", extensions=["md"])

    Args:
        urlpath: File, directory, or glob pattern, optionally protocol-prefixed.
        extensions: Keep only files with these extensions (with or without a
            leading dot). ``None`` keeps every file.
        storage_options: Backend-specific options forwarded to fsspec, e.g.
            credentials or an endpoint URL.
    """

    def __init__(
        self,
        urlpath: str,
        *,
        extensions: Optional[Sequence[str]] = None,
        storage_options: Optional[dict[str, Any]] = None,
    ) -> None:
        self.urlpath = urlpath
        self.extensions = (
            None
            if extensions is None
            else tuple(_normalise_extension(e) for e in extensions)
        )
        self.storage_options = storage_options or {}

    def read(self) -> Iterator[FsspecFile]:
        """Yield matching files in a stable (lexicographic) order."""
        fs, path = fsspec.core.url_to_fs(self.urlpath, **self.storage_options)
        for file_path in self._list(fs, path):
            file = FsspecFile(path=file_path, fs=fs)
            if self.extensions is None or file.suffix in self.extensions:
                yield file

    @staticmethod
    def _list(fs: AbstractFileSystem, path: str) -> list[str]:
        if any(character in path for character in _GLOB_CHARACTERS):
            matches = fs.glob(path, detail=True)
            return sorted(
                name for name, info in matches.items() if info.get("type") == "file"
            )
        if fs.isdir(path):
            return sorted(fs.find(path))
        return [path]

"""Ingestion: file loading, offset-preserving chunking, and pipeline orchestration.

Public API:

- :class:`FileLoader` — single parametrized loader for markdown/text files.
- :class:`RecursiveChunker` — offset-preserving recursive chunker.
- :class:`IngestionPipeline` — load -> chunk orchestration, single-source and
  directory-scale. Chunks only: it never writes to a collection, and it is
  deliberately *not* what ``grk ingest`` runs. Persisting to a collection is
  :class:`~groundkit.indexer.Indexer`'s job alone, and ADR-0010 records why
  the two stay separate.
- :class:`LoaderProtocol`, :class:`ChunkerProtocol` — the structural seams
  the above satisfy.
"""

from __future__ import annotations

from groundkit.ingestion.chunking import RecursiveChunker
from groundkit.ingestion.loaders import DEFAULT_EXTENSIONS, DEFAULT_MAX_BYTES, FileLoader
from groundkit.ingestion.pipeline import DEFAULT_MAX_CONCURRENT, IngestionPipeline
from groundkit.ingestion.protocols import ChunkerProtocol, LoaderProtocol

__all__ = [
    "DEFAULT_EXTENSIONS",
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_CONCURRENT",
    "ChunkerProtocol",
    "FileLoader",
    "IngestionPipeline",
    "LoaderProtocol",
    "RecursiveChunker",
]

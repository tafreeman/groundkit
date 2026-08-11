"""Persisted index: BM25, dense vectors (LanceDB), chunk metadata (SQLite).

Survives restarts; supports incremental re-index (Phase 1/3).
"""

from __future__ import annotations

from groundkit.index.bm25 import BM25Index
from groundkit.index.metadata import SQLiteMetadataStore
from groundkit.index.protocols import MetadataStoreProtocol, VectorStoreProtocol

__all__ = [
    "BM25Index",
    "MetadataStoreProtocol",
    "SQLiteMetadataStore",
    "VectorStoreProtocol",
]

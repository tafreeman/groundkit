"""Typed exception hierarchy with a local root (ADR-0001: no ported ancestry).

Unconfigured provider or malformed output is a typed error, never a silent
fallback or coercion (SPEC.md §2, fail closed).
"""

from __future__ import annotations


class GroundkitError(Exception):
    """Base exception for all groundkit errors."""


class ConfigurationError(GroundkitError):
    """Invalid, incomplete, or unknown configuration. Raised at startup."""


class IngestionError(GroundkitError):
    """Error while loading or ingesting a source."""


class ChunkingError(GroundkitError):
    """Error while chunking a document."""


class EmbeddingError(GroundkitError):
    """Error while generating embeddings."""


class ProviderNotConfiguredError(EmbeddingError):
    """A provider was requested but is not configured. Never falls back."""


class StorageError(GroundkitError):
    """Error in the persisted index (metadata store, BM25 store, vector store)."""


class RetrievalError(GroundkitError):
    """Error during retrieval or search."""

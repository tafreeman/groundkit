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


class IndexIdentityError(StorageError):
    """A collection was opened against an embedding identity it was not built with.

    Raised when the persisted collection manifest's
    ``(provider, model_name, dimensions)`` triple does not match the active
    :class:`~groundkit.config.EmbeddingConfig`, or when a store predates the
    manifest entirely (ADR-0004).

    Never a re-embed and never a fallback: mixing semantic spaces in one
    index corrupts it silently (SPEC.md §2), and vector width alone cannot
    detect the swap — distinct models share widths, so identity is the whole
    triple.
    """


class RetrievalError(GroundkitError):
    """Error during retrieval or search."""


class RerankerNotConfiguredError(RetrievalError):
    """A reranker was requested but its backend is unavailable. Never falls back.

    Raised when the optional ``rerank`` extra is not installed, or when the
    configured cross-encoder model cannot be loaded. Deliberately **not** a
    silent passthrough of the input ordering: a reranker that quietly returns
    what it was given is indistinguishable from one that worked, so a
    misconfigured deployment would report rerank-stage numbers that are really
    the upstream stage's (SPEC.md §2, fail closed).
    """


class EvalError(GroundkitError):
    """Error loading, validating, or resolving the golden eval corpus."""

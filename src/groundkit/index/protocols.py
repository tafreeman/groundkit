"""Structural seams for the persisted index (adapted from ARP per ADR-0001).

``VectorStoreProtocol`` keeps ARP's shape minus the ``**kwargs`` catch-all
that silently absorbed misspelled filter arguments (ADR-0001 hazard 3): the
signature is exact, so a wrong keyword is a ``TypeError`` at the call site,
and ``metadata_filter`` must actually filter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from groundkit.contracts import Chunk, RetrievalResult


@runtime_checkable
class MetadataStoreProtocol(Protocol):
    """Durable store for documents, chunks, and ingest state."""

    async def upsert_document(self, source: str, document_id: str, content_hash: str) -> None:
        """Record (or replace) a document's ingest state."""
        ...

    async def get_document_hash(self, source: str) -> str | None:
        """Return the stored content hash for ``source``, or None if unseen."""
        ...

    async def get_document_sources(self) -> dict[str, str]:
        """Return a ``document_id -> source`` map for every stored document."""
        ...

    async def add_chunks(self, chunks: list[Chunk], source: str) -> None:
        """Persist chunks for a document."""
        ...

    async def replace_document(
        self, source: str, document_id: str, content_hash: str, chunks: list[Chunk]
    ) -> None:
        """Atomically replace a document's row and its chunks in one transaction.

        The durable write path. An implementation must commit the document
        row and its chunks together or not at all: the two-call
        ``upsert_document`` + ``add_chunks`` sequence leaves a document row
        carrying a fresh content hash with zero chunks if it is interrupted
        between the calls, and incremental re-index then skips that document
        forever on hash match.
        """
        ...

    async def get_chunks(self) -> list[Chunk]:
        """Return all persisted chunks in the collection."""
        ...

    async def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Return one chunk by ID, or None."""
        ...

    async def delete_document(self, document_id: str) -> int:
        """Delete a document and its chunks. Returns deleted-chunk count."""
        ...


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Dense vector store (implementations arrive in Phase 3)."""

    async def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Store chunks with their embeddings.

        Raises:
            StorageError: On length/dimension mismatch or backend failure.
        """
        ...

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Return the most similar chunks. ``metadata_filter`` keeps only
        chunks whose metadata contains all specified key/value pairs — it is
        never accepted-and-ignored."""
        ...

    async def delete(self, document_id: str) -> int:
        """Delete all vectors for a document. Returns deleted count."""
        ...

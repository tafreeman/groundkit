"""Structural seams for ingestion components (ported from ARP per ADR-0001).

Every concrete implementation gets a conformance test asserting
``isinstance(impl, Protocol)`` and exact signature parity — the missing tests
that let ARP's implementations drift from its protocols (ADR-0001 hazards
3-4) are mandatory here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from groundkit.contracts import Chunk, Document


@runtime_checkable
class LoaderProtocol(Protocol):
    """Loads documents from a source locator."""

    @property
    def supported_extensions(self) -> list[str]:
        """Lowercase file extensions this loader accepts (e.g. ``[".md"]``)."""
        ...

    async def load(self, source: str) -> list[Document]:
        """Load ``source`` into documents.

        Raises:
            IngestionError: If the source cannot be read or is not permitted.
        """
        ...


@runtime_checkable
class ChunkerProtocol(Protocol):
    """Splits documents into offset-addressed chunks."""

    def chunk(self, document: Document, **kwargs: Any) -> list[Chunk]:
        """Split ``document`` into chunks.

        Every returned chunk's ``content`` must equal
        ``document.content[chunk.start_offset:chunk.end_offset]``.

        Raises:
            ChunkingError: If the document cannot be chunked.
        """
        ...

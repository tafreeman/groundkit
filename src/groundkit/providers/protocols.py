"""Structural seam for embedding providers (ported from ARP per ADR-0001)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProtocol(Protocol):
    """Generates embedding vectors for texts."""

    @property
    def provider(self) -> str:
        """Provider identity (e.g. ``"ollama"``)."""
        ...

    @property
    def model_name(self) -> str:
        """Model identity used to embed."""
        ...

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts``, preserving input order.

        Raises:
            EmbeddingError: On provider failure or a dimension mismatch.
            ProviderNotConfiguredError: If the provider is not configured —
                never a silent fallback.
        """
        ...

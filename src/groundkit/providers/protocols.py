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

    @property
    def dimensions(self) -> int:
        """Vector width this embedder produces.

        Part of the seam (rather than read off an ``EmbeddingConfig``
        alongside it) because ADR-0004's collection manifest binds a
        collection to the ``(provider, model_name, dimensions)`` triple of
        whatever actually produced its vectors. Sourcing all three from the
        embedder itself makes "the manifest describes a different model than
        the one that embedded" unrepresentable, instead of a divergence a
        caller has to be trusted not to introduce by passing a config that
        disagrees with the embedder built from a different one.
        """
        ...

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts``, preserving input order.

        Raises:
            EmbeddingError: On provider failure or a dimension mismatch.
            ProviderNotConfiguredError: If the provider is not configured —
                never a silent fallback.
        """
        ...

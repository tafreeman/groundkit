"""Structural seams for provider boundaries.

``EmbeddingProtocol`` was ported from ARP per ADR-0001. ``ChatProtocol`` is
new for Phase 5 (SPEC.md §9) and mirrors its shape deliberately: same
``provider``/``model_name`` identity properties, same fail-closed contract
(a typed error, never a silent fallback, on an unconfigured or failing
provider).
"""

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


@runtime_checkable
class ChatProtocol(Protocol):
    """Generates a chat/completion response for a prompt (Phase 5 boundary)."""

    @property
    def provider(self) -> str:
        """Provider identity (e.g. ``"ollama"``)."""
        ...

    @property
    def model_name(self) -> str:
        """Model identity used to complete."""
        ...

    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        """Complete ``prompt``, optionally under a ``system`` instruction.

        Raises:
            ChatError: On provider failure or an empty completion.
            ChatProviderNotConfiguredError: If the provider is not
                configured — never a silent fallback.
        """
        ...

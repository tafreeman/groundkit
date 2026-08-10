"""Structural seam for rerankers (ported from ARP per ADR-0001; used from
Phase 3). The parameter is named ``query`` — implementations are held to the
exact protocol signature by conformance tests (ADR-0001 hazard 4)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from groundkit.contracts import RetrievalResult


@runtime_checkable
class RerankerProtocol(Protocol):
    """Reorders retrieval results by relevance to the query."""

    async def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        *,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Return ``results`` reordered, truncated to ``top_k``. Scores must
        be normalized ``>= 0.0`` before constructing results."""
        ...

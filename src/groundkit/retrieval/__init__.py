"""Retrieval: search orchestration, citation resolution, and (Phase 3)
fusion + rerank. Deterministic — no LLM in this path."""

from __future__ import annotations

from groundkit.retrieval.citations import resolve_citation, verify_citation
from groundkit.retrieval.fusion import reciprocal_rank_fusion
from groundkit.retrieval.protocols import RerankerProtocol
from groundkit.retrieval.search import MAX_TOP_K, Retriever

__all__ = [
    "MAX_TOP_K",
    "RerankerProtocol",
    "Retriever",
    "reciprocal_rank_fusion",
    "resolve_citation",
    "verify_citation",
]

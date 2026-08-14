"""Reciprocal-rank fusion of lexical and dense result lists (Phase 3, ADR-0005).

Pure code: no I/O, no async, no logging, and no imports beyond stdlib and
:class:`~groundkit.contracts.Chunk` (imported only under ``TYPE_CHECKING``,
exactly as ``index/bm25.py`` does). This module never imports
``groundkit.config``, ``groundkit.errors``, or any store — invalid arguments
raise ``ValueError`` directly, because this is a pure-math seam, not a
pipeline stage.

Every decision below is ADR-0005's, cited here rather than re-derived:

- **Rank, not score (decision 1).** BM25 scores are unbounded and
  corpus-dependent; cosine similarity is bounded but its distribution
  depends on the embedding model. Neither is comparable to the other, so
  fusing by score would need an unprincipled normalization. RRF instead
  consumes only each ranking's permutation:
  ``RRFscore(d) = Σ_{r∈R} 1 / (k + r(d))``. Input scores are therefore
  ignored entirely — only a chunk's 1-based position within each ranking
  matters.
- **``rrf_k`` is a caller-supplied parameter; this module owns no default
  (decision 2).** The value 60 is pinned as ``RetrievalConfig.rrf_k``'s
  default (``config.py``), and its provenance — the original RRF paper's
  pilot investigation, corroborated by Elasticsearch's ``rank_constant``
  default — is recorded in ADR-0005, not here. :func:`reciprocal_rank_fusion`
  takes ``rrf_k`` as a required keyword argument and never substitutes a
  value of its own, so "60" has exactly one source of truth in the repo.
- **Tie-break is ascending ``chunk_id`` (decision 3).** Equal fused scores
  sort by ascending :attr:`~groundkit.contracts.Chunk.chunk_id` — a total,
  content-derived order that never depends on dict or set iteration order.
  This is deliberately distinct from the stores' own ``content_hash``
  tie-break (``index/bm25.py``, ``index/dense.py``); ADR-0005 names
  ``chunk_id`` specifically for fusion.
- **No clamping (decision 5).** With ``rrf_k > 0`` (enforced below) and
  ranks starting at 1, every summed term ``1 / (rrf_k + rank)`` is strictly
  positive by construction, so fused scores satisfy
  :class:`~groundkit.contracts.RetrievalResult`'s ``ge=0.0`` contract without
  a defensive ``max(0.0, ...)`` — a clamp here would mask a bug rather than
  prevent one.

This module builds ``(Chunk, score)`` pairs only, exactly as
:meth:`~groundkit.index.bm25.BM25Index.search` and the dense stores'
``search`` do — never :class:`~groundkit.contracts.RetrievalResult`. The
document-source join that turns a ranked chunk into a citation-bearing
result belongs to ``retrieval/search.py`` (ADR-0006); fusion has no store to
join against and performs no I/O.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from groundkit.contracts import Chunk


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[tuple[Chunk, float]]],
    *,
    rrf_k: int,
    top_k: int | None = None,
) -> list[tuple[Chunk, float]]:
    """Fuse multiple rankings of ``(Chunk, score)`` pairs by reciprocal rank.

    Each input score is ignored — only the 1-based rank of a chunk within
    its own ranking contributes to the fused score (ADR-0005 decision 1).
    A chunk's identity across rankings is its ``chunk_id``.

    Args:
        rankings: One sequence of ``(Chunk, score)`` pairs per retriever,
            each already sorted best-first. Input scores are ignored; only
            list position (rank) matters. Not mutated.
        rrf_k: RRF's rank-constant. Must be > 0. Callers pass
            ``RetrievalConfig.rrf_k`` (default 60; provenance in ADR-0005
            decision 2) — this function has no default of its own.
        top_k: Maximum number of fused results to return, applied after
            sorting. ``None`` returns every fused chunk. Must be > 0 if
            given.

    Returns:
        ``(Chunk, score)`` pairs sorted by descending fused score, ties
        broken by ascending ``chunk_id`` (ADR-0005 decision 3). When a
        chunk_id appears in more than one ranking, the emitted ``Chunk``
        object is the one from the first (lowest-index) ranking containing
        it. Empty ``rankings``, or rankings that are all empty, yield ``[]``.

    Raises:
        ValueError: ``rrf_k <= 0``; ``top_k`` is not ``None`` and
            ``top_k <= 0``; or a single ranking contains the same
            ``chunk_id`` more than once (a caller bug — fail loud rather
            than silently double-counting it).
    """
    if rrf_k <= 0:
        raise ValueError(f"rrf_k must be > 0, got {rrf_k}")
    if top_k is not None and top_k <= 0:
        raise ValueError(f"top_k must be > 0 if given, got {top_k}")

    fused_scores: dict[str, float] = {}
    first_chunk: dict[str, Chunk] = {}

    for ranking in rankings:
        seen_in_ranking: set[str] = set()
        for rank, (chunk, _score) in enumerate(ranking, start=1):
            chunk_id = chunk.chunk_id
            if chunk_id in seen_in_ranking:
                raise ValueError(f"duplicate chunk_id {chunk_id!r} within one ranking")
            seen_in_ranking.add(chunk_id)

            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            if chunk_id not in first_chunk:
                first_chunk[chunk_id] = chunk

    fused = [(first_chunk[chunk_id], score) for chunk_id, score in fused_scores.items()]
    fused.sort(key=lambda pair: (-pair[1], pair[0].chunk_id))

    if top_k is not None:
        fused = fused[:top_k]

    return fused

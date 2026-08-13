"""Deterministic retrieval eval metrics: recall@k, MRR, nDCG@k (SPEC.md §6).

Pure functions over opaque ``str`` IDs. This module does not import
:mod:`groundkit.contracts` or any other groundkit module, so it stays a
leaf: its own unit tests construct plain lists and sets, never Pydantic
objects. The runner (built separately) adapts real retrieval results via
``[r.chunk_id for r in response.results]``.

:func:`recall_at_k` and :func:`ndcg_at_k` look redundant but aren't.
``recall_at_k`` is **hit-rate**, not set-recall (``|retrieved ∩ gold| /
|gold|``): it scores ``1.0`` if *any* of the top-``k`` ranked IDs is in
``gold_ids``. Set-recall would let chunk-boundary artifacts decide the
score — a single gold quote straddling a chunk boundary produces two gold
chunks purely because of how the corpus was chunked, and set-recall would
reward a retriever for finding both fragments of *one* answer over a
retriever that finds a different, unfragmented answer. Hit-rate treats the
pair as a single opportunity, so chunk size stops leaking into the score.
Multi-passage credit — rewarding a retriever for surfacing *more* of the
relevant material, not just the same material split differently — is
nDCG's job instead.

Every function here is total over well-formed inputs: an empty
``gold_ids`` returns ``0.0`` rather than raising. These are scoring
primitives, not validators — rejecting a genuinely empty gold set (a
malformed eval example) is an aggregator's job, not this leaf's. A
non-positive ``k``, by contrast, is a caller bug rather than a data
condition, so it raises ``ValueError`` — matching how
``Indexer.index_directory`` treats ``max_concurrent < 1``.

There is no numpy/pandas/scipy dependency in this repo, and none may be
added — every computation here is hand-rolled from stdlib ``math``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from collections.abc import Set as AbstractSet

#: SPEC.md §6 — the k cutoffs reported for recall@k in the eval harness.
RECALL_K_VALUES: tuple[int, ...] = (1, 5, 10)

#: SPEC.md §6 — the cutoff reported for nDCG in the eval harness.
NDCG_K: int = 10


def recall_at_k(ranked_ids: Sequence[str], gold_ids: AbstractSet[str], *, k: int) -> float:
    """Hit-rate at ``k``: did any of the top-``k`` ranked IDs land in gold?

    This is hit-rate, not set-recall — see the module docstring for why a
    retriever that surfaces one of several gold IDs scores identically to
    one that surfaces all of them. Multi-passage credit is
    :func:`ndcg_at_k`'s job.

    Args:
        ranked_ids: Retrieved IDs in descending rank order (best first).
        gold_ids: The set of IDs considered relevant.
        k: Number of top-ranked IDs to consider. Must be ``>= 1``.

    Returns:
        ``1.0`` if any of ``ranked_ids[:k]`` is in ``gold_ids``, else
        ``0.0``. Also ``0.0`` when ``gold_ids`` is empty.

    Raises:
        ValueError: ``k`` is less than 1.
    """
    _reject_non_positive_k(k)
    if not gold_ids:
        return 0.0
    return 1.0 if any(rid in gold_ids for rid in ranked_ids[:k]) else 0.0


def reciprocal_rank(ranked_ids: Sequence[str], gold_ids: AbstractSet[str]) -> float:
    """Reciprocal rank of the first gold hit.

    TREC convention: a run with no gold hit contributes ``0.0`` rather
    than being excluded from the average — excluding misses would let a
    run look better simply by returning fewer results.

    Args:
        ranked_ids: Retrieved IDs in descending rank order (best first).
        gold_ids: The set of IDs considered relevant.

    Returns:
        ``1 / rank`` of the first (1-based) position in ``ranked_ids``
        that is in ``gold_ids``, or ``0.0`` if none is found or
        ``gold_ids`` is empty.
    """
    if not gold_ids:
        return 0.0
    for rank, rid in enumerate(ranked_ids, start=1):
        if rid in gold_ids:
            return 1.0 / rank
    return 0.0


def dcg_at_k(ranked_ids: Sequence[str], gold_ids: AbstractSet[str], *, k: int) -> float:
    """Discounted cumulative gain over the top ``k`` ranked IDs.

    ``DCG = sum(rel_i / log2(i + 1) for i in 1..k)`` with binary gain
    ``rel_i`` (``1`` if ``ranked_ids[i-1]`` is in ``gold_ids`` else ``0``)
    and 1-based rank ``i``. Binary gain is used rather than the
    exponential ``2**rel - 1`` form: the two are identical for binary
    relevance, so the exponential form adds nothing here.

    Args:
        ranked_ids: Retrieved IDs in descending rank order (best first).
        gold_ids: The set of IDs considered relevant.
        k: Number of top-ranked IDs to consider. Must be ``>= 1``.

    Returns:
        The DCG score, ``>= 0.0``. ``0.0`` when ``gold_ids`` is empty.

    Raises:
        ValueError: ``k`` is less than 1.
    """
    _reject_non_positive_k(k)
    if not gold_ids:
        return 0.0
    return sum(
        (
            1.0 / math.log2(rank + 1)
            for rank, rid in enumerate(ranked_ids[:k], start=1)
            if rid in gold_ids
        ),
        0.0,
    )


def ndcg_at_k(ranked_ids: Sequence[str], gold_ids: AbstractSet[str], *, k: int) -> float:
    """Normalized discounted cumulative gain over the top ``k`` ranked IDs.

    ``nDCG = DCG / IDCG`` where ``IDCG`` is the DCG of the best possible
    ranking: ``sum(1 / log2(i + 1) for i in 1..min(len(gold_ids), k))``.
    IDCG is capped at ``k`` (an ideal ranking can't place more relevant
    items in the top ``k`` than ``k`` allows) and at ``len(gold_ids)``
    (there's no gain left to earn past the last gold item).

    Args:
        ranked_ids: Retrieved IDs in descending rank order (best first).
        gold_ids: The set of IDs considered relevant.
        k: Number of top-ranked IDs to consider. Must be ``>= 1``.

    Returns:
        The nDCG score, clamped to ``[0.0, 1.0]``. ``0.0`` when
        ``gold_ids`` is empty (IDCG would be zero, so the division is
        skipped rather than performed).

    Raises:
        ValueError: ``k`` is less than 1.
    """
    _reject_non_positive_k(k)
    if not gold_ids:
        return 0.0
    dcg = dcg_at_k(ranked_ids, gold_ids, k=k)
    ideal_hits = min(len(gold_ids), k)
    idcg = sum((1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1)), 0.0)
    return _clamp_unit_interval(dcg / idcg)


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean of ``values``.

    Args:
        values: The values to average.

    Returns:
        The arithmetic mean.

    Raises:
        ValueError: ``values`` is empty — an empty average is meaningless,
            and silently returning ``0.0`` would corrupt an aggregate.
    """
    if not values:
        raise ValueError("values must be non-empty")
    return sum(values) / len(values)


def _reject_non_positive_k(k: int) -> None:
    """Raise ``ValueError`` if ``k`` is not a positive integer.

    Args:
        k: The cutoff value to validate.

    Raises:
        ValueError: ``k`` is less than 1.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")


def _clamp_unit_interval(value: float) -> float:
    """Clamp ``value`` into ``[0.0, 1.0]``.

    ``DCG <= IDCG`` holds exactly in real arithmetic, since IDCG sums the
    same per-rank gain terms in the best achievable order. But float
    summation is not associative, so summation order can drift the ratio
    a few ULPs past either boundary; clamping absorbs that drift instead
    of leaking it into callers as a nonsensical out-of-range score.

    Args:
        value: The ratio to clamp.

    Returns:
        ``value`` restricted to ``[0.0, 1.0]``.
    """
    return max(0.0, min(1.0, value))

"""Tests for the deterministic eval metrics engine (SPEC.md §6).

Every function under test is a pure function over plain ``str`` IDs, so
these tests construct only lists and sets — never Pydantic objects. That
mirrors the module's own leaf constraint: metrics.py must not import
``groundkit.contracts`` or any other groundkit module.
"""

from __future__ import annotations

import math

import pytest

from groundkit.evals.metrics import (
    dcg_at_k,
    mean,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestRecallAtK:
    """recall_at_k is HIT-RATE: 1.0 if any of the top-k ranked IDs is gold."""

    def test_hit_at_rank_one(self) -> None:
        """A gold ID at the very top of the ranking scores a full hit."""
        assert recall_at_k(["a", "b", "c"], {"a"}, k=1) == 1.0

    def test_hit_only_beyond_k_scores_zero(self) -> None:
        """A gold ID ranked below the cutoff does not count as a hit."""
        assert recall_at_k(["a", "b", "c"], {"c"}, k=2) == 0.0

    def test_no_hit_scores_zero(self) -> None:
        """No ranked ID is gold: the whole top-k window is a miss."""
        assert recall_at_k(["a", "b", "c"], {"z"}, k=3) == 0.0

    def test_any_single_hit_among_multiple_gold_counts_as_full_hit(self) -> None:
        """Pins hit-rate semantics rather than set-recall.

        Two gold IDs exist ("a" and "z"), but only "a" is retrieved
        inside k=1. Set-recall (|retrieved & gold| / |gold|) would score
        this 0.5; recall_at_k is hit-rate and scores it a full 1.0 —
        finding one fragment of a gold quote that straddled a chunk
        boundary is as good, for this metric, as finding every fragment.
        """
        assert recall_at_k(["a", "b", "c"], {"a", "z"}, k=1) == 1.0

    def test_empty_gold_returns_zero_without_raising(self) -> None:
        """An empty gold set is a legitimate (if degenerate) input."""
        assert recall_at_k(["a", "b"], set(), k=5) == 0.0

    def test_k_larger_than_ranked_list(self) -> None:
        """k beyond len(ranked_ids) degrades naturally via slicing."""
        assert recall_at_k(["a"], {"a"}, k=100) == 1.0

    def test_empty_ranked_list(self) -> None:
        """No results at all is a miss, not an error."""
        assert recall_at_k([], {"a"}, k=5) == 0.0

    @pytest.mark.parametrize("k", [0, -1, -50])
    def test_non_positive_k_raises(self, k: int) -> None:
        with pytest.raises(ValueError, match="k must be >= 1"):
            recall_at_k(["a"], {"a"}, k=k)


class TestReciprocalRank:
    """reciprocal_rank is 1/rank of the first gold hit, 0.0 if none."""

    def test_first_result_gold_scores_one(self) -> None:
        assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0

    def test_second_result_gold_scores_half(self) -> None:
        assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5

    def test_third_result_gold_scores_one_third(self) -> None:
        assert reciprocal_rank(["a", "b", "c"], {"c"}) == pytest.approx(1.0 / 3.0)

    def test_no_gold_found_scores_zero(self) -> None:
        """TREC convention: a miss contributes 0.0 rather than being excluded."""
        assert reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0

    def test_empty_gold_scores_zero(self) -> None:
        assert reciprocal_rank(["a", "b"], set()) == 0.0

    def test_empty_ranked_scores_zero(self) -> None:
        assert reciprocal_rank([], {"a"}) == 0.0


class TestDcgAndNdcgAtK:
    """dcg_at_k / ndcg_at_k: binary-gain DCG normalized by capped IDCG."""

    def test_perfect_ranking_scores_one(self) -> None:
        """Every top-k slot filled by a gold ID is the best possible ranking."""
        assert ndcg_at_k(["a", "b"], {"a", "b"}, k=2) == pytest.approx(1.0)

    def test_hand_computed_reference_value(self) -> None:
        """Pins a hand-computed nDCG@5 value.

        ranked = [a, b, c, d, e], gold = {b, d}, k=5.
        Binary relevance by 1-based rank: a=0, b=1, c=0, d=1, e=0.

            DCG  = 1/log2(3) + 1/log2(5)
                 = 0.6309297535714573 + 0.4306765580733931
                 = 1.0616063116448506
            IDCG = 1/log2(2) + 1/log2(3)   # best case: both gold hits at ranks 1, 2
                 = 1.0 + 0.6309297535714573
                 = 1.6309297535714573
            nDCG = DCG / IDCG
                 = 0.6509209298071326

        These values were produced by running ``math.log2`` through the
        interpreter (``python -c "import math; ..."``), not computed by
        hand — per repo convention, model-authored arithmetic in tests is
        untrustworthy and every reference value must be derived, not
        guessed.
        """
        ranked = ["a", "b", "c", "d", "e"]
        gold = {"b", "d"}

        assert dcg_at_k(ranked, gold, k=5) == pytest.approx(1.0616063116448506)
        assert ndcg_at_k(ranked, gold, k=5) == pytest.approx(0.6509209298071326)

    def test_idcg_capped_when_gold_smaller_than_k(self) -> None:
        """IDCG sums len(gold_ids) terms, not k, when gold is scarce.

        Only one gold ID exists ("a"), found at rank 2 of 5. An ideal
        ranking could place at most that one gold ID at rank 1 — it
        cannot manufacture a second relevant item to fill rank 2 of the
        ideal ranking too, so IDCG has exactly one term despite k=5.
        """
        ranked = ["x", "a", "c"]
        gold = {"a"}
        k = 5

        expected_dcg = 1.0 / math.log2(3)
        expected_idcg = 1.0 / math.log2(2)

        assert dcg_at_k(ranked, gold, k=k) == pytest.approx(expected_dcg)
        assert ndcg_at_k(ranked, gold, k=k) == pytest.approx(expected_dcg / expected_idcg)

    def test_idcg_capped_at_k_when_gold_larger_than_k(self) -> None:
        """IDCG sums only k terms, not len(gold_ids), when gold is plentiful.

        Six gold IDs exist but k=2: an ideal ranking can place at most 2
        relevant items in the top 2, so IDCG stops at 2 terms. Both
        top-2 ranked IDs here are gold, so DCG matches that capped IDCG
        exactly and the score is a perfect 1.0 despite 4 gold IDs never
        being retrieved at all.
        """
        ranked = ["a", "b", "x", "x", "x"]
        gold = {"a", "b", "c", "d", "e", "f"}
        k = 2

        assert ndcg_at_k(ranked, gold, k=k) == pytest.approx(1.0)

    def test_relevant_item_beyond_k_not_counted(self) -> None:
        """A gold ID ranked past the cutoff contributes no gain at all."""
        assert dcg_at_k(["x", "y", "gold"], {"gold"}, k=2) == 0.0
        assert ndcg_at_k(["x", "y", "gold"], {"gold"}, k=2) == 0.0

    def test_empty_gold_scores_zero(self) -> None:
        """An empty gold set returns 0.0 rather than dividing by zero IDCG."""
        assert dcg_at_k(["a", "b"], set(), k=2) == 0.0
        assert ndcg_at_k(["a", "b"], set(), k=2) == 0.0

    @pytest.mark.parametrize("k", [0, -1, -50])
    def test_non_positive_k_raises(self, k: int) -> None:
        with pytest.raises(ValueError, match="k must be >= 1"):
            dcg_at_k(["a"], {"a"}, k=k)
        with pytest.raises(ValueError, match="k must be >= 1"):
            ndcg_at_k(["a"], {"a"}, k=k)

    @pytest.mark.parametrize(
        ("ranked", "gold", "k"),
        [
            (["a", "b", "c"], {"a", "b", "c"}, 3),
            (["a", "b", "c"], {"z"}, 3),
            ([], {"a"}, 5),
            (["a"], set(), 1),
            (["a", "b", "c", "d", "e"], {"b", "d"}, 10),
        ],
    )
    def test_result_always_within_unit_interval(
        self, ranked: list[str], gold: set[str], k: int
    ) -> None:
        """ndcg_at_k never leaves [0, 1], including at k > len(ranked_ids)."""
        assert dcg_at_k(ranked, gold, k=k) >= 0.0
        assert 0.0 <= ndcg_at_k(ranked, gold, k=k) <= 1.0


class TestMean:
    """mean is a plain arithmetic mean that refuses an empty input."""

    def test_arithmetic_mean(self) -> None:
        assert mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_single_value(self) -> None:
        assert mean([5.0]) == pytest.approx(5.0)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="values must be non-empty"):
            mean([])

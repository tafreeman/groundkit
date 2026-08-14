"""Tests for reciprocal_rank_fusion (ADR-0005 decisions 1, 2, 3, 5).

``reciprocal_rank_fusion`` is a pure function over ``(Chunk, score)`` pairs —
no store, no async, no I/O — so these tests build minimal ``Chunk`` fixtures
directly and assert on plain (chunk_id, score) tuples, mirroring
``tests/test_bm25.py``'s conventions.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from groundkit.contracts import Chunk
from groundkit.retrieval.fusion import reciprocal_rank_fusion


def _make_chunk(
    content: str,
    *,
    chunk_id: str,
    document_id: str = "doc-1",
    chunk_index: int = 0,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        content=content,
        start_offset=0,
        end_offset=len(content),
    )


class TestHandComputedScores:
    """Fused scores match the RRF formula exactly, pinned via Fraction math."""

    def test_rank_one_in_both_rankings_scores_two_over_rrf_k_plus_one(self) -> None:
        """A chunk ranked first in both rankings: RRFscore = 2 / (rrf_k + 1)."""
        chunk = _make_chunk("shared content", chunk_id="shared")
        ranking_a = [(chunk, 10.0)]
        ranking_b = [(chunk, 3.0)]

        results = reciprocal_rank_fusion([ranking_a, ranking_b], rrf_k=60)

        assert len(results) == 1
        result_chunk, score = results[0]
        assert result_chunk.chunk_id == "shared"
        assert score == pytest.approx(float(Fraction(2, 61)), rel=1e-9)

    def test_disjoint_and_overlapping_chunks_sum_independently(self) -> None:
        """Each ranking contributes 1/(rrf_k + rank); a chunk present in
        both rankings sums the contribution from each one."""
        c1 = _make_chunk("one", chunk_id="c1")
        c2 = _make_chunk("two", chunk_id="c2")
        c3 = _make_chunk("three", chunk_id="c3")
        c4 = _make_chunk("four", chunk_id="c4")
        ranking_a = [(c1, 9.0), (c2, 8.0), (c3, 7.0)]  # c1@1, c2@2, c3@3
        ranking_b = [(c3, 5.0), (c1, 4.0), (c4, 3.0)]  # c3@1, c1@2, c4@3

        results = reciprocal_rank_fusion([ranking_a, ranking_b], rrf_k=60)
        scores = {chunk.chunk_id: score for chunk, score in results}

        assert scores["c1"] == pytest.approx(float(Fraction(1, 61) + Fraction(1, 62)), rel=1e-9)
        assert scores["c2"] == pytest.approx(float(Fraction(1, 62)), rel=1e-9)
        assert scores["c3"] == pytest.approx(float(Fraction(1, 63) + Fraction(1, 61)), rel=1e-9)
        assert scores["c4"] == pytest.approx(float(Fraction(1, 63)), rel=1e-9)


class TestOrderingCorrectness:
    def test_fused_order_differs_from_both_input_rankings(self) -> None:
        """A designed overlap where the fused order matches neither input
        ranking's order, proving fusion genuinely recombines the two."""
        c1 = _make_chunk("one", chunk_id="c1")
        c2 = _make_chunk("two", chunk_id="c2")
        c3 = _make_chunk("three", chunk_id="c3")
        c4 = _make_chunk("four", chunk_id="c4")
        ranking_a = [(c1, 1.0), (c2, 1.0), (c3, 1.0)]  # input order: c1, c2, c3
        ranking_b = [(c3, 1.0), (c1, 1.0), (c4, 1.0)]  # input order: c3, c1, c4

        results = reciprocal_rank_fusion([ranking_a, ranking_b], rrf_k=60)

        # Fused: c1 (1/61+1/62) > c3 (1/63+1/61) > c2 (1/62) > c4 (1/63).
        assert [chunk.chunk_id for chunk, _ in results] == ["c1", "c3", "c2", "c4"]


class TestDeterminism:
    def test_identical_inputs_from_fresh_objects_produce_byte_identical_output(self) -> None:
        """Rebuilding equal-but-distinct Chunk/list objects from scratch
        must reproduce an identical ordered (chunk_id, score) list — nothing
        may depend on object identity, dict iteration, or construction order."""

        def _build_rankings() -> list[list[tuple[Chunk, float]]]:
            c1 = _make_chunk("one", chunk_id="c1")
            c2 = _make_chunk("two", chunk_id="c2")
            c3 = _make_chunk("three", chunk_id="c3")
            ranking_a = [(c1, 1.0), (c2, 1.0), (c3, 1.0)]
            ranking_b = [(c3, 1.0), (c2, 1.0), (c1, 1.0)]
            return [ranking_a, ranking_b]

        first = reciprocal_rank_fusion(_build_rankings(), rrf_k=60)
        second = reciprocal_rank_fusion(_build_rankings(), rrf_k=60)

        first_view = [(chunk.chunk_id, score) for chunk, score in first]
        second_view = [(chunk.chunk_id, score) for chunk, score in second]
        assert first_view == second_view


class TestTieBreak:
    def test_equal_fused_scores_sort_by_ascending_chunk_id(self) -> None:
        """Two chunks that each rank first in exactly one ranking produce an
        exact score tie (both compute 1/(rrf_k + 1)); the tie must resolve
        by ascending chunk_id, not insertion order."""
        zeta = _make_chunk("zeta content", chunk_id="zeta")
        alpha = _make_chunk("alpha content", chunk_id="alpha")
        ranking_a = [(zeta, 1.0)]  # zeta inserted first
        ranking_b = [(alpha, 1.0)]

        results = reciprocal_rank_fusion([ranking_a, ranking_b], rrf_k=60)

        assert [chunk.chunk_id for chunk, _ in results] == ["alpha", "zeta"]
        assert results[0][1] == pytest.approx(results[1][1])


class TestInputScoresIgnored:
    def test_perturbing_input_scores_does_not_change_fused_output(self) -> None:
        """Only rank position matters; changing the numeric scores in the
        input pairs while preserving their order must not change the
        fused result at all."""
        c1 = _make_chunk("one", chunk_id="c1")
        c2 = _make_chunk("two", chunk_id="c2")
        ranking_a = [(c1, 100.0), (c2, 99.9)]
        ranking_b = [(c2, 0.001), (c1, 0.0009)]

        baseline = reciprocal_rank_fusion([ranking_a, ranking_b], rrf_k=60)

        perturbed_a = [(c1, 0.5), (c2, 0.4)]
        perturbed_b = [(c2, -3.0), (c1, -3.1)]
        perturbed = reciprocal_rank_fusion([perturbed_a, perturbed_b], rrf_k=60)

        baseline_view = [(chunk.chunk_id, score) for chunk, score in baseline]
        perturbed_view = [(chunk.chunk_id, score) for chunk, score in perturbed]
        assert baseline_view == perturbed_view


class TestChunkProvenance:
    def test_shared_chunk_id_emits_chunk_object_from_first_ranking(self) -> None:
        """When the same chunk_id appears in more than one ranking, the
        emitted Chunk must be the one from the FIRST (lowest-index) ranking
        containing it, even though both objects represent 'the same' chunk."""
        first_chunk = _make_chunk("first ranking's copy", chunk_id="shared")
        second_chunk = _make_chunk("second ranking's copy", chunk_id="shared")
        assert first_chunk is not second_chunk

        results = reciprocal_rank_fusion([[(first_chunk, 1.0)], [(second_chunk, 1.0)]], rrf_k=60)

        assert len(results) == 1
        result_chunk, _ = results[0]
        assert result_chunk is first_chunk
        assert result_chunk.content == "first ranking's copy"


class TestValueErrors:
    def test_rrf_k_zero_raises(self) -> None:
        chunk = _make_chunk("content", chunk_id="c1")
        with pytest.raises(ValueError, match="rrf_k"):
            reciprocal_rank_fusion([[(chunk, 1.0)]], rrf_k=0)

    def test_rrf_k_negative_raises(self) -> None:
        chunk = _make_chunk("content", chunk_id="c1")
        with pytest.raises(ValueError, match="rrf_k"):
            reciprocal_rank_fusion([[(chunk, 1.0)]], rrf_k=-1)

    def test_top_k_zero_raises(self) -> None:
        chunk = _make_chunk("content", chunk_id="c1")
        with pytest.raises(ValueError, match="top_k"):
            reciprocal_rank_fusion([[(chunk, 1.0)]], rrf_k=60, top_k=0)

    def test_top_k_negative_raises(self) -> None:
        chunk = _make_chunk("content", chunk_id="c1")
        with pytest.raises(ValueError, match="top_k"):
            reciprocal_rank_fusion([[(chunk, 1.0)]], rrf_k=60, top_k=-5)

    def test_duplicate_chunk_id_within_one_ranking_raises(self) -> None:
        """A ranking listing the same chunk_id twice is a caller bug —
        fusion fails loud rather than silently double-counting it."""
        dup_a = _make_chunk("copy a", chunk_id="dup")
        dup_b = _make_chunk("copy b", chunk_id="dup")
        with pytest.raises(ValueError, match="duplicate chunk_id"):
            reciprocal_rank_fusion([[(dup_a, 1.0), (dup_b, 1.0)]], rrf_k=60)


class TestTopKAndEmptyInputs:
    def test_top_k_truncates_after_sorting_not_before(self) -> None:
        """top_k must apply to the fused, sorted order — not to whatever
        order chunks were first encountered while merging rankings (which
        would, for this fixture, wrongly yield c1, c2)."""
        c1 = _make_chunk("one", chunk_id="c1")
        c2 = _make_chunk("two", chunk_id="c2")
        c3 = _make_chunk("three", chunk_id="c3")
        c4 = _make_chunk("four", chunk_id="c4")
        ranking_a = [(c1, 1.0), (c2, 1.0), (c3, 1.0)]
        ranking_b = [(c3, 1.0), (c1, 1.0), (c4, 1.0)]

        results = reciprocal_rank_fusion([ranking_a, ranking_b], rrf_k=60, top_k=2)

        assert [chunk.chunk_id for chunk, _ in results] == ["c1", "c3"]

    def test_top_k_none_returns_all_fused_results(self) -> None:
        chunks = [_make_chunk(f"content {i}", chunk_id=f"c{i}") for i in range(5)]
        ranking = [(chunk, 1.0) for chunk in chunks]

        results = reciprocal_rank_fusion([ranking], rrf_k=60, top_k=None)

        assert len(results) == 5

    def test_top_k_larger_than_result_count_returns_all(self) -> None:
        chunk = _make_chunk("content", chunk_id="c1")

        results = reciprocal_rank_fusion([[(chunk, 1.0)]], rrf_k=60, top_k=1000)

        assert len(results) == 1

    def test_empty_rankings_sequence_returns_empty_list(self) -> None:
        assert reciprocal_rank_fusion([], rrf_k=60) == []

    def test_all_empty_rankings_return_empty_list(self) -> None:
        assert reciprocal_rank_fusion([[], []], rrf_k=60) == []


class TestInputsNotMutated:
    def test_ranking_lists_and_tuples_unchanged_after_fusion(self) -> None:
        c1 = _make_chunk("one", chunk_id="c1")
        c2 = _make_chunk("two", chunk_id="c2")
        ranking_a = [(c1, 5.0), (c2, 4.0)]
        ranking_b = [(c2, 3.0), (c1, 2.0)]
        rankings = [ranking_a, ranking_b]
        snapshot_a = list(ranking_a)
        snapshot_b = list(ranking_b)

        reciprocal_rank_fusion(rankings, rrf_k=60, top_k=1)

        assert ranking_a == snapshot_a
        assert ranking_b == snapshot_b
        assert rankings == [ranking_a, ranking_b]

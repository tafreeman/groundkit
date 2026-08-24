from __future__ import annotations

import pytest

from groundkit.errors import EvalError
from groundkit.evals.schema import (
    EvalReport,
    GoldSpanResult,
    MetricSet,
    QueryMetrics,
    QueryResult,
    RunConfig,
    RunMetadata,
    StageResult,
)
from groundkit.evals.significance import compare_report_stages, paired_bootstrap


def test_paired_bootstrap_is_deterministic_and_defaults_to_ten_thousand_resamples() -> None:
    baseline = {f"q{i}": 0.0 for i in range(12)}
    candidate = {f"q{i}": 1.0 for i in range(12)}

    first = paired_bootstrap(
        baseline,
        candidate,
        baseline_name="bm25",
        candidate_name="dense",
        metric="ndcg_at_10",
    )
    second = paired_bootstrap(
        baseline,
        candidate,
        baseline_name="bm25",
        candidate_name="dense",
        metric="ndcg_at_10",
    )

    assert first == second
    assert first.resamples == 10_000
    assert first.query_count == 12
    assert first.effect == pytest.approx(1.0)
    assert first.confidence_interval_low == pytest.approx(1.0)
    assert first.confidence_interval_high == pytest.approx(1.0)
    # Never exactly zero: a Monte Carlo estimate over B draws cannot resolve a
    # probability below 1/B, so a zero tail count is reported at the floor
    # `2 * (0 + 1) / (B + 1)` rather than as an impossibility. This fixture is
    # the degenerate extreme -- every delta is identically 1.0, so no resample
    # can cross zero and the tail count is exactly zero on every run.
    assert first.p_value_two_sided == pytest.approx(2.0 / 10_001)
    assert first.significant is True


def test_p_value_is_never_zero_even_when_no_resample_crosses_zero() -> None:
    """The floor is a property of the estimator, not of this one fixture.

    Asserted separately from the determinism test above so that a future
    change to that fixture cannot quietly remove the only coverage of it.
    """
    baseline = {f"q{i}": 0.0 for i in range(30)}
    candidate = {f"q{i}": 5.0 for i in range(30)}

    for resamples in (100, 1_000, 10_000):
        result = paired_bootstrap(
            baseline,
            candidate,
            baseline_name="a",
            candidate_name="b",
            metric="ndcg_at_10",
            resamples=resamples,
        )
        assert result.p_value_two_sided > 0.0
        assert result.p_value_two_sided == pytest.approx(2.0 / (resamples + 1))
        # The interval, which is what `significant` actually reads, is
        # unaffected by the correction.
        assert result.significant is True


def test_paired_bootstrap_preserves_query_pairing_independent_of_mapping_order() -> None:
    baseline = {"q1": 0.1, "q2": 0.9, "q3": 0.2}
    candidate = {"q3": 0.4, "q1": 0.3, "q2": 0.8}

    result = paired_bootstrap(
        baseline,
        candidate,
        baseline_name="a",
        candidate_name="b",
        metric="mrr",
        resamples=200,
        seed=11,
    )

    assert result.effect == pytest.approx((0.3 - 0.1 + 0.8 - 0.9 + 0.4 - 0.2) / 3)


def test_paired_bootstrap_rejects_unpaired_queries() -> None:
    with pytest.raises(EvalError, match="query ids differ"):
        paired_bootstrap(
            {"q1": 0.5},
            {"q2": 0.5},
            baseline_name="a",
            candidate_name="b",
            metric="recall_at_10",
        )


@pytest.mark.parametrize("resamples", [0, -1])
def test_paired_bootstrap_rejects_non_positive_resamples(resamples: int) -> None:
    with pytest.raises(EvalError, match="resamples must be positive"):
        paired_bootstrap(
            {"q": 0.0},
            {"q": 1.0},
            baseline_name="a",
            candidate_name="b",
            metric="ndcg_at_10",
            resamples=resamples,
        )


@pytest.mark.parametrize("confidence_level", [0.0, 1.0, -0.5, 1.5])
def test_paired_bootstrap_rejects_a_confidence_level_outside_the_open_unit_interval(
    confidence_level: float,
) -> None:
    with pytest.raises(EvalError, match="confidence_level must be between 0 and 1"):
        paired_bootstrap(
            {"q": 0.0},
            {"q": 1.0},
            baseline_name="a",
            candidate_name="b",
            metric="ndcg_at_10",
            confidence_level=confidence_level,
        )


def test_paired_bootstrap_rejects_empty_scores() -> None:
    with pytest.raises(EvalError, match="at least one query score"):
        paired_bootstrap({}, {}, baseline_name="a", candidate_name="b", metric="mrr")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_paired_bootstrap_rejects_non_finite_scores(bad: float) -> None:
    """A NaN delta would propagate silently into every resample mean and make
    the comparison meaningless rather than wrong-looking."""
    with pytest.raises(EvalError, match="must be finite"):
        paired_bootstrap(
            {"q": 0.0},
            {"q": bad},
            baseline_name="a",
            candidate_name="b",
            metric="ndcg_at_10",
        )


# -- compare_report_stages ------------------------------------------------------


def _query(query_id: str, *, ndcg: float, rr: float = 0.5) -> QueryResult:
    return QueryResult(
        query_id=query_id,
        query=f"query {query_id}",
        category="normal",
        is_no_answer=False,
        gold=[GoldSpanResult(document="a.md", start_offset=0, end_offset=4, quote="text")],
        total_relevant_chunks=1,
        retrieved=[],
        metrics=QueryMetrics(
            recall_at_1=1.0,
            recall_at_5=1.0,
            recall_at_10=1.0,
            reciprocal_rank=rr,
            ndcg_at_10=ndcg,
        ),
        latency_ms=1.0,
    )


def _no_answer_query(query_id: str) -> QueryResult:
    """A no-answer query carries ``metrics=None`` and must be skipped by the
    pairing rather than counted as a zero."""
    return QueryResult(
        query_id=query_id,
        query=f"query {query_id}",
        category="no_answer",
        is_no_answer=True,
        gold=[],
        total_relevant_chunks=0,
        retrieved=[],
        metrics=None,
        latency_ms=1.0,
    )


def _stage(stage: str, queries: list[QueryResult], *, is_baseline: bool) -> StageResult:
    return StageResult(
        stage=stage,  # type: ignore[arg-type]
        is_baseline=is_baseline,
        aggregate=MetricSet(
            query_count=len([q for q in queries if q.metrics is not None]),
            recall_at_1=0.5,
            recall_at_5=0.5,
            recall_at_10=0.5,
            mrr=0.5,
            ndcg_at_10=0.5,
        ),
        by_category={},
        no_answer_query_count=len([q for q in queries if q.metrics is None]),
        no_answer_abstained_count=0,
        latency_p50_ms=1.0,
        latency_p95_ms=2.0,
        latency_p99_ms=3.0,
        queries=queries,
    )


def _report(stages: list[StageResult]) -> EvalReport:
    return EvalReport(
        run=RunMetadata(
            started_at="2026-08-24T00:00:00+00:00",
            groundkit_version="0.0.0",
            corpus_hash="c" * 64,
            judgments_hash="j" * 64,
            document_count=1,
            chunk_count=2,
            judgment_count=2,
            config=RunConfig(
                chunk_size=512,
                chunk_overlap=64,
                top_k=10,
                bm25_k1=1.5,
                bm25_b=0.75,
                score_threshold=None,
            ),
        ),
        stages=stages,
    )


def test_compare_report_stages_pairs_two_stages_of_one_report() -> None:
    report = _report(
        [
            _stage("bm25", [_query("q-1", ndcg=0.2), _query("q-2", ndcg=0.4)], is_baseline=True),
            _stage("dense", [_query("q-1", ndcg=0.6), _query("q-2", ndcg=0.5)], is_baseline=False),
        ]
    )

    result = compare_report_stages(
        report, baseline="bm25", candidate="dense", metric="ndcg_at_10", resamples=500
    )

    assert result.baseline == "bm25"
    assert result.candidate == "dense"
    assert result.query_count == 2
    assert result.effect == pytest.approx(((0.6 - 0.2) + (0.5 - 0.4)) / 2)


def test_compare_report_stages_reads_reciprocal_rank_for_the_mrr_metric() -> None:
    """``mrr`` is the one metric whose per-query field is named differently
    (`reciprocal_rank`), so the mapping is asserted rather than assumed."""
    report = _report(
        [
            _stage("bm25", [_query("q-1", ndcg=0.2, rr=0.25)], is_baseline=True),
            _stage("dense", [_query("q-1", ndcg=0.2, rr=0.75)], is_baseline=False),
        ]
    )

    result = compare_report_stages(
        report, baseline="bm25", candidate="dense", metric="mrr", resamples=100
    )

    assert result.effect == pytest.approx(0.5)


def test_compare_report_stages_skips_no_answer_queries() -> None:
    """A no-answer query has no metrics, so it must drop out of both sides
    rather than pair as a zero."""
    report = _report(
        [
            _stage(
                "bm25",
                [_query("q-1", ndcg=0.2), _no_answer_query("q-2")],
                is_baseline=True,
            ),
            _stage(
                "dense",
                [_query("q-1", ndcg=0.6), _no_answer_query("q-2")],
                is_baseline=False,
            ),
        ]
    )

    result = compare_report_stages(
        report, baseline="bm25", candidate="dense", metric="ndcg_at_10", resamples=100
    )

    assert result.query_count == 1
    assert result.effect == pytest.approx(0.4)


def test_compare_report_stages_rejects_a_stage_the_report_does_not_contain() -> None:
    report = _report([_stage("bm25", [_query("q-1", ndcg=0.2)], is_baseline=True)])

    with pytest.raises(EvalError, match="does not contain stage 'dense'"):
        compare_report_stages(report, baseline="bm25", candidate="dense", metric="ndcg_at_10")


def test_compare_report_stages_rejects_duplicate_query_ids() -> None:
    """``QueryResult.query_id`` is documented as unique but nothing in the
    schema enforces it. Assigning into a dict would silently keep the last of
    a duplicated pair, so the bootstrap would run over a smaller query set
    than the report claims and the effect would depend on their order."""
    report = _report(
        [
            _stage(
                "bm25",
                [_query("q-1", ndcg=0.2), _query("q-1", ndcg=0.9)],
                is_baseline=True,
            ),
            _stage("dense", [_query("q-1", ndcg=0.6)], is_baseline=False),
        ]
    )

    with pytest.raises(EvalError, match="duplicate query id"):
        compare_report_stages(report, baseline="bm25", candidate="dense", metric="ndcg_at_10")


def test_compare_report_stages_rejects_duplicate_stage_names() -> None:
    """``EvalReport`` pins exactly one baseline at position zero, but nothing
    stops two later stages sharing a name. A dict comprehension would keep the
    last silently, making the comparison depend on their order in the file."""
    report = _report(
        [
            _stage("bm25", [_query("q-1", ndcg=0.2)], is_baseline=True),
            _stage("dense", [_query("q-1", ndcg=0.6)], is_baseline=False),
            _stage("dense", [_query("q-1", ndcg=0.9)], is_baseline=False),
        ]
    )

    with pytest.raises(EvalError, match="more than one stage named"):
        compare_report_stages(report, baseline="bm25", candidate="dense", metric="ndcg_at_10")

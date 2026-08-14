"""Tests for read-time delta derivation (SPEC.md §6 baseline discipline).

Deltas are computed from hand-built reports here rather than from real eval
runs: the arithmetic and the direction reporting are what these tests are
about, and constructing the two ``MetricSet``s directly is the only way to
assert an exact expected delta. ``tests/test_runner.py`` covers the same
machinery driven by an actual multi-stage run.
"""

from __future__ import annotations

import pytest

from groundkit.evals.delta import (
    QUALITY_METRIC_FIELDS,
    MetricFieldMismatchError,
    StageDelta,
    assert_metric_fields_exist,
    derive_stage_deltas,
)
from groundkit.evals.schema import (
    EvalReport,
    MetricSet,
    RunConfig,
    RunMetadata,
    StageName,
    StageResult,
)


def _metrics(
    *,
    recall_at_1: float = 0.5,
    recall_at_5: float = 0.5,
    recall_at_10: float = 0.5,
    mrr: float = 0.5,
    ndcg_at_10: float = 0.5,
) -> MetricSet:
    """A ``MetricSet`` with every metric defaulting to 0.5, overridable one at a time."""
    return MetricSet(
        query_count=4,
        recall_at_1=recall_at_1,
        recall_at_5=recall_at_5,
        recall_at_10=recall_at_10,
        mrr=mrr,
        ndcg_at_10=ndcg_at_10,
    )


def _stage(
    stage: StageName,
    aggregate: MetricSet,
    *,
    is_baseline: bool = False,
    latency_p50_ms: float = 1.0,
    latency_p95_ms: float = 2.0,
    latency_p99_ms: float = 3.0,
) -> StageResult:
    """A ``StageResult`` carrying no per-query detail — deltas never read it."""
    return StageResult(
        stage=stage,
        is_baseline=is_baseline,
        aggregate=aggregate,
        by_category={},
        no_answer_query_count=0,
        no_answer_abstained_count=0,
        latency_p50_ms=latency_p50_ms,
        latency_p95_ms=latency_p95_ms,
        latency_p99_ms=latency_p99_ms,
        queries=[],
    )


def _report(stages: list[StageResult]) -> EvalReport:
    """An ``EvalReport`` over ``stages`` with placeholder provenance."""
    return EvalReport(
        run=RunMetadata(
            started_at="2026-08-14T00:00:00+00:00",
            groundkit_version="0.0.0",
            corpus_hash="c" * 64,
            judgments_hash="j" * 64,
            document_count=2,
            chunk_count=4,
            judgment_count=4,
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


class TestDeltaArithmetic:
    """A delta is stage minus baseline, per metric, signed."""

    def test_delta_is_stage_minus_baseline_per_metric(self) -> None:
        """Pins the subtraction direction on every quality field at once."""
        baseline = _stage("bm25", _metrics(), is_baseline=True)
        dense = _stage(
            "dense",
            _metrics(recall_at_1=0.75, recall_at_5=0.25, recall_at_10=0.5, mrr=1.0, ndcg_at_10=0.0),
        )

        (delta,) = derive_stage_deltas(_report([baseline, dense]))

        assert delta.quality["recall_at_1"] == pytest.approx(0.25)
        assert delta.quality["recall_at_5"] == pytest.approx(-0.25)
        assert delta.quality["recall_at_10"] == pytest.approx(0.0)
        assert delta.quality["mrr"] == pytest.approx(0.5)
        assert delta.quality["ndcg_at_10"] == pytest.approx(-0.5)

    def test_delta_covers_every_declared_quality_field(self) -> None:
        """The delta dict's keys are exactly ``QUALITY_METRIC_FIELDS``."""
        report = _report(
            [_stage("bm25", _metrics(), is_baseline=True), _stage("dense", _metrics())]
        )

        (delta,) = derive_stage_deltas(report)

        assert set(delta.quality) == set(QUALITY_METRIC_FIELDS)

    def test_latency_deltas_are_signed_with_positive_meaning_slower(self) -> None:
        """Latency keeps raw sign; a slower stage reports a positive delta."""
        baseline = _stage(
            "bm25",
            _metrics(),
            is_baseline=True,
            latency_p50_ms=1.0,
            latency_p95_ms=2.0,
            latency_p99_ms=3.0,
        )
        dense = _stage(
            "dense",
            _metrics(),
            latency_p50_ms=6.0,
            latency_p95_ms=1.5,
            latency_p99_ms=3.0,
        )

        (delta,) = derive_stage_deltas(_report([baseline, dense]))

        assert delta.latency_p50_delta_ms == pytest.approx(5.0)
        assert delta.latency_p95_delta_ms == pytest.approx(-0.5)
        assert delta.latency_p99_delta_ms == pytest.approx(0.0)

    def test_baseline_stage_is_recorded_on_every_delta(self) -> None:
        """A delta names what it was measured against, not just itself."""
        report = _report(
            [
                _stage("bm25", _metrics(), is_baseline=True),
                _stage("dense", _metrics()),
                _stage("fusion", _metrics()),
            ]
        )

        deltas = derive_stage_deltas(report)

        assert [d.stage for d in deltas] == ["dense", "fusion"]
        assert all(d.baseline_stage == "bm25" for d in deltas)


class TestHonestLossReporting:
    """SPEC.md §6: a stage that does not beat baseline is reported as such."""

    def test_a_stage_worse_on_every_metric_is_flagged_a_regression(self) -> None:
        """The whole point of the module: losses surface, they do not vanish."""
        baseline = _stage("bm25", _metrics(), is_baseline=True)
        loser = _stage(
            "dense",
            _metrics(recall_at_1=0.1, recall_at_5=0.1, recall_at_10=0.1, mrr=0.1, ndcg_at_10=0.1),
        )

        (delta,) = derive_stage_deltas(_report([baseline, loser]))

        assert delta.is_regression is True
        assert delta.is_improvement is False
        assert all(value < 0 for value in delta.quality.values())

    def test_losing_stage_is_never_filtered_out_of_the_derived_list(self) -> None:
        """Every non-baseline stage yields a delta, winner or not."""
        baseline = _stage("bm25", _metrics(), is_baseline=True)
        loser = _stage("dense", _metrics(recall_at_1=0.0, mrr=0.0, ndcg_at_10=0.0))
        winner = _stage("fusion", _metrics(recall_at_1=0.9, mrr=0.9, ndcg_at_10=0.9))

        deltas = derive_stage_deltas(_report([baseline, loser, winner]))

        assert [d.stage for d in deltas] == ["dense", "fusion"]
        assert deltas[0].is_regression is True
        assert deltas[1].is_improvement is True

    def test_mixed_result_reports_both_directions_not_a_single_verdict(self) -> None:
        """A stage that gains one metric and loses another is both, not neither.

        Collapsing this to one boolean is how a regression gets hidden behind
        an unrelated win, so both flags are asserted True simultaneously.
        """
        baseline = _stage("bm25", _metrics(), is_baseline=True)
        mixed = _stage("fusion", _metrics(recall_at_1=0.9, ndcg_at_10=0.1))

        (delta,) = derive_stage_deltas(_report([baseline, mixed]))

        assert delta.is_regression is True
        assert delta.is_improvement is True

    def test_identical_metrics_are_neither_regression_nor_improvement(self) -> None:
        """A wash reports as a wash — no tolerance band, no rounding to a win."""
        baseline = _stage("bm25", _metrics(), is_baseline=True)
        same = _stage("dense", _metrics())

        (delta,) = derive_stage_deltas(_report([baseline, same]))

        assert delta.is_regression is False
        assert delta.is_improvement is False
        assert all(value == 0.0 for value in delta.quality.values())

    def test_no_tolerance_band_swallows_a_tiny_regression(self) -> None:
        """A regression far below any plausible epsilon still reports as one.

        Guards the module's stated decision not to invent a noise threshold:
        on this corpus size a real effect can be small (R2), so the sign test
        must stay strict rather than acquiring a "close enough" band later.
        """
        baseline = _stage("bm25", _metrics(recall_at_1=0.5), is_baseline=True)
        barely_worse = _stage("dense", _metrics(recall_at_1=0.5 - 1e-9))

        (delta,) = derive_stage_deltas(_report([baseline, barely_worse]))

        assert delta.is_regression is True


class TestBaselineOnlyReport:
    """A single-stage report has nothing to compare against."""

    def test_baseline_only_report_yields_no_deltas(self) -> None:
        """``[]``, not a zero-valued self-comparison."""
        report = _report([_stage("bm25", _metrics(), is_baseline=True)])

        assert derive_stage_deltas(report) == []


class TestDeltasAreNeverStored:
    """The artifact must not gain a delta field (``schema.py``'s core promise)."""

    def test_eval_report_has_no_delta_field(self) -> None:
        """A stored delta can disagree with the numbers it came from."""
        assert "delta" not in EvalReport.model_fields
        assert "deltas" not in EvalReport.model_fields
        assert "delta" not in StageResult.model_fields

    def test_serialized_report_carries_no_delta_key(self) -> None:
        """Round-trips the actual JSON, not just the model's field list."""
        report = _report(
            [_stage("bm25", _metrics(), is_baseline=True), _stage("dense", _metrics())]
        )

        payload = report.model_dump_json()

        assert "delta" not in payload

    def test_stage_delta_is_rejected_as_report_input(self) -> None:
        """``extra="forbid"`` keeps a caller from smuggling one in."""
        baseline = _stage("bm25", _metrics(), is_baseline=True)
        report = _report([baseline])

        with pytest.raises(ValueError, match=r"[Ee]xtra"):
            EvalReport(**{**report.model_dump(), "deltas": []})


class TestFieldTupleIntegrity:
    """``QUALITY_METRIC_FIELDS`` must keep naming real ``MetricSet`` fields."""

    def test_every_declared_field_exists_on_metric_set(self) -> None:
        """A renamed metric would otherwise fail only inside a live report."""
        assert_metric_fields_exist()

    def test_the_guard_itself_detects_a_missing_field(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Proves the check above can fail, rather than passing vacuously."""
        monkeypatch.setattr(
            "groundkit.evals.delta.QUALITY_METRIC_FIELDS", ("recall_at_1", "not_a_real_metric")
        )

        with pytest.raises(MetricFieldMismatchError, match="not_a_real_metric"):
            assert_metric_fields_exist()


class TestNetQuality:
    """``net_quality`` is a glance across metrics, not a score."""

    def test_net_quality_sums_the_signed_deltas(self) -> None:
        """Pins it as a plain sum so a reader can predict it."""
        delta = StageDelta(
            stage="fusion",
            baseline_stage="bm25",
            quality={"recall_at_1": 0.25, "mrr": -0.5, "ndcg_at_10": 0.1},
            latency_p50_delta_ms=0.0,
            latency_p95_delta_ms=0.0,
            latency_p99_delta_ms=0.0,
            is_regression=True,
            is_improvement=True,
        )

        assert delta.net_quality == pytest.approx(-0.15)

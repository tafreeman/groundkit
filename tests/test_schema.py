"""Schema tests: JSON round-trip, frozen/extra=forbid, Field bounds, and the
artifact's intra-run baseline + no-answer invariants (SPEC.md §6, §9)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from groundkit.evals.schema import (
    EvalReport,
    GoldSpanResult,
    MetricSet,
    QueryMetrics,
    QueryResult,
    RetrievedHit,
    RunConfig,
    RunMetadata,
    StageResult,
)


def make_run_config(**overrides: object) -> RunConfig:
    defaults: dict[str, object] = {
        "chunk_size": 512,
        "chunk_overlap": 64,
        "top_k": 5,
        "bm25_k1": 1.5,
        "bm25_b": 0.75,
        "score_threshold": None,
    }
    defaults.update(overrides)
    return RunConfig(**defaults)  # type: ignore[arg-type]


def make_run_metadata(**overrides: object) -> RunMetadata:
    defaults: dict[str, object] = {
        "started_at": "2026-08-12T00:00:00+00:00",
        "groundkit_version": "0.1.0.dev0",
        "corpus_hash": "a" * 64,
        "judgments_hash": "b" * 64,
        "document_count": 8,
        "chunk_count": 40,
        "judgment_count": 40,
        "config": make_run_config(),
    }
    defaults.update(overrides)
    return RunMetadata(**defaults)  # type: ignore[arg-type]


def make_metric_set(**overrides: object) -> MetricSet:
    defaults: dict[str, object] = {
        "query_count": 40,
        "recall_at_1": 0.5,
        "recall_at_5": 0.8,
        "recall_at_10": 0.9,
        "mrr": 0.6,
        "ndcg_at_10": 0.7,
    }
    defaults.update(overrides)
    return MetricSet(**defaults)  # type: ignore[arg-type]


def make_gold_span(**overrides: object) -> GoldSpanResult:
    defaults: dict[str, object] = {
        "document": "docs/a.md",
        "start_offset": 0,
        "end_offset": 10,
        "quote": "0123456789",
    }
    defaults.update(overrides)
    return GoldSpanResult(**defaults)  # type: ignore[arg-type]


def make_retrieved_hit(**overrides: object) -> RetrievedHit:
    defaults: dict[str, object] = {
        "rank": 1,
        "document": "docs/a.md",
        "start_offset": 0,
        "end_offset": 10,
        "score": 1.5,
        "is_relevant": True,
    }
    defaults.update(overrides)
    return RetrievedHit(**defaults)  # type: ignore[arg-type]


def make_query_metrics(**overrides: object) -> QueryMetrics:
    defaults: dict[str, object] = {
        "recall_at_1": 1.0,
        "recall_at_5": 1.0,
        "recall_at_10": 1.0,
        "reciprocal_rank": 1.0,
        "ndcg_at_10": 1.0,
    }
    defaults.update(overrides)
    return QueryMetrics(**defaults)  # type: ignore[arg-type]


def make_query_result(**overrides: object) -> QueryResult:
    """An answerable query — has metrics and non-empty gold."""
    defaults: dict[str, object] = {
        "query_id": "q1",
        "query": "what is groundkit?",
        "category": "factual",
        "is_no_answer": False,
        "gold": [make_gold_span()],
        "total_relevant_chunks": 1,
        "retrieved": [make_retrieved_hit()],
        "metrics": make_query_metrics(),
        "latency_ms": 12.5,
    }
    defaults.update(overrides)
    return QueryResult(**defaults)  # type: ignore[arg-type]


def make_no_answer_query_result(**overrides: object) -> QueryResult:
    """A no-answer query — no metrics, no gold."""
    defaults: dict[str, object] = {
        "query_id": "q2",
        "query": "what is the capital of nowhere?",
        "category": "no_answer",
        "is_no_answer": True,
        "gold": [],
        "total_relevant_chunks": 0,
        "retrieved": [],
        "metrics": None,
        "latency_ms": 5.0,
    }
    defaults.update(overrides)
    return QueryResult(**defaults)  # type: ignore[arg-type]


def make_stage_result(**overrides: object) -> StageResult:
    defaults: dict[str, object] = {
        "stage": "bm25",
        "is_baseline": True,
        "aggregate": make_metric_set(),
        "by_category": {"factual": make_metric_set()},
        "no_answer_query_count": 1,
        "no_answer_abstained_count": 1,
        "latency_p50_ms": 3.0,
        "latency_p95_ms": 8.0,
        "latency_p99_ms": 10.0,
        "queries": [make_query_result(), make_no_answer_query_result()],
    }
    defaults.update(overrides)
    return StageResult(**defaults)  # type: ignore[arg-type]


def make_eval_report(**overrides: object) -> EvalReport:
    defaults: dict[str, object] = {
        "run": make_run_metadata(),
        "stages": [make_stage_result()],
    }
    defaults.update(overrides)
    return EvalReport(**defaults)  # type: ignore[arg-type]


class TestRunConfig:
    def test_frozen(self) -> None:
        c = make_run_config()
        with pytest.raises(ValidationError):
            c.top_k = 10

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_run_config(surprise=True)

    def test_threshold_none_accepted(self) -> None:
        assert make_run_config(score_threshold=None).score_threshold is None


class TestRunMetadata:
    def test_run_id_auto_generated_and_unique(self) -> None:
        a, b = make_run_metadata(), make_run_metadata()
        assert a.run_id != b.run_id

    def test_negative_document_count_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_run_metadata(document_count=-1)

    def test_negative_chunk_count_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_run_metadata(chunk_count=-1)

    def test_negative_judgment_count_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_run_metadata(judgment_count=-1)

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_run_metadata(surprise=True)

    def test_frozen(self) -> None:
        m = make_run_metadata()
        with pytest.raises(ValidationError):
            m.run_id = "other"


class TestMetricSet:
    def test_negative_query_count_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_metric_set(query_count=-1)

    def test_score_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_metric_set(recall_at_1=1.1)

    def test_negative_score_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_metric_set(mrr=-0.01)

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_metric_set(surprise=True)


class TestGoldSpanResult:
    def test_end_offset_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_gold_span(end_offset=0)

    def test_negative_start_offset_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_gold_span(start_offset=-1)

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_gold_span(surprise=True)

    def test_frozen(self) -> None:
        g = make_gold_span()
        with pytest.raises(ValidationError):
            g.quote = "other"


class TestRetrievedHit:
    def test_rank_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_retrieved_hit(rank=0)

    def test_negative_score_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_retrieved_hit(score=-0.5)

    def test_end_offset_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_retrieved_hit(end_offset=0)

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_retrieved_hit(surprise=True)


class TestQueryMetrics:
    def test_reciprocal_rank_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_query_metrics(reciprocal_rank=1.5)

    def test_negative_ndcg_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_query_metrics(ndcg_at_10=-0.1)

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_query_metrics(surprise=True)


class TestQueryResult:
    def test_answerable_query_accepted(self) -> None:
        q = make_query_result()
        assert q.metrics is not None
        assert q.gold != []

    def test_no_answer_query_accepted(self) -> None:
        q = make_no_answer_query_result()
        assert q.metrics is None
        assert q.gold == []

    def test_no_answer_with_metrics_rejected(self) -> None:
        with pytest.raises(ValidationError, match="is_no_answer"):
            make_no_answer_query_result(metrics=make_query_metrics())

    def test_no_answer_with_gold_rejected(self) -> None:
        with pytest.raises(ValidationError, match="is_no_answer"):
            make_no_answer_query_result(gold=[make_gold_span()])

    def test_answerable_without_metrics_rejected(self) -> None:
        with pytest.raises(ValidationError, match="is_no_answer"):
            make_query_result(metrics=None)

    def test_answerable_with_empty_gold_rejected(self) -> None:
        with pytest.raises(ValidationError, match="is_no_answer"):
            make_query_result(gold=[])

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_query_result(surprise=True)

    def test_frozen(self) -> None:
        q = make_query_result()
        with pytest.raises(ValidationError):
            q.query = "other"

    def test_negative_total_relevant_chunks_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_query_result(total_relevant_chunks=-1)

    def test_negative_latency_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_query_result(latency_ms=-1.0)


class TestStageResult:
    def test_abstained_exceeding_total_rejected(self) -> None:
        with pytest.raises(ValidationError, match="no_answer_abstained_count"):
            make_stage_result(no_answer_query_count=1, no_answer_abstained_count=2)

    def test_abstained_equal_to_total_accepted(self) -> None:
        s = make_stage_result(no_answer_query_count=1, no_answer_abstained_count=1)
        assert s.no_answer_abstained_count == s.no_answer_query_count

    def test_abstained_below_total_accepted(self) -> None:
        s = make_stage_result(no_answer_query_count=2, no_answer_abstained_count=1)
        assert s.no_answer_abstained_count < s.no_answer_query_count

    def test_unknown_stage_literal_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_stage_result(stage="cross_encoder")

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_stage_result(surprise=True)

    def test_frozen(self) -> None:
        s = make_stage_result()
        with pytest.raises(ValidationError):
            s.is_baseline = False


class TestEvalReport:
    def test_schema_version_default(self) -> None:
        assert make_eval_report().schema_version == 1

    def test_empty_stages_rejected(self) -> None:
        with pytest.raises(ValidationError, match="stages must not be empty"):
            make_eval_report(stages=[])

    def test_zero_baselines_rejected(self) -> None:
        with pytest.raises(ValidationError, match="exactly one stage"):
            make_eval_report(stages=[make_stage_result(is_baseline=False)])

    def test_two_baselines_rejected(self) -> None:
        stages = [
            make_stage_result(stage="bm25", is_baseline=True),
            make_stage_result(stage="dense", is_baseline=True),
        ]
        with pytest.raises(ValidationError, match="exactly one stage"):
            make_eval_report(stages=stages)

    def test_baseline_not_first_rejected(self) -> None:
        stages = [
            make_stage_result(stage="dense", is_baseline=False),
            make_stage_result(stage="bm25", is_baseline=True),
        ]
        with pytest.raises(ValidationError, match="stages\\[0\\]"):
            make_eval_report(stages=stages)

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_eval_report(surprise=True)

    def test_frozen(self) -> None:
        r = make_eval_report()
        with pytest.raises(ValidationError):
            r.schema_version = 2

    def test_json_round_trip(self) -> None:
        report = make_eval_report()
        restored = EvalReport.model_validate_json(report.model_dump_json())
        assert restored == report
        assert restored.run.config.top_k == report.run.config.top_k
        assert restored.stages[0].queries[0].query_id == report.stages[0].queries[0].query_id

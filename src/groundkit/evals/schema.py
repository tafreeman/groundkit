"""JSON artifact schema for the retrieval eval harness (SPEC.md §6, §9).

This is the reference artifact format every later phase reports deltas
against: Phase 3 appends ``dense``/``fusion``/``rerank`` stages to the same
:class:`EvalReport` shape without a breaking change. Written to
``evals/results/latest.json``, which is gitignored by design — see
``EvalReport``'s docstring for why that makes intra-run comparison the only
kind this schema supports.

All models are frozen and reject unknown fields, matching ``contracts.py``
and ``config.py``.
"""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RunConfig(BaseModel):
    """The settings that produced a run — what makes two runs comparable.

    This is a record of the retrieval settings in effect, not itself a
    comparability key: SPEC.md keys comparability on ``corpus_hash`` +
    ``judgments_hash`` (see :class:`RunMetadata`). A config diff between two
    otherwise-comparable runs explains *why* their numbers might disagree.

    Attributes:
        chunk_size: Target chunk size in characters
            (``ChunkingConfig.chunk_size``).
        chunk_overlap: Overlap between consecutive chunks, in characters.
        top_k: Number of results requested per query.
        bm25_k1: BM25 term-frequency saturation parameter.
        bm25_b: BM25 length-normalization parameter.
        score_threshold: Minimum score filter in effect, or ``None`` if
            disabled.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_size: int
    chunk_overlap: int
    top_k: int
    bm25_k1: float
    bm25_b: float
    score_threshold: float | None


class RunMetadata(BaseModel):
    """Identity and provenance of one eval run.

    Deliberately carries no ``git_sha``. Comparability keys on
    ``corpus_hash`` + ``judgments_hash`` instead: an unrelated commit
    elsewhere in the repo changes the SHA without touching golden data, so a
    SHA would be a misleading comparability signal sitting next to the
    hashes that are the honest one.

    Attributes:
        run_id: Unique identifier for this run (auto-generated hex UUID).
        started_at: ISO-8601 UTC timestamp the run began, matching
            ``index/metadata.py``'s ``_now_iso()`` format.
        groundkit_version: The ``groundkit`` package version that produced
            this report.
        corpus_hash: SHA-256 over the sorted ``(relpath, bytes)`` pairs of
            every corpus file, so any content or membership change is
            detectable.
        judgments_hash: SHA-256 of the judgments file's raw bytes.
        document_count: Number of documents in the corpus at run time.
        chunk_count: Number of chunks produced from the corpus.
        judgment_count: Number of query judgments evaluated.
        config: The retrieval settings that produced this run.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    started_at: str
    groundkit_version: str
    corpus_hash: str
    judgments_hash: str
    document_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    judgment_count: int = Field(ge=0)
    config: RunConfig


class MetricSet(BaseModel):
    """Aggregate retrieval quality metrics over a set of queries.

    Attributes:
        query_count: Number of queries the metrics were computed over.
        recall_at_1: Fraction of queries with a relevant chunk in the top 1.
        recall_at_5: Fraction of queries with a relevant chunk in the top 5.
        recall_at_10: Fraction of queries with a relevant chunk in the top
            10.
        mrr: Mean reciprocal rank of the first relevant chunk.
        ndcg_at_10: Mean normalized discounted cumulative gain at 10.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_count: int = Field(ge=0)
    recall_at_1: float = Field(ge=0.0, le=1.0)
    recall_at_5: float = Field(ge=0.0, le=1.0)
    recall_at_10: float = Field(ge=0.0, le=1.0)
    mrr: float = Field(ge=0.0, le=1.0)
    ndcg_at_10: float = Field(ge=0.0, le=1.0)


class GoldSpanResult(BaseModel):
    """A judgment's gold span, resolved against the corpus at run time.

    ``document`` is corpus-relative, never the absolute realpath that
    ``RetrievalResult.source`` carries — an absolute path would make the
    artifact machine-specific and undiffable across checkouts. The runner
    is responsible for that conversion.

    Attributes:
        document: Corpus-relative path of the document the judgment points
            at.
        start_offset: Character offset where the gold span starts.
        end_offset: One past the last character of the gold span.
        quote: The gold span's text, for human review.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    document: str
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)
    quote: str


class RetrievedHit(BaseModel):
    """One retrieved result at a given rank, scored against the gold set.

    ``document`` is corpus-relative for the same reason as
    :class:`GoldSpanResult.document` — never the absolute realpath that
    ``RetrievalResult.source`` carries, so the artifact stays diffable
    across machines. The runner does the conversion.

    Attributes:
        rank: 1-indexed position of this hit in the retrieved list.
        document: Corpus-relative path of the retrieved chunk's document.
        start_offset: Character offset of the retrieved chunk in its
            document.
        end_offset: One past the last character of the retrieved chunk.
        score: Relevance score assigned by the stage (>= 0.0; BM25 scores
            are unbounded above).
        is_relevant: Whether this hit matches a gold span for the query.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int = Field(ge=1)
    document: str
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)
    score: float = Field(ge=0.0)
    is_relevant: bool


class QueryMetrics(BaseModel):
    """Retrieval quality metrics for a single query.

    ``None`` on the owning :class:`QueryResult` for no-answer queries,
    since recall/MRR/nDCG are undefined against an empty gold set — see
    ``QueryResult``'s validator.

    Attributes:
        recall_at_1: 1.0 if a relevant chunk is in the top 1, else 0.0.
        recall_at_5: 1.0 if a relevant chunk is in the top 5, else 0.0.
        recall_at_10: 1.0 if a relevant chunk is in the top 10, else 0.0.
        reciprocal_rank: 1 / rank of the first relevant chunk, or 0.0 if
            none was retrieved.
        ndcg_at_10: Normalized discounted cumulative gain at 10.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    recall_at_1: float = Field(ge=0.0, le=1.0)
    recall_at_5: float = Field(ge=0.0, le=1.0)
    recall_at_10: float = Field(ge=0.0, le=1.0)
    reciprocal_rank: float = Field(ge=0.0, le=1.0)
    ndcg_at_10: float = Field(ge=0.0, le=1.0)


class QueryResult(BaseModel):
    """One query's judgment, retrieval, and metrics for a single stage.

    ``metrics`` is ``None`` exactly for no-answer queries, because
    recall/MRR/nDCG are undefined with an empty gold set — computing them
    against ``[]`` would silently produce a meaningless 0.0 or 1.0 instead
    of surfacing that the question doesn't apply. The validator below
    enforces the three-way biconditional (``is_no_answer`` is True iff
    ``metrics`` is ``None`` iff ``gold`` is empty) and fails closed on a
    record that disagrees with itself.

    Attributes:
        query_id: Unique identifier for this query.
        query: The query text.
        category: Judgment category (e.g. factual, ambiguous, adversarial,
            no_answer).
        is_no_answer: Whether this query has no relevant chunk in the
            corpus.
        gold: Resolved gold spans; empty iff ``is_no_answer``.
        total_relevant_chunks: Count of distinct relevant chunks for this
            query.
        retrieved: Ranked hits returned by the stage.
        metrics: Per-query metrics, or ``None`` iff ``is_no_answer``.
        latency_ms: Wall-clock time to answer this query, in milliseconds.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_id: str
    query: str
    category: str
    is_no_answer: bool
    gold: list[GoldSpanResult]
    total_relevant_chunks: int = Field(ge=0)
    retrieved: list[RetrievedHit]
    metrics: QueryMetrics | None
    latency_ms: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _validate_no_answer_biconditional(self) -> QueryResult:
        metrics_is_none = self.metrics is None
        gold_is_empty = len(self.gold) == 0
        if not (self.is_no_answer == metrics_is_none == gold_is_empty):
            raise ValueError(
                "is_no_answer, metrics is None, and gold == [] must all agree "
                f"(is_no_answer={self.is_no_answer}, metrics_is_none={metrics_is_none}, "
                f"gold_is_empty={gold_is_empty})"
            )
        return self


class StageResult(BaseModel):
    """One retrieval stage's results over the full judgment set.

    ``no_answer_abstained_count`` counts no-answer queries that returned
    zero results. There is deliberately no score threshold: BM25 scores are
    unbounded, so any fixed cutoff would report noise from an arbitrary
    number rather than a real measurement of abstention.

    Attributes:
        stage: Which retrieval stage produced this result.
        is_baseline: Whether this is the intra-run baseline stage. Exactly
            one stage in an :class:`EvalReport` may set this, and it must
            be ``stages[0]``.
        aggregate: Metrics aggregated over all answerable queries.
        by_category: Aggregate metrics broken out per judgment category.
        no_answer_query_count: Count of no-answer queries evaluated.
        no_answer_abstained_count: Count of no-answer queries that returned
            zero results — never exceeds ``no_answer_query_count``.
        latency_p50_ms: Median per-query latency for this stage.
        latency_p95_ms: 95th-percentile per-query latency for this stage.
        latency_p99_ms: 99th-percentile per-query latency for this stage.
        queries: Per-query results for this stage.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: Literal["bm25", "dense", "fusion", "rerank"]
    is_baseline: bool
    aggregate: MetricSet
    by_category: dict[str, MetricSet]
    no_answer_query_count: int = Field(ge=0)
    no_answer_abstained_count: int = Field(ge=0)
    latency_p50_ms: float = Field(ge=0.0)
    latency_p95_ms: float = Field(ge=0.0)
    latency_p99_ms: float = Field(ge=0.0)
    queries: list[QueryResult]

    @model_validator(mode="after")
    def _validate_abstained_bound(self) -> StageResult:
        if self.no_answer_abstained_count > self.no_answer_query_count:
            raise ValueError(
                f"no_answer_abstained_count ({self.no_answer_abstained_count}) must not "
                f"exceed no_answer_query_count ({self.no_answer_query_count})"
            )
        return self


class EvalReport(BaseModel):
    """The complete artifact for one eval run — the reference report shape.

    No ``DeltaMetricSet`` and no ``git_sha``, both deliberately. A delta
    (``stages[i].aggregate`` minus ``stages[0].aggregate``) is DERIVED at
    read time, not stored: a stored delta is redundant data that can
    disagree with the two numbers it was computed from. ``git_sha`` is
    omitted for the reason given on :class:`RunMetadata` — comparability
    keys on ``corpus_hash``/``judgments_hash``, and a SHA would be a
    misleading comparability signal alongside them.

    Baseline is intra-run: ``stages[0]`` is always ``stage="bm25"``,
    ``is_baseline=True``. Phase 3 appends stages to this same report and
    diffs each new stage against ``stages[0]`` *within that report*.
    ``evals/results/`` is gitignored by design, so no historical artifact
    reliably exists in CI to diff across runs — that is intentional, not a
    gap for a future phase to "fix" by adding cross-run diffing.

    Attributes:
        schema_version: Version of this artifact schema. Pinned to the one
            version this module can actually parse: an artifact labelled
            with any other version is rejected rather than silently read as
            though it had this shape, which is what an unconstrained ``int``
            would do the first time the format changes.
        run: Provenance and settings for this run.
        stages: Per-stage results; non-empty, with exactly one baseline
            stage, at index 0, and that stage is BM25.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    run: RunMetadata
    stages: list[StageResult]

    @model_validator(mode="after")
    def _validate_baseline_invariants(self) -> EvalReport:
        if not self.stages:
            raise ValueError("stages must not be empty")
        baseline_count = sum(1 for stage in self.stages if stage.is_baseline)
        if baseline_count != 1:
            raise ValueError(
                f"exactly one stage must have is_baseline=True, found {baseline_count}"
            )
        if not self.stages[0].is_baseline:
            raise ValueError("the baseline stage must be stages[0]")
        # SPEC.md §6 fixes BM25-only as *the* baseline, and readers derive
        # every delta against stages[0]. Checking only the flag and the
        # position would accept a report whose stages[0] was, say, dense —
        # and silently measure every later feature against the wrong
        # reference instead of failing.
        if self.stages[0].stage != "bm25":
            raise ValueError(
                f"the baseline stage must be 'bm25' (SPEC.md §6), got {self.stages[0].stage!r}"
            )
        return self

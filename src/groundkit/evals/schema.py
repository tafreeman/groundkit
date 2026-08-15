"""JSON artifact schema for the retrieval eval harness (SPEC.md §6, §9).

This is the reference artifact format every later phase reports deltas
against: Phase 3 appends ``dense``/``fusion``/``rerank`` stages to the same
:class:`EvalReport` shape without a breaking change. Written to
``evals/results/latest.json``, which is gitignored by design — see
``EvalReport``'s docstring for why that makes intra-run comparison the only
kind this schema supports.

Wave E kept that promise for stages and extended :class:`RunConfig` by two
optional, defaulted fields (``embedding``, ``rrf_k``) rather than by a
``schema_version`` bump. Additive-with-default is not a breaking change in
the direction that exists: a current reader parses a pre-Wave-E artifact
unchanged, both fields defaulting to ``None``. The reverse direction — an
older reader meeting a newer artifact — would fail on ``extra="forbid"``,
and is accepted precisely because no such artifact is kept: ``evals/results/``
is gitignored, so every artifact in existence is produced by the code
reading it. Should that ever stop being true, the next field is a version
bump, not another default.

The ``rerank`` stage (ADR-0012) extends :class:`RunConfig` by three more
fields on the same terms and for the same reason it is safe: still no kept
artifact, still additive-with-default. That condition has now been relied on
twice, so it is worth restating as the live constraint it is rather than a
historical note — the moment an artifact outlives the code that wrote it,
the next field is a ``schema_version`` bump.

Deltas are **not** here. They are derived at read time by
:mod:`groundkit.evals.delta`, which returns a ``StageDelta`` that is never a
field of :class:`EvalReport` — see that module and :class:`EvalReport` for
why a stored delta is the one thing this schema refuses to carry.

All models are frozen and reject unknown fields, matching ``contracts.py``
and ``config.py``.
"""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from groundkit.contracts import EmbeddingIdentity

#: The retrieval stages a report may contain, in the order Phase 3 produces
#: them. Defined once here and referenced by :class:`StageResult`, the
#: runner, and :mod:`groundkit.evals.delta`: three copies of the same
#: ``Literal`` would let a stage added to one drift out of the others, and
#: the artifact schema is the right single owner of what a stage may be.
StageName = Literal["bm25", "dense", "fusion", "rerank"]


class RunConfig(BaseModel):
    """The settings that produced a run — what makes two runs comparable.

    This is a record of the retrieval settings in effect, not itself a
    comparability key: SPEC.md keys comparability on ``corpus_hash`` +
    ``judgments_hash`` (see :class:`RunMetadata`). A config diff between two
    otherwise-comparable runs explains *why* their numbers might disagree.

    ``embedding`` and ``rrf_k`` arrived with Phase 3 Wave E's dense and
    fusion stages, both optional and both ``None``-meaning-absent rather
    than ``None``-meaning-defaulted: a BM25-only run has no embedding
    identity and no fusion constant, and inventing one would put a number in
    the artifact that nothing produced. ``embedding`` in particular is what
    stops the corpus/judgment hashes from overstating comparability — two
    runs over identical golden data with *different* embedders agree on
    every existing field here while measuring two different semantic spaces,
    which is ADR-0004's silent-mixing failure one layer above the index.

    The three ``rerank_*`` fields exist for a sharper version of that same
    problem, and are not optional documentation (ADR-0012). A ``rerank``
    stage reorders **the best upstream stage available** — ``fusion`` when
    the run had a dense pair, ``bm25`` when it did not — so unlike every
    other stage in this schema, what the ``rerank`` row measured is a
    function of how the run was configured. Two reports can hold a ``rerank``
    stage each, agree on ``corpus_hash``, ``judgments_hash`` and every other
    field here, and describe two different experiments. ``rerank_input`` is
    what makes that difference visible; without it the incomparability is
    silent, which is the one outcome ADR-0012 refuses. ``rerank_candidates``
    matters for a reason that is easy to miss: the reranker truncates to
    ``top_k`` *after* reordering, so a run whose candidate depth equals
    ``top_k`` hands the model a set it can only permute, and that run's
    ``recall_at_10`` is pinned to the upstream stage's by construction rather
    than measured. A reader comparing two ``rerank`` rows needs the depth to
    know whether a flat recall delta was a finding or an arithmetic
    inevitability. ``rerank_model`` closes the ADR-0004 argument one more
    layer up: two cross-encoders are two different measurements.

    Attributes:
        chunk_size: Target chunk size in characters
            (``ChunkingConfig.chunk_size``).
        chunk_overlap: Overlap between consecutive chunks, in characters.
        top_k: Number of results requested per query.
        bm25_k1: BM25 term-frequency saturation parameter.
        bm25_b: BM25 length-normalization parameter.
        score_threshold: Minimum score filter in effect, or ``None`` if
            disabled.
        embedding: The ADR-0004 identity triple of the embedder that
            produced this run's dense vectors, or ``None`` for a BM25-only
            run that embedded nothing. A ``provider`` of ``"inmemory"``
            marks every dense-derived number in the report as structurally
            valid and semantically meaningless (SPEC.md §2) — recorded in
            the artifact so that judgment is machine-checkable rather than
            a footnote a reader has to remember.
        rrf_k: The RRF constant used by the fusion stage (ADR-0005), or
            ``None`` if this run produced no fusion stage.
        rerank_input: The stage whose results the ``rerank`` stage
            reordered, or ``None`` if this run produced no rerank stage.
            Never ``"rerank"`` — a stage cannot be its own input.
        rerank_candidates: How many candidates the upstream stage was asked
            for before reranking truncated them back to ``top_k``, or
            ``None`` if this run produced no rerank stage. Never below
            ``top_k``.
        rerank_model: Identity of the reranker that produced the stage, or
            ``None`` if this run produced no rerank stage. A value that
            names a class rather than a model — anything other than a real
            cross-encoder identifier — marks the stage's numbers as
            structurally valid and semantically meaningless, exactly as
            ``embedding.provider == "inmemory"`` does for the dense stages.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_size: int
    chunk_overlap: int
    top_k: int
    bm25_k1: float
    bm25_b: float
    score_threshold: float | None
    embedding: EmbeddingIdentity | None = None
    rrf_k: int | None = Field(default=None, gt=0)
    rerank_input: StageName | None = None
    rerank_candidates: int | None = Field(default=None, gt=0)
    rerank_model: str | None = None

    @model_validator(mode="after")
    def _validate_rerank_fields(self) -> RunConfig:
        """The three ``rerank_*`` fields are all present or all absent.

        Half a rerank record is worse than none: a ``rerank`` stage whose
        ``rerank_input`` is ``None`` is precisely the silently-incomparable
        artifact ADR-0012 exists to prevent, and a ``rerank_input`` with no
        stage to describe claims a measurement the run never made. Neither
        is representable.
        """
        present = [
            name
            for name in ("rerank_input", "rerank_candidates", "rerank_model")
            if getattr(self, name) is not None
        ]
        if present and len(present) != 3:
            raise ValueError(
                "rerank_input, rerank_candidates and rerank_model must be set together "
                f"or not at all; got only {sorted(present)}"
            )
        if self.rerank_input == "rerank":
            raise ValueError("rerank_input must name the stage rerank reordered, not 'rerank'")
        if self.rerank_candidates is not None and self.rerank_candidates < self.top_k:
            raise ValueError(
                f"rerank_candidates ({self.rerank_candidates}) must be at least top_k "
                f"({self.top_k}); reranking cannot return more results than it was given"
            )
        return self


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

    stage: StageName
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

"""Read-time delta derivation over an :class:`EvalReport` (SPEC.md §6).

Nothing in this module is ever persisted. ``schema.py`` deliberately stores
no delta — a stored delta is redundant data that can disagree with the two
numbers it was computed from — so the delta is computed *here*, from
``stages[i].aggregate`` minus ``stages[0].aggregate``, every time a reader
wants one. :class:`StageDelta` is a return type, never a field of
:class:`~groundkit.evals.schema.EvalReport`; adding it to the artifact would
reintroduce exactly the drift ``schema.py`` was written to prevent.

**No noise threshold, deliberately.** A "wash" band — treating a delta
smaller than some epsilon as no change — would require a number this repo
has no basis to pick. The golden corpus is small enough that a real delta
may sit inside any epsilon anyone chose (R2,
``docs/specs/phase-3-hybrid-retrieval.md``), so an invented threshold would
launder a measurement into a verdict. :attr:`StageDelta.is_regression` is
therefore a strict sign test on the observed numbers, and the magnitudes are
reported alongside it so the reader can judge whether a delta that small
means anything on a corpus that size.

**Direction is never suppressed.** SPEC.md §6's baseline discipline requires
that a stage which does not beat baseline is *reported as such*. Every
non-baseline stage in a report gets a :class:`StageDelta` whether it won,
lost, or tied; there is no filtering step here, and callers that render
deltas must not add one.

**Two derivations, not one, once a rerank stage exists.**
:func:`derive_stage_deltas` answers SPEC.md §6's question — every stage
against the BM25 baseline — and that stays the report's spine. But a rerank
stage reorders the best upstream stage available (ADR-0012 decision 1), so on
a dense run its baseline delta is ``fusion``'s gain *plus* the reranker's,
summed into one number with nothing separating them.
:func:`derive_rerank_attribution` diffs the rerank stage against the stage
named by ``RunConfig.rerank_input`` instead. Neither is stored; both are
recomputed from the report's own numbers on every call.

**How clean that attribution is depends on the input stage, and the
difference is not cosmetic.** The rerank stage fetches ``rerank_candidates``
results where its input stage fetched ``top_k``:

- **BM25 input: a clean isolation.** BM25 scores each document independently
  of ``top_k`` (``index/bm25.py``), which only truncates an already-sorted
  list. ``bm25@50`` is therefore a strict prefix of ``bm25@10``, the
  reranker sees a superset of exactly what the baseline stage scored, and
  the delta is attributable to the cross-encoder alone.
- **Fusion input: NOT a clean isolation.** RRF sums ``1/(rrf_k + rank)``
  over the rankings a chunk is *visible in* at the fetched depth
  (``retrieval/fusion.py``), so widening the fetch adds contributions that
  did not exist at the narrower one. ``fusion@50`` is a different ranking
  function, not a deeper slice of ``fusion@10`` — a chunk absent from the
  depth-10 fused list can rank first at depth 50. The resulting delta
  therefore measures *the rerank pipeline at candidate depth N versus the
  fusion stage as reported*, which is a real and useful production
  comparison, and it is **not** the cross-encoder's isolated contribution.
  Do not read it as one.

Closing that gap needs a fourth comparison stage — fusion recomputed at the
rerank candidate depth — which is deliberately not built. On the current
golden corpus it would sharpen a number that stays uninterpretable anyway:
the noise floor is set by corpus size rather than by this confound, and
``recall@10`` is already saturated. The revisit trigger is a corpus large
enough that a two-to-three query difference clears noise (R2,
``docs/specs/phase-3-hybrid-retrieval.md``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from groundkit.errors import EvalError
from groundkit.evals.schema import StageName

if TYPE_CHECKING:
    from groundkit.evals.schema import EvalReport, MetricSet, StageResult

#: The quality metrics a delta is computed over, in report order. Latency is
#: deliberately excluded: it is reported separately on :class:`StageDelta`
#: because "lower is better" inverts the sign convention every quality metric
#: here shares, and folding both into one mapping would make
#: :attr:`StageDelta.is_regression` mean two different things at once.
QUALITY_METRIC_FIELDS: tuple[str, ...] = (
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "mrr",
    "ndcg_at_10",
)


class StageDelta(BaseModel):
    """One non-baseline stage's signed difference against the baseline stage.

    Derived, never stored — see the module docstring. Every value is
    ``stage`` minus ``baseline``, so a positive quality delta is an
    improvement and a negative one is a regression. Latency inverts that:
    ``latency_p50_delta_ms`` positive means the stage got *slower*, which is
    why it takes no part in :attr:`is_regression`.

    Attributes:
        stage: The stage this delta describes.
        baseline_stage: The stage it was measured against — always
            ``stages[0]``, which :class:`~groundkit.evals.schema.EvalReport`
            pins to ``"bm25"``.
        quality: Signed per-metric quality deltas, keyed by the field names
            in :data:`QUALITY_METRIC_FIELDS`.
        latency_p50_delta_ms: Change in median latency; positive is slower.
        latency_p95_delta_ms: Change in p95 latency; positive is slower.
        latency_p99_delta_ms: Change in p99 latency; positive is slower.
        is_regression: True if *any* quality metric is strictly below
            baseline. A strict sign test with no tolerance band, for the
            reason given in the module docstring.
        is_improvement: True if *any* quality metric is strictly above
            baseline. Not the negation of :attr:`is_regression`: a stage can
            gain recall@1 while losing nDCG@10, and both flags are then True.
            Reporting that as a single verdict would hide half the result.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: StageName
    baseline_stage: StageName
    quality: dict[str, float]
    latency_p50_delta_ms: float
    latency_p95_delta_ms: float
    latency_p99_delta_ms: float
    is_regression: bool
    is_improvement: bool

    @property
    def net_quality(self) -> float:
        """Sum of the signed quality deltas — a glance, not a verdict.

        Useful for ordering several stages in a summary table. Deliberately
        not a score: the metrics it sums are not commensurable (recall@1 and
        nDCG@10 do not move on the same scale), so a positive sum is not
        evidence a stage is better. Read :attr:`quality` for that.
        """
        return sum(self.quality.values())


def derive_stage_deltas(report: EvalReport) -> list[StageDelta]:
    """Derive one :class:`StageDelta` per non-baseline stage in ``report``.

    Computed fresh from the report's own numbers on every call — the delta
    is never read from the artifact, because the artifact never stores it.

    Args:
        report: A validated report. Its own validators already guarantee
            ``stages[0]`` is the baseline and is ``"bm25"``, so this
            function trusts that rather than re-checking it.

    Returns:
        Deltas for ``stages[1:]``, in report order. A single-stage
        (baseline-only) report yields ``[]`` — there is nothing to compare
        it against, which is the correct empty answer rather than a
        zero-valued self-comparison.
    """
    baseline = report.stages[0]
    return [_delta_for(stage, baseline) for stage in report.stages[1:]]


def derive_rerank_attribution(report: EvalReport) -> StageDelta | None:
    """Derive the ``rerank`` stage's delta against the stage it reordered.

    The companion to :func:`derive_stage_deltas`, and the reason
    :class:`~groundkit.evals.schema.RunConfig.rerank_input` is recorded
    rather than inferred. Against ``stages[0]`` a rerank stage that reordered
    ``fusion`` reports fusion's gain and the reranker's added together; this
    separates them as far as the input stage's ranking allows.

    **Read the module docstring before interpreting the fusion-input case.**
    This is the reranker's isolated contribution only when the input stage's
    ranking is depth-invariant. That holds for ``bm25`` and does not hold for
    ``fusion``: the rerank stage fetches a wider candidate pool, and RRF is a
    function of the fetch depth, so part of a fusion-input delta belongs to
    the wider pool rather than to the cross-encoder.

    The returned :class:`StageDelta` carries ``baseline_stage =
    rerank_input``, so a renderer that prints ``delta[{stage} vs
    {baseline_stage}]`` labels it correctly with no special-casing — the two
    deltas are the same type saying different true things, not one of them
    mislabelled.

    On a BM25-only run the input stage *is* ``stages[0]``, so this returns
    the same numbers :func:`derive_stage_deltas` already produced for that
    stage. That duplication is deliberate: suppressing it would make the
    presence of an attribution depend on the run's configuration, and a
    caller would have to reimplement this function's logic to know whether
    to expect one.

    Args:
        report: A validated report.

    Returns:
        The rerank stage's delta against its input stage, or ``None`` if the
        report has no rerank stage.

    Raises:
        RerankInputMissingError: The report has a rerank stage whose
            ``rerank_input`` names a stage the report does not contain.
            Unreachable through :func:`~groundkit.evals.runner.run_eval`,
            which derives the name from the plan it executed — but this
            function is the one place that assumption is load-bearing, and
            silently returning ``None`` would report "no rerank ran" for an
            artifact that plainly contains one.
    """
    rerank_stage = next((stage for stage in report.stages if stage.stage == "rerank"), None)
    if rerank_stage is None:
        return None
    input_name = report.run.config.rerank_input
    input_stage = next((stage for stage in report.stages if stage.stage == input_name), None)
    if input_stage is None:
        raise RerankInputMissingError(
            f"report has a 'rerank' stage whose rerank_input is {input_name!r}, which is "
            f"not among its stages ({[stage.stage for stage in report.stages]}). The "
            "reranker's own contribution cannot be attributed against a stage that is "
            "not there."
        )
    return _delta_for(rerank_stage, input_stage)


def _delta_for(stage: StageResult, baseline: StageResult) -> StageDelta:
    """Build one stage's delta against ``baseline``.

    Args:
        stage: The non-baseline stage to describe.
        baseline: The report's ``stages[0]``.

    Returns:
        The signed :class:`StageDelta`.
    """
    quality = _quality_deltas(stage.aggregate, baseline.aggregate)
    return StageDelta(
        stage=stage.stage,
        baseline_stage=baseline.stage,
        quality=quality,
        latency_p50_delta_ms=stage.latency_p50_ms - baseline.latency_p50_ms,
        latency_p95_delta_ms=stage.latency_p95_ms - baseline.latency_p95_ms,
        latency_p99_delta_ms=stage.latency_p99_ms - baseline.latency_p99_ms,
        is_regression=any(value < 0.0 for value in quality.values()),
        is_improvement=any(value > 0.0 for value in quality.values()),
    )


def _quality_deltas(stage: MetricSet, baseline: MetricSet) -> dict[str, float]:
    """Signed ``stage - baseline`` for every field in :data:`QUALITY_METRIC_FIELDS`.

    Read via ``getattr`` off the field-name tuple rather than five hand-written
    subtractions so that a metric added to
    :class:`~groundkit.evals.schema.MetricSet` and to
    :data:`QUALITY_METRIC_FIELDS` is picked up here automatically instead of
    being silently omitted from every delta.

    Args:
        stage: The non-baseline stage's aggregate metrics.
        baseline: The baseline stage's aggregate metrics.

    Returns:
        ``field name -> signed delta``.
    """
    return {
        field: float(getattr(stage, field)) - float(getattr(baseline, field))
        for field in QUALITY_METRIC_FIELDS
    }


class RerankInputMissingError(EvalError):
    """A report's ``rerank_input`` names a stage the report does not contain.

    Raised by :func:`derive_rerank_attribution`. Separate from
    :class:`MetricFieldMismatchError` because it describes an inconsistent
    *artifact* rather than an inconsistent *module*, and a caller may
    reasonably want to catch one without the other.

    **Rooted in** :class:`~groundkit.errors.GroundkitError`, via
    :class:`~groundkit.errors.EvalError`, and that is load-bearing rather
    than tidy. This class began as a bare ``ValueError`` — which made it the
    only reachable exception in the package outside the local root, so the
    one guard written to *fail closed* on an inconsistent artifact was
    itself the untyped escape every other failure here is wrapped to
    prevent. ``cli.py``'s ``except GroundkitError`` would have missed it and
    ``grk eval`` would have died on a raw traceback while reporting a
    perfectly ordinary artifact problem.
    """


class MetricFieldMismatchError(EvalError):
    """:data:`QUALITY_METRIC_FIELDS` names a field ``MetricSet`` does not have.

    Raised by :func:`assert_metric_fields_exist`, which
    ``tests/test_delta.py`` calls so the tuple cannot silently drift away
    from the model it indexes into — a ``getattr`` against a renamed field
    would otherwise fail only at run time, inside a report nobody reads
    until the numbers are already wrong.
    """


def assert_metric_fields_exist() -> None:
    """Check every name in :data:`QUALITY_METRIC_FIELDS` is a ``MetricSet`` field.

    Raises:
        MetricFieldMismatchError: A name in the tuple is not a field of
            :class:`~groundkit.evals.schema.MetricSet`.
    """
    from groundkit.evals.schema import MetricSet as _MetricSet

    missing = [field for field in QUALITY_METRIC_FIELDS if field not in _MetricSet.model_fields]
    if missing:
        raise MetricFieldMismatchError(
            f"QUALITY_METRIC_FIELDS names field(s) MetricSet does not define: {missing}"
        )

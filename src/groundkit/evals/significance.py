"""Deterministic paired-bootstrap comparisons for retrieval experiments.

The unit of resampling is a query, not an individual retrieved hit.  Callers
must therefore provide one scalar score per query for both systems.  Pairing
by query id is checked explicitly so a missing or reordered query cannot turn
into an unpaired comparison by accident.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from groundkit.errors import EvalError
from groundkit.evals.schema import EvalReport, QueryResult, StageName

MetricName = Literal["recall_at_1", "recall_at_5", "recall_at_10", "mrr", "ndcg_at_10"]


class BootstrapResult(BaseModel):
    """Machine-readable result of one paired-bootstrap comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline: str
    candidate: str
    metric: MetricName
    query_count: int = Field(gt=0)
    resamples: int = Field(gt=0)
    seed: int
    confidence_level: float = Field(gt=0.0, lt=1.0)
    effect: float
    confidence_interval_low: float
    confidence_interval_high: float
    #: Twice the smaller tail proportion of the resampling distribution about
    #: zero, with a ``(r + 1) / (B + 1)`` finite-sample correction. This is a
    #: Monte Carlo *achieved significance level* in the Smucker/Allan/Carterette
    #: (2007) style -- the resamples are drawn from the observed deltas, not
    #: from a null-centred distribution -- so it is not a null-hypothesis
    #: p-value in the Neyman-Pearson sense and should not be described as one.
    #: It carries a floor of ``2 / (B + 1)`` by construction and can never be
    #: zero. :attr:`significant` is the primary verdict; it is computed from
    #: the confidence interval and never reads this field.
    p_value_two_sided: float = Field(gt=0.0, le=1.0)
    #: Whether the percentile confidence interval excludes zero. Computed
    #: independently of :attr:`p_value_two_sided`.
    significant: bool
    method: str = "paired bootstrap over query-level candidate-minus-baseline deltas"


def paired_bootstrap(
    baseline_scores: Mapping[str, float],
    candidate_scores: Mapping[str, float],
    *,
    baseline_name: str,
    candidate_name: str,
    metric: MetricName,
    resamples: int = 10_000,
    seed: int = 7,
    confidence_level: float = 0.95,
) -> BootstrapResult:
    """Compare two systems by resampling paired query-level score deltas.

    The confidence interval uses linear interpolation between ordered
    bootstrap means, and is what :attr:`BootstrapResult.significant` reads.

    :attr:`BootstrapResult.p_value_two_sided` is twice the smaller empirical
    tail proportion around zero, carrying a ``(r + 1) / (B + 1)``
    finite-sample correction so it is never reported as exactly zero. It is
    an achieved significance level computed from resamples of the *observed*
    deltas, not from a null-centred distribution -- see that field's own note.
    The two are computed independently: correcting one cannot move the other.
    """

    if resamples <= 0:
        raise EvalError(f"resamples must be positive, got {resamples}")
    if not 0.0 < confidence_level < 1.0:
        raise EvalError(f"confidence_level must be between 0 and 1, got {confidence_level}")
    if not baseline_scores:
        raise EvalError("paired bootstrap requires at least one query score")
    if baseline_scores.keys() != candidate_scores.keys():
        missing_candidate = sorted(baseline_scores.keys() - candidate_scores.keys())
        missing_baseline = sorted(candidate_scores.keys() - baseline_scores.keys())
        raise EvalError(
            "paired bootstrap query ids differ "
            f"(missing from candidate={missing_candidate}, "
            f"missing from baseline={missing_baseline})"
        )

    query_ids = sorted(baseline_scores)
    deltas = [candidate_scores[qid] - baseline_scores[qid] for qid in query_ids]
    if not all(math.isfinite(delta) for delta in deltas):
        raise EvalError("paired bootstrap scores must be finite")

    # A seeded simulation needs reproducibility, not cryptographic randomness.
    rng = random.Random(seed)  # noqa: S311
    query_count = len(deltas)
    bootstrap_means = [
        sum(deltas[rng.randrange(query_count)] for _ in range(query_count)) / query_count
        for _ in range(resamples)
    ]
    bootstrap_means.sort()

    alpha = (1.0 - confidence_level) / 2.0
    low = _quantile(bootstrap_means, alpha)
    high = _quantile(bootstrap_means, 1.0 - alpha)
    # Finite-sample correction, (r + 1) / (B + 1) rather than r / B. A Monte
    # Carlo estimate over B draws cannot resolve a probability below 1/B, so a
    # zero tail count means "smaller than this simulation can measure", never
    # "impossible". Reporting the uncorrected 0.0 states a certainty no finite
    # resampling can support, and it is reachable in practice, not just in
    # theory: any effect large and consistent enough that no resample crosses
    # zero produces it -- which is exactly what the strongest real contrasts do.
    non_positive = sum(value <= 0.0 for value in bootstrap_means)
    non_negative = sum(value >= 0.0 for value in bootstrap_means)
    p_value = min(1.0, 2.0 * (min(non_positive, non_negative) + 1) / (resamples + 1))

    return BootstrapResult(
        baseline=baseline_name,
        candidate=candidate_name,
        metric=metric,
        query_count=query_count,
        resamples=resamples,
        seed=seed,
        confidence_level=confidence_level,
        effect=sum(deltas) / query_count,
        confidence_interval_low=low,
        confidence_interval_high=high,
        p_value_two_sided=p_value,
        significant=low > 0.0 or high < 0.0,
    )


def compare_report_stages(
    report: EvalReport,
    *,
    baseline: StageName,
    candidate: StageName,
    metric: MetricName,
    resamples: int = 10_000,
    seed: int = 7,
    confidence_level: float = 0.95,
) -> BootstrapResult:
    """Compare two stages from the same eval artifact."""

    # `EvalReport`'s validator pins exactly one baseline, at position zero,
    # named "bm25" -- but nothing stops two later stages sharing a name. A
    # dict comprehension would keep the last silently, so which `dense` stage
    # a comparison actually measured would depend on their order in the file.
    duplicate_stages = sorted(
        {stage.stage for stage in report.stages if _count_stage(report, stage.stage) > 1}
    )
    if duplicate_stages:
        raise EvalError(
            f"report contains more than one stage named {duplicate_stages}; a comparison "
            "cannot say which of them it measured"
        )
    stages = {stage.stage: stage for stage in report.stages}
    try:
        baseline_stage = stages[baseline]
        candidate_stage = stages[candidate]
    except KeyError as exc:
        raise EvalError(f"report does not contain stage {exc.args[0]!r}") from exc

    baseline_scores = _query_scores(baseline_stage.queries, metric)
    candidate_scores = _query_scores(candidate_stage.queries, metric)
    return paired_bootstrap(
        baseline_scores,
        candidate_scores,
        baseline_name=baseline,
        candidate_name=candidate,
        metric=metric,
        resamples=resamples,
        seed=seed,
        confidence_level=confidence_level,
    )


def _count_stage(report: EvalReport, name: StageName) -> int:
    return sum(1 for stage in report.stages if stage.stage == name)


def _query_scores(query_results: Sequence[QueryResult], metric: MetricName) -> dict[str, float]:
    """Map query id to one metric value, refusing a duplicate id.

    ``QueryResult.query_id`` is documented as unique but nothing in the schema
    enforces it across a stage's ``queries`` list. Assigning into a dict would
    silently keep the last of a duplicated pair, which is worse than it
    sounds: the bootstrap would then run over a *smaller* query set than the
    report claims, and the effect would depend on the order the duplicates
    happened to appear in. This module's whole premise is that pairing by
    query id is checked rather than assumed, so a report that cannot be paired
    unambiguously is rejected instead of quietly averaged.
    """
    scores: dict[str, float] = {}
    for result in query_results:
        if result.metrics is None:
            continue
        if result.query_id in scores:
            raise EvalError(
                f"duplicate query id {result.query_id!r} in a stage's results; "
                "query ids must be unique for a paired comparison to mean anything"
            )
        value = (
            result.metrics.reciprocal_rank if metric == "mrr" else getattr(result.metrics, metric)
        )
        scores[result.query_id] = float(value)
    return scores


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise EvalError("cannot calculate a quantile over an empty sample")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight

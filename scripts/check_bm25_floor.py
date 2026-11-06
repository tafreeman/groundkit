"""Fail if offline BM25 retrieval quality drops below the committed floor.

Runs the BM25-only eval (:func:`groundkit.evals.runner.run_eval` with no
embedder) over the committed golden corpus, ``evals/corpus`` +
``evals/judgments.jsonl``, and compares the baseline stage's aggregate metrics
to ``evals/bm25-floor.json``. Deterministic, offline, credential-free: no model
is pulled and nothing touches the network, so it can run on every pull request
and inside ``release-gates.yml``.

This is a regression tripwire, not a quality claim. The floor was computed by
running this script with ``--write-floor`` (each metric rounded *down* to four
decimals), never typed in by hand. Real-model dense/fusion deltas stay in
``eval-gated.yml`` (SPEC.md §6); the BEIR subset under ``evals/beir/`` needs a
dataset download and so cannot run offline from committed data.

Exit codes: 0 every metric is at or above its floor; 1 at least one metric
regressed below it; 2 the check could not run (missing or invalid floor file,
unreadable report, import failure).

Usage::

    uv run python scripts/check_bm25_floor.py
    uv run python scripts/check_bm25_floor.py --write-floor   # deliberate re-baseline
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path

try:
    from groundkit.errors import GroundkitError
    from groundkit.evals.runner import run_eval
except ImportError as _exc:  # e.g. run with a Python that is not the project venv
    print(f"error: cannot import groundkit ({_exc}); run via `uv run`", file=sys.stderr)
    raise SystemExit(2) from None

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO_ROOT / "evals" / "corpus"
JUDGMENTS_PATH = REPO_ROOT / "evals" / "judgments.jsonl"
FLOOR_PATH = REPO_ROOT / "evals" / "bm25-floor.json"

#: Aggregate metrics gated. nDCG@10 and recall@10 are the audit's named pair;
#: recall@5 and MRR catch a ranking regression that recall@10 would absorb.
GATED_METRICS: tuple[str, ...] = ("ndcg_at_10", "recall_at_5", "recall_at_10", "mrr")
#: Decimal places the floor is rounded down to when written.
FLOOR_DECIMALS = 4
#: Absolute slack for float comparison, below the floor's own rounding step.
EPSILON = 1e-9


def floor_down(value: float, decimals: int = FLOOR_DECIMALS) -> float:
    """Round ``value`` down to ``decimals`` places (never up)."""
    scale = 10**decimals
    return math.floor(value * scale + EPSILON) / scale


def _non_negative_int(value: object, key: str) -> int:
    # bool is an int subclass; a floor of `true` is a corrupt file, not 1.
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer, got {value!r}")
    return value


def parse_floor(raw: object) -> tuple[dict[str, float], tuple[int, int]]:
    """Validate a decoded floor file; raise ``ValueError`` if it cannot gate.

    Every metric in :data:`GATED_METRICS` must be present (and nothing else), and
    each value must be a finite number in ``[0, 1]``: a missing key would skip its
    comparison and ``NaN`` makes every ``<`` comparison false, so either would let
    the gate pass silently.
    """
    if not isinstance(raw, dict):
        raise ValueError("floor file must be a JSON object")
    metrics = raw.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != set(GATED_METRICS):
        got = sorted(metrics) if isinstance(metrics, dict) else metrics
        raise ValueError(f"metrics must have exactly {sorted(GATED_METRICS)}, got {got!r}")
    floors: dict[str, float] = {}
    for name in GATED_METRICS:
        value = metrics[name]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"metrics.{name} must be a number, got {value!r}")
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"metrics.{name} must be finite and in [0, 1], got {value!r}")
        floors[name] = float(value)
    abstained = _non_negative_int(raw.get("no_answer_abstained"), "no_answer_abstained")
    total = _non_negative_int(raw.get("no_answer_queries"), "no_answer_queries")
    if abstained > total:
        raise ValueError(f"no_answer_abstained {abstained} > no_answer_queries {total}")
    return floors, (abstained, total)


def find_regressions(
    measured: dict[str, float],
    floors: dict[str, float],
    no_answer: tuple[int, int],
    floor_no_answer: tuple[int, int],
) -> list[str]:
    """Return one message per metric that is below its floor; empty means pass.

    ``no_answer`` and ``floor_no_answer`` are ``(abstained, total)``. The total is
    pinned too: a count alone would accept 8/9 against a floor of 8/8, hiding a
    new no-answer judgment that retrieves a result.
    """
    problems: list[str] = []
    for name, floor in floors.items():
        if name not in measured:
            problems.append(f"{name}: not in the report")
        elif measured[name] + EPSILON < floor:
            problems.append(f"{name}: measured {measured[name]:.4f} < floor {floor:.4f}")
    abstained, total = no_answer
    floor_abstain, floor_total = floor_no_answer
    if total != floor_total:
        problems.append(
            f"no_answer_queries: {total} != floor {floor_total} "
            "(judgments changed; re-baseline with --write-floor)"
        )
    if abstained < floor_abstain or total - abstained > floor_total - floor_abstain:
        problems.append(
            f"no_answer_abstained: {abstained}/{total} < floor {floor_abstain}/{floor_total}"
        )
    return problems


async def _measure() -> tuple[dict[str, float], tuple[int, int]]:
    report = await run_eval(CORPUS_DIR, JUDGMENTS_PATH)
    stage = report.stages[0]
    if stage.stage != "bm25":
        raise RuntimeError(f"expected a bm25 baseline stage, got {stage.stage!r}")
    values = stage.aggregate.model_dump()
    measured = {name: float(values[name]) for name in GATED_METRICS}
    return measured, (stage.no_answer_abstained_count, stage.no_answer_query_count)


def _write_floor(measured: dict[str, float], no_answer: tuple[int, int]) -> None:
    payload = {
        "comment": (
            "Offline BM25 regression floor, generated by "
            "`uv run python scripts/check_bm25_floor.py --write-floor`. "
            "Re-baseline only deliberately, in a reviewed change."
        ),
        "metrics": {name: floor_down(value) for name, value in measured.items()},
        "no_answer_abstained": no_answer[0],
        "no_answer_queries": no_answer[1],
    }
    with FLOOR_PATH.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {FLOOR_PATH}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--write-floor", action="store_true", help="Re-baseline the floor file.")
    args = parser.parse_args()
    try:
        measured, no_answer = asyncio.run(_measure())
    except (GroundkitError, RuntimeError, OSError) as exc:
        print(f"error: BM25 eval could not run: {exc}", file=sys.stderr)
        return 2
    summary = ", ".join(f"{k}={v:.4f}" for k, v in measured.items())
    print(f"measured: {summary}, no_answer_abstained={no_answer[0]}/{no_answer[1]}")
    if args.write_floor:
        _write_floor(measured, no_answer)
        return 0
    try:
        floors, floor_no_answer = parse_floor(json.loads(FLOOR_PATH.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:  # json.JSONDecodeError is a ValueError
        print(f"error: invalid floor file {FLOOR_PATH}: {exc}", file=sys.stderr)
        return 2
    problems = find_regressions(measured, floors, no_answer, floor_no_answer)
    if problems:
        print("BM25 retrieval quality regressed below the committed floor:", file=sys.stderr)
        for line in problems:
            print(f"  - {line}", file=sys.stderr)
        return 1
    print(f"BM25 floor OK ({FLOOR_PATH.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

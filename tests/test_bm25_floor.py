"""The offline BM25 floor gate (scripts/check_bm25_floor.py, evals/bm25-floor.json)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "check_bm25_floor.py"
_FLOOR = _ROOT / "evals" / "bm25-floor.json"


def _load_gate() -> Any:
    # The script lives outside the package, so load it by path.
    spec = importlib.util.spec_from_file_location("check_bm25_floor", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load_gate()


def test_floor_down_never_rounds_up() -> None:
    assert gate.floor_down(0.80559) == 0.8055
    assert gate.floor_down(0.5) == 0.5


def test_find_regressions_passes_at_and_above_the_floor() -> None:
    floors = {"ndcg_at_10": 0.6, "mrr": 0.55}
    assert gate.find_regressions({"ndcg_at_10": 0.6, "mrr": 0.7}, floors, (8, 8), (8, 8)) == []


def test_find_regressions_flags_each_metric_below_floor() -> None:
    floors = {"ndcg_at_10": 0.6, "mrr": 0.55}
    problems = gate.find_regressions({"ndcg_at_10": 0.59, "mrr": 0.2}, floors, (8, 8), (8, 8))
    assert len(problems) == 2


def test_find_regressions_flags_lost_abstention_and_missing_metric() -> None:
    problems = gate.find_regressions({}, {"mrr": 0.5}, (7, 8), (8, 8))
    assert any("not in the report" in p for p in problems)
    assert any("no_answer_abstained" in p for p in problems)


def test_find_regressions_flags_a_new_no_answer_query_that_retrieves() -> None:
    # 8/9 against a floor of 8/8: the count held, but a ninth no-answer
    # judgment returned a result, so abstention regressed.
    problems = gate.find_regressions({}, {}, (8, 9), (8, 8))
    assert any("no_answer_queries" in p for p in problems)
    assert any("no_answer_abstained" in p for p in problems)


def _valid_floor() -> dict[str, Any]:
    return {
        "metrics": dict.fromkeys(gate.GATED_METRICS, 0.5),
        "no_answer_abstained": 8,
        "no_answer_queries": 8,
    }


def test_parse_floor_accepts_a_complete_floor() -> None:
    floors, no_answer = gate.parse_floor(_valid_floor())
    assert set(floors) == set(gate.GATED_METRICS)
    assert no_answer == (8, 8)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda f: f["metrics"].pop("mrr"), id="missing-metric"),
        pytest.param(lambda f: f["metrics"].update(extra=0.1), id="unknown-metric"),
        pytest.param(lambda f: f["metrics"].update(mrr=float("nan")), id="nan"),
        pytest.param(lambda f: f["metrics"].update(mrr=float("inf")), id="inf"),
        pytest.param(lambda f: f["metrics"].update(mrr=1.5), id="above-one"),
        pytest.param(lambda f: f["metrics"].update(mrr="0.5"), id="string"),
        pytest.param(lambda f: f["metrics"].update(mrr=True), id="bool"),
        pytest.param(lambda f: f.pop("no_answer_queries"), id="no-denominator"),
        pytest.param(lambda f: f.update(no_answer_abstained=9), id="abstained-gt-total"),
        pytest.param(lambda f: f.update(no_answer_abstained=-1), id="negative"),
    ],
)
def test_parse_floor_rejects_a_floor_that_cannot_gate(mutate: Any) -> None:
    floor = _valid_floor()
    mutate(floor)
    with pytest.raises(ValueError):
        gate.parse_floor(floor)


def test_committed_floor_file_is_valid() -> None:
    floors, _ = gate.parse_floor(json.loads(_FLOOR.read_text(encoding="utf-8")))
    assert set(floors) == set(gate.GATED_METRICS)


def test_committed_corpus_meets_the_committed_floor() -> None:
    completed = subprocess.run(  # noqa: S603 - fixed interpreter and repo script
        [sys.executable, str(_SCRIPT)], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr


def test_script_rejects_unknown_flags() -> None:
    completed = subprocess.run(  # noqa: S603 - fixed interpreter and repo script
        [sys.executable, str(_SCRIPT), "--no-such-flag"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "evals" / "beir" / "embed_bakeoff.py"


def _load_bakeoff() -> ModuleType:
    spec = importlib.util.spec_from_file_location("embed_bakeoff_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bakeoff = _load_bakeoff()


def _valid_result() -> dict[str, Any]:
    """A fresh ``CachedResult``-shaped dict with every field valid.

    Called anew by each test so mutating the returned dict (or its nested
    ``per_query`` entries) never leaks between tests.
    """
    return {
        "model": "nomic-embed-text",
        "dimensions": 768,
        "chunks": 100,
        "reused_index": False,
        "embed_minutes": 12.5,
        "query_p50_ms": 45.0,
        "query_p95_ms": 90.0,
        "vector_store_mb": 12.3,
        "per_query": {
            "q-1": {
                "ndcg_at_10": 0.5,
                "mrr": 0.5,
                "recall_at_1": 0.0,
                "recall_at_10": 1.0,
            },
        },
    }


_IDS: set[str] = {"q-1"}


def test_a_valid_result_is_accepted_as_the_positive_control() -> None:
    """Guards the guard: a validator that rejects everything would still pass
    every negative test below, for the wrong reason. This pins that a
    genuinely well-formed cache entry is still usable, so the rejections
    that follow are meaningful.
    """
    assert bakeoff.usable_result(_valid_result(), _IDS) is True


def test_a_non_numeric_dimensions_is_rejected() -> None:
    """``main()`` prints ``f"{r['dimensions']:5d}"`` once every model has
    scored. A non-int ``dimensions`` reaching that far raises
    ``ValueError: Unknown format code 'd' for object of type 'str'`` and
    kills the whole run's summary output after every model already paid the
    cost of scoring.
    """
    result = _valid_result()
    result["dimensions"] = "x"
    assert bakeoff.usable_result(result, _IDS) is False


def test_a_numeric_string_is_rejected_rather_than_coerced() -> None:
    """The case a declared model does *not* close by itself.

    Pydantic's default lax mode coerces ``"768"`` to ``768`` and ``"0.5"`` to
    ``0.5``, so a cache whose numbers are quoted -- what any tool that
    round-trips JSON through a stringifying layer produces, and what a hand
    edit produces -- validated cleanly and fed the leaderboard. A test using
    an *un*coercible string like ``"x"`` passes with or without strict mode
    and therefore proves nothing about it; these values are the ones that
    distinguish the two.
    """
    quoted_dimensions = _valid_result()
    quoted_dimensions["dimensions"] = "768"
    assert bakeoff.usable_result(quoted_dimensions, _IDS) is False

    quoted_metric = _valid_result()
    quoted_metric["per_query"]["q-1"]["ndcg_at_10"] = "0.5"
    assert bakeoff.usable_result(quoted_metric, _IDS) is False


def test_an_integer_reused_index_is_rejected_rather_than_coerced() -> None:
    """``1`` is not ``True``. Lax mode accepts it as one, which turns a
    corrupt flag into a confident claim that the index was reused -- the one
    field that decides whether an embedding run is skipped."""
    result = _valid_result()
    result["reused_index"] = 1
    assert bakeoff.usable_result(result, _IDS) is False


def test_a_metric_value_that_is_a_string_is_rejected() -> None:
    """A non-numeric metric reaches ``np.mean`` over the per-query values.
    Depending on numpy's dtype inference that either raises deep inside the
    leaderboard sort or silently coerces into a nonsense aggregate -- neither
    is a trustworthy number to publish.
    """
    result = _valid_result()
    result["per_query"]["q-1"]["ndcg_at_10"] = "bad"
    assert bakeoff.usable_result(result, _IDS) is False


def test_an_integer_metric_is_accepted_because_json_returns_ints_for_whole_numbers() -> None:
    """Strict mode must not be so strict it rejects the writer's own output:
    ``json.load`` gives back ``0`` for a metric written as ``0``, and a recall
    of exactly zero or one is the common case, not an edge case."""
    result = _valid_result()
    result["per_query"]["q-1"]["recall_at_1"] = 0
    result["per_query"]["q-1"]["recall_at_10"] = 1
    assert bakeoff.usable_result(result, _IDS) is True


def test_a_nan_metric_is_rejected_because_it_poisons_every_average() -> None:
    """NaN never raises. It silently poisons ``np.mean`` and every comparison
    (including the paired bootstrap) drawn from that average, corrupting the
    leaderboard with no visible error anywhere in the run.
    """
    result = _valid_result()
    result["per_query"]["q-1"]["recall_at_1"] = math.nan
    assert bakeoff.usable_result(result, _IDS) is False


def test_an_infinite_metric_is_rejected() -> None:
    """An infinite metric is finite-looking enough to sort and average without
    raising, but it dominates every mean it enters and silently wins (or
    loses) the leaderboard regardless of the other 999 queries' scores.
    """
    result = _valid_result()
    result["per_query"]["q-1"]["mrr"] = math.inf
    assert bakeoff.usable_result(result, _IDS) is False


def test_per_query_with_wrong_query_ids_is_rejected_because_it_pairs_falsely_in_bootstrap() -> None:
    """The paired bootstrap and every average pair per-query values by query
    id. A result carrying the right shape but scored against different
    queries than the ones actually requested would still pair cleanly and
    report a comparison nobody ran -- the schema alone cannot catch this,
    since it only checks shape, not identity.
    """
    result = _valid_result()
    result["per_query"] = {"q-999": result["per_query"]["q-1"]}
    assert bakeoff.usable_result(result, _IDS) is False


def test_per_query_missing_one_of_several_expected_ids_is_rejected() -> None:
    """A cache scored against a subset of the current judgment set would
    silently drop the missing queries from every metric's average, quietly
    understating or overstating a model with no indication that anything
    was left out.
    """
    result = _valid_result()
    result["per_query"] = {
        "q-1": {"ndcg_at_10": 0.5, "mrr": 0.5, "recall_at_1": 0.0, "recall_at_10": 1.0},
        "q-2": {"ndcg_at_10": 0.4, "mrr": 0.4, "recall_at_1": 0.0, "recall_at_10": 1.0},
    }
    assert bakeoff.usable_result(result, {"q-1", "q-2", "q-3"}) is False


def test_a_result_with_an_extra_unknown_key_is_rejected() -> None:
    """``extra="forbid"`` makes drift bidirectional: a field the writer starts
    producing without the model being updated to match must fail loudly here,
    not be silently dropped and hide that the writer and the model disagree
    on what a result contains.
    """
    result = _valid_result()
    result["unexpected_field"] = "surprise"
    assert bakeoff.usable_result(result, _IDS) is False


def test_a_result_missing_a_required_key_is_rejected() -> None:
    """A required field dropped by a truncated write (or an older build of
    the writer) must not be treated as a complete, publishable measurement --
    every downstream reader of a ``CachedResult`` assumes every field is
    present.
    """
    result = _valid_result()
    del result["chunks"]
    assert bakeoff.usable_result(result, _IDS) is False


@pytest.mark.parametrize("non_dict", [None, [], {}], ids=["none", "empty_list", "empty_dict"])
def test_a_non_dict_result_is_rejected(non_dict: object) -> None:
    """A non-mapping (or a mapping missing every field) must fail inside
    ``usable_result``'s own try/except. Letting it raise instead would
    propagate through ``asyncio.gather`` and abort every other model in the
    wave, rather than just dropping this one model from the leaderboard.
    """
    assert bakeoff.usable_result(non_dict, _IDS) is False


def test_embed_minutes_none_is_accepted_because_it_is_legitimately_optional() -> None:
    """``embed_minutes`` is ``None`` whenever a run rescored against a reused
    index without re-measuring embedding time -- a legitimate, common case,
    not a defect. This pins that the model was not accidentally tightened to
    require it, which would reject every rescore-only cache entry.
    """
    result = _valid_result()
    result["embed_minutes"] = None
    assert bakeoff.usable_result(result, _IDS) is True


def test_cached_result_fields_match_the_writer_dict_exactly() -> None:
    """Anti-drift: ``CachedResult``'s declared fields must equal the keys the
    writer actually produces.

    Hardcoded from the dict literal returned by ``index_and_score`` in
    ``evals/beir/embed_bakeoff.py`` (the writer), not derived from it -- the
    point is to catch drift between the two, so the expectation must live
    independently of both. If this assertion fails, the writer and the model
    have drifted: a field was added to, removed from, or renamed on one side
    without the other. Update BOTH the writer's dict and ``CachedResult``
    together; do not just adjust this test's expected set.
    """
    writer_keys = {
        "model",
        "dimensions",
        "chunks",
        "reused_index",
        "embed_minutes",
        "query_p50_ms",
        "query_p95_ms",
        "vector_store_mb",
        "per_query",
    }
    assert set(bakeoff.CachedResult.model_fields) == writer_keys

from __future__ import annotations

import importlib.util
import math
import random
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


def test_a_negative_metric_is_rejected() -> None:
    """``allow_inf_nan=False`` alone admitted this. An out-of-range float is
    as unpublishable as a NaN and quieter about it: a NaN propagates visibly
    through the leaderboard, while ``-1.0`` prints as a plausible number."""
    result = _valid_result()
    result["per_query"]["q-1"]["ndcg_at_10"] = -1.0
    assert bakeoff.usable_result(result, _IDS) is False


def test_a_metric_above_one_is_rejected() -> None:
    """No metric ``score_ranking`` computes can exceed 1.0, so a cached value
    that does is a corrupt file rather than a good run."""
    result = _valid_result()
    result["per_query"]["q-1"]["recall_at_10"] = 2.0
    assert bakeoff.usable_result(result, _IDS) is False


def test_the_unit_interval_endpoints_are_still_accepted() -> None:
    """The bound must be inclusive at both ends. A perfect ranking scores
    exactly 1.0 and a miss scores exactly 0.0 -- both are the common case, so
    an exclusive bound would reject good runs and silently force a rescore."""
    perfect = _valid_result()
    perfect["per_query"]["q-1"] = dict.fromkeys(bakeoff.METRICS, 1.0)
    assert bakeoff.usable_result(perfect, _IDS) is True

    missed = _valid_result()
    missed["per_query"]["q-1"] = dict.fromkeys(bakeoff.METRICS, 0.0)
    assert bakeoff.usable_result(missed, _IDS) is True


def test_impossible_scalars_are_rejected() -> None:
    """The same defect class as an out-of-range metric, one field out. A
    negative latency, a negative store size or a zero-dimension embedding are
    all finite, all impossible, and all published without complaint before."""
    for field, value in (
        ("dimensions", 0),
        ("dimensions", -8),
        ("query_p50_ms", -5.0),
        ("query_p95_ms", -5.0),
        ("vector_store_mb", -1.0),
        ("embed_minutes", -3.0),
        ("model", ""),
    ):
        result = _valid_result()
        result[field] = value
        assert bakeoff.usable_result(result, _IDS) is False, f"{field}={value!r} was accepted"


def test_a_zero_embed_minutes_is_still_accepted() -> None:
    """``ge``, not ``gt``: SCORING_VERSION 2's note records that entries
    written before the build-timing fix legitimately carry ``0.0``."""
    result = _valid_result()
    result["embed_minutes"] = 0.0
    assert bakeoff.usable_result(result, _IDS) is True


def test_a_p95_below_p50_is_rejected_but_an_equal_pair_is_not() -> None:
    """Both percentiles are read out of one sorted latency array at monotone
    indices and rounded to the same precision, so p95 below p50 cannot happen
    in a run the writer produced -- it is proof the file was edited. An equal
    pair, by contrast, is ordinary on a short or uniform battery."""
    swapped = _valid_result()
    swapped["query_p50_ms"], swapped["query_p95_ms"] = 9.0, 2.0
    assert bakeoff.usable_result(swapped, _IDS) is False

    equal = _valid_result()
    equal["query_p50_ms"] = equal["query_p95_ms"] = 4.0
    assert bakeoff.usable_result(equal, _IDS) is True


def test_every_ranking_score_ranking_can_produce_validates() -> None:
    """Ties the model to the writer rather than to my reading of it.

    If a bound here is tighter than what ``score_ranking`` actually returns,
    real runs start failing validation and rescoring forever. Fuzzing the
    scorer and validating each row is the check that catches that, and it is
    the direction the per-field tests above cannot cover.
    """
    # S311: fuzzing the scorer, seeded so a failure is reproducible.
    rng = random.Random(7)  # noqa: S311
    for _ in range(2000):
        size = rng.randint(1, 12)
        docs = [f"d{i}" for i in range(size)]
        gold = set(rng.sample(docs, rng.randint(1, size)))
        order = docs[:]
        rng.shuffle(order)
        bakeoff.CachedMetrics.model_validate(bakeoff.score_ranking(order, gold))

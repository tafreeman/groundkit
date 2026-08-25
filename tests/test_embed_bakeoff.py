from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from groundkit.errors import EvalError

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


def _usable(
    result: Any,
    ids: set[str] | None = None,
    *,
    model: str | None = None,
    dimensions: int | None = None,
) -> bool:
    """``usable_result`` with the identity arguments defaulted to the result.

    Defaulted *from the result* rather than to fixed constants, deliberately:
    with constants, mutating ``dimensions`` to ``0`` would trip the identity
    check as well as the ``gt=0`` bound, and every range test below would pass
    for a reason other than the one it names. Tests that are about identity
    pass ``model``/``dimensions`` explicitly instead.
    """
    body = result if isinstance(result, dict) else {}
    return bool(
        bakeoff.usable_result(
            result,
            _IDS if ids is None else ids,
            model=body.get("model", "nomic-embed-text") if model is None else model,
            dimensions=body.get("dimensions", 768) if dimensions is None else dimensions,
        )
    )


def test_a_valid_result_is_accepted_as_the_positive_control() -> None:
    """Guards the guard: a validator that rejects everything would still pass
    every negative test below, for the wrong reason. This pins that a
    genuinely well-formed cache entry is still usable, so the rejections
    that follow are meaningful.
    """
    assert _usable(_valid_result()) is True


def test_a_non_numeric_dimensions_is_rejected() -> None:
    """``main()`` prints ``f"{r['dimensions']:5d}"`` once every model has
    scored. A non-int ``dimensions`` reaching that far raises
    ``ValueError: Unknown format code 'd' for object of type 'str'`` and
    kills the whole run's summary output after every model already paid the
    cost of scoring.
    """
    result = _valid_result()
    result["dimensions"] = "x"
    assert _usable(result) is False


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
    assert _usable(quoted_dimensions) is False

    quoted_metric = _valid_result()
    quoted_metric["per_query"]["q-1"]["ndcg_at_10"] = "0.5"
    assert _usable(quoted_metric) is False


def test_an_integer_reused_index_is_rejected_rather_than_coerced() -> None:
    """``1`` is not ``True``. Lax mode accepts it as one, which turns a
    corrupt flag into a confident claim that the index was reused -- the one
    field that decides whether an embedding run is skipped."""
    result = _valid_result()
    result["reused_index"] = 1
    assert _usable(result) is False


def test_a_metric_value_that_is_a_string_is_rejected() -> None:
    """A non-numeric metric reaches ``np.mean`` over the per-query values.
    Depending on numpy's dtype inference that either raises deep inside the
    leaderboard sort or silently coerces into a nonsense aggregate -- neither
    is a trustworthy number to publish.
    """
    result = _valid_result()
    result["per_query"]["q-1"]["ndcg_at_10"] = "bad"
    assert _usable(result) is False


def test_an_integer_metric_is_accepted_because_json_returns_ints_for_whole_numbers() -> None:
    """Strict mode must not be so strict it rejects the writer's own output:
    ``json.load`` gives back ``0`` for a metric written as ``0``, and a recall
    of exactly zero or one is the common case, not an edge case."""
    result = _valid_result()
    result["per_query"]["q-1"]["recall_at_1"] = 0
    result["per_query"]["q-1"]["recall_at_10"] = 1
    assert _usable(result) is True


def test_a_nan_metric_is_rejected_because_it_poisons_every_average() -> None:
    """NaN never raises. It silently poisons ``np.mean`` and every comparison
    (including the paired bootstrap) drawn from that average, corrupting the
    leaderboard with no visible error anywhere in the run.
    """
    result = _valid_result()
    result["per_query"]["q-1"]["recall_at_1"] = math.nan
    assert _usable(result) is False


def test_an_infinite_metric_is_rejected() -> None:
    """An infinite metric is finite-looking enough to sort and average without
    raising, but it dominates every mean it enters and silently wins (or
    loses) the leaderboard regardless of the other 999 queries' scores.
    """
    result = _valid_result()
    result["per_query"]["q-1"]["mrr"] = math.inf
    assert _usable(result) is False


def test_per_query_with_wrong_query_ids_is_rejected_because_it_pairs_falsely_in_bootstrap() -> None:
    """The paired bootstrap and every average pair per-query values by query
    id. A result carrying the right shape but scored against different
    queries than the ones actually requested would still pair cleanly and
    report a comparison nobody ran -- the schema alone cannot catch this,
    since it only checks shape, not identity.
    """
    result = _valid_result()
    result["per_query"] = {"q-999": result["per_query"]["q-1"]}
    assert _usable(result) is False


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
    assert _usable(result, {"q-1", "q-2", "q-3"}) is False


def test_a_result_with_an_extra_unknown_key_is_rejected() -> None:
    """``extra="forbid"`` makes drift bidirectional: a field the writer starts
    producing without the model being updated to match must fail loudly here,
    not be silently dropped and hide that the writer and the model disagree
    on what a result contains.
    """
    result = _valid_result()
    result["unexpected_field"] = "surprise"
    assert _usable(result) is False


def test_a_result_missing_a_required_key_is_rejected() -> None:
    """A required field dropped by a truncated write (or an older build of
    the writer) must not be treated as a complete, publishable measurement --
    every downstream reader of a ``CachedResult`` assumes every field is
    present.
    """
    result = _valid_result()
    del result["chunks"]
    assert _usable(result) is False


@pytest.mark.parametrize("non_dict", [None, [], {}], ids=["none", "empty_list", "empty_dict"])
def test_a_non_dict_result_is_rejected(non_dict: object) -> None:
    """A non-mapping (or a mapping missing every field) must fail inside
    ``usable_result``'s own try/except. Letting it raise instead would
    propagate through ``asyncio.gather`` and abort every other model in the
    wave, rather than just dropping this one model from the leaderboard.
    """
    assert _usable(non_dict) is False


def test_embed_minutes_none_is_accepted_because_it_is_legitimately_optional() -> None:
    """``embed_minutes`` is ``None`` whenever a run rescored against a reused
    index without re-measuring embedding time -- a legitimate, common case,
    not a defect. This pins that the model was not accidentally tightened to
    require it, which would reject every rescore-only cache entry.
    """
    result = _valid_result()
    result["embed_minutes"] = None
    assert _usable(result) is True


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
    assert _usable(result) is False


def test_a_metric_above_one_is_rejected() -> None:
    """No metric ``score_ranking`` computes can exceed 1.0, so a cached value
    that does is a corrupt file rather than a good run."""
    result = _valid_result()
    result["per_query"]["q-1"]["recall_at_10"] = 2.0
    assert _usable(result) is False


def test_the_unit_interval_endpoints_are_still_accepted() -> None:
    """The bound must be inclusive at both ends. A perfect ranking scores
    exactly 1.0 and a miss scores exactly 0.0 -- both are the common case, so
    an exclusive bound would reject good runs and silently force a rescore."""
    perfect = _valid_result()
    perfect["per_query"]["q-1"] = dict.fromkeys(bakeoff.METRICS, 1.0)
    assert _usable(perfect) is True

    missed = _valid_result()
    missed["per_query"]["q-1"] = dict.fromkeys(bakeoff.METRICS, 0.0)
    assert _usable(missed) is True


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
    ):
        result = _valid_result()
        result[field] = value
        assert _usable(result) is False, f"{field}={value!r} was accepted"


def test_a_zero_embed_minutes_is_still_accepted() -> None:
    """``ge``, not ``gt``: SCORING_VERSION 2's note records that entries
    written before the build-timing fix legitimately carry ``0.0``."""
    result = _valid_result()
    result["embed_minutes"] = 0.0
    assert _usable(result) is True


def test_a_p95_below_p50_is_rejected_but_an_equal_pair_is_not() -> None:
    """Both percentiles are read out of one sorted latency array at monotone
    indices and rounded to the same precision, so p95 below p50 cannot happen
    in a run the writer produced -- it is proof the file was edited. An equal
    pair, by contrast, is ordinary on a short or uniform battery."""
    swapped = _valid_result()
    swapped["query_p50_ms"], swapped["query_p95_ms"] = 9.0, 2.0
    assert _usable(swapped) is False

    equal = _valid_result()
    equal["query_p50_ms"] = equal["query_p95_ms"] = 4.0
    assert _usable(equal) is True


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


def test_a_result_whose_body_names_another_model_is_rejected() -> None:
    """The sentinel binds the model, but it binds it in ``payload["experiment"]``,
    written *beside* the result and not derived from it -- so a body disagreeing
    with its own sentinel passed every check there was.

    The consequence is not a crash. ``main`` finds the bootstrap baseline with
    ``r["model"] == args.baseline``, so a wrong name there makes the baseline
    look absent and silently skips every paired comparison in the run.
    """
    assert _usable(_valid_result(), model="bge-m3") is False


def test_a_result_whose_body_disagrees_on_dimensions_is_rejected() -> None:
    """Same cross-field gap one field over. A body claiming 1024 dimensions
    for a 768-dimension run is published under the wrong metadata, and the
    dimension is exactly what the leaderboard reports as the model's cost."""
    assert _usable(_valid_result(), dimensions=1024) is False


def test_an_empty_model_name_is_rejected() -> None:
    """Belt and braces with the identity check above: a body carrying no model
    name at all cannot match any requested model, and would be unattributable
    in the leaderboard even if it did."""
    result = _valid_result()
    result["model"] = ""
    assert _usable(result, model="") is False


def test_write_cache_file_round_trips_a_payload() -> None:
    """Guards the guard: a writer that always reported failure would pass every
    negative test below for the wrong reason."""
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "cache.json"
        assert (
            bakeoff.write_cache_file(target, {"experiment": 1, "result": _valid_result()}) is True
        )
        assert json.loads(target.read_text(encoding="utf-8"))["result"]["dimensions"] == 768


def test_a_blocked_cache_write_reports_failure_instead_of_raising() -> None:
    """This ran outside ``one_model``'s handler and *after* scoring succeeded,
    so a stale ``.tmp`` directory or a locked target did not merely lose this
    model's cache -- it propagated through ``asyncio.gather`` and cancelled the
    other models' index builds in the same wave, discarding tens of minutes of
    embedding work over a failed write of a file whose only purpose is to save
    time on the *next* run.
    """
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "cache.json"
        (Path(directory) / "cache.tmp").mkdir()  # a directory where the temp file goes

        assert bakeoff.write_cache_file(target, {"experiment": 1, "result": {}}) is False
        assert not target.exists()


def test_an_unserialisable_payload_reports_failure_and_leaves_no_debris() -> None:
    """``json.dumps`` raises ``TypeError``, not ``OSError``, so an ``OSError``
    clause alone would still have let this through. The temp file must not
    survive either, or the next attempt fails the same way for a new reason."""
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "cache.json"

        assert bakeoff.write_cache_file(target, {"result": {1, 2}}) is False
        assert not (Path(directory) / "cache.tmp").exists()
        assert not target.exists()


def test_a_missing_parent_directory_reports_failure_instead_of_raising() -> None:
    """The remaining ordinary I/O failure: the cache directory was never
    created, or was removed under a long run."""
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "absent" / "cache.json"

        assert bakeoff.write_cache_file(target, {"experiment": 1, "result": {}}) is False


def _write_judgments(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _judgment(query_id: str, doc: str) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "query": f"query for {query_id}",
        "category": "normal",
        "gold": [{"doc": doc, "quote": "a quote"}],
    }


def test_a_repeated_query_id_is_refused_before_any_scoring() -> None:
    """The two consumers of a raw-loaded judgments file disagreed about what a
    repeated id meant. The gold map is a dict comprehension, so it kept only
    the last row; the scoring loop iterated *every* row and wrote each result
    under the same ``per_query`` key. The earlier query was scored against the
    later row's relevance set and then overwritten, so the published aggregate
    covered fewer queries than the ``n=len(judgments)`` printed beside it --
    wrong in a way that reads as correct.
    """
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "judgments.jsonl"
        _write_judgments(path, [_judgment("q-1", "a.txt"), _judgment("q-1", "b.txt")])

        with pytest.raises(EvalError, match="duplicate query_id"):
            bakeoff.load_bakeoff_judgments(path)


def test_a_well_formed_judgments_file_pairs_every_id_with_its_gold() -> None:
    """Guards the guard, and pins the pairing the scorer depends on: every
    judgment must have an entry in the gold map, or ``score_ranking`` raises
    ``KeyError`` mid-run."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "judgments.jsonl"
        _write_judgments(path, [_judgment("q-1", "a.txt"), _judgment("q-2", "b.txt")])

        judgments, gold = bakeoff.load_bakeoff_judgments(path)

        assert [j.query_id for j in judgments] == ["q-1", "q-2"]
        assert set(gold) == {j.query_id for j in judgments}
        assert gold["q-2"] == {"b.txt"}


def _sentinel_dir(root: Path, name: str, build: Any) -> Path:
    index_dir = root / name
    index_dir.mkdir()
    (index_dir / bakeoff.SENTINEL_NAME).write_text(
        json.dumps({"inputs": {}, "build": build}), encoding="utf-8"
    )
    return index_dir


def test_a_usable_sentinel_timing_is_returned() -> None:
    """Guards the guard: a reader that always reported "not measured" would
    pass every negative case below while silently erasing the ingest cost that
    is one of this bake-off's headline comparisons."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        assert bakeoff._sentinel_embed_seconds(_sentinel_dir(root, "a", {"embed_seconds": 120.0}))
        assert (
            bakeoff._sentinel_embed_seconds(_sentinel_dir(root, "b", {"embed_seconds": 120})) == 120
        )
        # 0.0 is a real measurement and must survive, not collapse to None.
        assert (
            bakeoff._sentinel_embed_seconds(_sentinel_dir(root, "c", {"embed_seconds": 0.0})) == 0.0
        )


def test_unusable_sentinel_timings_degrade_to_not_measured() -> None:
    """``_sentinel_valid`` compares ``inputs`` and says nothing about
    ``build``, so these reached the published result unchecked.

    A negative value became a negative embedding time in the leaderboard --
    impossible, and a value the cached-result schema would have caught had it
    arrived through the cache instead of straight out of this run. A string
    was worse: it raised at ``embed_seconds / 60``, inside ``one_model``'s
    handler but *after* every query had been rescored, dropping the model
    having already paid the full cost.
    """
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name, build in (
            ("negative", {"embed_seconds": -60}),
            ("string", {"embed_seconds": "x"}),
            ("numeric_string", {"embed_seconds": "120"}),
            ("nan", {"embed_seconds": math.nan}),
            ("infinite", {"embed_seconds": math.inf}),
        ):
            index_dir = _sentinel_dir(root, name, build)
            assert bakeoff._sentinel_embed_seconds(index_dir) is None, f"{name} was accepted"


def test_a_sentinel_predating_timings_reads_as_not_measured_rather_than_zero() -> None:
    """The distinction the result's ``null``-vs-``0.0`` split exists to keep:
    an index built before this file recorded timings was not measured as free,
    it was not measured at all."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        assert (
            bakeoff._sentinel_embed_seconds(_sentinel_dir(root, "legacy", {"chunks": 10})) is None
        )
        missing = root / "no-sentinel"
        missing.mkdir()
        assert bakeoff._sentinel_embed_seconds(missing) is None

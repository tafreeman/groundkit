"""Tests for the dense vector-index usage check.

``assert_search_uses_vector_index`` must pass only when the query
``LanceDBVectorStore.search`` actually runs is served by an ANN index, and
must fail loudly on every way that silently stops being true: no index, an
index built for the wrong metric, and a stale index with unindexed rows.

Also pins the companion guarantee: once an index exists, a *filtered* search
still scores every row, so filter-then-truncate cannot come back short.

Async methods are driven with ``asyncio.run()`` inside sync tests, matching
``test_dense.py`` (pytest-asyncio is not configured in this repo).
"""

from __future__ import annotations

import asyncio
import inspect
import random
import re
import subprocess
import sys
from pathlib import Path

import pytest

from groundkit.contracts import Chunk
from groundkit.errors import StorageError
from groundkit.index.dense import (
    LanceDBVectorStore,
    VectorIndexFallbackError,
    assert_search_uses_vector_index,
)

lancedb_index = pytest.importorskip("lancedb.index")

#: Enough rows to train a tiny IVF_PQ index; small enough to stay fast.
_ROWS: int = 512
_DIMS: int = 16


def _chunk(i: int, *, group: str) -> Chunk:
    content = f"chunk {i}"
    return Chunk(
        chunk_id=f"c{i}",
        document_id=f"d{i}",
        chunk_index=0,
        content=content,
        start_offset=0,
        end_offset=len(content),
        metadata={"source": "doc.md", "group": group},
    )


def _vectors(n: int, seed: int) -> list[list[float]]:
    rng = random.Random(seed)  # noqa: S311 - deterministic test vectors, not crypto
    return [[rng.random() for _ in range(_DIMS)] for _ in range(n)]


async def _seeded_store(tmp_path: Path, *, start: int = 0) -> LanceDBVectorStore:
    store = await LanceDBVectorStore.open(tmp_path / "lancedb")
    chunks = [_chunk(i, group="a" if i % 2 else "b") for i in range(start, start + _ROWS)]
    await store.add(chunks, _vectors(_ROWS, seed=start))
    return store


def _build_index(store: LanceDBVectorStore, *, metric: str = "cosine", partitions: int = 2) -> None:
    # Test seam: product code builds no index yet. The `config=` signature
    # arrived after the supported floor (lancedb 0.18), so choose the call
    # form by inspecting the signature rather than catching TypeError, which
    # would silently reroute an unrelated error into the old form. Both build
    # the same IVF_PQ index.
    table = store._table
    if "config" in inspect.signature(table.create_index).parameters:
        table.create_index(
            "vector",
            config=lancedb_index.IvfPq(
                distance_type=metric, num_partitions=partitions, num_sub_vectors=2
            ),
        )
    else:
        table.create_index(
            metric=metric,
            vector_column_name="vector",
            num_partitions=partitions,
            num_sub_vectors=2,
        )


def _query() -> list[float]:
    return _vectors(1, seed=999)[0]


def test_passes_when_search_is_served_by_a_cosine_index(tmp_path: Path) -> None:
    async def run() -> str:
        store = await _seeded_store(tmp_path)
        _build_index(store)
        return await assert_search_uses_vector_index(store, _query())

    plan = asyncio.run(run())
    assert "ANNSubIndex" in plan


def test_fails_loudly_when_no_index_exists(tmp_path: Path) -> None:
    async def run() -> None:
        store = await _seeded_store(tmp_path)
        await assert_search_uses_vector_index(store, _query())

    with pytest.raises(VectorIndexFallbackError, match="brute-force scan") as excinfo:
        asyncio.run(run())
    assert "KNNVectorDistance" in str(excinfo.value)  # the plan travels with the error


def test_unindexed_table_whose_path_contains_the_marker_still_fails(tmp_path: Path) -> None:
    # The plan prints the dataset URI; a collection literally named after the
    # plan operator must not make a brute-force scan look like an ANN search.
    async def run() -> None:
        store = await LanceDBVectorStore.open(tmp_path / "ANNSubIndex")
        await store.add([_chunk(i, group="a") for i in range(_ROWS)], _vectors(_ROWS, seed=1))
        await assert_search_uses_vector_index(store, _query())

    with pytest.raises(VectorIndexFallbackError, match="no index exists"):
        asyncio.run(run())


def test_fails_loudly_when_index_metric_does_not_match_the_query(tmp_path: Path) -> None:
    # An L2 index on a cosine query. LanceDB 0.37 logs a warning and scans;
    # 0.18 uses the L2 index and ranks by the wrong metric, with a plan that
    # looks healthy. Both must fail, which is why the check reads the index
    # metric from its stats rather than trusting the plan.
    async def run() -> None:
        store = await _seeded_store(tmp_path)
        _build_index(store, metric="l2")
        await assert_search_uses_vector_index(store, _query())

    with pytest.raises(VectorIndexFallbackError, match="built for metric 'l2'"):
        asyncio.run(run())


def test_fails_loudly_when_rows_were_added_after_the_index(tmp_path: Path) -> None:
    async def run() -> None:
        store = await _seeded_store(tmp_path)
        _build_index(store)
        await store.add([_chunk(10_000, group="a")], _vectors(1, seed=7))
        await assert_search_uses_vector_index(store, _query())

    with pytest.raises(VectorIndexFallbackError, match="1 row"):
        asyncio.run(run())


def test_unindexed_allowance_is_respected(tmp_path: Path) -> None:
    async def run() -> None:
        store = await _seeded_store(tmp_path)
        _build_index(store)
        await store.add([_chunk(10_000, group="a")], _vectors(1, seed=7))
        await assert_search_uses_vector_index(store, _query(), max_unindexed_rows=1)

    asyncio.run(run())


def test_error_is_a_storage_error_so_existing_handlers_catch_it() -> None:
    assert issubclass(VectorIndexFallbackError, StorageError)


def test_explain_search_requires_a_table(tmp_path: Path) -> None:
    async def run() -> None:
        store = await LanceDBVectorStore.open(tmp_path / "empty")
        await store.explain_search(_query())

    with pytest.raises(StorageError, match="no table"):
        asyncio.run(run())


def test_filtered_search_scores_every_row_even_with_an_index(tmp_path: Path) -> None:
    # Many partitions so an ANN probe would cover only part of the table.
    # Without bypassing the index, limit=count_rows returns a subset and
    # filter-then-truncate silently comes back short.
    async def run() -> tuple[int, str]:
        store = await _seeded_store(tmp_path)
        _build_index(store, partitions=32)
        results = await store.search(_query(), top_k=_ROWS, metadata_filter={"group": "a"})
        plan = await store.explain_search(_query(), top_k=5, metadata_filter={"group": "a"})
        return len(results), plan

    returned, plan = asyncio.run(run())
    assert returned == _ROWS // 2
    assert "ANNSubIndex" not in plan


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_dense_index.py"


def _run_script(*args: str) -> int:
    completed = subprocess.run(  # noqa: S603 - fixed interpreter and repo script, test args only
        [sys.executable, str(_SCRIPT), *args], capture_output=True, text=True, check=False
    )
    return completed.returncode


@pytest.mark.parametrize("bad_name", ["../escape", "has space", "a/b"])
def test_script_exits_2_not_1_for_an_invalid_collection_name(bad_name: str, tmp_path: Path) -> None:
    # Exit 1 is reserved for a confirmed fallback; an input error must not look like one.
    assert _run_script("--index-dir", str(tmp_path), "--collection", bad_name) == 2


def test_script_exits_2_when_the_collection_has_no_dense_store(tmp_path: Path) -> None:
    assert _run_script("--index-dir", str(tmp_path), "--collection", "absent") == 2


def test_script_exits_1_on_fallback_and_0_once_indexed(tmp_path: Path) -> None:
    async def seed() -> LanceDBVectorStore:
        store = await LanceDBVectorStore.open(tmp_path / "demo.lance")
        await store.add([_chunk(i, group="a") for i in range(_ROWS)], _vectors(_ROWS, seed=3))
        return store

    store = asyncio.run(seed())
    assert _run_script("--index-dir", str(tmp_path), "--collection", "demo") == 1
    _build_index(store)
    assert _run_script("--index-dir", str(tmp_path), "--collection", "demo") == 0


def _seed_unindexed_demo(tmp_path: Path) -> None:
    async def seed() -> None:
        store = await LanceDBVectorStore.open(tmp_path / "demo.lance")
        await store.add([_chunk(i, group="a") for i in range(8)], _vectors(8, seed=5))

    asyncio.run(seed())


@pytest.mark.parametrize(
    "bad_args",
    [
        ("--top-k", "0"),
        ("--top-k", "-3"),
        ("--max-unindexed-rows", "-1"),
        ("--top-k", "five"),
    ],
)
def test_script_exits_2_not_1_for_out_of_range_numbers(
    bad_args: tuple[str, str], tmp_path: Path
) -> None:
    # Seeded and unindexed, so a wrongly accepted value would reach the plan
    # check and report a fallback (exit 1) that the real search never runs.
    _seed_unindexed_demo(tmp_path)
    assert _run_script("--index-dir", str(tmp_path), "--collection", "demo", *bad_args) == 2


@pytest.mark.parametrize("top_k", [0, -1])
def test_explain_search_refuses_a_search_that_runs_no_query(top_k: int, tmp_path: Path) -> None:
    async def run() -> None:
        store = await _seeded_store(tmp_path)
        await store.explain_search(_query(), top_k=top_k)

    with pytest.raises(StorageError, match="top_k must be positive"):
        asyncio.run(run())


def test_assert_rejects_a_negative_unindexed_allowance(tmp_path: Path) -> None:
    # Must be an input error, not a VectorIndexFallbackError: with -1, even a
    # fully indexed table (0 unindexed rows) would be reported as stale.
    async def run() -> None:
        store = await _seeded_store(tmp_path)
        _build_index(store)
        await assert_search_uses_vector_index(store, _query(), max_unindexed_rows=-1)

    with pytest.raises(StorageError, match="max_unindexed_rows") as excinfo:
        asyncio.run(run())
    assert not isinstance(excinfo.value, VectorIndexFallbackError)


# --- several indexes on the vector column -----------------------------------
#
# LanceDB serves a query from the first-*created* index and lists them
# alphabetically, so the two can differ. The check must judge the index the
# plan names, not the first one listed (issue #53).

_PLAN_INDEX_RE = re.compile(r"^\s*ANNSubIndex:\s*name=([^,\s]+)", re.MULTILINE)


def _add_named_index(store: LanceDBVectorStore, name: str, *, metric: str = "cosine") -> None:
    # Several named indexes on one column need `config=`, `name=` and
    # `replace=`, none of which the supported floor (lancedb 0.18) has. Unlike
    # `_build_index` there is no legacy equivalent to fall back to -- the old
    # form builds one unnamed index and replaces it -- so skip instead of
    # failing in setup. Chosen by signature, for the reason `_build_index` gives.
    params = inspect.signature(store._table.create_index).parameters
    if not {"config", "name", "replace"} <= params.keys():
        pytest.skip("installed lancedb cannot create several named indexes on one column")
    config = lancedb_index.IvfPq(distance_type=metric, num_partitions=2, num_sub_vectors=2)
    store._table.create_index("vector", config=config, name=name, replace=False)


async def _planned_vs_listed(store: LanceDBVectorStore) -> tuple[list[str], list[str]]:
    """``(names the plan reads, names list_indices() reports)``."""
    plan = await store.explain_search(_query())
    listed = [i.name for i in store._table.list_indices()]
    return _PLAN_INDEX_RE.findall(plan), listed


def test_stale_plan_selected_index_fails_even_when_first_listed_index_is_fresh(
    tmp_path: Path,
) -> None:
    # 'old' is built first and so serves the query; the row added afterwards
    # leaves it one row stale. 'new' is built after the row, is fresh, and
    # sorts first in list_indices(), which is all the old check looked at.
    async def run() -> None:
        store = await _seeded_store(tmp_path)
        _add_named_index(store, "old")
        await store.add([_chunk(10_000, group="a")], _vectors(1, seed=7))
        _add_named_index(store, "new")
        # Preconditions: without them this test could pass for the wrong reason.
        planned, listed = await _planned_vs_listed(store)
        assert planned == ["old"]
        assert listed[0] == "new"
        assert await store.vector_index_stats_by_name() == {
            "new": ("cosine", 0),
            "old": ("cosine", 1),
        }
        await assert_search_uses_vector_index(store, _query())

    with pytest.raises(VectorIndexFallbackError, match=r"'old' is stale: 1 row"):
        asyncio.run(run())


def test_fresh_plan_selected_index_passes_despite_a_wrong_metric_index_listed_first(
    tmp_path: Path,
) -> None:
    # 'zz' (cosine, fresh) is built first and serves the query; 'aa' (l2) is
    # built later, sorts first, and would have condemned a healthy setup when
    # the check read the first-listed index.
    async def run() -> str:
        store = await _seeded_store(tmp_path)
        _add_named_index(store, "zz")
        _add_named_index(store, "aa", metric="l2")
        planned, listed = await _planned_vs_listed(store)
        assert planned == ["zz"]
        assert listed[0] == "aa"
        assert (await store.vector_index_stats_by_name())["aa"][0] == "l2"
        return await assert_search_uses_vector_index(store, _query())

    assert "ANNSubIndex: name=zz" in asyncio.run(run())


def test_no_ann_operator_with_several_indexes_names_all_of_them(tmp_path: Path) -> None:
    # An l2 index built first means the cosine query is not served by any
    # index; the message must not attribute one index's metric silently.
    async def run() -> None:
        store = await _seeded_store(tmp_path)
        _add_named_index(store, "aa", metric="l2")
        _add_named_index(store, "zz")
        planned, _ = await _planned_vs_listed(store)
        assert planned == []
        await assert_search_uses_vector_index(store, _query())

    with pytest.raises(VectorIndexFallbackError) as excinfo:
        asyncio.run(run())
    message = str(excinfo.value)
    assert "'aa': metric 'l2'" in message
    assert "'zz': metric 'cosine'" in message


@pytest.mark.parametrize(
    "plan_lines",
    [
        pytest.param("ANNSubIndex: k=5, metric=Cosine", id="no-parseable-name"),
        pytest.param(
            "ANNSubIndex: name=old, k=5\n  ANNSubIndex: name=new, k=5", id="two-distinct-indexes"
        ),
        pytest.param("ANNSubIndex: name=ghost, k=5", id="index-not-in-list-indices"),
    ],
)
def test_ambiguous_plan_is_refused_rather_than_guessed(plan_lines: str, tmp_path: Path) -> None:
    async def run() -> None:
        store = await _seeded_store(tmp_path)
        _add_named_index(store, "old")
        _add_named_index(store, "new")

        async def fake_plan(*_: object, **__: object) -> str:
            return "ProjectionExec\n  " + plan_lines

        store.explain_search = fake_plan  # type: ignore[method-assign]
        await assert_search_uses_vector_index(store, _query())

    with pytest.raises(VectorIndexFallbackError, match="Cannot tell which vector index"):
        asyncio.run(run())


def test_vector_index_stats_by_name_selects_a_named_index(tmp_path: Path) -> None:
    async def run() -> tuple[object, object, object]:
        store = await _seeded_store(tmp_path)
        _add_named_index(store, "old")
        await store.add([_chunk(10_000, group="a")], _vectors(1, seed=7))
        _add_named_index(store, "new")
        return (
            await store.vector_index_stats("old"),
            await store.vector_index_stats("missing"),
            await store.vector_index_stats(),
        )

    named, missing, first_listed = asyncio.run(run())
    assert named == ("cosine", 1)
    assert missing is None
    assert first_listed == ("cosine", 0)  # 'new' is listed first: the legacy behaviour


def test_unnamed_ann_plan_over_a_single_index_is_judged_from_that_index(tmp_path: Path) -> None:
    # A plan with no `name=` (older LanceDB) is ambiguous only when several
    # indexes exist; over a sole index it must be judged, not refused -- here
    # the sole index is stale, so the staleness verdict proves it was judged.
    async def run() -> None:
        store = await _seeded_store(tmp_path)
        _add_named_index(store, "only")
        await store.add([_chunk(10_000, group="a")], _vectors(1, seed=7))

        async def fake_plan(*_: object, **__: object) -> str:
            return "ProjectionExec\n  ANNSubIndex: k=5, metric=Cosine"

        store.explain_search = fake_plan  # type: ignore[method-assign]
        await assert_search_uses_vector_index(store, _query())

    with pytest.raises(VectorIndexFallbackError, match="'only' is stale: 1 row"):
        asyncio.run(run())

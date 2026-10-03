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
import random
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
    store._table.create_index(  # test seam: product code builds no index yet
        "vector",
        config=lancedb_index.IvfPq(
            distance_type=metric, num_partitions=partitions, num_sub_vectors=2
        ),
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


def test_fails_loudly_when_index_metric_does_not_match_the_query(tmp_path: Path) -> None:
    # An L2 index on a cosine query: LanceDB only logs a warning and scans.
    async def run() -> None:
        store = await _seeded_store(tmp_path)
        _build_index(store, metric="l2")
        await assert_search_uses_vector_index(store, _query())

    with pytest.raises(VectorIndexFallbackError, match="brute-force scan"):
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

"""Tests for the dense vector stores (Phase 3, ADR-0001, ADR-0004).

Async methods are driven with ``asyncio.run()`` inside sync test functions
(pytest-asyncio is not configured in this repo), matching
``test_metadata_store.py``.

Nearly every behavioral test is parametrized across both store
implementations via the ``store_kind`` fixture parameter, so hazard 3
(metadata filtering) and hazard 5 (delete-predicate safety) are held to
identical standards on both the in-memory and LanceDB paths, per SPEC.md
§5.3 ("Metadata filtering is implemented on both in-memory and LanceDB
paths from the first dense-store commit, with regression tests on both
paths").
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from groundkit.contracts import Chunk
from groundkit.errors import StorageError
from groundkit.index.dense import InMemoryVectorStore, LanceDBVectorStore
from groundkit.index.protocols import VectorStoreProtocol
from test_protocol_conformance import assert_signature_parity

#: Both store kinds under test; nearly every test is parametrized over this.
_STORE_KINDS: list[str] = ["memory", "lancedb"]

#: Dimensionality used by every test's vectors — small, one-hot-friendly.
_DIMS: int = 4

#: A top_k large enough to return every row a test ever seeds, used as a
#: black-box "how many rows survive" probe via search() rather than
#: reaching into a store's private state.
_PROBE_TOP_K: int = 10_000


async def _make_store(store_kind: str, tmp_path: Path) -> VectorStoreProtocol:
    """Construct a fresh store of the given kind.

    Typed as ``VectorStoreProtocol`` (structural, not inheritance) rather
    than ``Any`` so every caller gets ``search()``'s real
    ``list[tuple[Chunk, float]]`` return type back instead of ``Any``.
    """
    if store_kind == "memory":
        return InMemoryVectorStore()
    if store_kind == "lancedb":
        return await LanceDBVectorStore.open(tmp_path / "lancedb")
    raise AssertionError(f"unknown store_kind: {store_kind!r}")  # pragma: no cover


def _make_chunk(
    chunk_id: str,
    document_id: str,
    content: str,
    *,
    chunk_index: int = 0,
    start_offset: int = 0,
    source: str = "doc.md",
    extra_metadata: dict[str, Any] | None = None,
) -> Chunk:
    """Build a Chunk whose metadata always carries "source", matching what
    ingestion/chunking.py actually produces (metadata={"source": ..., **doc.metadata})."""
    metadata: dict[str, Any] = {"source": source, **(extra_metadata or {})}
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        content=content,
        start_offset=start_offset,
        end_offset=start_offset + len(content),
        metadata=metadata,
    )


def _unit_vector(index: int, dimensions: int = _DIMS) -> list[float]:
    """A one-hot vector: cosine similarity 1.0 to another one-hot vector at
    the same index, 0.0 (orthogonal) to any other index."""
    vector = [0.0] * dimensions
    vector[index] = 1.0
    return vector


async def _row_count(store: VectorStoreProtocol, dimensions: int = _DIMS) -> int:
    """Black-box row count via search(), rather than reaching into private state.

    A dense search never drops zero-scored results (unlike BM25, which
    excludes score == 0), so an unfiltered search with a large top_k
    returns every stored row.
    """
    results = await store.search(_unit_vector(0, dimensions), top_k=_PROBE_TOP_K)
    return len(results)


# ── Protocol conformance ───────────────────────────────────────────────────


class TestVectorStoreProtocolConformance:
    def test_in_memory_vector_store_matches_protocol(self) -> None:
        assert_signature_parity(VectorStoreProtocol, InMemoryVectorStore)
        assert isinstance(InMemoryVectorStore(), VectorStoreProtocol)

    def test_lancedb_vector_store_matches_protocol(self) -> None:
        assert_signature_parity(VectorStoreProtocol, LanceDBVectorStore)


# ── Lazy/guarded lancedb import ─────────────────────────────────────────────


def test_dense_module_imports_without_lancedb() -> None:
    """``import groundkit.index.dense`` must not require lancedb — verified
    by blocking the import in a fresh subprocess (patching sys.modules in
    this process wouldn't prove anything about the module's own top-level
    import statements, since it's already imported by the time this test
    runs)."""
    script = (
        "import sys; sys.modules['lancedb'] = None; sys.modules['pyarrow'] = None; "
        "import groundkit.index.dense"
    )
    result = subprocess.run(  # noqa: S603 - sys.executable + a fixed literal script, no untrusted input
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr


def test_lancedb_open_raises_storage_error_when_lancedb_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Constructing a LanceDBVectorStore without lancedb installed raises a
    clear StorageError, not a bare ImportError — simulated by hiding the
    module in sys.modules rather than actually uninstalling it."""
    monkeypatch.setitem(sys.modules, "lancedb", None)

    async def _run() -> None:
        await LanceDBVectorStore.open(tmp_path / "col")

    with pytest.raises(StorageError, match="dense"):
        asyncio.run(_run())


# ── add -> search round-trip, correct ordering ──────────────────────────────


@pytest.mark.parametrize("store_kind", _STORE_KINDS)
def test_add_then_search_orders_by_descending_similarity(store_kind: str, tmp_path: Path) -> None:
    async def _run() -> list[tuple[Chunk, float]]:
        store = await _make_store(store_kind, tmp_path)
        chunks = [
            _make_chunk("a", "doc-1", "alpha content"),
            _make_chunk("b", "doc-1", "beta content"),
            _make_chunk("c", "doc-2", "gamma content"),
        ]
        embeddings = [
            [1.0, 0.0, 0.0, 0.0],  # similarity to query = 1.0
            [0.8, 0.6, 0.0, 0.0],  # similarity to query = 0.8 (unit vector)
            [0.0, 1.0, 0.0, 0.0],  # similarity to query = 0.0 (orthogonal)
        ]
        await store.add(chunks, embeddings)
        return await store.search([1.0, 0.0, 0.0, 0.0], top_k=3)

    results = asyncio.run(_run())
    assert [c.chunk_id for c, _ in results] == ["a", "b", "c"]
    assert results[0][1] == pytest.approx(1.0, abs=1e-4)
    assert results[1][1] == pytest.approx(0.8, abs=1e-4)
    assert results[2][1] == pytest.approx(0.0, abs=1e-4)
    # The store returns chunks, never citations: resolving a chunk to its
    # document's source is Retriever's job, against the metadata store that
    # owns that mapping. A store that built citations here would be trusting
    # chunk.metadata's ingest-time snapshot over the durable truth.
    for chunk, _ in results:
        assert chunk.document_id in ("doc-1", "doc-2")


# ── Metadata filtering actually filters (hazard 3) ─────────────────────────


@pytest.mark.parametrize("store_kind", _STORE_KINDS)
def test_metadata_filter_removes_non_matching_chunks(store_kind: str, tmp_path: Path) -> None:
    async def _run() -> list[tuple[Chunk, float]]:
        store = await _make_store(store_kind, tmp_path)
        chunks = [
            _make_chunk("keep-1", "doc-1", "keep this one", extra_metadata={"category": "keep"}),
            _make_chunk("drop-1", "doc-1", "drop this one", extra_metadata={"category": "drop"}),
        ]
        embeddings = [[1.0, 0.0, 0.0, 0.0], [0.9, 0.1, 0.0, 0.0]]
        await store.add(chunks, embeddings)
        return await store.search(
            [1.0, 0.0, 0.0, 0.0], top_k=10, metadata_filter={"category": "keep"}
        )

    results = asyncio.run(_run())
    assert [c.chunk_id for c, _ in results] == ["keep-1"]


@pytest.mark.parametrize("store_kind", _STORE_KINDS)
def test_metadata_filter_requires_all_pairs(store_kind: str, tmp_path: Path) -> None:
    """A filter with two keys only matches a chunk carrying BOTH."""

    async def _run() -> list[tuple[Chunk, float]]:
        store = await _make_store(store_kind, tmp_path)
        chunks = [
            _make_chunk(
                "both", "doc-1", "matches both", extra_metadata={"category": "a", "tier": "hot"}
            ),
            _make_chunk(
                "one-only",
                "doc-1",
                "matches one",
                extra_metadata={"category": "a", "tier": "cold"},
            ),
        ]
        embeddings = [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
        await store.add(chunks, embeddings)
        return await store.search(
            [1.0, 0.0, 0.0, 0.0],
            top_k=10,
            metadata_filter={"category": "a", "tier": "hot"},
        )

    results = asyncio.run(_run())
    assert [c.chunk_id for c, _ in results] == ["both"]


# ── Filtered search still returns top_k when enough matches exist ─────────


@pytest.mark.parametrize("store_kind", _STORE_KINDS)
def test_filtered_search_still_returns_top_k_when_enough_matches_exist(
    store_kind: str, tmp_path: Path
) -> None:
    """The filter-then-truncate trap: the two best-scoring chunks are
    filtered OUT. A naive "vector-search top_k then filter" implementation
    would return fewer than top_k "keep" results here (or even zero);
    filtering before truncation must not."""

    async def _run() -> list[tuple[Chunk, float]]:
        store = await _make_store(store_kind, tmp_path)
        chunks = [
            _make_chunk("drop-1", "doc-1", "dropped a", extra_metadata={"category": "drop"}),
            _make_chunk("drop-2", "doc-1", "dropped b", extra_metadata={"category": "drop"}),
            _make_chunk(
                "keep-1", "doc-2", "keep a", extra_metadata={"category": "keep"}, chunk_index=0
            ),
            _make_chunk(
                "keep-2", "doc-2", "keep b", extra_metadata={"category": "keep"}, chunk_index=1
            ),
            _make_chunk(
                "keep-3", "doc-2", "keep c", extra_metadata={"category": "keep"}, chunk_index=2
            ),
            _make_chunk(
                "keep-4", "doc-2", "keep d", extra_metadata={"category": "keep"}, chunk_index=3
            ),
            _make_chunk(
                "keep-5", "doc-2", "keep e", extra_metadata={"category": "keep"}, chunk_index=4
            ),
        ]
        embeddings = [
            [1.0, 0.0, 0.0, 0.0],  # drop-1, similarity 1.0 (best match, filtered out)
            [0.99, 0.01, 0.0, 0.0],  # drop-2, similarity ~0.9999 (filtered out)
            [0.9, 0.1, 0.0, 0.0],
            [0.8, 0.2, 0.0, 0.0],
            [0.7, 0.3, 0.0, 0.0],
            [0.6, 0.4, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
        ]
        await store.add(chunks, embeddings)
        return await store.search(
            [1.0, 0.0, 0.0, 0.0], top_k=3, metadata_filter={"category": "keep"}
        )

    results = asyncio.run(_run())
    assert len(results) == 3
    assert [c.chunk_id for c, _ in results] == ["keep-1", "keep-2", "keep-3"]
    assert all(c.metadata["category"] == "keep" for c, _ in results)


# ── delete: correct count, correct scope ────────────────────────────────────


@pytest.mark.parametrize("store_kind", _STORE_KINDS)
def test_delete_returns_count_and_removes_only_target_document(
    store_kind: str, tmp_path: Path
) -> None:
    async def _run() -> tuple[int, list[tuple[Chunk, float]]]:
        store = await _make_store(store_kind, tmp_path)
        chunks = [
            _make_chunk("d1-a", "doc-1", "a"),
            _make_chunk("d1-b", "doc-1", "b"),
            _make_chunk("d1-c", "doc-1", "c"),
            _make_chunk("d2-a", "doc-2", "d"),
            _make_chunk("d2-b", "doc-2", "e"),
        ]
        embeddings = [_unit_vector(0), _unit_vector(1), _unit_vector(2), _unit_vector(3), [0.5] * 4]
        await store.add(chunks, embeddings)
        removed = await store.delete("doc-1")
        remaining = await store.search(_unit_vector(0), top_k=_PROBE_TOP_K)
        return removed, remaining

    removed, remaining = asyncio.run(_run())
    assert removed == 3
    assert {c.chunk_id for c, _ in remaining} == {"d2-a", "d2-b"}
    assert all(c.document_id == "doc-2" for c, _ in remaining)


@pytest.mark.parametrize("store_kind", _STORE_KINDS)
def test_delete_of_unknown_document_removes_nothing(store_kind: str, tmp_path: Path) -> None:
    async def _run() -> tuple[int, int]:
        store = await _make_store(store_kind, tmp_path)
        await store.add([_make_chunk("a", "doc-1", "x")], [_unit_vector(0)])
        removed = await store.delete("doc-does-not-exist")
        return removed, await _row_count(store)

    removed, remaining = asyncio.run(_run())
    assert removed == 0
    assert remaining == 1


# ── The hostile-ID regression test (ADR-0004 decision 6) ──────────────────


@pytest.mark.parametrize("store_kind", _STORE_KINDS)
def test_hostile_document_id_does_not_wipe_the_table(store_kind: str, tmp_path: Path) -> None:
    """Verified live against this repo's pinned lancedb 0.37.1: naively
    interpolating this exact document_id into ``f"document_id = '{id}'"``
    and calling ``table.delete(pred)`` empties the table (3 rows -> 0),
    even though it names no real document. Both stores must refuse it
    before it ever reaches a predicate (ADR-0004 decision 6)."""
    hostile_id = "nope' OR '1'='1"

    async def _run() -> int:
        store = await _make_store(store_kind, tmp_path)
        chunks = [
            _make_chunk("a", "doc-1", "alpha"),
            _make_chunk("b", "doc-1", "beta"),
            _make_chunk("c", "doc-2", "gamma"),
        ]
        embeddings = [_unit_vector(0), _unit_vector(1), _unit_vector(2)]
        await store.add(chunks, embeddings)

        with pytest.raises(StorageError):
            await store.delete(hostile_id)

        return await _row_count(store)

    surviving = asyncio.run(_run())
    assert surviving == 3


# ── Length/dimension mismatch on add() ──────────────────────────────────────


@pytest.mark.parametrize("store_kind", _STORE_KINDS)
def test_add_length_mismatch_raises_storage_error(store_kind: str, tmp_path: Path) -> None:
    async def _run() -> None:
        store = await _make_store(store_kind, tmp_path)
        chunks = [_make_chunk("a", "doc-1", "x"), _make_chunk("b", "doc-1", "y")]
        await store.add(chunks, [_unit_vector(0)])

    with pytest.raises(StorageError):
        asyncio.run(_run())


@pytest.mark.parametrize("store_kind", _STORE_KINDS)
def test_add_inconsistent_width_within_call_raises_storage_error(
    store_kind: str, tmp_path: Path
) -> None:
    async def _run() -> None:
        store = await _make_store(store_kind, tmp_path)
        chunks = [_make_chunk("a", "doc-1", "x"), _make_chunk("b", "doc-1", "y")]
        await store.add(chunks, [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    with pytest.raises(StorageError):
        asyncio.run(_run())


@pytest.mark.parametrize("store_kind", _STORE_KINDS)
def test_add_width_disagreeing_with_established_width_raises(
    store_kind: str, tmp_path: Path
) -> None:
    async def _run() -> None:
        store = await _make_store(store_kind, tmp_path)
        await store.add([_make_chunk("a", "doc-1", "x")], [_unit_vector(0)])
        await store.add([_make_chunk("b", "doc-1", "y")], [[1.0, 0.0, 0.0]])

    with pytest.raises(StorageError):
        asyncio.run(_run())


@pytest.mark.parametrize("store_kind", _STORE_KINDS)
def test_add_empty_lists_is_a_no_op(store_kind: str, tmp_path: Path) -> None:
    async def _run() -> int:
        store = await _make_store(store_kind, tmp_path)
        await store.add([], [])
        return await _row_count(store)

    # An empty store's search short-circuits to [] before ever needing a
    # dimension to probe with, for both implementations.
    assert asyncio.run(_run()) == 0


# ── Scores are always >= 0.0 ────────────────────────────────────────────────


@pytest.mark.parametrize("store_kind", _STORE_KINDS)
def test_scores_are_always_non_negative(store_kind: str, tmp_path: Path) -> None:
    async def _run() -> list[tuple[Chunk, float]]:
        store = await _make_store(store_kind, tmp_path)
        chunks = [_make_chunk("opposite", "doc-1", "anti-aligned")]
        # Exactly opposite the query vector: cosine similarity == -1.0.
        await store.add(chunks, [[-1.0, 0.0, 0.0, 0.0]])
        return await store.search([1.0, 0.0, 0.0, 0.0], top_k=5)

    results = asyncio.run(_run())
    assert len(results) == 1
    assert results[0][1] == 0.0
    # The producer clamps, so a negative similarity never leaves the store.
    # Retriever builds RetrievalResult (Field(ge=0.0)) downstream from these
    # scores, and would raise if one slipped through — but that is a backstop,
    # not the assertion; this pins the clamp at its source.
    assert all(s >= 0.0 for _, s in results)


# ── Determinism: repeated identical searches, tie-break by content hash ────


@pytest.mark.parametrize("store_kind", _STORE_KINDS)
def test_repeated_search_returns_identical_ordering(store_kind: str, tmp_path: Path) -> None:
    async def _run() -> tuple[list[str], list[str]]:
        store = await _make_store(store_kind, tmp_path)
        # Every chunk gets the identical vector, so every score ties exactly
        # -- only the content-hash tie-break can determine ordering.
        chunks = [
            _make_chunk("z-chunk", "doc-1", "zzz content", chunk_index=0),
            _make_chunk("a-chunk", "doc-1", "aaa content", chunk_index=1),
            _make_chunk("m-chunk", "doc-1", "mmm content", chunk_index=2),
        ]
        embeddings = [[1.0, 0.0, 0.0, 0.0]] * 3
        await store.add(chunks, embeddings)
        first = await store.search([1.0, 0.0, 0.0, 0.0], top_k=3)
        second = await store.search([1.0, 0.0, 0.0, 0.0], top_k=3)
        return [c.chunk_id for c, _ in first], [c.chunk_id for c, _ in second]

    first_order, second_order = asyncio.run(_run())
    assert first_order == second_order
    # The tie-break key is ascending Chunk.content_hash (index/bm25.py's
    # convention) -- independently recompute it here and assert the order
    # matches, not just that two runs agree with each other.
    content_by_id = {"z-chunk": "zzz content", "a-chunk": "aaa content", "m-chunk": "mmm content"}

    def _content_hash(chunk_id: str) -> str:
        content = content_by_id[chunk_id]
        chunk = Chunk(
            chunk_id=chunk_id,
            document_id="doc-1",
            chunk_index=0,
            content=content,
            start_offset=0,
            end_offset=len(content),
        )
        return chunk.content_hash

    expected = sorted(content_by_id, key=_content_hash)
    assert first_order == expected


# ── Additional edge cases (fail-closed paths, guard branches) ──────────────


def test_zero_vector_embedding_scores_zero_in_memory() -> None:
    """InMemoryVectorStore: an all-zero embedding has no direction;
    ``_cosine_similarity``'s guard defines its similarity as 0.0 rather
    than raising ``ZeroDivisionError``."""

    async def _run() -> list[tuple[Chunk, float]]:
        store = InMemoryVectorStore()
        await store.add([_make_chunk("zero", "doc-1", "no direction")], [[0.0, 0.0, 0.0, 0.0]])
        return await store.search([1.0, 0.0, 0.0, 0.0], top_k=5)

    results = asyncio.run(_run())
    assert len(results) == 1
    assert results[0][1] == 0.0


def test_zero_vector_embedding_lancedb_excludes_it(tmp_path: Path) -> None:
    """LanceDB-specific, documented divergence (verified live against the
    pinned 0.37.1 client): a zero-magnitude stored vector is silently
    excluded from cosine-metric search results entirely, rather than
    reported as a 0.0-similarity match the way InMemoryVectorStore's own
    ``_cosine_similarity`` guard behaves. A genuinely all-zero embedding
    essentially never occurs from a real embedding model, so this is
    documented as a known backend-specific edge case (see dense.py's
    ``_cosine_similarity`` docstring) rather than worked around."""

    async def _run() -> list[tuple[Chunk, float]]:
        store = await LanceDBVectorStore.open(tmp_path / "lancedb")
        await store.add([_make_chunk("zero", "doc-1", "no direction")], [[0.0, 0.0, 0.0, 0.0]])
        return await store.search([1.0, 0.0, 0.0, 0.0], top_k=5)

    assert asyncio.run(_run()) == []


@pytest.mark.parametrize("store_kind", _STORE_KINDS)
def test_search_top_k_zero_returns_empty(store_kind: str, tmp_path: Path) -> None:
    async def _run() -> list[tuple[Chunk, float]]:
        store = await _make_store(store_kind, tmp_path)
        await store.add([_make_chunk("a", "doc-1", "x")], [_unit_vector(0)])
        return await store.search(_unit_vector(0), top_k=0)

    assert asyncio.run(_run()) == []


@pytest.mark.parametrize("store_kind", _STORE_KINDS)
def test_search_query_embedding_width_mismatch_raises(store_kind: str, tmp_path: Path) -> None:
    async def _run() -> None:
        store = await _make_store(store_kind, tmp_path)
        await store.add([_make_chunk("a", "doc-1", "x")], [_unit_vector(0)])
        await store.search([1.0, 0.0, 0.0], top_k=5)  # width 3, established width 4

    with pytest.raises(StorageError):
        asyncio.run(_run())


@pytest.mark.parametrize("document_id", ["", "   ", "bad\0id"])
@pytest.mark.parametrize("store_kind", _STORE_KINDS)
def test_delete_rejects_empty_and_null_byte_document_ids(
    store_kind: str, document_id: str, tmp_path: Path
) -> None:
    """_validate_document_id's empty/whitespace and null-byte branches,
    distinct from the character-class rejection the hostile-ID test covers."""

    async def _run() -> None:
        store = await _make_store(store_kind, tmp_path)
        await store.delete(document_id)

    with pytest.raises(StorageError):
        asyncio.run(_run())


def test_lancedb_search_with_filter_on_emptied_table_returns_empty(tmp_path: Path) -> None:
    """LanceDB-specific: a table that exists but currently holds zero rows
    (everything in it was deleted) with a metadata_filter set must hit the
    fetch_limit <= 0 short-circuit, not call into LanceDB with a 0 limit."""

    async def _run() -> list[tuple[Chunk, float]]:
        store = await LanceDBVectorStore.open(tmp_path / "lancedb")
        await store.add([_make_chunk("a", "doc-1", "x")], [_unit_vector(0)])
        await store.delete("doc-1")
        return await store.search(_unit_vector(0), top_k=5, metadata_filter={"category": "keep"})

    assert asyncio.run(_run()) == []


def test_lancedb_delete_on_uninitialized_store_returns_zero(tmp_path: Path) -> None:
    """LanceDB-specific: delete() on a store that has never had add()
    called (its table is still None) returns 0 rather than erroring."""

    async def _run() -> int:
        store = await LanceDBVectorStore.open(tmp_path / "lancedb")
        return await store.delete("some-doc")

    assert asyncio.run(_run()) == 0


def test_lancedb_open_wraps_os_error_as_storage_error(tmp_path: Path) -> None:
    """LanceDB-specific: a connection failure (verified live: connecting to
    a path blocked by a pre-existing regular file raises OSError) is
    wrapped as a StorageError, never left as a bare OSError."""
    blocked_path = tmp_path / "blocked"
    blocked_path.write_text("not a directory")

    async def _run() -> None:
        await LanceDBVectorStore.open(blocked_path)

    with pytest.raises(StorageError):
        asyncio.run(_run())


def test_lancedb_store_persists_and_reopens_across_sessions(tmp_path: Path) -> None:
    """LanceDB-specific: data written by one LanceDBVectorStore.open() is
    visible to a second .open() over the same path (a fresh process
    reopening the collection), and the reopened store infers its
    dimensions from the existing table's schema (_infer_dimensions)
    instead of requiring a fresh add() first."""
    db_path = tmp_path / "lancedb"

    async def _write() -> None:
        store = await LanceDBVectorStore.open(db_path)
        await store.add(
            [_make_chunk("a", "doc-1", "alpha"), _make_chunk("b", "doc-2", "beta")],
            [_unit_vector(0), _unit_vector(1)],
        )

    async def _reopen_and_search() -> list[tuple[Chunk, float]]:
        store = await LanceDBVectorStore.open(db_path)
        return await store.search(_unit_vector(0), top_k=10)

    asyncio.run(_write())
    results = asyncio.run(_reopen_and_search())
    assert {c.chunk_id for c, _ in results} == {"a", "b"}


@pytest.mark.parametrize("store_kind", _STORE_KINDS)
def test_add_rejects_a_document_id_that_could_never_be_deleted(
    store_kind: str, tmp_path: Path
) -> None:
    """A document ID accepted by ``add`` must be one ``delete`` can act on.

    ``delete`` validates ``document_id`` against ``_DOCUMENT_ID_PATTERN``
    (ADR-0001 hazard 5, keeping it out of LanceDB's SQL-like delete
    predicate), but ``Chunk.document_id`` is a plain ``str``. A custom loader
    supplying ``tenant/doc`` therefore used to ingest cleanly and only fail
    later, on the first replace or prune that tried to delete it — leaving a
    collection whose vectors could be written but never maintained or
    removed through the protocol.

    Validating on the way in makes the accepted set identical at both ends,
    and identical across both store implementations, which share
    ``_validate_add_shapes``.
    """

    async def run() -> None:
        store = await _make_store(store_kind, tmp_path)
        undeletable = _make_chunk("c1", "tenant/doc", "content that will be refused")

        with pytest.raises(StorageError, match="document_id"):
            await store.add([undeletable], [[1.0, 0.0, 0.0, 0.0]])

        # Nothing was written on the way to the error.
        assert await store.search([1.0, 0.0, 0.0, 0.0], top_k=_PROBE_TOP_K) == []

        # And the ID that *is* deletable still works, so the guard is not
        # simply rejecting everything.
        fine = _make_chunk("c2", "tenant.doc-1_v2", "content that is accepted")
        await store.add([fine], [[1.0, 0.0, 0.0, 0.0]])
        assert len(await store.search([1.0, 0.0, 0.0, 0.0], top_k=_PROBE_TOP_K)) == 1

    asyncio.run(run())


@pytest.mark.parametrize("store_kind", _STORE_KINDS)
def test_duplicate_content_ties_break_deterministically_on_chunk_index(
    store_kind: str, tmp_path: Path
) -> None:
    """Duplicate content must not rank backend-dependently.

    ``content_hash`` alone is not a total order. ``index/bm25.py`` can fall
    back to insertion order because its postings are rebuilt from SQLite in a
    fixed order, but this sort is fed by LanceDB row order, which carries no
    such guarantee. ``chunk_index`` completes the order, matching ADR-0005
    decision 3 and ``retrieval/fusion.py``.
    """

    async def run() -> None:
        store = await _make_store(store_kind, tmp_path)
        vector = [1.0, 0.0, 0.0, 0.0]

        # Identical content, so identical content_hash and identical score.
        seeded = [
            _make_chunk("c-late", "doc-b", "duplicated passage", chunk_index=9),
            _make_chunk("c-early", "doc-a", "duplicated passage", chunk_index=1),
        ]
        await store.add(seeded, [vector, vector])

        ranked = await store.search(vector, top_k=_PROBE_TOP_K)

        assert [chunk.content_hash for chunk, _ in ranked].count(ranked[0][0].content_hash) == 2
        assert [chunk.chunk_index for chunk, _ in ranked] == [1, 9]

    asyncio.run(run())


def test_empty_metadata_filter_does_not_trigger_the_over_fetch_path(tmp_path: Path) -> None:
    """``{}`` is a documented no-op filter, so it must not pay the O(N) fetch.

    ``_matches_filter`` treats an empty dict as matching everything. The
    LanceDB path keyed its over-fetch on ``metadata_filter is None``, so
    ``{}`` counted the whole table to apply a filter that cannot remove a
    single row. The two must agree on what counts as a filter.
    """

    async def run() -> None:
        store = await LanceDBVectorStore.open(tmp_path / "lancedb")
        chunks = [
            _make_chunk(f"c{i}", "doc-a", f"passage number {i}", chunk_index=i) for i in range(5)
        ]
        await store.add(chunks, [[1.0, 0.0, 0.0, 0.0] for _ in chunks])

        counted: list[int] = []
        original = store._table.count_rows

        def _counting() -> int:
            counted.append(1)
            return int(original())

        store._table.count_rows = _counting

        assert len(await store.search([1.0, 0.0, 0.0, 0.0], top_k=2, metadata_filter={})) == 2
        assert counted == []

        # A real filter still takes the over-fetch path.
        await store.search([1.0, 0.0, 0.0, 0.0], top_k=2, metadata_filter={"source": "doc.md"})
        assert counted == [1]

    asyncio.run(run())

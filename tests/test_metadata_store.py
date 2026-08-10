"""Tests for SQLiteMetadataStore — the persisted document/chunk truth store.

Async methods are driven with ``asyncio.run()`` inside sync test functions
(pytest-asyncio is not configured in this repo).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from groundkit.contracts import Chunk
from groundkit.errors import StorageError
from groundkit.index.metadata import SQLiteMetadataStore
from groundkit.index.protocols import MetadataStoreProtocol


def _make_chunk(
    chunk_id: str,
    document_id: str,
    content: str,
    *,
    chunk_index: int = 0,
    start_offset: int = 0,
    metadata: dict[str, object] | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        content=content,
        start_offset=start_offset,
        end_offset=start_offset + len(content),
        metadata=metadata or {},
    )


def test_sqlite_metadata_store_conforms_to_protocol(tmp_path: Path) -> None:
    """SQLiteMetadataStore satisfies MetadataStoreProtocol structurally."""

    async def _run() -> SQLiteMetadataStore:
        return await SQLiteMetadataStore.open(tmp_path, "col")

    store = asyncio.run(_run())
    try:
        assert isinstance(store, MetadataStoreProtocol)
    finally:
        asyncio.run(store.close())


def test_round_trip_document_and_chunks(tmp_path: Path) -> None:
    """Chunks persisted with metadata and offsets read back unchanged."""

    async def _run() -> tuple[list[Chunk], str | None]:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            await store.upsert_document(source="a.md", document_id="doc-1", content_hash="h1")
            chunks = [
                _make_chunk(
                    "c1",
                    "doc-1",
                    "hello world",
                    chunk_index=0,
                    start_offset=0,
                    metadata={"page": 1, "tags": ["a", "b"]},
                ),
                _make_chunk("c2", "doc-1", "second chunk", chunk_index=1, start_offset=11),
            ]
            await store.add_chunks(chunks, source="a.md")
            persisted = await store.get_chunks()
            doc_hash = await store.get_document_hash("a.md")
            return persisted, doc_hash
        finally:
            await store.close()

    persisted, doc_hash = asyncio.run(_run())

    assert doc_hash == "h1"
    assert len(persisted) == 2
    by_id = {c.chunk_id: c for c in persisted}
    assert by_id["c1"].content == "hello world"
    assert by_id["c1"].start_offset == 0
    assert by_id["c1"].end_offset == 11
    assert by_id["c1"].metadata == {"page": 1, "tags": ["a", "b"]}
    assert by_id["c2"].content == "second chunk"
    assert by_id["c2"].start_offset == 11
    assert by_id["c2"].metadata == {}


def test_add_chunks_rejects_source_with_no_document(tmp_path: Path) -> None:
    """Chunks cannot be attached to a source that was never upserted."""

    async def _run() -> None:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            chunk = _make_chunk("c1", "doc-1", "orphan")
            await store.add_chunks([chunk], source="never-ingested.md")
        finally:
            await store.close()

    with pytest.raises(StorageError):
        asyncio.run(_run())


def test_upsert_replaces_existing_source_and_its_chunks(tmp_path: Path) -> None:
    """Re-ingesting a source drops its old document_id, hash, and chunks."""

    async def _run() -> tuple[list[Chunk], str | None, Chunk | None]:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            await store.upsert_document(source="a.md", document_id="doc-1", content_hash="h1")
            old_chunk = _make_chunk("old", "doc-1", "old text")
            await store.add_chunks([old_chunk], source="a.md")

            # Re-ingest under a new document_id, as a chunker producing
            # fresh UUIDs on every run would.
            await store.upsert_document(source="a.md", document_id="doc-2", content_hash="h2")
            new_chunk = _make_chunk("new", "doc-2", "new text here")
            await store.add_chunks([new_chunk], source="a.md")

            chunks = await store.get_chunks()
            doc_hash = await store.get_document_hash("a.md")
            old = await store.get_chunk("old")
            return chunks, doc_hash, old
        finally:
            await store.close()

    chunks, doc_hash, old = asyncio.run(_run())

    assert doc_hash == "h2"
    assert [c.chunk_id for c in chunks] == ["new"]
    assert old is None


def test_get_document_hash_unseen_source_returns_none(tmp_path: Path) -> None:
    """An unseen source has no stored hash."""

    async def _run() -> str | None:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            return await store.get_document_hash("missing.md")
        finally:
            await store.close()

    assert asyncio.run(_run()) is None


def test_get_chunk_hit_and_miss(tmp_path: Path) -> None:
    """get_chunk returns the chunk when present and None when absent."""

    async def _run() -> tuple[Chunk | None, Chunk | None]:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            await store.upsert_document(source="a.md", document_id="doc-1", content_hash="h1")
            chunk = _make_chunk("c1", "doc-1", "hi")
            await store.add_chunks([chunk], source="a.md")
            hit = await store.get_chunk("c1")
            miss = await store.get_chunk("nope")
            return hit, miss
        finally:
            await store.close()

    hit, miss = asyncio.run(_run())

    assert hit is not None
    assert hit.content == "hi"
    assert miss is None


def test_delete_document_cascades_and_returns_count(tmp_path: Path) -> None:
    """Deleting a document removes its chunks and reports how many."""

    async def _run() -> tuple[int, list[Chunk], str | None]:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            await store.upsert_document(source="a.md", document_id="doc-1", content_hash="h1")
            chunks = [_make_chunk(f"c{i}", "doc-1", f"chunk {i}", chunk_index=i) for i in range(3)]
            await store.add_chunks(chunks, source="a.md")

            deleted = await store.delete_document("doc-1")
            remaining = await store.get_chunks()
            doc_hash = await store.get_document_hash("a.md")
            return deleted, remaining, doc_hash
        finally:
            await store.close()

    deleted, remaining, doc_hash = asyncio.run(_run())

    assert deleted == 3
    assert remaining == []
    assert doc_hash is None


def test_delete_document_unknown_id_returns_zero(tmp_path: Path) -> None:
    """Deleting a document that never existed is a harmless no-op."""

    async def _run() -> int:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            return await store.delete_document("nonexistent")
        finally:
            await store.close()

    assert asyncio.run(_run()) == 0


def test_persists_across_close_and_reopen(tmp_path: Path) -> None:
    """State survives closing the store and reopening the same file.

    This is the ARP gap this repo exists to close: ARP's CLI held RAG state
    in module-level globals that reset every process (ADR-0001, claim #1).
    """

    async def _write() -> None:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            await store.upsert_document(source="a.md", document_id="doc-1", content_hash="h1")
            chunk = _make_chunk("c1", "doc-1", "durable", metadata={"k": "v"})
            await store.add_chunks([chunk], source="a.md")
        finally:
            await store.close()

    async def _reopen() -> tuple[str | None, list[Chunk]]:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            doc_hash = await store.get_document_hash("a.md")
            chunks = await store.get_chunks()
            return doc_hash, chunks
        finally:
            await store.close()

    asyncio.run(_write())
    doc_hash, chunks = asyncio.run(_reopen())

    assert doc_hash == "h1"
    assert len(chunks) == 1
    assert chunks[0].content == "durable"
    assert chunks[0].metadata == {"k": "v"}


def test_open_raises_storage_error_on_unusable_path(tmp_path: Path) -> None:
    """A path SQLite cannot open as a database surfaces as StorageError."""
    # The store expects to create/open a *file* at <index_dir>/col.sqlite3;
    # putting a directory there makes that impossible regardless of platform.
    (tmp_path / "col.sqlite3").mkdir()

    async def _open() -> SQLiteMetadataStore:
        return await SQLiteMetadataStore.open(tmp_path, "col")

    with pytest.raises(StorageError):
        asyncio.run(_open())


def test_open_raises_storage_error_on_corrupted_file(tmp_path: Path) -> None:
    """A file that exists but isn't a valid SQLite database is a StorageError."""
    db_path = tmp_path / "col.sqlite3"
    db_path.write_bytes(b"not a sqlite database, just garbage bytes")

    async def _open() -> SQLiteMetadataStore:
        return await SQLiteMetadataStore.open(tmp_path, "col")

    with pytest.raises(StorageError):
        asyncio.run(_open())

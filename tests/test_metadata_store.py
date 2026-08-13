"""Tests for SQLiteMetadataStore — the persisted document/chunk truth store.

Async methods are driven with ``asyncio.run()`` inside sync test functions
(pytest-asyncio is not configured in this repo).
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from groundkit.config import EmbeddingConfig
from groundkit.contracts import Chunk, CollectionManifest
from groundkit.errors import ConfigurationError, IndexIdentityError, StorageError
from groundkit.index.metadata import APPLICATION_ID, SCHEMA_VERSION, SQLiteMetadataStore
from groundkit.index.protocols import MetadataStoreProtocol
from test_protocol_conformance import assert_signature_parity


def _write_legacy_schema(tmp_path: Path, collection: str) -> None:
    """Create a pre-ADR-0004 store file: documents/chunks tables, no manifest stamp.

    Mimics a collection created before the embedding-identity manifest
    existed — same documents/chunks schema :class:`SQLiteMetadataStore`
    would create, but with no ``collection_manifest`` table and, crucially,
    none of ``PRAGMA application_id``/``user_version`` set (both stay at
    SQLite's default of 0). Built with a raw ``sqlite3`` connection, never
    through :class:`SQLiteMetadataStore`, so the file exists *before*
    ``SQLiteMetadataStore.open`` ever sees it — matching how a real
    pre-existing store is only ever encountered on reopen, not on creation.
    """
    db_path = tmp_path / f"{collection}.sqlite3"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(
            """
            CREATE TABLE documents (
                document_id TEXT PRIMARY KEY,
                source TEXT UNIQUE NOT NULL,
                content_hash TEXT NOT NULL,
                ingested_at TEXT NOT NULL
            );
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                start_offset INTEGER NOT NULL,
                end_offset INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                metadata TEXT NOT NULL
            );
            CREATE INDEX idx_chunks_document_id ON chunks (document_id);
            """
        )
        conn.commit()
    finally:
        conn.close()


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


def test_replace_document_round_trip(tmp_path: Path) -> None:
    """replace_document writes the document row and its chunks in one commit."""

    async def _run() -> tuple[str | None, list[Chunk]]:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            chunks = [
                _make_chunk("c1", "doc-1", "hello world", chunk_index=0),
                _make_chunk("c2", "doc-1", "second chunk", chunk_index=1, start_offset=11),
            ]
            await store.replace_document(
                source="a.md", document_id="doc-1", content_hash="h1", chunks=chunks
            )
            doc_hash = await store.get_document_hash("a.md")
            persisted = await store.get_chunks()
            return doc_hash, persisted
        finally:
            await store.close()

    doc_hash, persisted = asyncio.run(_run())

    assert doc_hash == "h1"
    assert {c.chunk_id for c in persisted} == {"c1", "c2"}


def test_replace_document_rolls_back_document_row_on_chunk_mismatch(tmp_path: Path) -> None:
    """A mismatched chunk's document_id must roll back the document row too.

    This is exactly the atomicity ``replace_document`` exists for: under the
    old ``upsert_document`` + ``add_chunks`` sequence, ``upsert_document``'s
    row was already committed by the time a mismatched chunk was
    discovered, leaving it durably orphaned with zero chunks.
    """

    async def _run() -> str | None:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            mismatched = _make_chunk("c1", "wrong-doc-id", "hello world")
            with pytest.raises(StorageError):
                await store.replace_document(
                    source="a.md", document_id="doc-1", content_hash="h1", chunks=[mismatched]
                )
            return await store.get_document_hash("a.md")
        finally:
            await store.close()

    assert asyncio.run(_run()) is None


def test_add_chunks_rolls_back_partial_write_on_metadata_error(tmp_path: Path) -> None:
    """A non-serializable chunk mid-batch must not leave earlier chunks durable.

    Regression test: ``_run`` used to wrap ``sqlite3.Error`` into
    ``StorageError`` but never called ``rollback()`` — not even for a
    ``StorageError`` raised deliberately inside ``_op`` itself, as the
    non-serializable-metadata check does. Because this connection uses
    sqlite3's default (legacy) transaction-control mode, the good chunk's
    INSERT stayed uncommitted-but-visible on this connection after the bad
    chunk failed, and would become durable on disk the moment any later,
    unrelated ``commit()`` ran on the same connection.
    """

    async def _run() -> tuple[list[Chunk], list[Chunk]]:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            await store.upsert_document(source="a.md", document_id="doc-1", content_hash="h1")
            good = _make_chunk("good", "doc-1", "good text", chunk_index=0)
            bad = _make_chunk(
                "bad",
                "doc-1",
                "bad text",
                chunk_index=1,
                start_offset=9,
                metadata={"oops": {1, 2, 3}},  # a set is not JSON-serializable
            )
            with pytest.raises(StorageError):
                await store.add_chunks([good, bad], source="a.md")
            immediately_after_failure = await store.get_chunks()

            # A later, unrelated successful write commits the shared
            # connection — this is exactly what made an un-rolled-back
            # partial write durable in the original defect.
            await store.upsert_document(source="b.md", document_id="doc-2", content_hash="h2")
            other = _make_chunk("other", "doc-2", "unrelated chunk")
            await store.add_chunks([other], source="b.md")
        finally:
            await store.close()

        reopened = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            after_reopen = await reopened.get_chunks()
        finally:
            await reopened.close()

        return immediately_after_failure, after_reopen

    immediately_after_failure, after_reopen = asyncio.run(_run())

    assert immediately_after_failure == []
    assert {c.chunk_id for c in after_reopen} == {"other"}


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


def test_open_rejects_parent_traversal_collection_and_creates_nothing_outside(
    tmp_path: Path,
) -> None:
    """'../outside' is rejected and must not create a database outside index_dir.

    This is the P2 path-containment bug: an unvalidated ``collection`` used
    to resolve ``<index_dir>/../outside.sqlite3``, landing the database one
    level above the configured index directory.
    """
    index_dir = tmp_path / "index_dir"

    async def _open() -> SQLiteMetadataStore:
        return await SQLiteMetadataStore.open(index_dir, "../outside")

    with pytest.raises(ConfigurationError):
        asyncio.run(_open())

    assert not (tmp_path / "outside.sqlite3").exists()


def test_open_rejects_absolute_collection(tmp_path: Path) -> None:
    """An absolute collection value is rejected instead of discarding index_dir.

    ``Path("/a") / "/b"`` is ``Path("/b")`` — an absolute ``collection``
    would otherwise silently ignore ``index_dir`` entirely.
    """

    async def _open(collection: str) -> SQLiteMetadataStore:
        return await SQLiteMetadataStore.open(tmp_path, collection)

    for bad in ("/etc/passwd", "C:\\Windows\\evil", "\\\\server\\share\\evil"):
        with pytest.raises(ConfigurationError):
            asyncio.run(_open(bad))


def test_open_rejects_path_separator_in_collection(tmp_path: Path) -> None:
    """A collection name containing a path separator is rejected, '/' or '\\'."""

    async def _open(collection: str) -> SQLiteMetadataStore:
        return await SQLiteMetadataStore.open(tmp_path, collection)

    for bad in ("sub/dir", "sub\\dir"):
        with pytest.raises(ConfigurationError):
            asyncio.run(_open(bad))


def test_open_rejects_dot_and_dotdot_collection(tmp_path: Path) -> None:
    """A collection name of exactly '.' or '..' is rejected."""

    async def _open(collection: str) -> SQLiteMetadataStore:
        return await SQLiteMetadataStore.open(tmp_path, collection)

    for bad in (".", ".."):
        with pytest.raises(ConfigurationError):
            asyncio.run(_open(bad))


def test_open_rejects_empty_or_whitespace_collection(tmp_path: Path) -> None:
    """An empty or whitespace-only collection name is rejected."""

    async def _open(collection: str) -> SQLiteMetadataStore:
        return await SQLiteMetadataStore.open(tmp_path, collection)

    for bad in ("", "   ", "\t"):
        with pytest.raises(ConfigurationError):
            asyncio.run(_open(bad))


def test_open_accepts_valid_collection_names_with_dash_underscore_dot(tmp_path: Path) -> None:
    """Collection names with '-', '_', and '.' are valid and round-trip a document.

    Guards against an over-tight fix: legitimate collection names like
    ``my-docs.v2`` must keep working end to end.
    """

    async def _run(collection: str) -> tuple[str | None, list[Chunk]]:
        store = await SQLiteMetadataStore.open(tmp_path, collection)
        try:
            await store.upsert_document(source="a.md", document_id="doc-1", content_hash="h1")
            chunk = _make_chunk("c1", "doc-1", "hello")
            await store.add_chunks([chunk], source="a.md")
            doc_hash = await store.get_document_hash("a.md")
            chunks = await store.get_chunks()
            return doc_hash, chunks
        finally:
            await store.close()

    doc_hash, chunks = asyncio.run(_run("my-docs.v2"))

    assert doc_hash == "h1"
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "c1"


# ── ADR-0004: collection manifest (embedding identity binding) ─────────────


def test_manifest_write_and_read_round_trip(tmp_path: Path) -> None:
    """A written manifest reads back with the identity it was written with."""

    async def _run() -> CollectionManifest | None:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            embedding = EmbeddingConfig(
                provider="ollama", model_name="nomic-embed-text", dimensions=768
            )
            await store.write_manifest(embedding)
            return await store.get_manifest()
        finally:
            await store.close()

    manifest = asyncio.run(_run())

    assert manifest is not None
    assert manifest.provider == "ollama"
    assert manifest.model_name == "nomic-embed-text"
    assert manifest.dimensions == 768
    assert manifest.created_at  # a real timestamp was recorded, not left blank


def test_manifest_rewrite_same_identity_is_noop(tmp_path: Path) -> None:
    """Re-ingesting into an already-bound collection must keep working.

    A second write with the *same* identity is a no-op, not an error — and
    genuinely a no-op, not a silent rewrite: the manifest (including its
    original ``created_at``) is unchanged by the second call.
    """

    async def _run() -> tuple[CollectionManifest | None, CollectionManifest | None]:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            embedding = EmbeddingConfig(
                provider="ollama", model_name="nomic-embed-text", dimensions=768
            )
            await store.write_manifest(embedding)
            first = await store.get_manifest()
            await store.write_manifest(embedding)
            second = await store.get_manifest()
            return first, second
        finally:
            await store.close()

    first, second = asyncio.run(_run())

    assert first is not None
    assert second is not None
    assert first == second


@pytest.mark.parametrize(
    "second",
    [
        pytest.param(
            EmbeddingConfig(
                provider="openai_compatible", model_name="nomic-embed-text", dimensions=768
            ),
            id="different_provider",
        ),
        pytest.param(
            EmbeddingConfig(provider="ollama", model_name="mxbai-embed-large", dimensions=768),
            id="different_model_name",
        ),
        pytest.param(
            EmbeddingConfig(provider="ollama", model_name="nomic-embed-text", dimensions=1024),
            id="different_dimensions",
        ),
    ],
)
def test_manifest_rewrite_different_identity_raises(
    tmp_path: Path, second: EmbeddingConfig
) -> None:
    """A second write() differing in any single field is refused, not silently applied."""
    first = EmbeddingConfig(provider="ollama", model_name="nomic-embed-text", dimensions=768)

    async def _run() -> None:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            await store.write_manifest(first)
            await store.write_manifest(second)
        finally:
            await store.close()

    with pytest.raises(IndexIdentityError):
        asyncio.run(_run())


def test_manifest_rejects_same_dimensions_different_model(tmp_path: Path) -> None:
    """768-vs-768: two distinct models sharing a vector width must still conflict.

    ``nomic-embed-text`` and ``all-mpnet-base-v2`` are both 768-dimensional
    — the exact case ADR-0004 exists to close. A width-only identity check
    would let this swap through silently; identity is the full
    ``(provider, model_name, dimensions)`` triple, so a match on width alone
    is not a match.
    """
    built_with = EmbeddingConfig(provider="ollama", model_name="nomic-embed-text", dimensions=768)
    swapped_to = EmbeddingConfig(provider="ollama", model_name="all-mpnet-base-v2", dimensions=768)

    async def _run() -> None:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            await store.write_manifest(built_with)
            await store.write_manifest(swapped_to)
        finally:
            await store.close()

    with pytest.raises(IndexIdentityError, match="dimensions=768"):
        asyncio.run(_run())


def test_verify_manifest_passes_when_no_manifest_written_yet(tmp_path: Path) -> None:
    """A collection with no dense write yet has nothing to conflict with.

    ``write_manifest`` establishes the manifest; ``verify_manifest`` never
    does, so it must not raise merely because no dense write has happened.
    """

    async def _run() -> None:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            embedding = EmbeddingConfig(
                provider="ollama", model_name="nomic-embed-text", dimensions=768
            )
            await store.verify_manifest(embedding)
        finally:
            await store.close()

    asyncio.run(_run())  # must not raise


def test_verify_manifest_passes_for_matching_identity(tmp_path: Path) -> None:
    """verify_manifest is a no-op when the active config matches the stored manifest."""
    embedding = EmbeddingConfig(provider="ollama", model_name="nomic-embed-text", dimensions=768)

    async def _run() -> None:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            await store.write_manifest(embedding)
            await store.verify_manifest(embedding)
        finally:
            await store.close()

    asyncio.run(_run())  # must not raise


def test_verify_manifest_raises_on_mismatch(tmp_path: Path) -> None:
    """verify_manifest refuses a mismatch exactly like write_manifest — never re-embeds."""
    built_with = EmbeddingConfig(provider="ollama", model_name="nomic-embed-text", dimensions=768)
    different = EmbeddingConfig(provider="ollama", model_name="all-mpnet-base-v2", dimensions=768)

    async def _run() -> None:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            await store.write_manifest(built_with)
            await store.verify_manifest(different)
        finally:
            await store.close()

    with pytest.raises(IndexIdentityError):
        asyncio.run(_run())


def test_fresh_store_stamps_application_id_and_user_version(tmp_path: Path) -> None:
    """A freshly created store is stamped with groundkit's application_id/user_version."""

    async def _open_and_close() -> None:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        await store.close()

    asyncio.run(_open_and_close())

    conn = sqlite3.connect(str(tmp_path / "col.sqlite3"))
    try:
        app_id = conn.execute("PRAGMA application_id").fetchone()[0]
        version = conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()

    assert app_id == APPLICATION_ID
    assert app_id != 0  # sanity: not SQLite's unset default
    assert version == SCHEMA_VERSION
    assert version != 0


def test_legacy_store_without_manifest_stamp_is_refused_for_dense_work(tmp_path: Path) -> None:
    """A store predating ADR-0004 refuses manifest writes/verifies but still opens fine.

    BM25-only collections must keep working: a store created before the
    embedding-identity manifest existed has no PRAGMA application_id/
    user_version stamp, so opening it and doing ordinary document/chunk work
    succeeds unchanged. Only the manifest-specific operations — which would
    otherwise trust an identity this store never recorded — are refused,
    with a clear IndexIdentityError rather than a guess. Pre-1.0 there is no
    migration path: the fix is delete-and-reingest.
    """
    _write_legacy_schema(tmp_path, "col")

    async def _bm25_only_work() -> tuple[str | None, CollectionManifest | None]:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            await store.upsert_document(source="a.md", document_id="doc-1", content_hash="h1")
            doc_hash = await store.get_document_hash("a.md")
            manifest = await store.get_manifest()  # a plain read: never raises
            return doc_hash, manifest
        finally:
            await store.close()

    doc_hash, manifest = asyncio.run(_bm25_only_work())
    assert doc_hash == "h1"
    assert manifest is None

    embedding = EmbeddingConfig(provider="ollama", model_name="nomic-embed-text", dimensions=768)

    async def _try_write() -> None:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            await store.write_manifest(embedding)
        finally:
            await store.close()

    with pytest.raises(IndexIdentityError):
        asyncio.run(_try_write())

    async def _try_verify() -> None:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            await store.verify_manifest(embedding)
        finally:
            await store.close()

    with pytest.raises(IndexIdentityError):
        asyncio.run(_try_verify())


def test_bm25_only_round_trip_with_no_manifest_present(tmp_path: Path) -> None:
    """A collection that never does a dense write behaves exactly as before ADR-0004."""

    async def _run() -> tuple[list[Chunk], CollectionManifest | None]:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            await store.upsert_document(source="a.md", document_id="doc-1", content_hash="h1")
            chunk = _make_chunk("c1", "doc-1", "hello world")
            await store.add_chunks([chunk], source="a.md")
            chunks = await store.get_chunks()
            manifest = await store.get_manifest()
            return chunks, manifest
        finally:
            await store.close()

    chunks, manifest = asyncio.run(_run())

    assert len(chunks) == 1
    assert chunks[0].content == "hello world"
    assert manifest is None


def test_sqlite_metadata_store_has_signature_parity_with_protocol() -> None:
    """SQLiteMetadataStore's manifest methods match MetadataStoreProtocol exactly.

    ``test_sqlite_metadata_store_conforms_to_protocol`` above only confirms
    (via ``isinstance``) that members of the right name exist — it would not
    catch a parameter rename or an accidental sync/async mismatch on
    ``write_manifest``/``verify_manifest``/``get_manifest``.
    ``assert_signature_parity`` closes that gap by comparing parameter
    names, kinds, order, defaults, and resolved type hints.
    """
    assert_signature_parity(MetadataStoreProtocol, SQLiteMetadataStore)

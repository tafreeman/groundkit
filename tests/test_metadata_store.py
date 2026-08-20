"""Tests for SQLiteMetadataStore — the persisted document/chunk truth store.

Async methods are driven with ``asyncio.run()`` inside sync test functions
(pytest-asyncio is not configured in this repo).
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from pathlib import Path

import pytest

from groundkit.contracts import Chunk, CollectionManifest, EmbeddingIdentity
from groundkit.errors import ConfigurationError, IndexIdentityError, StorageError
from groundkit.index.metadata import (
    APPLICATION_ID,
    BUSY_TIMEOUT_MS,
    SCHEMA_VERSION,
    SQLiteMetadataStore,
)
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


def test_add_chunks_rejects_chunk_with_mismatched_document_id(tmp_path: Path) -> None:
    """``add_chunks`` carries its own copy of the document-id guard
    ``replace_document`` enforces (exercised by
    ``test_generation_bump_is_atomic_with_the_write`` below via a real
    failure path, not a patched one). ``add_chunks`` has no production
    caller today -- ``indexer.py`` uses ``replace_document`` exclusively --
    but remains a required ``MetadataStoreProtocol`` member a third-party
    implementation could rely on, so a refactor of ``add_chunks`` alone
    dropping this guard would otherwise go unnoticed.
    """

    async def _run() -> None:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            await store.upsert_document(source="a.md", document_id="doc-1", content_hash="h1")
            mismatched = _make_chunk("c1", "WRONG-DOC", "orphaned content")
            await store.add_chunks([mismatched], source="a.md")
        finally:
            await store.close()

    with pytest.raises(StorageError, match="does not match"):
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

    ``bad`` is built with :meth:`~pydantic.BaseModel.model_construct`,
    bypassing validation, rather than through ``_make_chunk`` /
    ``Chunk(...)``: GK-017 added a ``metadata`` validator that itself
    rejects a non-JSON-serializable value with ``ValidationError`` at
    construction, so the normal constructor can no longer produce the chunk
    this test needs. That is exactly the improvement GK-017 made — but this
    test's actual subject is ``add_chunks``'s own rollback discipline, which
    must hold regardless of what already validated the data reaching it
    (defense in depth: an old row predating the validator, or any future
    ``MetadataStoreProtocol`` implementation, can still hand it a ``Chunk``
    Pydantic never checked).
    """

    async def _run() -> tuple[list[Chunk], list[Chunk]]:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            await store.upsert_document(source="a.md", document_id="doc-1", content_hash="h1")
            good = _make_chunk("good", "doc-1", "good text", chunk_index=0)
            bad = Chunk.model_construct(
                chunk_id="bad",
                document_id="doc-1",
                chunk_index=1,
                content="bad text",
                start_offset=9,
                end_offset=9 + len("bad text"),
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


def test_get_document_id_returns_stored_id_and_none_for_unseen(tmp_path: Path) -> None:
    """get_document_id resolves a source to its stored id, matching get_document_sources.

    The dense write path (Phase 3) needs this to look up a source's current
    document_id in O(1) rather than scanning the whole get_document_sources()
    map, so the id it returns must agree with what that map reports for the
    same source.
    """

    async def _run() -> tuple[str | None, str | None, dict[str, str]]:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            await store.upsert_document(source="a.md", document_id="doc-1", content_hash="h1")
            found = await store.get_document_id("a.md")
            missing = await store.get_document_id("missing.md")
            sources = await store.get_document_sources()
            return found, missing, sources
        finally:
            await store.close()

    found, missing, sources = asyncio.run(_run())

    assert found == "doc-1"
    assert missing is None
    assert found is not None
    assert sources[found] == "a.md"


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


def test_open_refuses_a_foreign_sqlite_file_and_leaves_it_byte_identical(
    tmp_path: Path,
) -> None:
    """Opening an unrelated database must not write groundkit's schema into it.

    ``open`` applies ``_SCHEMA`` unconditionally, so before the read-only
    identity probe it would *create* ``documents``, ``chunks``,
    ``collection_manifest`` and ``collection_state`` inside any pre-existing
    ``*.sqlite3`` it was pointed at. Combined with ``list_collections``
    advertising every such file, a read-only service surface mutated a
    stranger's database: list it, then ask ``index_status`` for it.

    Asserting the bytes are unchanged is deliberately stronger than asserting
    the table set: a stamped ``PRAGMA application_id`` alters the header
    without adding a table, and that is a write too.
    """
    foreign = tmp_path / "notes.sqlite3"
    conn = sqlite3.connect(str(foreign))
    try:
        conn.execute("CREATE TABLE journal (id INTEGER PRIMARY KEY, body TEXT)")
        conn.execute("INSERT INTO journal (body) VALUES ('private')")
        conn.commit()
    finally:
        conn.close()
    before = foreign.read_bytes()
    siblings_before = sorted(p.name for p in tmp_path.iterdir())

    async def _open() -> SQLiteMetadataStore:
        return await SQLiteMetadataStore.open(tmp_path, "notes")

    with pytest.raises(StorageError, match="not a groundkit collection"):
        asyncio.run(_open())

    assert foreign.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == siblings_before

    conn = sqlite3.connect(str(foreign))
    try:
        tables = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    finally:
        conn.close()
    assert tables == {"journal"}


def test_open_still_accepts_a_legacy_unstamped_groundkit_store(tmp_path: Path) -> None:
    """A store predating ``PRAGMA application_id`` must keep opening.

    Keying the probe on the stamp alone would lock out exactly the older
    stores ADR-0016's narrow write-guard was written to keep readable, so the
    unstamped case is recognized by its marker tables instead.
    """

    async def _make() -> None:
        store = await SQLiteMetadataStore.open(tmp_path, "legacy")
        await store.close()

    asyncio.run(_make())

    # A real store with its stamp cleared -- the actual shape of a pre-stamp
    # store, rather than a hand-built file that only shares the table names.
    legacy = tmp_path / "legacy.sqlite3"
    conn = sqlite3.connect(str(legacy))
    try:
        conn.execute("PRAGMA application_id = 0")
        conn.commit()
    finally:
        conn.close()
    conn = sqlite3.connect(str(legacy))
    try:
        assert int(conn.execute("PRAGMA application_id").fetchone()[0]) == 0
    finally:
        conn.close()

    async def _reopen() -> None:
        store = await SQLiteMetadataStore.open(tmp_path, "legacy")
        await store.close()

    asyncio.run(_reopen())


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
            embedding = EmbeddingIdentity(
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
            embedding = EmbeddingIdentity(
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
            EmbeddingIdentity(
                provider="openai_compatible", model_name="nomic-embed-text", dimensions=768
            ),
            id="different_provider",
        ),
        pytest.param(
            EmbeddingIdentity(provider="ollama", model_name="mxbai-embed-large", dimensions=768),
            id="different_model_name",
        ),
        pytest.param(
            EmbeddingIdentity(provider="ollama", model_name="nomic-embed-text", dimensions=1024),
            id="different_dimensions",
        ),
    ],
)
def test_manifest_rewrite_different_identity_raises(
    tmp_path: Path, second: EmbeddingIdentity
) -> None:
    """A second write() differing in any single field is refused, not silently applied."""
    first = EmbeddingIdentity(provider="ollama", model_name="nomic-embed-text", dimensions=768)

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
    built_with = EmbeddingIdentity(provider="ollama", model_name="nomic-embed-text", dimensions=768)
    swapped_to = EmbeddingIdentity(
        provider="ollama", model_name="all-mpnet-base-v2", dimensions=768
    )

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
            embedding = EmbeddingIdentity(
                provider="ollama", model_name="nomic-embed-text", dimensions=768
            )
            await store.verify_manifest(embedding)
        finally:
            await store.close()

    asyncio.run(_run())  # must not raise


def test_verify_manifest_passes_for_matching_identity(tmp_path: Path) -> None:
    """verify_manifest is a no-op when the active config matches the stored manifest."""
    embedding = EmbeddingIdentity(provider="ollama", model_name="nomic-embed-text", dimensions=768)

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
    built_with = EmbeddingIdentity(provider="ollama", model_name="nomic-embed-text", dimensions=768)
    different = EmbeddingIdentity(provider="ollama", model_name="all-mpnet-base-v2", dimensions=768)

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
    """A store predating ADR-0004 refuses manifest writes/verifies and, since
    schema v3, refuses document writes too — while reads still open fine.

    **This test's promise narrowed at ADR-0016, and the change is recorded
    rather than silently applied.** It used to assert that "BM25-only
    collections must keep working" — that ordinary document/chunk work on an
    unstamped legacy store "succeeds unchanged", with only manifest-specific
    operations refused. ADR-0016 added ``source_class``/``extractor`` to the
    ``documents`` table and raised ``SCHEMA_VERSION`` to 3, and states that
    existing collections must be deleted and re-ingested. A pre-v3 store has
    no column to record a source class in — ``CREATE TABLE IF NOT EXISTS``
    does not add one to a table that already exists — so a write must be
    refused rather than silently dropping the class, which is precisely the
    fail-open defect ADR-0016 closes.

    What did **not** change, and is asserted below: **reads still work.** The
    guard is scoped to the three methods that touch the new columns, so an
    existing collection stays openable and searchable; it just cannot be
    written to. And the manifest operations still fail with their own
    ``IndexIdentityError`` for ADR-0004's reason rather than being folded
    into the schema refusal, so the two failures stay distinguishable.
    """
    _write_legacy_schema(tmp_path, "col")

    async def _write_is_refused() -> None:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            with pytest.raises(StorageError, match="re-ingest"):
                await store.upsert_document(source="a.md", document_id="doc-1", content_hash="h1")
        finally:
            await store.close()

    asyncio.run(_write_is_refused())

    async def _reads_still_work() -> tuple[dict[str, str], CollectionManifest | None]:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            sources = await store.get_document_sources()
            manifest = await store.get_manifest()  # a plain read: never raises
            return sources, manifest
        finally:
            await store.close()

    sources, manifest = asyncio.run(_reads_still_work())
    assert sources == {}
    assert manifest is None

    embedding = EmbeddingIdentity(provider="ollama", model_name="nomic-embed-text", dimensions=768)

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


def test_cancelled_operation_rolls_back_its_partial_writes(tmp_path: Path) -> None:
    """A cancelled operation must roll back, not leave writes pending.

    G2's second half. The connection runs in sqlite3's legacy transaction
    mode: an implicit transaction opens at the first DML statement and stays
    open, its writes visible to later reads on this same connection and
    durable the moment any *other* commit runs on it. ``_run`` rolls back on
    the way out for exactly that reason -- but under ``except Exception`` it
    skipped the path that needs it most, since ``CancelledError`` derives
    from ``BaseException``, not ``Exception``. A sibling cancelled by
    ``asyncio.gather`` therefore left its half-written statements armed to
    leak out on the next unrelated commit.

    Driven through ``_run`` directly: the invariant is about how ``_run``
    unwinds, and every public method commits atomically, so there is no
    public seam that can be interrupted mid-transaction on purpose.
    """

    async def run() -> None:
        store = await SQLiteMetadataStore.open(tmp_path / "idx", "default")
        try:

            def _op() -> None:
                store._conn.execute(
                    "INSERT INTO documents (document_id, source, content_hash, ingested_at) "
                    "VALUES (?, ?, ?, ?)",
                    ("doc-cancelled", "/corpus/cancelled.md", "hash", "2026-01-01T00:00:00Z"),
                )
                raise asyncio.CancelledError()

            with pytest.raises(asyncio.CancelledError):
                await store._run(_op)

            # The implicit transaction was undone, not left pending.
            assert store._conn.in_transaction is False

            # And the write is genuinely gone -- not merely uncommitted and
            # waiting for some later commit to make it durable.
            await store.upsert_document("/corpus/other.md", "doc-other", "hash-other")
            assert await store.get_document_sources() == {"doc-other": "/corpus/other.md"}
        finally:
            await store.close()

    asyncio.run(run())


def test_cancellation_waits_for_the_worker_before_releasing_the_lock(tmp_path: Path) -> None:
    """Cancelling an operation must not free the connection while it is in use.

    Cancelling ``await asyncio.to_thread(fn)`` does not stop ``fn`` -- the
    worker thread runs to completion regardless. So neither the rollback nor
    the lock release can be driven off the awaiting coroutine unwinding: a
    rollback issued from the event loop would race statements the worker is
    still executing, and releasing the lock on unwind would let the next
    operation interleave with an abandoned one, breaking exactly the
    one-commit atomicity ``replace_document`` promises.

    Pinned directly on ``_run``: the invariant is about how ``_run`` unwinds,
    and no public method can be interrupted mid-transaction on purpose.
    """

    async def run() -> None:
        store = await SQLiteMetadataStore.open(tmp_path / "idx", "default")
        try:
            entered = threading.Event()
            may_finish = threading.Event()

            def _op() -> None:
                store._conn.execute(
                    "INSERT INTO documents (document_id, source, content_hash, ingested_at) "
                    "VALUES (?, ?, ?, ?)",
                    ("doc-slow", "/corpus/slow.md", "hash", "2026-01-01T00:00:00Z"),
                )
                entered.set()
                may_finish.wait(timeout=10)
                raise sqlite3.OperationalError("worker failed after cancellation")

            task = asyncio.create_task(store._run(_op))
            await asyncio.to_thread(entered.wait, 10)

            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            # The worker is still inside _op. The lock must still be held.
            assert store._lock.locked()

            may_finish.set()
            for _ in range(200):
                if not store._lock.locked():
                    break
                await asyncio.sleep(0.05)

            # Released only once the worker finished, and it rolled back its
            # own partial write on its own thread.
            assert not store._lock.locked()
            assert store._conn.in_transaction is False

            await store.upsert_document("/corpus/other.md", "doc-other", "hash-other")
            assert await store.get_document_sources() == {"doc-other": "/corpus/other.md"}
        finally:
            await store.close()

    asyncio.run(run())


class _SlowCloseConnection(sqlite3.Connection):
    """A connection whose ``close()`` blocks until released.

    ``sqlite3.Connection`` is an immutable C type: neither
    ``conn.close = ...`` (instance attribute) nor
    ``sqlite3.Connection.close = ...`` (class attribute) is legal --
    both raise (``AttributeError`` / ``TypeError``) because the type has no
    ``__dict__`` and forbids attribute assignment. Subclassing is the only
    way to make ``close()`` itself block, which is what
    ``test_cancelled_close_waits_for_the_worker_before_releasing_the_lock``
    needs to reproduce C2: a worker genuinely still inside
    ``Connection.close()`` when the awaiting coroutine is cancelled.
    """

    entered: threading.Event
    may_finish: threading.Event

    def close(self) -> None:
        self.entered.set()
        self.may_finish.wait(timeout=10)
        super().close()


def test_cancelled_close_waits_for_the_worker_before_releasing_the_lock(tmp_path: Path) -> None:
    """C2 regression: ``close()`` must not release the lock before its worker finishes closing.

    The original ``close()`` awaited ``asyncio.to_thread(self._conn.close)``
    directly inside ``async with self._lock:``, bypassing the cancellation-safety
    pattern every other method on this class uses. Cancelling that await does not
    stop the worker thread -- ``Connection.close()`` keeps running regardless --
    but the bare ``async with`` releases the lock the moment the *awaiting*
    coroutine unwinds, not when the worker actually finishes. A second coroutine
    could then acquire the lock and touch the same ``check_same_thread=False``
    connection from a second thread while the first close was still in flight.

    Mirrors ``test_cancellation_waits_for_the_worker_before_releasing_the_lock``'s
    harness exactly (block a worker on a real ``threading.Event``, cancel the
    outer task while it is genuinely still running, assert the lock survives
    that cancellation), substituting a ``_SlowCloseConnection`` for the
    plain-Python ``_op`` that harness uses -- there is no Python-level ``_op``
    to block inside ``close()``, since ``self._conn.close`` is the callable
    passed straight to ``_run``.
    """

    async def run() -> None:
        db_path = tmp_path / "default.sqlite3"
        conn = sqlite3.connect(str(db_path), factory=_SlowCloseConnection, check_same_thread=False)
        conn.entered = threading.Event()
        conn.may_finish = threading.Event()
        store = SQLiteMetadataStore(conn, db_path, schema_current=True)

        task = asyncio.create_task(store.close())
        await asyncio.to_thread(conn.entered.wait, 10)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # The worker is still inside Connection.close(). If the lock were
        # already free here, a second coroutine could acquire it and touch
        # the connection while this close() is still running on its thread --
        # exactly the race C2 describes.
        assert store._lock.locked()

        conn.may_finish.set()
        for _ in range(200):
            if not store._lock.locked():
                break
            await asyncio.sleep(0.05)

        # Released only once the worker actually finished closing the connection.
        assert not store._lock.locked()

    asyncio.run(run())


# --------------------------------------------------------------------------
# ADR-0013: the persisted staleness marker
# --------------------------------------------------------------------------

#: Every public MetadataStoreProtocol member that commits durable state.
_MUTATING_MEMBERS = frozenset(
    {
        "upsert_document",
        "add_chunks",
        "replace_document",
        "delete_document",
        "write_manifest",
    }
)

#: Every public MetadataStoreProtocol member that does not.
_READ_ONLY_MEMBERS = frozenset(
    {
        "get_document_hash",
        "get_document_id",
        "get_document_sources",
        "get_chunks",
        "get_chunk",
        "verify_manifest",
        "get_manifest",
        "get_generation",
    }
)


def _protocol_members(protocol: type) -> frozenset[str]:
    """Public callable members declared on ``protocol`` itself."""
    return frozenset(
        name
        for name in vars(protocol)
        if not name.startswith("_") and callable(getattr(protocol, name, None))
    )


def test_metadata_store_protocol_members_are_all_classified() -> None:
    """Every protocol member is classified mutating or read-only.

    NOT a regression test and must never be reported as one: nothing is
    being fixed, and it cannot be shown to fail against unfixed code. It is
    a completeness guard that fails when a *future* member is added without
    a decision about whether it advances the ADR-0013 marker.

    It carries real weight despite that. The marker's *read* lives in
    ``runtime.py``, inside the coverage core subset; the marker's *bump*
    lives here in ``index/metadata.py``, which ``pyproject.toml``
    deliberately keeps outside it. The half whose omission causes silent
    staleness is therefore ungoverned by the core gate, and this assertion
    plus the parameterized bump test below are what stand in for it.
    """
    declared = _protocol_members(MetadataStoreProtocol)
    classified = _MUTATING_MEMBERS | _READ_ONLY_MEMBERS
    assert declared == classified, (
        "MetadataStoreProtocol member set changed. Classify each new member as "
        "mutating (it must call _bump_generation before its commit) or read-only, "
        "then update _MUTATING_MEMBERS / _READ_ONLY_MEMBERS here."
    )


def test_new_store_seeds_the_generation_at_zero(tmp_path: Path) -> None:
    """A freshly created store is cacheable immediately, with a real 0."""

    async def run() -> None:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            assert await store.get_generation() == 0
        finally:
            await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("method", sorted(_MUTATING_MEMBERS))
def test_every_committing_mutation_advances_the_generation(tmp_path: Path, method: str) -> None:
    """Each mutating method advances the marker. Parameterized so a gap is visible per method.

    Fail-first: removing ``self._bump_generation()`` from any one method's
    ``_op`` fails that method's parameterization while the others still pass,
    which is exactly the granularity a reviewer needs.
    """

    async def run() -> None:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            # Preconditions each mutation needs, established before the
            # generation is sampled so setup writes are not what is measured.
            if method in {"add_chunks", "delete_document"}:
                await store.upsert_document("/a.md", "doc-1", "h1")

            before = await store.get_generation()
            assert before is not None

            if method == "upsert_document":
                await store.upsert_document("/b.md", "doc-2", "h2")
            elif method == "add_chunks":
                await store.add_chunks([_make_chunk("c1", "doc-1", "hello")], source="/a.md")
            elif method == "replace_document":
                await store.replace_document(
                    "/c.md", "doc-3", "h3", [_make_chunk("c2", "doc-3", "world")]
                )
            elif method == "delete_document":
                await store.delete_document("doc-1")
            elif method == "write_manifest":
                await store.write_manifest(
                    EmbeddingIdentity(provider="ollama", model_name="m", dimensions=8)
                )
            else:  # pragma: no cover - guarded by the classification test
                raise AssertionError(f"unclassified mutating method {method!r}")

            after = await store.get_generation()
            assert after is not None
            assert after > before, f"{method} committed without advancing the marker"
        finally:
            await store.close()

    asyncio.run(run())


def test_delete_of_an_absent_document_still_advances_the_generation(tmp_path: Path) -> None:
    """Over-bumping is deliberate: no branch decides whether a write 'really' changed anything.

    ``delete_document`` on an id that is not present deletes zero rows and
    still commits, so it advances. The cost is one redundant rebuild; the
    cost of the opposite policy is a stale cache served silently, so where
    the two are traded, bump.
    """

    async def run() -> None:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            before = await store.get_generation()
            assert await store.delete_document("no-such-doc") == 0
            after = await store.get_generation()
            assert before is not None and after is not None
            assert after > before
        finally:
            await store.close()

    asyncio.run(run())


def test_read_only_methods_do_not_advance_the_generation(tmp_path: Path) -> None:
    """Reads leave the marker alone, so a read-heavy service never invalidates its own cache."""

    async def run() -> None:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            await store.replace_document("/a.md", "doc-1", "h1", [_make_chunk("c1", "doc-1", "hi")])
            before = await store.get_generation()

            await store.get_document_hash("/a.md")
            await store.get_document_id("/a.md")
            await store.get_document_sources()
            await store.get_chunks()
            await store.get_chunk("c1")
            await store.get_manifest()
            await store.get_generation()

            assert await store.get_generation() == before
        finally:
            await store.close()

    asyncio.run(run())


def test_generation_bump_is_atomic_with_the_write(tmp_path: Path) -> None:
    """A rolled-back write leaves the marker where it was.

    The bump is a statement inside the caller's ``_op``, before its commit,
    so a failure anywhere in that op rolls both back together. Issued as its
    own operation instead, the marker would advance while the content did
    not — and a reader would then observe "changed" over unchanged data
    (wasteful) or, with the opposite ordering, "unchanged" over changed data
    (silent staleness, the failure the marker exists to prevent).

    ``replace_document`` is driven to raise *after* its document INSERT by a
    chunk whose ``document_id`` does not match, which is a real failure path
    rather than a patched one.
    """

    async def run() -> None:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            before = await store.get_generation()
            assert before is not None

            with pytest.raises(StorageError, match="does not match"):
                await store.replace_document(
                    "/a.md", "doc-1", "h1", [_make_chunk("c1", "WRONG-DOC", "hi")]
                )

            # Neither half landed.
            assert await store.get_generation() == before
            assert await store.get_document_sources() == {}
        finally:
            await store.close()

    asyncio.run(run())


def test_rewriting_the_same_manifest_does_not_advance_the_generation(tmp_path: Path) -> None:
    """The no-op branch of ``write_manifest`` commits nothing, so it advances nothing.

    The invariant is one bump per *commit*, not one per method call.
    Re-ingesting into an already-bound collection re-calls ``write_manifest``
    with the identity it already holds; that branch returns before any INSERT
    and before any commit, so durable state is untouched and a cached
    retriever built against it is still valid. Bumping here would force a
    write on a path taken by every re-ingest of a bound collection, to
    invalidate a cache that nothing invalidated.
    """

    async def run() -> None:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            identity = EmbeddingIdentity(provider="ollama", model_name="m", dimensions=8)
            await store.write_manifest(identity)
            after_first = await store.get_generation()

            await store.write_manifest(identity)  # same triple: a no-op
            assert await store.get_generation() == after_first
        finally:
            await store.close()

    asyncio.run(run())


def test_legacy_store_reports_an_unanswerable_generation(tmp_path: Path) -> None:
    """A pre-ADR-0004 store returns None, not 0 — freshness is unanswerable, not unchanged.

    Returning 0 would be the dangerous failure: a caller comparing 0 == 0
    across two requests would conclude "unchanged" and serve a cached index
    over a store it has no basis to vouch for. None forces the caller to
    rebuild every time — correct and slow — which is the whole point of
    distinguishing the two values.
    """
    _write_legacy_schema(tmp_path, "legacy")

    async def run() -> None:
        store = await SQLiteMetadataStore.open(tmp_path, "legacy")
        try:
            assert await store.get_generation() is None
        finally:
            await store.close()

    asyncio.run(run())


def test_store_stamped_with_an_older_schema_version_reports_unanswerable(tmp_path: Path) -> None:
    """A v1-stamped store is refused the cache, which is what makes the version bump load-bearing.

    ``_SCHEMA`` is applied on every open with ``CREATE TABLE IF NOT EXISTS``,
    so ``collection_state`` appears in an older collection regardless. Without
    the ``SCHEMA_VERSION`` bump the marker would therefore start answering
    over it, and an older ``grk`` binary with no bump logic could then write
    to that collection without advancing the counter — a long-lived service
    would serve its cached retriever forever. The bump is what turns that
    into "uncacheable" instead of "silently stale".
    """

    async def run_setup() -> None:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        await store.close()

    asyncio.run(run_setup())

    # Roll the stamp back to the pre-ADR-0013 version, leaving everything else.
    conn = sqlite3.connect(str(tmp_path / "col.sqlite3"))
    try:
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
    finally:
        conn.close()

    async def run() -> None:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            assert await store.get_generation() is None
            # The collection_state row is still physically present — it is the
            # version stamp, not a missing table, that withholds the answer.
            cur = store._conn.execute("SELECT generation FROM collection_state WHERE id = 1")
            assert cur.fetchone() is not None
        finally:
            await store.close()

    asyncio.run(run())


def test_schema_version_is_current(tmp_path: Path) -> None:
    """A store this build creates is stamped with the version this build expects."""

    async def run() -> None:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            version = int(store._conn.execute("PRAGMA user_version").fetchone()[0])
            app_id = int(store._conn.execute("PRAGMA application_id").fetchone()[0])
            assert version == SCHEMA_VERSION
            assert app_id == APPLICATION_ID
        finally:
            await store.close()

    asyncio.run(run())


def test_busy_timeout_is_this_module_s_value_not_the_connect_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lock-wait belongs to groundkit, not to a stdlib default.

    Correcting the finding this closes: it asserted SQLite's own default of
    0 was in force and that a contending writer therefore failed on contact.
    Measured, the connection was already waiting five seconds --
    ``sqlite3.connect`` passes ``timeout=5.0`` unless told otherwise, and
    ``_connect`` never told it otherwise. The behaviour was fine; its
    *provenance* was the defect. Nothing in the repository named the value,
    stated it, or tested it, and ``timeout=0`` is one keyword away from
    being added to that ``connect`` call by an unrelated change.

    Demonstrated by injection, since a passing assertion on 5000 alone
    proves nothing: ``connect`` is forced to the one value that would leave
    a bare connection at 0, and the store must still report
    ``BUSY_TIMEOUT_MS``. Against a ``_connect`` with no explicit pragma this
    reads back 0 and fails.
    """
    real_connect = sqlite3.connect

    def _zero_timeout_connect(
        database: str, *, check_same_thread: bool = True, uri: bool = False
    ) -> sqlite3.Connection:
        """Every ``connect`` shape ``metadata.py`` uses, forced to ``timeout=0``."""
        return real_connect(database, timeout=0, check_same_thread=check_same_thread, uri=uri)

    # The same module object ``metadata.py`` resolves ``sqlite3.connect`` on,
    # at call time; monkeypatch restores it.
    monkeypatch.setattr(sqlite3, "connect", _zero_timeout_connect)

    async def run() -> None:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            effective = store._conn.execute("PRAGMA busy_timeout").fetchone()[0]
            assert effective == BUSY_TIMEOUT_MS, (
                f"busy_timeout is {effective}, not this module's {BUSY_TIMEOUT_MS} -- "
                "the value is being inherited from sqlite3.connect rather than set"
            )
        finally:
            await store.close()

    asyncio.run(run())

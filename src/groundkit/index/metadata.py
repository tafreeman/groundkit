"""Document/chunk metadata store backed by SQLite, behind an interface (Phase 1).

Implements :class:`~groundkit.index.protocols.MetadataStoreProtocol`. SQLite
is the durable truth for documents and chunks (ADR-0002); the BM25 index is
rebuilt from this store at process start rather than persisted itself.

Concurrency model (ADR-0002): a single ``sqlite3.Connection`` opened with
``check_same_thread=False``, with every call serialized through one
``asyncio.Lock``. sqlite3 connections are not safe to use concurrently from
multiple threads, and ``asyncio.to_thread`` hands each call to a fresh worker
thread — the lock, held for the duration of the coroutine (including the
awaited ``to_thread`` call), guarantees only one thread ever touches the
connection at a time. This was chosen over per-call connections to avoid
repeated file-handle churn and "database is locked" contention under WAL for
groundkit's expected single-process local deployment.

Also implements the ADR-0004 collection manifest: a single-row
``collection_manifest`` table binding a collection to the embedding
``(provider, model_name, dimensions)`` triple it was built with, plus the
``PRAGMA application_id``/``user_version`` stamp that lets :meth:`open`
recognize a store created before that manifest existed and refuse it for
dense work rather than trust an identity it never recorded.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, TypeVar

from groundkit.contracts import Chunk, CollectionManifest, EmbeddingIdentity
from groundkit.errors import ConfigurationError, IndexIdentityError, StorageError
from groundkit.utils.path_safety import ensure_within_base

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

#: Allowed characters for a collection name: ASCII letters, digits, and the
#: conservative separators ``-``, ``_``, ``.``. The character class alone
#: already excludes path separators (``/``, ``\``), drive/UNC prefixes
#: (``:``), and whitespace — but a lone ``.`` or ``..`` matches this pattern
#: too, so :func:`validate_collection_name` rejects those explicitly.
_COLLECTION_NAME_PATTERN: re.Pattern[str] = re.compile(r"^[A-Za-z0-9._-]+$")

#: Schema for the persisted metadata store. ``IF NOT EXISTS`` makes this safe
#: to run on every open, whether the file is fresh or already populated.
#:
#: ``collection_manifest`` (ADR-0004) is pinned to exactly one row at the
#: schema level, not by convention: ``id`` is the primary key (unique by
#: definition) and ``CHECK (id = 1)`` forces the only legal value, so a
#: second ``INSERT`` collides on the primary key rather than relying on
#: application code to never issue one.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    source TEXT UNIQUE NOT NULL,
    content_hash TEXT NOT NULL,
    ingested_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    start_offset INTEGER NOT NULL,
    end_offset INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    metadata TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks (document_id);

CREATE TABLE IF NOT EXISTS collection_manifest (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    provider TEXT NOT NULL,
    model_name TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS collection_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    generation INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);
"""

#: Fixed identifier stamped into ``PRAGMA application_id`` on every SQLite
#: file this store creates (ADR-0004 decision 4). SQLite reserves this field
#: for exactly this purpose and makes no use of it itself
#: (https://www.sqlite.org/pragma.html#pragma_application_id): it lets a
#: reopen recognize "this file is groundkit's" without inspecting its
#: tables. Value is the ASCII bytes of "GRK1" packed big-endian — it only
#: needs to be a fixed, recognizable 32-bit signed integer, and this stays
#: comfortably inside that range.
APPLICATION_ID: Final[int] = 0x47524B31  # "GRK1"

#: Schema version stamped into ``PRAGMA user_version`` alongside
#: :data:`APPLICATION_ID`, written together as the last statements before
#: commit when a store is first created (ADR-0004 decision 4) so a
#: failed schema application can never leave a store claiming a version it
#: does not have. A store lacking either stamp — including every store
#: created before this ADR landed — predates the collection manifest;
#: :meth:`SQLiteMetadataStore.open` detects that and
#: :meth:`SQLiteMetadataStore.write_manifest` /
#: :meth:`SQLiteMetadataStore.verify_manifest` refuse it for dense work
#: rather than guessing at an identity it never recorded (ADR-0004 decision
#: 5: pre-1.0, every index is reproducible from ``grk ingest`` in seconds,
#: so a migration path here would be code written to preserve data that
#: costs seconds to regenerate).
#: Bumped 1 -> 2 by ADR-0013, which added ``collection_state``. The bump is
#: load-bearing rather than bookkeeping. ``_SCHEMA`` is applied on *every*
#: open with ``CREATE TABLE IF NOT EXISTS``, so without a version change the
#: new table would appear in a v1 collection transparently and
#: :meth:`SQLiteMetadataStore.get_generation` would start answering over it —
#: after which an older ``grk`` binary, which has no bump logic, could write
#: to that collection without advancing the counter and a long-lived service
#: would serve its cached retriever forever. With the bump, such a store is
#: not schema-current, ``get_generation`` returns ``None``, and the runtime
#: degrades to rebuilding every request instead of caching a stale index.
SCHEMA_VERSION: Final[int] = 2


class SQLiteMetadataStore:
    """Durable store for documents, chunks, and ingest state, backed by SQLite.

    Construct via :meth:`open`, not directly — the constructor takes an
    already-configured connection and its precomputed schema-currency verdict.

    Attributes:
        db_path: Path to the collection's SQLite file, kept for diagnostics.
    """

    def __init__(
        self, connection: sqlite3.Connection, db_path: Path, *, schema_current: bool
    ) -> None:
        self._conn = connection
        self._lock = asyncio.Lock()
        self.db_path = db_path
        #: True when this store's PRAGMA application_id/user_version match
        #: groundkit's current stamp exactly (set by :meth:`open`). False for
        #: a store written against an older schema, one that predates the
        #: ADR-0004 stamp entirely, or one groundkit never created.
        #:
        #: Named for the schema rather than for the manifest because it now
        #: gates two unrelated capabilities: :meth:`write_manifest` and
        #: :meth:`verify_manifest` refuse a non-current store (ADR-0004), and
        #: :meth:`get_generation` reports "freshness unanswerable" for one
        #: (ADR-0013). Such a store still works for BM25-only reads and
        #: writes; it is uncacheable and closed to dense work, not broken.
        self._schema_current = schema_current

    @classmethod
    async def open(cls, index_dir: Path, collection: str) -> SQLiteMetadataStore:
        """Open (creating if absent) the SQLite store for a collection.

        Also determines whether this store predates the ADR-0004 collection
        manifest, by comparing its ``PRAGMA application_id``/``user_version``
        against :data:`APPLICATION_ID`/:data:`SCHEMA_VERSION`. A freshly
        created file is stamped with both, inside the same transaction that
        applies the schema, as the last statements before commit. This
        never blocks opening the store — BM25-only collections must keep
        working unchanged — it only gates :meth:`write_manifest` and
        :meth:`verify_manifest`.

        Args:
            index_dir: Directory holding the collection's persisted state.
                Created (with parents) if it does not exist.
            collection: Collection name; backs the file ``<collection>.sqlite3``.
                Validated against ``_COLLECTION_NAME_PATTERN`` and, once
                resolved, checked to stay contained within ``index_dir`` —
                an unchecked value could otherwise create or open a database
                anywhere on disk (e.g. ``collection="../outside"`` or an
                absolute path).

        Returns:
            A ready-to-use store with the schema applied.

        Raises:
            ConfigurationError: ``collection`` is empty or whitespace-only,
                contains a null byte or a path separator, is ``.`` or
                ``..``, contains characters outside the allowed set, or
                otherwise resolves to a database path outside ``index_dir``.
            StorageError: The directory cannot be created, the database file
                cannot be opened, or the schema cannot be applied (e.g. a
                corrupted file or a path occupied by something other than a
                regular SQLite file).
        """
        validate_collection_name(collection)
        db_path = index_dir / f"{collection}.sqlite3"

        def _connect() -> tuple[sqlite3.Connection, bool]:
            index_dir.mkdir(parents=True, exist_ok=True)
            _chmod_best_effort(index_dir, 0o700)
            try:
                ensure_within_base(db_path, index_dir)
            except ValueError as exc:
                raise ConfigurationError(
                    f"collection {collection!r} resolves outside index_dir {index_dir}"
                ) from exc
            # Captured before connect() — sqlite3 creates the file on first
            # connection, so this is the only point at which "did this file
            # already exist" is still observable.
            is_new_file = not db_path.exists()
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            try:
                conn.execute("PRAGMA foreign_keys = ON")
                conn.execute("PRAGMA journal_mode = WAL")
                conn.executescript(_SCHEMA)
                if is_new_file:
                    # Seed the ADR-0013 generation marker at 0, in the same
                    # transaction as the schema and the stamps below. A v2
                    # store therefore always has a row to read: a *missing*
                    # row is then indistinguishable from a legacy store to
                    # get_generation(), and both correctly answer "freshness
                    # unanswerable" rather than a fabricated 0 that would let
                    # a cache engage over an index it cannot vouch for.
                    conn.execute(
                        "INSERT INTO collection_state (id, generation, updated_at) "
                        "VALUES (1, 0, ?)",
                        (_now_iso(),),
                    )
                    # Stamp groundkit's identity as the last statements
                    # before commit (ADR-0004 decision 4): both
                    # PRAGMA writes land in the same transaction as the
                    # schema application above, so a failure anywhere in
                    # this block (caught below) leaves nothing committed —
                    # never a store claiming a version it does not have.
                    # PRAGMA does not accept bound parameters; the
                    # interpolated values are fixed module constants, never
                    # externally supplied data.
                    conn.execute(f"PRAGMA application_id = {APPLICATION_ID}")
                    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                conn.commit()
            except Exception:
                # Schema application failed (e.g. a corrupted pre-existing
                # file) — close explicitly rather than leaving the handle to
                # CPython refcounting, which is not a guarantee.
                conn.close()
                raise
            _chmod_best_effort(db_path, 0o600)
            app_id = int(conn.execute("PRAGMA application_id").fetchone()[0])
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            schema_current = app_id == APPLICATION_ID and version == SCHEMA_VERSION
            return conn, schema_current

        try:
            connection, schema_current = await asyncio.to_thread(_connect)
        except (OSError, sqlite3.Error) as exc:
            raise StorageError(f"failed to open metadata store at {db_path}") from exc

        logger.debug(
            "opened metadata store (collection=%s, schema_current=%s)",
            collection,
            schema_current,
        )
        return cls(connection, db_path, schema_current=schema_current)

    async def close(self) -> None:
        """Close the underlying connection."""
        async with self._lock:
            await asyncio.to_thread(self._conn.close)
        logger.debug("closed metadata store")

    async def upsert_document(self, source: str, document_id: str, content_hash: str) -> None:
        """Record (or replace) a document's ingest state.

        If ``source`` already has a document row (possibly under a different
        ``document_id`` — e.g. re-ingestion that regenerates IDs), that row
        and all of its chunks are explicitly deleted before the new row is
        inserted, so no stale chunks from a prior version of the document
        survive.

        Args:
            source: File path, URL, or other source identifier. Unique.
            document_id: Identifier to store the document under.
            content_hash: Hash of the document's current content.

        Raises:
            StorageError: On a backend failure.
        """
        ingested_at = _now_iso()

        def _op() -> None:
            self._delete_existing_source(source)
            self._conn.execute(
                "INSERT INTO documents (document_id, source, content_hash, ingested_at) "
                "VALUES (?, ?, ?, ?)",
                (document_id, source, content_hash, ingested_at),
            )
            self._bump_generation()
            self._conn.commit()

        await self._run(_op)

    async def get_document_hash(self, source: str) -> str | None:
        """Return the stored content hash for ``source``, or None if unseen.

        Args:
            source: File path, URL, or other source identifier.

        Returns:
            The stored content hash, or ``None`` if ``source`` has never
            been ingested.

        Raises:
            StorageError: On a backend failure.
        """

        def _op() -> str | None:
            cur = self._conn.execute(
                "SELECT content_hash FROM documents WHERE source = ?", (source,)
            )
            row = cur.fetchone()
            return str(row[0]) if row is not None else None

        return await self._run(_op)

    async def get_document_id(self, source: str) -> str | None:
        """Return the stored document ID for ``source``, or None if unseen.

        Args:
            source: File path, URL, or other source identifier.

        Returns:
            The stored document ID, or ``None`` if ``source`` has never
            been ingested.

        Raises:
            StorageError: On a backend failure.
        """

        def _op() -> str | None:
            cur = self._conn.execute(
                "SELECT document_id FROM documents WHERE source = ?", (source,)
            )
            row = cur.fetchone()
            return str(row[0]) if row is not None else None

        return await self._run(_op)

    async def get_document_sources(self) -> dict[str, str]:
        """Return a ``document_id -> source`` map for every stored document.

        Raises:
            StorageError: On a backend failure.
        """

        def _op() -> dict[str, str]:
            cur = self._conn.execute("SELECT document_id, source FROM documents")
            return {str(row[0]): str(row[1]) for row in cur.fetchall()}

        return await self._run(_op)

    async def add_chunks(self, chunks: list[Chunk], source: str) -> None:
        """Persist chunks for a document.

        Args:
            chunks: Chunks to persist. A no-op if empty.
            source: The source identifier the document was upserted under —
                used to resolve the expected ``document_id`` so a caller
                cannot silently attach chunks to the wrong document.

        Raises:
            StorageError: ``source`` has no upserted document, a chunk's
                ``document_id`` does not match the document registered for
                ``source``, or a backend failure occurs (including a
                duplicate ``chunk_id``).
        """
        if not chunks:
            return

        def _op() -> None:
            cur = self._conn.execute(
                "SELECT document_id FROM documents WHERE source = ?", (source,)
            )
            row = cur.fetchone()
            if row is None:
                raise StorageError(f"cannot add chunks: no document upserted for source {source!r}")
            expected_document_id = row[0]
            for chunk in chunks:
                if chunk.document_id != expected_document_id:
                    raise StorageError(
                        f"chunk.document_id {chunk.document_id!r} does not match "
                        f"document_id {expected_document_id!r} registered for source {source!r}"
                    )
            self._insert_chunks(chunks)
            self._bump_generation()
            self._conn.commit()

        await self._run(_op)

    async def replace_document(
        self, source: str, document_id: str, content_hash: str, chunks: list[Chunk]
    ) -> None:
        """Atomically replace a document's row and its chunks in one transaction.

        Equivalent to calling :meth:`upsert_document` followed by
        :meth:`add_chunks`, but as a single commit. That matters because the
        two-call sequence has a durable-partial-write hazard: a crash (or any
        exception) between the two calls leaves the document row committed
        with its new content hash but zero chunks, and the incremental
        re-index skip (``get_document_hash`` matching) then treats that
        document as already up to date forever. This method closes that gap
        by doing the delete-old-row, insert-document, and insert-chunks steps
        inside one :meth:`_run` call, so any failure — including a chunk with
        a mismatched ``document_id`` or non-JSON-serializable metadata —
        rolls back the whole thing and leaves the prior state (if any)
        untouched. Callers that need document and chunks written together
        (the indexer) should use this instead of the two-call sequence.

        Args:
            source: File path, URL, or other source identifier. Unique. If
                ``source`` already has a document row, that row and all of
                its chunks are deleted before the new ones are inserted.
            document_id: Identifier to store the document under.
            content_hash: Hash of the document's current content.
            chunks: Chunks to persist for this document. May be empty.

        Raises:
            StorageError: A chunk's ``document_id`` does not match
                ``document_id``, a chunk has non-JSON-serializable metadata,
                or a backend failure occurs.
        """
        ingested_at = _now_iso()

        def _op() -> None:
            self._delete_existing_source(source)
            self._conn.execute(
                "INSERT INTO documents (document_id, source, content_hash, ingested_at) "
                "VALUES (?, ?, ?, ?)",
                (document_id, source, content_hash, ingested_at),
            )
            for chunk in chunks:
                if chunk.document_id != document_id:
                    raise StorageError(
                        f"chunk.document_id {chunk.document_id!r} does not match "
                        f"document_id {document_id!r} for source {source!r}"
                    )
            self._insert_chunks(chunks)
            self._bump_generation()
            self._conn.commit()

        await self._run(_op)

    async def get_chunks(self) -> list[Chunk]:
        """Return all persisted chunks in the collection.

        Returns:
            Chunks ordered by document then position, for deterministic
            downstream indexing (e.g. BM25 rebuild).

        Raises:
            StorageError: On a backend failure.
        """

        def _op() -> list[Chunk]:
            cur = self._conn.execute(
                "SELECT chunk_id, document_id, chunk_index, content, start_offset, "
                "end_offset, metadata FROM chunks ORDER BY document_id, chunk_index"
            )
            return [_row_to_chunk(row) for row in cur.fetchall()]

        return await self._run(_op)

    async def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Return one chunk by ID, or None.

        Args:
            chunk_id: The chunk's unique identifier.

        Raises:
            StorageError: On a backend failure.
        """

        def _op() -> Chunk | None:
            cur = self._conn.execute(
                "SELECT chunk_id, document_id, chunk_index, content, start_offset, "
                "end_offset, metadata FROM chunks WHERE chunk_id = ?",
                (chunk_id,),
            )
            row = cur.fetchone()
            return _row_to_chunk(row) if row is not None else None

        return await self._run(_op)

    async def delete_document(self, document_id: str) -> int:
        """Delete a document and its chunks.

        Chunks are deleted explicitly rather than relying solely on the
        ``ON DELETE CASCADE`` foreign key, so the deleted-chunk count is
        known without a second query.

        Args:
            document_id: The document's unique identifier.

        Returns:
            The number of chunks deleted (0 if the document did not exist).

        Raises:
            StorageError: On a backend failure.
        """

        def _op() -> int:
            cur = self._conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE document_id = ?", (document_id,)
            )
            count = int(cur.fetchone()[0])
            self._conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            self._conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
            self._bump_generation()
            self._conn.commit()
            return count

        return await self._run(_op)

    async def write_manifest(self, identity: EmbeddingIdentity) -> None:
        """Write the collection's embedding-identity manifest, once (ADR-0004).

        Called on the collection's first dense write. The manifest is
        immutable for the collection's lifetime thereafter: a later call
        with the *same* ``(provider, model_name, dimensions)`` triple as the
        stored manifest is a no-op — re-ingesting into an already-bound
        collection must keep working — but a call with a *different*
        triple is refused outright. Never a silent overwrite, never a
        re-embed.

        Args:
            identity: The embedding identity establishing (or, on a
                later call, being checked against) this collection's
                identity.

        Raises:
            IndexIdentityError: This store predates the embedding-identity
                manifest (ADR-0004) and cannot be used for dense work, or a
                manifest already exists with a different identity triple.
            StorageError: On a backend failure.
        """
        self._require_manifest_capable("write")
        created_at = _now_iso()

        def _op() -> None:
            existing = self._select_manifest()
            if existing is None:
                self._conn.execute(
                    "INSERT INTO collection_manifest "
                    "(id, provider, model_name, dimensions, created_at) "
                    "VALUES (1, ?, ?, ?, ?)",
                    (identity.provider, identity.model_name, identity.dimensions, created_at),
                )
                self._bump_generation()
                self._conn.commit()
                return
            if _manifest_matches(existing, identity):
                return  # same identity: re-ingesting a bound collection is a no-op
            raise IndexIdentityError(_identity_mismatch_message(existing, identity))

        await self._run(_op)

    async def verify_manifest(self, identity: EmbeddingIdentity) -> CollectionManifest | None:
        """Verify ``identity`` matches the collection's stored identity manifest.

        A collection with no manifest yet — no dense write has ever
        happened — has nothing to conflict with, so verification passes
        trivially; :meth:`write_manifest` is what establishes the manifest,
        not this method. Never a re-embed, never a fallback, never a
        warn-and-continue (SPEC.md §2): a real mismatch always raises.

        Args:
            identity: The active embedding identity to check.

        Returns:
            The manifest this call verified ``identity`` against, or
            ``None`` when the collection has none. The verdict and the
            manifest come out of the *same* read, inside one :meth:`_run`,
            which is what lets a caller decide "is this collection
            dense-bound" without a second read that could observe a
            different state — see ``Retriever.open`` for the concrete race
            this closes.

        Raises:
            IndexIdentityError: This store predates the embedding-identity
                manifest (ADR-0004) and cannot be used for dense work, or
                the stored manifest's identity triple does not match
                ``identity``.
            StorageError: On a backend failure.
        """
        self._require_manifest_capable("verify")

        def _op() -> CollectionManifest | None:
            existing = self._select_manifest()
            if existing is None or _manifest_matches(existing, identity):
                return existing
            raise IndexIdentityError(_identity_mismatch_message(existing, identity))

        return await self._run(_op)

    async def get_manifest(self) -> CollectionManifest | None:
        """Return the collection's embedding-identity manifest, or None if unset.

        Unlike :meth:`write_manifest` and :meth:`verify_manifest`, this is a
        plain read and does not require the store to be manifest-capable: a
        legacy pre-ADR-0004 store never has a manifest row, so it reports
        ``None`` here rather than raising — useful for diagnostics (e.g. a
        future ``index_status`` tool) without needing to catch
        :class:`~groundkit.errors.IndexIdentityError` just to check.

        Raises:
            StorageError: On a backend failure.
        """

        def _op() -> CollectionManifest | None:
            return self._select_manifest()

        return await self._run(_op)

    async def get_generation(self) -> int | None:
        """Return the collection's staleness marker, or ``None`` if unanswerable (ADR-0013).

        The marker advances on every commit that changes durable state, and
        the only operation defined on it is **equality against a previously
        observed value**. Nothing compares generations for ordering or
        distance; a caller that has one may ask "is this still the state I
        built against", and nothing else.

        ``None`` means *freshness cannot be asserted*, never *unchanged*. It
        is returned for a store that is not schema-current — one predating
        ADR-0013, or predating ADR-0004's stamp entirely — and for a v2 store
        whose row is somehow absent. A caller must treat ``None`` as "assume
        it changed" and rebuild: answering a freshness question that cannot
        be answered is how a cache ends up serving an index it has no basis
        to vouch for. This deliberately degrades to reopen-per-request rather
        than refusing, because nothing is *wrong* on a legacy store — only
        uncacheable — and SPEC.md §2's fail-closed rule governs wrong
        answers, not slow ones.

        Returns:
            The current generation, or ``None`` when freshness is
            unanswerable.

        Raises:
            StorageError: On a backend failure.
        """
        if not self._schema_current:
            return None

        def _op() -> int | None:
            cur = self._conn.execute("SELECT generation FROM collection_state WHERE id = 1")
            row = cur.fetchone()
            return None if row is None else int(row[0])

        return await self._run(_op)

    def _delete_existing_source(self, source: str) -> None:
        """Delete the document (and its chunks) currently registered under ``source``, if any.

        Shared by :meth:`upsert_document` and :meth:`replace_document` — both
        replace whatever was previously stored for a source before writing
        the new row. Must be called from within an ``_op`` passed to
        :meth:`_run`; it does not commit.
        """
        cur = self._conn.execute("SELECT document_id FROM documents WHERE source = ?", (source,))
        row = cur.fetchone()
        if row is not None:
            existing_document_id = row[0]
            self._conn.execute("DELETE FROM chunks WHERE document_id = ?", (existing_document_id,))
            self._conn.execute(
                "DELETE FROM documents WHERE document_id = ?", (existing_document_id,)
            )

    def _insert_chunks(self, chunks: list[Chunk]) -> None:
        """Insert ``chunks`` in order. Must be called from within an ``_op``; does not commit.

        Raises:
            StorageError: A chunk's ``metadata`` is not JSON-serializable.
        """
        for chunk in chunks:
            try:
                metadata_json = json.dumps(chunk.metadata)
            except TypeError as exc:
                raise StorageError(
                    f"chunk {chunk.chunk_id!r} has non-JSON-serializable metadata"
                ) from exc
            self._conn.execute(
                "INSERT INTO chunks "
                "(chunk_id, document_id, chunk_index, content, start_offset, "
                "end_offset, content_hash, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.chunk_index,
                    chunk.content,
                    chunk.start_offset,
                    chunk.end_offset,
                    chunk.content_hash,
                    metadata_json,
                ),
            )

    def _select_manifest(self) -> CollectionManifest | None:
        """Read the single manifest row, if present. Must be called from within an ``_op``."""
        cur = self._conn.execute(
            "SELECT provider, model_name, dimensions, created_at "
            "FROM collection_manifest WHERE id = 1"
        )
        row = cur.fetchone()
        if row is None:
            return None
        provider, model_name, dimensions, created_at = row
        return CollectionManifest(
            provider=provider,
            model_name=model_name,
            dimensions=dimensions,
            created_at=created_at,
        )

    def _bump_generation(self) -> None:
        """Advance the ADR-0013 staleness marker. Call from within an ``_op``, before its commit.

        The bump must land in the **same transaction** as the write it
        describes, which is why this is a plain statement inside the caller's
        ``_op`` rather than its own :meth:`_run` call. Issued separately after
        a write, it would open a window in which content is newer than the
        marker — and a reader that observes "unchanged" over changed data is
        exactly the silent staleness the marker exists to prevent. The
        reverse ordering (marker committed, write rolled back) is merely a
        wasted rebuild, so atomicity rather than statement order is what
        makes this safe.

        The invariant is **one bump per commit**, not one per method call:
        every ``_op`` that commits durable state advances the marker, and a
        branch that returns without committing (``write_manifest`` re-called
        with the identity it already holds) changes nothing and correctly
        does not advance it. Tying the bump to the commit rather than to the
        method makes that provable by reading the code instead of relying on
        each caller to classify its own branches.

        The ``ON CONFLICT`` upsert rather than a bare ``UPDATE``: a v2 store
        is seeded at :meth:`open`, but a store whose row is missing for any
        reason still advances correctly instead of silently no-op-ing, which
        an ``UPDATE`` matching zero rows would do without raising.
        """
        self._conn.execute(
            "INSERT INTO collection_state (id, generation, updated_at) VALUES (1, 1, ?) "
            "ON CONFLICT(id) DO UPDATE SET "
            "generation = generation + 1, updated_at = excluded.updated_at",
            (_now_iso(),),
        )

    def _require_manifest_capable(self, action: str) -> None:
        """Guard dense-identity operations against a store that predates them.

        Checked outside :meth:`_run` (like the empty-``chunks`` guard in
        :meth:`add_chunks`): ``_schema_current`` is a plain attribute fixed
        at :meth:`open` and never touches the connection, so there is
        nothing here that needs the lock.

        Args:
            action: Short verb describing the attempted operation (e.g.
                ``"write"`` or ``"verify"``), folded into the error message.

        Raises:
            IndexIdentityError: This store's ``PRAGMA application_id``/
                ``user_version`` do not match groundkit's current stamp —
                it predates ADR-0004, it was written against an older schema
                (ADR-0013 raised the version to 2), or it was never created
                by groundkit at all. Pre-1.0 there is no migration path:
                every index is reproducible from ``grk ingest`` in seconds,
                so the remedy is to delete the collection and re-ingest it,
                not to write code that preserves data this cheap to
                regenerate.
        """
        if not self._schema_current:
            raise IndexIdentityError(
                f"cannot {action} the embedding-identity manifest for the store at "
                f"{self.db_path}: it predates the manifest (ADR-0004) or was not "
                "created by groundkit. Pre-1.0 there is no migration path for dense "
                "data — delete this collection and re-ingest it."
            )

    async def _run(self, fn: Callable[[], _T]) -> _T:
        """Run a synchronous sqlite3 operation off the event loop, serialized.

        On any exception, the connection is rolled back before the exception
        propagates. This matters because the connection uses sqlite3's
        default (legacy) transaction-control mode: an implicit transaction
        opens before the first DML statement and stays open — its writes
        visible to later reads on this same connection, and durable as soon
        as some *other*, unrelated ``commit()`` runs on it — until an
        explicit ``commit()``. Without a rollback here, a multi-statement
        ``_op`` that fails partway through (e.g. one bad chunk in a batch)
        would leave the statements that already ran uncommitted but not
        undone, ready to leak out on the next unrelated commit. Read-only
        ``_op``s roll back too; that is harmless (nothing pending) rather
        than pointless to special-case away.

        Cancellation is the case that dictates the shape of this method.
        Cancelling ``await asyncio.to_thread(fn)`` does not stop ``fn`` — the
        worker thread runs to completion regardless — so neither the rollback
        nor the lock release can be driven off this coroutine unwinding:

        - **The rollback runs on the worker thread**, inside ``_guarded``,
          strictly after ``fn``'s last statement. Rolling back from the event
          loop thread instead would race a worker still executing statements
          on this same connection, potentially undoing part of an operation
          while the rest went on to commit.
        - **The lock is released when the worker finishes**, not when this
          coroutine returns. Releasing on unwind would let the next operation
          start while an abandoned worker was still mid-statement, and
          ``replace_document``'s one-commit atomicity is exactly what that
          would break.
        - ``asyncio.shield`` keeps the worker alive through an outer
          cancellation, so the connection is always left consistent by the
          thread that was using it. The cancellation itself still propagates.

        Raises:
            StorageError: Wraps any ``sqlite3.Error`` raised by ``fn``.
                Errors of other types (e.g. a deliberate ``StorageError``
                raised by ``fn`` itself) propagate unchanged — both trigger
                the same rollback first.
        """

        def _guarded() -> _T:
            try:
                return fn()
            except BaseException:
                try:
                    self._conn.rollback()
                except sqlite3.Error:
                    # Already closed (an operation racing store teardown) or
                    # otherwise unusable. Nothing is left to undo in that
                    # state, and raising here would replace the exception
                    # being propagated with a less informative one.
                    logger.debug("Rollback skipped: connection unusable", exc_info=True)
                raise

        await self._lock.acquire()
        try:
            worker: asyncio.Task[_T] = asyncio.ensure_future(asyncio.to_thread(_guarded))
        except BaseException:
            self._lock.release()
            raise
        worker.add_done_callback(self._release_after_worker)

        try:
            return await asyncio.shield(worker)
        except sqlite3.Error as exc:
            raise StorageError(str(exc)) from exc

    def _release_after_worker(self, worker: asyncio.Future[Any]) -> None:
        """Release the operation lock once the worker thread is done with the connection.

        Wired as :meth:`_run`'s done-callback rather than a ``finally``: on
        cancellation the awaiting coroutine unwinds while the thread is still
        running, and the lock must outlive it. Also retrieves the worker's
        exception so an abandoned failure does not surface as an
        "exception was never retrieved" warning.
        """
        self._lock.release()
        if not worker.cancelled():
            worker.exception()


def _row_to_chunk(row: tuple[str, str, int, str, int, int, str]) -> Chunk:
    """Reconstruct a :class:`Chunk` from a chunk-select query row.

    ``content_hash`` is not read back — it is a computed field derived from
    ``content`` on every :class:`Chunk` construction, never a stored input.
    """
    chunk_id, document_id, chunk_index, content, start_offset, end_offset, metadata_json = row
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        content=content,
        start_offset=start_offset,
        end_offset=end_offset,
        metadata=json.loads(metadata_json),
    )


def _manifest_matches(manifest: CollectionManifest, identity: EmbeddingIdentity) -> bool:
    """True when ``manifest``'s identity triple matches ``identity``'s (ADR-0004).

    Identity is the triple ``(provider, model_name, dimensions)``, checked
    as a triple — never dimensions alone. Vector width by itself is
    insufficient: ``nomic-embed-text`` and ``all-mpnet-base-v2`` are both
    768-dimensional, so a width-only check would admit exactly the model
    swap this whole mechanism exists to reject.
    """
    return (
        manifest.provider == identity.provider
        and manifest.model_name == identity.model_name
        and manifest.dimensions == identity.dimensions
    )


def _identity_mismatch_message(manifest: CollectionManifest, identity: EmbeddingIdentity) -> str:
    """Build the shared ``IndexIdentityError`` message for an identity conflict.

    Names all three fields of both triples explicitly, even when only one
    differs, because the 768-vs-768 case (two models sharing a width) is
    exactly what this check exists to catch — a caller should see all three
    fields compared at once rather than guess which one moved.
    """
    return (
        "embedding identity mismatch: collection was built with "
        f"(provider={manifest.provider!r}, model_name={manifest.model_name!r}, "
        f"dimensions={manifest.dimensions}) but the active configuration is "
        f"(provider={identity.provider!r}, model_name={identity.model_name!r}, "
        f"dimensions={identity.dimensions}). Vector width alone is not identity — "
        "distinct models can share a width. Mixing embedding spaces in one "
        "collection corrupts it silently; the fix is to delete this collection and "
        "re-ingest it under the new embedding configuration, never to re-embed or "
        "fall back automatically."
    )


def validate_collection_name(collection: str) -> None:
    """Validate that ``collection`` is safe to interpolate into a file path.

    This is the first of two containment layers for :meth:`SQLiteMetadataStore.open`
    (the second is the resolved-path check via
    :func:`~groundkit.utils.path_safety.ensure_within_base`): it states the
    contract on the name itself, up front, so a bad value fails with a
    specific message rather than relying solely on the later path check.

    Args:
        collection: Candidate collection name.

    Raises:
        ConfigurationError: ``collection`` is empty or whitespace-only,
            contains a null byte, is ``.`` or ``..``, or contains characters
            outside ``_COLLECTION_NAME_PATTERN`` (which itself excludes path
            separators, drive/UNC prefixes, and absolute paths — all of
            those require a character the pattern disallows).
    """
    if "\0" in collection:
        raise ConfigurationError("collection name must not contain a null byte")
    if not collection.strip():
        raise ConfigurationError("collection name must not be empty or whitespace-only")
    if collection in (".", ".."):
        raise ConfigurationError(f"collection name must not be {collection!r}")
    if not _COLLECTION_NAME_PATTERN.fullmatch(collection):
        raise ConfigurationError(
            f"collection name {collection!r} contains characters outside the allowed "
            "set (letters, digits, '-', '_', '.')"
        )


def _now_iso() -> str:
    """Current UTC time as an ISO-8601 string, for ``ingested_at``."""
    return datetime.now(UTC).isoformat()


def _chmod_best_effort(path: Path, mode: int) -> None:
    """Tighten ``path``'s permissions on POSIX; a tolerated no-op elsewhere.

    The store persists full chunk *content*, so a world-readable database
    file or index directory is a real exposure under a permissive umask
    (e.g. Docker/Kubernetes, Phase 6's target deployment). This is
    defense-in-depth, not a security boundary anything depends on: it never
    raises, and it is gated off on Windows (``os.chmod`` there does not
    express POSIX-style permission bits).
    """
    if os.name == "nt":
        return
    try:
        os.chmod(path, mode)
    except OSError:
        logger.debug("Could not chmod %s", path)

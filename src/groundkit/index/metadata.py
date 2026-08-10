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
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

from groundkit.contracts import Chunk
from groundkit.errors import StorageError

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

#: Schema for the persisted metadata store. ``IF NOT EXISTS`` makes this safe
#: to run on every open, whether the file is fresh or already populated.
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
"""


class SQLiteMetadataStore:
    """Durable store for documents, chunks, and ingest state, backed by SQLite.

    Construct via :meth:`open`, not directly — the constructor takes an
    already-configured connection.

    Attributes:
        db_path: Path to the collection's SQLite file, kept for diagnostics.
    """

    def __init__(self, connection: sqlite3.Connection, db_path: Path) -> None:
        self._conn = connection
        self._lock = asyncio.Lock()
        self.db_path = db_path

    @classmethod
    async def open(cls, index_dir: Path, collection: str) -> SQLiteMetadataStore:
        """Open (creating if absent) the SQLite store for a collection.

        Args:
            index_dir: Directory holding the collection's persisted state.
                Created (with parents) if it does not exist.
            collection: Collection name; backs the file ``<collection>.sqlite3``.

        Returns:
            A ready-to-use store with the schema applied.

        Raises:
            StorageError: The directory cannot be created, the database file
                cannot be opened, or the schema cannot be applied (e.g. a
                corrupted file or a path occupied by something other than a
                regular SQLite file).
        """
        db_path = index_dir / f"{collection}.sqlite3"

        def _connect() -> sqlite3.Connection:
            index_dir.mkdir(parents=True, exist_ok=True)
            _chmod_best_effort(index_dir, 0o700)
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            try:
                conn.execute("PRAGMA foreign_keys = ON")
                conn.execute("PRAGMA journal_mode = WAL")
                conn.executescript(_SCHEMA)
                conn.commit()
            except Exception:
                # Schema application failed (e.g. a corrupted pre-existing
                # file) — close explicitly rather than leaving the handle to
                # CPython refcounting, which is not a guarantee.
                conn.close()
                raise
            _chmod_best_effort(db_path, 0o600)
            return conn

        try:
            connection = await asyncio.to_thread(_connect)
        except (OSError, sqlite3.Error) as exc:
            raise StorageError(f"failed to open metadata store at {db_path}") from exc

        logger.debug("opened metadata store (collection=%s)", collection)
        return cls(connection, db_path)

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
            self._conn.commit()
            return count

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

        Raises:
            StorageError: Wraps any ``sqlite3.Error`` raised by ``fn``.
                Errors of other types (e.g. a deliberate ``StorageError``
                raised by ``fn`` itself) propagate unchanged — both trigger
                the same rollback first.
        """
        async with self._lock:
            try:
                return await asyncio.to_thread(fn)
            except sqlite3.Error as exc:
                await asyncio.to_thread(self._conn.rollback)
                raise StorageError(str(exc)) from exc
            except Exception:
                await asyncio.to_thread(self._conn.rollback)
                raise


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

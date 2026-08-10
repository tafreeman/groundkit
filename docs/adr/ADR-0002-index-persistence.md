# ADR-0002 — Index persistence: SQLite truth, BM25 rebuild-at-open

- **Status:** Accepted (owner, 2026-08-10)
- **Date:** 2026-08-10
- **Deciders:** Andy Freeman (owner)

## Context

ADR-0001 confirmed ARP's verified gap #1 — no persistence across processes —
and decided the BM25 *library* question (pure-Python, ported near-verbatim
over `rank-bm25`). It explicitly deferred the persistence *design* around
that index to this ADR. SPEC.md §5.3 requires a persistence layout of one
index directory per collection, SQLite for documents/chunks/offsets/ingest
state, content-hash dedup on write, and — per SPEC.md §2 — no in-process
state may shadow persisted state (ADR-0001 hazard 7, the `memory.py`
`_key_map` defect this repo exists partly to avoid repeating).

Scope: this ADR covers **persistence design only** — the SQLite schema, the
BM25-rebuild strategy, the incremental-re-ingest trigger, and the
single-connection concurrency model for `SQLiteMetadataStore`. It does not
revisit the BM25 library choice (ADR-0001) or the dense store (Phase 3).

## Decision

1. **SQLite is the durable truth for documents and chunks.** A
   `SQLiteMetadataStore` per collection (`<index_dir>/<collection>.sqlite3`)
   holds `documents(document_id, source, content_hash, ingested_at)` and
   `chunks(chunk_id, document_id, chunk_index, content, start_offset,
   end_offset, content_hash, metadata)`, with `chunks.document_id` a foreign
   key `ON DELETE CASCADE` and an index on `chunks(document_id)`. `metadata`
   is stored as a JSON text column. `PRAGMA foreign_keys=ON` and
   `PRAGMA journal_mode=WAL` are set on every connection open.

2. **BM25 is rebuilt in memory at open, never pickled.** `BM25Index` holds no
   on-disk representation of its own. `BM25Index.from_store(store)`
   constructs a fresh index and calls `index_chunks()` once with every chunk
   read from `SQLiteMetadataStore.get_chunks()`. The postings, document
   frequencies, and length statistics are pure functions of the persisted
   chunk set, so there is nothing to serialize independently — and nothing
   that can drift out of sync with SQLite the way `memory.py`'s in-process
   `_key_map` drifted from its backing store.

3. **Incremental re-ingest is content-hash gated.** `upsert_document` records
   a `content_hash` per `source`; the ingestion pipeline (outside this
   module's scope) calls `get_document_hash(source)` first and skips
   re-chunking/re-embedding when the hash is unchanged. When a source *is*
   re-ingested, `upsert_document` explicitly deletes the prior document row
   and its chunks (by the previously-registered `document_id`, which may
   differ from the new one) before inserting the replacement — explicit
   deletion rather than relying solely on the cascade, so the deleted state
   is never implicit in a `INSERT OR REPLACE` conflict-resolution side effect.

4. **Concurrency: one connection, one asyncio.Lock.** `SQLiteMetadataStore`
   opens a single `sqlite3.Connection` with `check_same_thread=False` and
   serializes every operation through one `asyncio.Lock`, held for the
   duration of the coroutine including the awaited `asyncio.to_thread` call.
   sqlite3 connections are not safe to share across OS threads without this
   guard, and `asyncio.to_thread` dispatches each call to a fresh worker
   thread from the default executor. The lock, not `check_same_thread=False`
   alone, is what makes concurrent `await`s from multiple coroutines safe.

## Alternatives considered

- **Persisted BM25 postings tables in SQLite** (term → document → frequency
  rows, avoiding the O(corpus) rebuild). Deferred, not rejected outright: it
  adds real complexity (keeping postings consistent with `chunks` under
  concurrent writes, migration surface) for a cost that is currently
  hypothetical. Revisit if rebuild-at-open time is *measured* to be a
  problem for a real corpus size — that measurement is the trigger, not a
  guess made now.
- **Pickled BM25 index files.** Rejected: `pickle.load` on an index file is
  unsafe deserialization of untrusted-by-default input (the file could have
  been swapped or corrupted between writes), the format is opaque to
  anything but this exact Python/BM25Index version, and a version bump to
  `BM25Index`'s internal shape silently breaks every existing pickle with no
  schema check. SPEC.md's fail-closed principle argues against a persistence
  format with no validation step.
- **LanceDB for everything, including chunk/document metadata.** Rejected
  for this layer: document/chunk truth wants relational queries (`upsert`
  by unique `source`, `ON DELETE CASCADE` to keep chunks and documents in
  lockstep, indexed lookups by `document_id`) that a vector table is not
  designed to express or enforce. SPEC.md §3 already commits to LanceDB +
  SQLite as separate concerns behind separate interfaces; this ADR does not
  reopen that split, only the SQLite side of it.

## Consequences

**Positive:** the metadata store is durable and independently verifiable —
closing and reopening a `SQLiteMetadataStore` against the same file recovers
every document and chunk, closing exactly the gap ARP's gap-audit claim #1
confirmed. BM25 can never hold state that SQLite doesn't also have, by
construction — there is no BM25-only write path. Incremental re-ingest is a
single indexed lookup (`source` is `UNIQUE`), not a corpus scan.

**Negative:** `BM25Index.from_store` is an O(corpus) rebuild on every process
start — for a large collection this is a real, measurable startup cost that
this ADR accepts for v1 rather than building the postings-persistence
alternative above pre-emptively. The single-connection-plus-lock concurrency
model serializes all metadata-store access; under WAL, SQLite itself would
allow concurrent readers, but groundkit does not exploit that concurrency in
v1 — acceptable for a single-process local service, revisited only if
metadata-store contention is measured to matter.

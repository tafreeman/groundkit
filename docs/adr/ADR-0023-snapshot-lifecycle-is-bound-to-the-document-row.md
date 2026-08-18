# ADR-0023 — A snapshot's lifetime is bound to its document row

- **Status:** Accepted (owner, 2026-08-17)
- **Date:** 2026-08-17
- **Deciders:** Andy Freeman (owner)

## Context

ADR-0016 decision 4 gave URL ingestion a local snapshot: the fetched, decoded text is
written under `<index_dir>/<collection>.snapshots/<document_id>`, and
`resolve_citation`'s `snapshot` branch reads it back rather than re-fetching, because a
re-fetch is a different observation at a different time and a mismatch could not
distinguish "the index is stale" from "the server changed" from "the network lied".

That ADR closed the verifiability question and explicitly left one open:

> URL ingestion consumes disk proportional to what is fetched, under the index directory.
> Retention and cleanup for those snapshots are **not** decided here and are owed before
> any deployment that is not a single user's local machine, in the same sense SPEC.md §7
> already says of SQLite.

Phase 6 made exactly such a deployment real — a Kubernetes PVC and an EBS volume carrying
`prevent_destroy`. The decision came due, and an audit found that in the absence of one
the implementation had defaulted to the worst available answer: **snapshots were written
but never removed, by any code path at all.** Two concrete defects followed, neither of
which is a coding slip so much as the absence of this decision:

1. **`UrlLoader.load()` writes the snapshot itself, before the caller can decide whether
   the document will be stored.** `Indexer._process` computes a processing fingerprint
   (ADR-0009) and skips an unchanged document — but the loader has already written a file
   by then, and `Document.document_id` defaults to a fresh `uuid.uuid4().hex` per load
   (`contracts.py`), so the name is different every time. Re-running `grk ingest <url>` —
   the documented incremental workflow, and the one an operator is most likely to
   automate — left a full, permanently orphaned copy of the fetched text on every run.
   Nothing referenced it, nothing reported it, and nothing would ever remove it.

2. **Deleting a document left its snapshot on disk.** `_delete_document_everywhere`
   removes vectors and then the SQLite row (ADR-0004 decision 6); `_prune_emptied_source`
   routes through it. Neither touched the snapshot. The full text of a fetched remote
   document therefore survived an explicit deletion, indefinitely, on a volume the
   operator believed they had cleared.

The second is the sharper one. A user who deletes a document has stated an intent about
*content*, and honouring it only in the index while the bytes remain in a sibling
directory is the kind of gap that turns a retention policy into a false statement.

## Decision

### 1. A snapshot exists if and only if a `documents` row references it

This is the whole policy, and everything below follows from it. The snapshot is not an
independent cache with its own expiry; it is part of the on-disk footprint of one
document, in the same way that document's chunk rows and its dense vectors are. SPEC.md §7
already treats SQLite as content-bearing data rather than a rebuildable index; a snapshot
is content-bearing for the same reason and by a stronger argument, since it is the *only*
copy of what was fetched.

Rejected alternative: time-based or size-based retention (an LRU cap, a TTL). Both make a
citation's resolvability depend on when it is asked rather than on whether its document is
still indexed — a `snapshot` citation would resolve on Monday and report `unresolvable` on
Friday with nothing having changed in the index. That trades a disk-usage problem for a
correctness problem, and this repo's whole claim is that a citation either verifies or
says why it cannot.

### 2. Cleanup is the `Indexer`'s job, not the loader's

The loader writes the snapshot because only the loader has the bytes. But only the
`Indexer` knows whether the document was stored, replaced, skipped or deleted, so only the
`Indexer` can decide when a snapshot has become unreferenced. `Indexer` therefore takes an
optional `snapshot_dir`, and removes a snapshot at the three points where a document stops
referencing one:

- **skipped as unchanged** — the loader wrote a snapshot under a throwaway `document_id`;
  the stored document keeps its own, which is still byte-correct because the fingerprint
  matched. The new one is removed.
- **replaced** — the prior document's snapshot is removed *after* `replace_document`
  commits, so a failed replace never strands the row that is still pointing at it.
- **deleted or pruned** — removed after the SQLite row is gone, for the same ordering
  reason.

`snapshot_dir=None` (the default, and what every `FileLoader` caller passes) disables all
of it. A collection that never ingested a URL has no snapshot directory and pays nothing.

### 3. Deletion is best-effort and logged, never fatal to the ingest

A snapshot that cannot be unlinked — a permission error, a file already gone, a
Windows share-violation — is a warning naming the document, not a raised exception. The
ordering in decision 2 is what makes this safe: the durable state (SQLite, vectors) is
already correct by the time the unlink is attempted, so a failed unlink leaves disk
litter, which the next ingest of the same source will retry, and never an index that
disagrees with itself. Raising instead would fail an otherwise-complete ingest over
cleanup of a file that no longer matters.

### 4. The path is containment-checked before every unlink

`snapshots.snapshot_path_for` performs no containment check and says so: `document_id` is
a plain string field with no character class, so it is attacker-influenced in principle.
Reading a snapshot already guards against this (`resolve_citation` runs the path through
`ensure_within_base`). Unlinking is strictly more dangerous than reading, so it takes the
same guard — `is_within_base`, non-raising to match decision 3 — before it touches the
filesystem. A `document_id` that escapes the snapshot directory is refused and logged, not
followed.

### 5. Content-derived snapshot names were rejected

The obvious alternative fix for defect 1 is to name the snapshot after a hash of its
content, so a re-fetch of unchanged bytes overwrites in place and nothing accumulates.
It is rejected because `resolve_citation` locates a snapshot by `citation.document_id`
(`citations.py`, via `snapshots.snapshot_path_for`), so changing the naming convention
changes the read side too — and every snapshot written before the change becomes
unfindable, which ADR-0004 decision 5 says is answered by delete-and-re-ingest rather than
migration. Paying a re-ingest of every remote document to avoid passing one directory into
`Indexer` is the wrong trade. It also would not have fixed defect 2 at all.

## Alternatives considered

- **Move the snapshot write out of the loader entirely**, into the `Indexer` after the
  fingerprint gate. This is arguably the cleanest shape, and it fixes defect 1 by
  construction rather than by compensation. Rejected for now because it changes the
  `LoaderProtocol` contract — the loader would have to return the bytes *and* a promise
  that someone else will persist them — and because `UrlLoader` is usable standalone,
  where nothing would ever write the snapshot. Worth revisiting if a second snapshotting
  loader appears; recorded here so that revisit starts from the reasoning rather than
  rediscovering it.
- **A `grk gc` / `grk vacuum` verb** that sweeps unreferenced snapshots. Rejected as the
  primary mechanism: it makes correctness depend on an operator remembering to run
  something, and defect 2 (content surviving deletion) would persist until they did. A
  sweep remains a reasonable *addition* later for litter left by decision 3's best-effort
  failures.
- **Refusing to skip an unchanged URL document**, so the snapshot write is always
  meaningful. Rejected: it would re-chunk, re-embed and rewrite every URL document on
  every run, discarding ADR-0009's incremental skip for a disk-hygiene reason.

## Consequences

- `Indexer.__init__` gains an optional keyword-only `snapshot_dir`. Existing callers are
  unaffected; `grk ingest <url>` passes the directory it already computes for `UrlLoader`,
  so the write side and the cleanup side cannot disagree about where snapshots live.
- `_persist_document` now resolves the prior document id on the BM25-only path too when
  snapshot cleanup is enabled — one extra `get_document_id` per document, and only for a
  collection that stores snapshots.
- SPEC.md §7's statement that deleting a collection means deleting its `.sqlite3` *and*
  its `.lance` sibling now names a third artifact, the `.snapshots` directory. ADR-0013
  decision 9's standing obligation on a future `delete_collection` inherits the same third
  artifact.
- Disk usage for URL ingestion becomes proportional to the *stored* corpus rather than to
  the number of times ingest has been run.
- This closes the retention question ADR-0016 left open for the document-level case. It
  does **not** decide collection-level deletion, which has no implementation to attach to
  yet (there is still no `delete_collection`), and it does not decide backup scope —
  `docs/guides/deployment.md`'s backup paragraph names only the SQLite store and should
  name all three artifacts.

## References

- ADR-0016 — citation verifiability for extracted and remote sources (decision 4; the
  Consequences paragraph deferring retention)
- ADR-0009 — the incremental skip key is a processing fingerprint
- ADR-0004 — embedding identity binding (decision 5, no migration; decision 6, deletion is
  verified by count)
- ADR-0013 — collection runtime persisted staleness marker (decision 9)
- SPEC.md §7 — SQLite is content-bearing data, not just an index

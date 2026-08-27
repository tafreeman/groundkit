# ADR-0027 — The rebuild-cliff reading selects persisted postings, and retires GK-020's incremental rebuild

- **Status:** Accepted (owner, 2026-08-27)
- **Acts on:** ADR-0026 decision 5's trigger, which required a reading before either
  remedy could be chosen
- **Amends:** ADR-0026's re-deferral of ADR-0002's persisted postings; re-scopes
  `BACKLOG.md`'s GK-020
- **Date:** 2026-08-27
- **Deciders:** Andy Freeman (owner)

## Context

ADR-0026 landed rebuild observability first and then re-deferred ADR-0002's persisted
postings against a trigger it stated in full:

> **New trigger.** A recorded reading of `index_status`'s `retriever_acquires` /
> `retriever_rebuilds` / `rebuild_seconds_total` / `last_rebuild_seconds` from a real
> corpus under a real workload, quoted in the ADR that acts on it. No threshold is
> written here; writing one now would be the guess ADR-0002 refused to make.

It also made the reading *decide the remedy*, which is the reason this record exists
rather than a commit implementing GK-020's remaining criteria:

> - `retriever_rebuilds` low while `last_rebuild_seconds` is large → few
>   invalidations, each expensive. The corpus is the problem; **persisted postings**
>   (ADR-0002's alternative) is the answer.
> - `retriever_rebuilds` tracking the ingested document count, with
>   `rebuild_seconds_total` dominated by many small rebuilds → the invalidation
>   *rate* is the problem; **incremental rebuild** (GK-020 criteria 2–3) is the
>   answer.

That reading had never been taken. GK-020 criteria 2–3 describe the second remedy.
This record reports the reading, which selects the first.

## The reading

Taken 2026-08-27 on the owner's Windows workstation, against a corpus built by fanning
out this repository's own `docs/` tree, ingested BM25-only. Load is `handle_search` —
the operation that calls `CollectionRuntime.acquire()` — driven in-process so that
per-request process startup cannot mask the commit burst. The writer is a real
`grk ingest` subprocess over documents mutated first, because ingestion skips unchanged
files by content hash and **a skip does not bump the generation**; re-ingesting an
untouched corpus reports a perfect hit rate that means nothing.

Counters are deltas across the ingest window, read from `CollectionRuntime.rebuild_stats()`
— the same values `index_status` surfaces.

| Corpus | Docs changed (= commits = bumps) | Acquires | Rebuilds | Cache-hit rate | Mean rebuild | Rebuild s in window | Ingest wall |
|---|---|---|---|---|---|---|---|
| 102 docs / 3,349 chunks | 40 | 305 | **2** | 99.3% | 165 ms | 0.329 s | 1.02 s |
| 510 docs / 16,580 chunks | 100 | 75 | **2** | 97.3% | **1,208 ms** | **2.417 s** | 1.74 s |

These are a dated observation on one machine, not a standing claim, and nothing in the
docs derives a number from them. They are quoted here because ADR-0026's trigger
requires exactly that.

Three things follow, and the second is the one that decides this record:

1. **Rebuilds do not track the bump count.** Two rebuilds whether 40 or 100 documents
   changed — 0.05 and 0.02 rebuilds per bump. A real ingest spends most of its wall
   time loading and chunking; the commits land in a short burst, so most acquires find
   the generation unchanged since the last rebuild.
2. **Each rebuild is expensive and grows with the corpus** — 165 ms at 3.3k chunks,
   1,208 ms at 16.6k. At the larger size the window's two rebuilds cost **more wall
   time than the entire ingest that provoked them**, and the load loop's completed
   searches fell from 305 to 75 as a direct result, because ADR-0013 decision 5 forbids
   serving a waiter stale.
3. **The cache is not broken.** The hit rate stayed above 97% in both runs.

### Why this contradicts GK-020's stated premise, and which instrument to believe

GK-020 records that "during an ingest the hit rate approaches zero." Measured, it did
not: 99.3% and 97.3%.

That claim traces to `scripts/measure_retriever_open.py --sections acquire`, which
reports `cache hits=0` at every corpus size. Both are correct measurements of different
things, and the difference is not noise:

- The script's `acquire` section commits rows **back to back with no intervening work**
  (its own docstring notes a commit there is cheaper than a real one, "so the contention
  it shows is a floor"). Every acquire therefore lands between two bumps. That is a
  synthetic worst case and a legitimate floor for contention.
- A real ingest interleaves loading, chunking and hashing between commits, so the
  window in which an acquire is guaranteed to find a bumped generation is a small
  fraction of the ingest.

The script's reading is the right instrument for *cost per rebuild*. It is the wrong
instrument for *invalidation rate*, and invalidation rate is the quantity ADR-0026's
decision table branches on. GK-020's premise took the floor for the expected case.

## Decision

### 1. The reading selects persisted postings; GK-020 criteria 2–3 are retired

Rebuilds are few and do not scale with the write volume; each is expensive and scales
with the corpus. That is ADR-0026's first row verbatim, so ADR-0002's persisted postings
is the indicated remedy and **incremental rebuild is not**.

This is not a deferral of criteria 2–3. A watermark, `get_chunks_since`, and a
`remove_document` on the lexical index attack the invalidation *rate*, which this reading
shows is not the binding cost — two rebuilds per ingest would remain two rebuilds. They
are withdrawn rather than re-deferred, and `BACKLOG.md` records the withdrawal with the
reading as its reason, so a later reader does not restore them as an oversight.

The cost avoided is worth stating, because it is what made criteria 2–3 expensive:
`SCHEMA_VERSION` is at 3, and a v4 that adds a watermark column has the same shape as
v3 — `CREATE TABLE IF NOT EXISTS` cannot supply a column to an existing table — so it
would force delete-and-re-ingest on every existing collection under ADR-0004 decision 5.
`BM25Index` also has no `remove_document`, and its postings are keyed by position in the
chunk list, so removal renumbers every posting after it. That is an index redesign
carrying GK-018's score-identity obligation. None of that is now owed.

### 2. Persisted postings is *indicated*, not adopted here

This record does not adopt it, and accepting this record does not adopt it either.
Adopting it reverses ADR-0002's central choice — that
SQLite is the sole durable truth and BM25 postings are never persisted, precisely so no
second on-disk structure can drift from the chunk set — and that reversal deserves its
own record with its own alternatives, not a paragraph inside a measurement report.

What this record does is discharge the trigger: the reading exists, it is quoted, and it
points at one remedy rather than the other.

### 3. The threshold question stays open, deliberately

ADR-0026 declined to write a threshold and this record does the same. At 3.3k chunks a
165 ms rebuild twice per ingest is defensible; at 16.6k chunks a 1,208 ms rebuild that
outlasts the ingest is not obviously so, and the growth is roughly linear in chunk count.
ADR-0026 explicitly permitted the outcome that neither remedy is worth its complexity —
"that is a result, not a failure" — and at the smaller corpus that remains a defensible
reading of these numbers. What has changed is that the question is now answerable by
re-running a script rather than by argument.

## Consequences

- GK-020 closes as measured rather than as built. `BACKLOG.md` keeps a single item
  pointing at the persisted-postings decision, not at a watermark.
- `KNOWN_LIMITATIONS.md` should stop implying the ingest-time hit rate collapses; the
  honest statement is that rebuild *cost* grows with the corpus while rebuild *count*
  stays low.
- `scripts/measure_retriever_open.py --sections acquire` keeps its 0%-hit-rate result,
  which is correct for what it measures. Its docstring is the right place to say that it
  reports a contention floor rather than an expected-case rate, so the next reader does
  not repeat GK-020's inference.
- Re-taking this reading is cheap and should precede any postings work, since the
  numbers above are one machine and two corpus sizes.

## Alternatives considered

**Build criteria 2–3 anyway, because they were already specified.** Rejected: it is the
guess ADR-0026 exists to prevent, and the reading indicates it would not move the binding
cost. Two rebuilds per ingest would still be two rebuilds; only their price would change,
and incremental rebuild changes that price by less than postings does.

**Adopt persisted postings in this record.** Rejected: see decision 2. Reversing ADR-0002
needs its own alternatives section, particularly on how a persisted postings table is
kept from drifting from the chunk set it derives from — the exact failure ADR-0002 chose
rebuild-at-open to make structurally impossible.

**Write a threshold now.** Rejected for the reason ADR-0026 gave, which has not changed.

**Treat the `acquire` section's 0% as the reading and skip this.** Rejected: it measures a
contention floor under back-to-back commits, not the invalidation rate under a real
ingest, and ADR-0026's decision table branches on the latter.

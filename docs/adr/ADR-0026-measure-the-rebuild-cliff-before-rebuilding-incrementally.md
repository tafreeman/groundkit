# ADR-0026 — Measure the staleness cache's rebuild cliff before rebuilding incrementally, and re-defer ADR-0002's persisted postings against a trigger the measurement can satisfy

- **Status:** Accepted (owner, 2026-08-19)
- **Date:** 2026-08-19
- **Deciders:** Andy Freeman (owner)

## Context

[ADR-0013](ADR-0013-collection-runtime-persisted-staleness-marker.md) decision 1
advances a collection's generation counter **once per commit**, and its own text
argues that rule at length: tying the bump to the commit rather than to a method
is what makes it checkable by reading the code, and over-bumping costs one
redundant rebuild while under-bumping serves stale results silently and forever.
That reasoning is not reopened here. It is correct as specified.

What it did not record is a consequence of composing it with how `grk ingest`
writes. The ingest path commits once per document — `replace_document` is one
`_op`, one commit, one bump — so an ingest over N changed files publishes N
distinct generations. `CollectionRuntime.acquire`'s validity predicate is
equality against the generation the cached artifact was stamped with, so **every
one of those N generations invalidates the cache**, and each invalidation costs
a full `BM25Index.from_store` rebuild over the whole corpus, taken while holding
the same metadata-store lock the ingest writer is contending for. ADR-0013
decision 5 additionally rules that a request arriving during a rebuild waits
rather than being served the stale artifact, so concurrent requests do not
amortize: each absorbs the rebuild.

Stated as a shape rather than a number: for the duration of an ingest, the cache
hit rate tends toward zero, the fallback is the reopen-per-request baseline
ADR-0013 rejected on measurement, and the contention runs in both directions —
the reads slow the ingest that is invalidating them.

**That paragraph is an argument, not a reading, and nothing in the process could
turn it into one.** This is the pivot the whole ADR turns on. A running
groundkit service exposed the *availability* of the cache — `index_status`
reports `generation` and `cache_enabled` — and nothing at all about its
*behaviour*. `cache_enabled: true` is compatible with a cache that hits on every
request and with one that has not hit since the process started. The only
external symptom of the difference is latency, which on this path has several
other causes (an O(corpus) aggregate, a cold page cache, a dense probe, a
reranker). SPEC.md §2's rule that no doc may carry a number that was not
generated has a sharper corollary when the number is the basis for a design
decision: a remedy chosen against an unmeasured cost is chosen against a guess,
and the backlog entry that raised this said so itself — *"GK-019 and GK-020
should follow measurement rather than this file's guess."*

**Three different remedies are in play and they are routinely conflated.** They
attack different terms of the same product, so distinguishing them is the
precondition for choosing one:

| Remedy | Reduces | Leaves alone |
| --- | --- | --- |
| Persisted BM25 postings ([ADR-0002](ADR-0002-index-persistence.md)'s deferred alternative) | the cost *of* a rebuild | how many rebuilds happen |
| Incremental rebuild (a watermark, `get_chunks_since`, `remove_document`) | how much each rebuild *reads* | how many rebuilds happen |
| A coarser bump (per ingest run rather than per document) | how many rebuilds happen | the cost of each one |

ADR-0013's own alternatives section already ruled that persisted postings are
"not an alternative" to the staleness marker — they attack rebuild cost, not
staleness, and the two compose. The same sentence, read the other way, is why
they are very much an alternative *to incremental rebuild*: both spend
complexity to make an invalidation cheaper, and a reading of the counters below
distinguishes which one the workload is actually asking for.

**ADR-0002's deferred alternative is half-triggered, not triggered.** Its
recorded trigger is "revisit if rebuild-at-open time is *measured* to be a
problem for a real corpus size — that measurement is the trigger, not a guess
made now." `scripts/measure_retriever_open.py` measured the per-open cost for
ADR-0013, which settles the first factor. Whether that cost is *a problem*
depends on the second factor — how often an open happens — and no instrument in
this repo could observe that. So the trigger has been sitting in a state its
author did not anticipate: not unmet, but unmeetable, because half of what it
asks for was unobservable by construction. Leaving it in that state is the thing
this ADR must not do.

## Decision

### 1. Rebuild observability lands first, as in-process counters on `CollectionRuntime`

`CollectionRuntime` records four values and hands them out as a frozen
`RebuildStats` snapshot from a synchronous `rebuild_stats()`:

- `acquires` — every `acquire()` past the closed check.
- `rebuilds` — every rebuild body that *started*.
- `rebuild_seconds_total` — wall-clock seconds inside those bodies.
- `last_rebuild_seconds` — the most recent one, or `None`.

Four choices inside that are load-bearing, and each rejects a plausible
implementation that would have reported the wrong thing:

**`acquires` exists because a rebuild count has no denominator.** The quantity
the cliff is *about* is a fraction of requests, and `rebuilds` alone cannot
express one: the same value is consistent with a runtime that served one request
and one that served a thousand. Counting acquires before the fast-path return,
rather than after, is what makes `rebuilds / acquires` the fraction of requests
that paid — counting after would make the denominator "requests that missed",
against which the ratio is identically 1.

**The ratio is derived by the reader, never stored.** This is
`evals/schema.py`'s rule about stage deltas applied unchanged: a ratio persisted
beside the two numbers that determine it is redundant state that can disagree
with them.

**`rebuilds` counts at entry, not at publication.** A counter advanced next to
the `self._cached` assignment reports a runtime whose every rebuild raises as
one that never rebuilds: count zero, total zero, a perfect hit rate — while it
is in fact taking the rebuild lock and doing an O(corpus) read on every single
request. The cost is highest exactly where that version reports none. For the
same reason the timer is stopped in a `finally`, so a failed rebuild is charged
for the lock it held and the waiters it blocked.

**`last_rebuild_seconds` is kept alongside the total** because a total divided by
a count is an average, and an average hides the spike that one O(corpus) rebuild
is — which is the shape of the event a tail-latency complaint is actually about.

Timing uses `time.perf_counter`. A wall clock can step backwards, and a negative
duration added to a monotonically growing total is the one arithmetic a
cumulative counter cannot survive.

### 2. The counters are process-local and are deliberately not persisted

They describe one `CollectionRuntime` object's history, not the collection.
Persisting them was rejected on three independent grounds, any one of which is
sufficient, and the third is the interesting one:

- It needs a schema bump to hold them, which under ADR-0004 decision 5 means
  every existing collection must be deleted and re-ingested — a real cost, paid
  for observability.
- It puts a **write** on the path of an unauthenticated read. That is the same
  defect `list_collections` closed when it stopped confirming a candidate
  collection by opening it, and `index_status` closed when the registry began
  refusing a collection that does not exist rather than creating it.
- It is self-defeating in both branches. A write to `collection_state` that
  bumped the generation would invalidate, on every request, the very cache the
  counter exists to measure — the observer would create the phenomenon. A write
  that did *not* bump would be a mutating store `_op` with no bump in it, which
  breaks ADR-0013 decision 1's structural rule that the bump lives inside the
  store's mutating `_op`s precisely so no caller has to be trusted to classify
  its own branches. There is no third branch.

The consequence is stated rather than hidden: a runtime evicted by
`CollectionRegistry`'s LRU bound and later reopened starts from zero, and so
does a restarted process. These are not lifetime totals for a collection, and
`KNOWN_LIMITATIONS.md` records that.

### 3. `index_status` is the surface, and reading the meter must not move it

The four values are reported as `retriever_acquires`, `retriever_rebuilds`,
`rebuild_seconds_total` and `last_rebuild_seconds` on `IndexStatusResponse`.
That operation already holds the runtime and already reports `generation` and
`cache_enabled`, so the marker's *cost* belongs beside the marker's
*availability*; the read is four attributes off an object already checked out,
with no store round-trip.

`handle_index_status` builds no retriever and must not start. An implementation
that called `runtime.acquire()` — for a retriever it does not need, or by
copying the shape of `handle_search` — would inflate its own denominator on
every call and drive the reported hit rate toward 1.0 in proportion to how often
anyone looked. A monitor that improves the number it reads is worse than no
monitor, and a test asserts that two consecutive `index_status` calls leave
`retriever_acquires` unmoved.

Nothing new is disclosed. A rebuild happens exactly when the generation moved,
and `generation` is already on this same response — a strictly finer-grained
signal of the same write activity. No added field names a source, a query, a
path, or a byte of content, so `IndexStatusResponse`'s standing promise holds
unchanged, including against the OpenAPI walk that enforces it.

No fifth service operation is added: SPEC.md §1.2 names four tools and a test
pins that set, so a fifth is a deliberate act owing its own argument, and this
data does not need one.

### 4. Incremental rebuild and its schema bump are **not** taken in this change

GK-020's second and third acceptance criteria — a monotonic per-document
watermark, `get_chunks_since`, `remove_document` on the lexical index, and the
`SCHEMA_VERSION` bump to carry the watermark — remain open. Two reasons, and the
second is the one that would have been discovered late:

**A partial schema bump strands real indexes.** `SCHEMA_VERSION` is already at 3,
and v3 is a harder break than v2 was: v2 added a *table*, which
`CREATE TABLE IF NOT EXISTS` supplies to an older store transparently, while v3
added *columns to an existing table*, which it does not — so a pre-v3 store has
its writes refused. A watermark column is v3's shape exactly. A v4 landed
halfway has the full blast radius and none of the benefit, so it is taken once,
whole, with its tests, or not at all.

**The watermark answers only half the question.** "Which chunks arrived after
generation G" is the easy half. The other half is which documents *left*:
`replace_document` deletes then inserts, and `delete_document` removes outright,
and no ascending watermark can represent a row that is gone. That is why GK-020
names `remove_document` on the lexical index, and why that method is the real
work. `BM25Index`'s postings map — added by GK-018 and recorded as an erratum
against ADR-0002 decision 2 — maps a term to *positions in `self._chunks`*, so
removing a chunk from the middle invalidates every stored position above it.
Removal therefore needs either tombstoning inside the index (with `_doc_freqs`,
`_doc_lengths` and `_avg_doc_length` all decremented consistently, plus a
compaction policy) or a position remap, and whichever is chosen must be
**score-identical** to a full rebuild — because if it is not, ADR-0002 decision
2's invariant that the in-memory index is a pure function of the persisted chunk
set becomes false, and that invariant is this repo's structural guard against
repeating ARP's `memory.py` `_key_map` drift (ADR-0001 hazard 7). That is a
redesign of the index's internals, not an added method.

### 5. ADR-0002's deferred alternative is re-deferred, against a new trigger

Persisted BM25 postings tables in SQLite are **not adopted**, and the old trigger
is **replaced** rather than merely restated, because the old one asked for
something that could not be observed:

> **New trigger.** A recorded reading of `index_status`'s
> `retriever_acquires` / `retriever_rebuilds` / `rebuild_seconds_total` /
> `last_rebuild_seconds` from a real corpus under a real workload, quoted in the
> ADR that acts on it. No threshold is written here; writing one now would be
> the guess ADR-0002 refused to make.

The trigger is deliberately "a reading exists and is quoted", not "a number
exceeds a constant". What changed today is not that the cost got worse — it is
that the reading became *takeable*, which is what the old trigger silently
presupposed.

The reading also **decides which remedy applies**, which is the concrete payoff
of sequencing observability first:

- `retriever_rebuilds` low while `last_rebuild_seconds` is large → few
  invalidations, each expensive. The corpus is the problem; **persisted
  postings** (ADR-0002's alternative) is the answer.
- `retriever_rebuilds` tracking the ingested document count, with
  `rebuild_seconds_total` dominated by many small rebuilds → the invalidation
  *rate* is the problem; **incremental rebuild** (GK-020 criteria 2–3) is the
  answer.
- Both large → they compose, and the order is postings first, since incremental
  rebuild over persisted postings is a smaller change than the reverse.

A remedy that does not survive its own reading is the outcome this sequencing
exists to permit: it is entirely possible that on a real corpus the ingest-time
cliff is short enough that neither is worth its complexity, and that is a result,
not a failure.

### 6. No new Protocol seam, so no conformance entry is owed

ADR-0013 decision 10 requires a signature-parity entry in
`tests/test_protocol_conformance.py` for any `typing.Protocol` the runtime or
registry introduces. This change introduces none: `RebuildStats` is a frozen
dataclass and `rebuild_stats()` is a concrete method on a concrete class.
`MetadataStoreProtocol`, `DocumentRecordStoreProtocol`, `VectorStoreProtocol`
and `LexicalIndexProtocol` are untouched, and no store double needs an edit.
Recorded because the obligation is standing, and "we checked and none was owed"
is a different statement from silence.

## Alternatives considered

- **Land the watermark, `get_chunks_since`, `remove_document` and the v4 bump
  now.** Rejected per decision 4: a half-landed column bump has v3's full blast
  radius, and the removal half is a redesign of `BM25Index`'s internals with a
  score-identity obligation, not an added method. Deferred with the work order
  intact, not dropped.
- **Persist the counters in `collection_state`.** Rejected per decision 2 — and
  most sharply because both branches fail: bumping the generation on the write
  makes the observer create the phenomenon it measures, and not bumping breaks
  ADR-0013 decision 1's structural rule.
- **Emit the counters as OpenTelemetry spans or metrics instead.** Not taken as
  the primary surface. ADR-0022 makes the OTel *SDK* an extra, so on a default
  install these numbers would not exist, while `index_status` needs no extra and
  is already the operation an operator calls to ask about a collection. A span
  around the rebuild is compatible with everything decided here and is a later
  addition, not a substitute — and it would arrive under ADR-0022's attribute
  allowlist, which no counter here violates.
- **A fifth read-only service operation carrying runtime statistics.** Rejected
  per decision 3: SPEC.md §1.2's four tools are pinned by a test so that a fifth
  is a deliberate edit, and this data belongs on the operation that already
  reports the marker.
- **Bump the generation once per ingest *run* rather than per document.** This
  is the fix that looks obvious, and it is recorded because it looks obvious. It
  requires the store to know an ingest is in progress — transaction-scope
  semantics on `MetadataStoreProtocol`, which ADR-0013 decision 4 already
  declined to add for an optimization on a signature-parity-tested seam — and it
  widens the staleness window from one commit to one entire ingest run, so a
  service would serve results missing every document ingested so far until the
  run finished. ADR-0013 chose bounded-and-conservative over fast-and-stale on
  exactly this axis; this trades in the opposite direction on the same axis
  while pointing at a cost nobody has measured.
- **Serve the stale artifact for the duration of a rebuild, behind a flag.**
  Not reopened: ADR-0013 decision 5 rejected it because a flag nobody is
  required to read is not fail-closed, and nothing in this change bears on that
  argument.
- **Debounce or coalesce rebuilds with a short timer.** Rejected for TTL's
  reason, restated at a smaller scale: it serves stale results for the length of
  the window after every commit, with a staleness bound nobody chose on
  evidence. The evidence is what this ADR is for.

## Consequences

**Positive.** The cliff is a reading rather than an argument, taken from an
operation that already exists, needing no extra dependency and no schema change.
The choice between ADR-0002's persisted postings and GK-020's incremental
rebuild now has a decision rule stated in terms of observable quantities, so the
next ADR on this subject argues from data. ADR-0013 is unmodified: the marker,
the bump rule, the stamp ordering, the single-flight and the wait-rather-than-
serve-stale rule all stand exactly as written, and nothing on the retrieval path
changed.

**Costs.** Four integers and a float per runtime, one increment per acquire, and
one `perf_counter` pair per rebuild — negligible against the O(corpus) work
being measured, but not zero. Four new fields on `IndexStatusResponse`, which is
a public response shape. The counters reset on LRU eviction and on process
restart, so a scrape immediately after either reports zeros that mean "no data",
not "no rebuilds" — a real trap for anyone graphing them, and the reason
`KNOWN_LIMITATIONS.md` names it.

**What this does NOT fix.** The cliff itself. An ingest over N documents still
costs N invalidations and N full rebuilds, still serialized against the ingest
writer on one store lock, and every concurrent request still waits through one
rather than being served stale. GK-020's second and third acceptance criteria
are open, with the obstacle named in decision 4 rather than left to be
rediscovered. The counters are per-process, so a multi-process deployment
requires reading each process's own `index_status`, and nothing aggregates them.

## References

- [ADR-0002](ADR-0002-index-persistence.md) — SQLite as durable truth,
  rebuild-at-open, the deferred persisted-postings alternative this ADR
  re-defers against a new trigger, and the GK-018 postings erratum whose
  position-keyed map makes `remove_document` a redesign.
- [ADR-0004](ADR-0004-embedding-identity-binding.md) — decision 5's
  rebuild-not-migrate rule, which prices the schema bump decision 4 declines.
- [ADR-0013](ADR-0013-collection-runtime-persisted-staleness-marker.md) —
  decision 1's per-commit bump (unmodified), decision 4's declined
  transaction-scope widening, decision 5's wait-rather-than-serve-stale rule,
  decision 9's LRU bound that resets these counters, and decision 10's
  Protocol-conformance obligation.
- [ADR-0014](ADR-0014-read-only-service-surface-and-outbound-endpoint-safety.md)
  — the read-only surface these fields are published on.
- [ADR-0022](ADR-0022-observability-dependency-shape-and-span-attribute-allowlist.md)
  — why the OTel SDK is an extra, and therefore why `index_status` rather than a
  span is the primary surface.
- `scripts/measure_retriever_open.py` — the per-open half of ADR-0002's original
  trigger, and the script ADR-0013 already owes a mode measuring its own
  per-request cost.
- `BACKLOG.md` GK-020 — the entry whose first acceptance criterion this closes
  and whose second and third it defers.
- SPEC.md §1.2, §2, §7, §8.

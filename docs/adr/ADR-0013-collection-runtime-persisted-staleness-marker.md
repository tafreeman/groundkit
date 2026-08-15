# ADR-0013 — A cached `CollectionRuntime` whose validity is a marker persisted in SQLite

- **Status:** Accepted (owner, 2026-08-15)
- **Date:** 2026-08-15
- **Deciders:** Andy Freeman (owner)

## Context

Phase 4 puts a REST API and an MCP server in front of the retrieval path. Both are
long-lived processes answering many requests against a collection that a *different*
process — `grk ingest`, the documented first half of `grk ingest ./docs && grk
serve-mcp` (SPEC.md §9) — may write to at any moment. Nothing in the current design
survives that.

**`Retriever.open()` is a snapshot that never refreshes.** ADR-0002 accepted this
deliberately: `BM25Index.from_store()` rebuilds postings in memory from the persisted
chunk set, so the index cannot drift from SQLite, and it costs O(corpus) at every
open. ADR-0002's Consequences then name the asymmetry that makes a long-lived
retriever unacceptable rather than merely stale: a hit against modified or deleted
content **fails closed** (`RetrievalError`, no stored source), but a query that should
match content ingested *after* `open()` has no representation in the stale index at
all and **silently returns zero results**. Wave C gave the dense path the same
semantics by filtering rather than by rebuild (`_dense_candidates` restricts hits to
`documents_at_open`), so both paths are silently blind to new content in exactly the
same way. A server that opens one retriever at startup is wrong the moment anyone
ingests, and wrong in the one direction this repo's fail-closed rule cannot detect.

**The deferred design's sequencing constraint is met.** The Codex repository-review
document (`GroundKit_Repository_Simplification_System_Design.docx`, repo root,
untracked) proposed a `CollectionRuntime` composition root and a versioned collection
snapshot. That proposal was deferred on 2026-08-14. Its own rollout constraint reads:
*"land correctness fixes first; add CollectionRuntime behind current CLI behavior;
migrate one command at a time; require regression, lifecycle, and gated-provider
evidence before Phase 4 exposure"*, and its decision section sequences *"a focused
CollectionRuntime composition root and a versioned collection snapshot … and only then
connect REST and MCP adapters."* Phase 4 is the exposure that constraint names.

That document must be read with care rather than cited as authority: its milestone M1
was already delivered by `04dfc5b`, and it cites two ADRs that did not exist when it
was written. The quotes above were extracted from the file itself for this ADR rather
than taken from any summary of it. Its M2 exit criterion — *"CLI ingest/search/eval
use one lifecycle path with no behavior regression"* — is **not** adopted in full here;
see decision 7.

**SPEC.md §2 constrains the shape of the fix.** "No in-process state shadowing
persisted state. Anything the index knows must be recoverable from disk (ADR-0001
hazard 7)." A cached `Retriever` *is* in-process state shadowing persisted state —
unless its validity is derived from something persisted. This is why an
invalidate-on-my-own-writes cache is not the answer: its validity is decided
in-process, so it is correct exactly until someone runs `grk ingest` in another
terminal, which is the workflow SPEC.md §9 documents.

**Reopen-per-request was rejected on measurement, not on taste.**
`scripts/measure_retriever_open.py` is the method; it exists because the Phase 3 spec
(R4) required measuring `open()` cost "rather than discovering it in Phase 4 when the
service opens retrievers per request." Run on the owner's development machine on
2026-08-15, 10 repeats per configuration, the dense figures rose from tens of
milliseconds at 100 documents to roughly half a second at 2 000 documents / 8 000
chunks. Regenerate with `uv run python scripts/measure_retriever_open.py --sizes 100
500 2000 --repeats 10`; the numbers are point-in-time script output on one machine,
not facts about groundkit, and are deliberately not restated here (SPEC.md §2, no
number in any doc that wasn't generated). Two things make them worse than they look:
the rebuild serializes on the metadata store's single `asyncio.Lock`, so concurrent
requests queue behind one another rather than paying the cost in parallel, and the
dense figure already includes the manifest verification, the LanceDB connect, and the
`documents_at_open` query that Wave C added.

**The schema-version mechanism already exists; this ADR extends it rather than
inventing one.** ADR-0004 decision 4 stamps `PRAGMA application_id` (`GRK1`) and
`PRAGMA user_version` (`SCHEMA_VERSION`) as the last statements before commit on a
newly created store, `SQLiteMetadataStore.open` verifies both, and a
`collection_manifest` table already exists. Any claim that the schema records no
identity and no version is pre-Phase-3 text and is stale.

**The TOCTOU discipline this ADR must not break.** ADR-0004 decision 3, and the
docstring on `Retriever.open()`, establish that `store.verify_manifest()` returns the
manifest it checked so that **one read decides both** "does the identity match" and
"is this collection dense-bound." Two reads admitted a real silent-corruption race. A
staleness check adds another store read to that sequence, so its placement is a
correctness decision, not plumbing.

## Decision

### 1. The marker is a monotonic per-collection generation counter, bumped inside the same transaction as every write

`SQLiteMetadataStore` gains an integer `generation`, starting at `0` on creation and
incremented by **every mutating store method** — `upsert_document`, `add_chunks`,
`replace_document`, `delete_document`, `write_manifest` — as a statement inside that
method's existing `_op`, before its `commit()`. `MetadataStoreProtocol` gains one read,
`get_generation() -> int | None`.

**In-transaction, not adjacent to it.** The bump commits with the write or not at all.
A bump issued as a separate call after the write opens a window in which content is
newer than the marker — a reader then observes "unchanged" over changed data, the
precise failure this mechanism exists to prevent. The reverse is merely wasteful.

**One bump per commit, not one per method call.** Every `_op` that commits durable state
advances the marker, and no branch inside it decides whether the write "really" changed
anything — `delete_document` on an absent id deletes zero rows, commits, and advances.
Over-bumping costs one redundant rebuild; under-bumping serves stale results silently and
forever, so where the two are traded, bump.

The rule is tied to the **commit** rather than to the method because that is the form
that can be checked by reading the code instead of trusting each caller to classify its
own branches. It also settles the one case where the two formulations disagree:
`write_manifest` re-called with the identity the collection already holds returns before
any `INSERT` and before any `commit`, so durable state is untouched, a retriever built
against it is still valid, and it correctly advances nothing. Bumping there would force a
write on the path every re-ingest of a bound collection takes, in order to invalidate a
cache nothing invalidated.

**Monotonic and opaque.** The only operation is equality against a previously observed
value. A counter beats `(doc_count, max_rowid)`, which the most common write defeats —
`replace_document` deletes then inserts, and SQLite reuses freed rowids without
`AUTOINCREMENT`, so a replacement can land on the same rowid with an unchanged count.
It beats a content-derived digest over chunk hashes, which is O(corpus) per read: the
cost being removed.

**What the read costs, named.** A one-row primary-key lookup against a one-page table
is effectively always in SQLite's page cache. The cost is not SQLite: it is one
`asyncio.to_thread` dispatch and one acquisition of the store's single `asyncio.Lock`
per request, contending with in-flight ingest writes on that lock. It is bounded work
per request against unbounded work avoided, but it is not free, and
`scripts/measure_retriever_open.py` is owed a mode that times a warm acquire against a
rebuild so this claim is measured rather than asserted.

**Dense mutations are covered transitively.** `Indexer` maintains the invariant that
SQLite is never ahead of the dense store, and `write_manifest` bumps on its own. Because
the SQLite commit is *last*, an observed bump implies the dense write already landed.
The standing obligation: a future dense-only mutation path with no SQLite counterpart
would be invisible to the marker. Enforcement is therefore structural — the bump lives
inside the store's mutating `_op`s, not at call sites — plus the classification test in
the implementation notes.

### 2. The marker lives in a new single-row `collection_state` table, and `SCHEMA_VERSION` goes to 2

```sql
CREATE TABLE IF NOT EXISTS collection_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    generation INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);
```

The `CHECK (id = 1)` single-row pin mirrors `collection_manifest` exactly, so a second
row is unrepresentable in the schema rather than prevented by discipline. It lives in
SQLite because ADR-0002 makes SQLite the durable truth. `updated_at` is **diagnostic
only**, for `index_status`; it is never consulted for validity.

**This extends ADR-0004's existing verification; it does not add a parallel one.**
Adding a table is a schema change, so `SCHEMA_VERSION` becomes `2` and the existing
`open()`-time check covers it unchanged. Declining the bump would make the version a
decoration and leave v1-shaped and v2-shaped files indistinguishable.

The blast radius is real and is accepted on ADR-0004 decision 5's own reasoning.
`manifest_capable` is an **exact** equality check, so every collection created before
this change becomes non-schema-current: `write_manifest` and `verify_manifest` refuse
it, and **existing dense collections must be deleted and re-ingested.** Pre-1.0, every
index is reproducible from `grk ingest` in seconds, so a migration path would be code
written to preserve data that costs seconds to regenerate — the argument
`index/metadata.py` already records against `SCHEMA_VERSION` itself.

The bump is also load-bearing rather than cosmetic, and this is the sharper reason.
`executescript(_SCHEMA)` runs on **every** open with `CREATE TABLE IF NOT EXISTS`, so
without a version bump the new table would appear in a legacy collection
transparently and the cache would engage over it. An older `grk` binary lacking the
bump logic could then write to that collection without moving the counter, and the
service would serve stale results silently and indefinitely — exactly the defect this
ADR exists to close. With the bump, such a store reads as non-current, the cache
disables, and the failure degrades to correct-and-slow. The exact-equality check is
**not** widened to `>=`: that would reopen ADR-0004 decision 5, and a real migration
story is owed at 1.0, not here.

Internally `_manifest_capable` is renamed `_schema_current`, since it now gates two
unrelated capabilities.

### 3. A store that is not schema-current has no marker; the runtime treats it as permanently stale

`get_generation()` returns `None` — it does not raise — when `_schema_current` is
false. `CollectionRuntime` treats `None` as "cannot assert freshness," rebuilds on
**every** request, and logs a warning **once per runtime** naming the remedy.

This is fail-closed in the correct direction, and the precision matters: the marker
answers one question — *has anything changed since I last looked?* — and a missing
marker makes that question unanswerable. The safe answer to an unanswerable freshness
question is "assume it changed." That degrades to reopen-per-request: correct, slow,
loudly labelled. Refusing outright was rejected — nothing is *wrong* on a legacy
BM25-only store, only slow, and SPEC.md §2's fail-closed rule governs wrong answers,
not slow ones. Backfilling a marker was rejected on a sharper ground: it would write
v2 tables into a file stamped v1, producing a store whose version stamp lies about its
shape.

`index_status` reports the disabled state explicitly, so it is visible rather than
inferred from latency.

### 4. The marker is read **first**, on the runtime's own store handle, and the artifact is stamped with that pre-build value

The rebuild sequence, entirely on one store handle:

1. `G = await store.get_generation()` — **before anything else**
2. `manifest = await store.verify_manifest(identity_of(embedder))` — unchanged
3. `await verify_dense_side_present(...)` — unchanged
4. `bm25 = await BM25Index.from_store(store, ...)` — unchanged
5. `documents_at_open = frozenset(await store.get_document_sources())` — unchanged
6. publish `(G, retriever)`

Steps 2–5 are `Retriever.open()` exactly as it stands. **`retrieval/search.py` gains no
behaviour change from this ADR** — step 1 happens in the runtime before the call, step
6 after it.

**Why ADR-0004's race is not reintroduced.** That hazard is two reads *of the manifest*
producing verdicts that can disagree. The generation read touches a different table,
produces neither verdict, and is consulted by neither. `verify_manifest` remains the
single read answering both questions together. If a concurrent dense ingest binds the
collection between steps 1 and 2, step 2 observes the *new* manifest and verifies
against it — raising `IndexIdentityError` or producing a consistent pair from that one
read, exactly as today. The artifact is then stamped `G` while its content reflects
`G+1`, so the next request rebuilds. Conservative, never wrong.

**Why the ordering is load-bearing.** Every read at steps 2–5 happens at or after step
1, so the artifact reflects a store state at generation ≥ `G`. The validity predicate
is `get_generation() == G`. Because the counter is monotonic, equality implies no
commit landed after step 1, so the artifact is exactly the state at `G`. If a commit
landed during the build, `current > G`, the predicate is false, and the artifact is
rebuilt even though it may already contain that write. **The stamp is a lower bound on
freshness, and the error is always toward a redundant rebuild.**

Reading the marker *last* inverts that and is unsafe. Stamping a value read after step
5 means a commit landing during steps 2–5 raised the counter without necessarily being
reflected in reads that already completed — `BM25Index.from_store` ran at step 4,
before the commit. The predicate then holds indefinitely and the runtime serves an
index missing that write until the *next* write happens to bump again. That is
ADR-0002's silent-zero-results outcome made permanent by caching: worse than the
problem being fixed. Stamp before build; never after.

**Deliberately not done:** a single read transaction spanning steps 1–5. It would make
the stamp exact rather than a lower bound, but requires widening
`MetadataStoreProtocol` with transaction-scope semantics — a signature-parity-tested
seam changed for an optimization — and blocks WAL checkpointing across an O(corpus)
rebuild. Revisit if redundant rebuilds under write-heavy load are *measured* to matter,
borrowing ADR-0002's discipline that the measurement is the trigger.

### 5. Single-flight: one rebuild per observed generation, and a waiting request is never served the stale artifact

The runtime holds the cached `(generation, retriever)` pair and one `asyncio.Lock`.
Each request reads the generation, returns the cache on a match without taking the
lock, and otherwise queues, re-checks under the lock against **the generation it
observed before queueing**, and rebuilds only if still unsatisfied.

**N concurrent stale requests produce exactly one rebuild.** The leader rebuilds and
publishes; waiters re-check, find the cache stamped with the generation *they*
observed, and return.

**The re-check deliberately does not take a fresh read inside the lock.** Re-reading
would make each waiter chase a moving target under sustained writes: every waiter would
observe a newer generation than the leader just satisfied and rebuild again, converting
write pressure into an unbounded rebuild storm. Reusing the pre-queue observation caps
rebuilds at one per observed generation and gives every waiter a well-defined freshness
bound.

**A request arriving during a rebuild waits; it is not served the stale artifact.**
This costs real tail latency — concurrent requests during a rebuild each absorb it —
and is chosen anyway, because serving stale "just for the duration of the rebuild"
reintroduces silent zero results for new content at the moment of highest probability
that someone is actively ingesting. Serving stale behind a `stale: true` flag was
rejected: a flag nobody is required to read is not fail-closed.

**Cancellation.** The rebuild body runs shielded with the lock released by a
done-callback, mirroring `SQLiteMetadataStore._run`'s proven pattern, so a client
disconnecting does not abandon a rebuild others are blocked on. The distinction is
recorded honestly: in `_run` the shield is a *correctness* property; here it is a
*throughput* property, since a `Retriever` owns nothing closable and an abandoned
half-built one leaks nothing.

**A failed rebuild does not publish.** The cache is assigned only after
`Retriever.open()` returns. On failure the exception propagates and the previous
artifact — still cached but stamped with a non-matching generation — is unreachable
through the fast path. That is structural, not defensive code.

### 6. Ownership: process-lifetime resources versus collection-derived ones

**Process-lifetime:** the `SQLiteMetadataStore` handle; the `EmbeddingProtocol` (an
HTTP client independent of collection state, whose identity is re-verified against the
manifest on every rebuild); and the `RerankerProtocol` when configured — a
multi-gigabyte model that must never be reconstructed because a collection changed.

**Collection-derived, rebuilt on every generation change:** the `Retriever`, and the
**vector-store handle with it**.

The vector store belongs on the derived side because of a specific defect, not for
symmetry. `LanceDBVectorStore.open()` caches `self._table` and returns `[]` from
`search()` when it is `None`, never re-checking for a table created later. A runtime
opened dense-paired over a collection whose `.lance` table does not yet exist therefore
holds `_table = None` permanently; when an out-of-process `grk ingest --dense` later
writes vectors, a Retriever-only rebuild would probe the stale handle, get `[]`, and
raise `StorageError` naming a lost dense side on a healthy collection. The runtime
therefore takes a **vector-store factory**, not an instance. The embedder/factory pair
is both-or-neither, validated through the existing `groundkit.identity` helpers.

**Who closes what.** `Retriever` has no `close()`, so a rebuild is a pointer swap.
Standing obligation: if a future `Retriever` acquires a closable resource, the rebuild
path is where the superseded one must be closed. The same applies to the superseded
vector-store handle, which exposes no `close()` today and is simply dropped.
`aclose()` is idempotent and ordered — drain the rebuild lock, drop the artifact,
close the embedder, close the store **last**, since the store's lock is what in-flight
operations serialize on. Acquiring after close raises a typed error.

### 7. The runtime absorbs the CLI's read lifecycle; the eval runner keeps its own

Today there are two production callers of `Retriever.open()` — `cli.py` (`_cmd_search`)
and `evals/runner.py`. Naively, Phase 4 makes four.

**`_cmd_search` is absorbed.** The cache never hits in a one-shot process, which is
fine: the value is one lifecycle, and the default suite then exercises the runtime's
open path on every CLI test rather than only in service tests.

**`_cmd_ingest` is not absorbed.** It is a write lifecycle over an `Indexer`, not a read
lifecycle over a `Retriever`; folding it in would make the runtime mean two things.

**The eval runner keeps its own lifecycle, and this is a deliberate divergence from the
design document's M2 exit criterion** (*"CLI ingest/search/eval use one lifecycle
path"*). `run_eval` opens exactly one `Retriever` after ingestion completes, and the
comment there states the requirement: one retriever serves every stage, so stages
compare retrieval strategy rather than index state. Routed through the runtime,
behaviour would be identical — nothing writes between stages, so every acquire would
return the same cached object — but the *guarantee* would weaken from structural ("one
`open()` call, one object") to contingent ("the cache must hit, and it will because
nothing writes"). For the one path in this repo that emits a hash-pinned artifact whose
cross-run comparability is the entire point, that is the wrong trade, and it buys
nothing: the runner's store lives in a per-run temp directory no other process can
reach.

Net: three prospective new copies collapse into one; the eval runner's survives with
its reason recorded. **Revisit trigger:** if `run_eval` ever writes to its store
*between* stages, its single-retriever assumption breaks and it needs the runtime.

### 8. The module is `src/groundkit/runtime.py`, added explicitly to the coverage core subset

Placement follows `identity.py`, whose docstring records the precedent: a module holding
invariants that `Indexer`, `Retriever` and `run_eval` all need identically, living
outside whichever caller happened to need it first "precisely so that sharing them
creates no dependency between ingest and retrieval." `CollectionRuntime` has the same
three callers and the same hazard — per-caller copies that stay identical in logic while
diverging in prose.

One property does **not** carry over and is stated rather than implied: `identity.py` is
an import *leaf*, importing only contracts, errors and the protocol modules with nothing
importing it back. A `CollectionRuntime` is a composition root — it must import
`Retriever`, the metadata store and the vector store, because composing them is its
job. What it keeps from the precedent is top-level placement and the
nothing-imports-it-back property: only `cli.py` and `service/` depend on it, so ingest
and retrieval gain no dependency on each other through it.

`pyproject.toml`'s `[tool.groundkit.coverage].core_subset` already mixes one glob with
three explicit file paths, so `src/groundkit/runtime.py` is added as a fourth explicit
entry rather than relying on placement to catch it. **The gate is the reason for the
entry, not a side effect:** this module decides whether an answer is computed over
current data, and its hardest code — the single-flight and cancellation paths — is
exactly the class that is unreachable on a non-failing path and therefore invisible to a
green suite.

**The honest asymmetry:** the marker *read* is gated; the marker *bump* lives in
`index/metadata.py`, which `pyproject.toml` deliberately keeps outside the subset. The
half whose omission causes silent staleness is ungoverned by the core gate, and its
named regression tests carry that weight.

### 9. One runtime per collection, behind a registry that will not create a collection it was merely asked about

A registry holds one runtime per collection name and hands them out as a refcounted
async context manager — the natural shape for both a FastAPI dependency and an MCP tool
handler.

**The registry must not create collections implicitly.** `SQLiteMetadataStore.open()`
creates the file if absent, so a registry that opened on demand would let a request for
an arbitrary name create an empty SQLite file — a trivial disk-fill against a service
whose reads are unauthenticated. The registry checks the file exists before opening and
refuses otherwise, raising `ConfigurationError` — the type `SQLiteMetadataStore.open`
already raises for a malformed name, so bad-name and unknown-name failures are one type
at one boundary.

**What bounds the number held open is memory, not file handles.** `BM25Index` retains
its chunk list, so a cached retriever pins the collection's entire chunk text resident.
N open collections means N corpora in memory. The bound is an explicit
`max_open_collections` setting enforced by LRU eviction of zero-refcount entries.
Eviction never closes a runtime with an in-flight request: the registry exceeds its
bound temporarily with a logged warning rather than closing a store out from under a
live request. A future `delete_collection` must go through the registry so it can evict
before unlinking both the `.sqlite3` and its `.lance` sibling (SPEC.md §7).

### 10. Any new Protocol seam gets a signature-parity conformance entry

If the runtime or registry introduces a `typing.Protocol`, it gains an entry in
`tests/test_protocol_conformance.py` using `assert_signature_parity`, never
`isinstance`. `isinstance` on a runtime-checkable Protocol compares member *names* only
and would pass straight through the `query` → `_query` rename that caused ARP's
signature drift (ADR-0001 hazard 4). The added `MetadataStoreProtocol.get_generation`
member is covered by the existing conformance test for that protocol.

## Alternatives considered

- **Reopen a `Retriever` per request.** Rejected on the measurement above, and worse
  than the raw figures suggest because the rebuild serializes on the store's single
  lock. It remains the correctness baseline this design must match, and decision 3
  falls back to exactly it.
- **Invalidate on the service's own writes.** Rejected: validity decided in-process is
  what SPEC.md §2 forbids, and it misses out-of-process `grk ingest` — correct precisely
  until the tool is used as documented.
- **Time-based refresh (TTL).** Rejected: wrong in both directions at once. It serves
  stale results for up to its own length after every ingest and rebuilds an unchanged
  corpus on every expiry for nothing, with a staleness bound nobody chose on evidence.
- **The SQLite file's mtime.** Rejected on four grounds: under WAL the commit lands in
  `-wal` and the main file's mtime need not move; timestamp granularity is coarse; mtime
  is not monotonic — a clock adjustment or restore can move it *backwards*, and a
  decreasing marker makes a stale cache look fresh; and it observes the file rather than
  the transaction.
- **`PRAGMA data_version`.** A close fit, rejected on a documented clause: it changes
  only for commits made by *other* connections and deliberately not for commits on the
  same connection — so it would miss exactly the writes a service's own ingest route
  makes. It is also per-connection rather than persisted, failing SPEC.md §2's
  recoverable-from-disk test. **Confirm against SQLite's pragma documentation before
  implementing anything that assumes otherwise.**
- **`(doc_count, max_rowid)`**, and **a content digest over chunk hashes.** Rejected per
  decision 1.
- **A per-collection `instance_id` to detect delete-and-recreate.** Rejected because it
  does not work: a runtime holding an open handle to an unlinked file keeps reading the
  old inode, including its `instance_id`. Decision 9's registry-mediated delete is the
  supported path; out-of-band `rm` is documented as a limitation rather than half-solved.
- **Folding the generation into the existing `get_document_sources()` read.** Rejected:
  the freshness decision must be made *before* a retriever is selected, not inside one,
  and it would widen a signature-parity-tested return type for a micro-optimization.
- **Placing the module in `retrieval/`** to catch the `retrieval/*` glob automatically.
  Rejected per decision 8: it puts a composition root that imports `index.metadata`
  inside the retrieval package, and an explicit `core_subset` entry achieves the same
  gate without the layering compromise.
- **Persisted BM25 postings** (ADR-0002's own deferred alternative). Not an alternative:
  it attacks rebuild *cost*, not staleness, and persisted postings still need a validity
  signal isomorphic to this marker. The two compose.

## Consequences

**Positive.** The cache's validity is a persisted fact, satisfying SPEC.md §2 rather
than being excused from it. Out-of-process `grk ingest` is observed, which is what makes
`grk ingest ./docs && grk serve-mcp` work as documented. Steady-state request cost drops
from an O(corpus) rebuild to one indexed single-row read. The `Retriever` contract is
untouched, so ADR-0002, ADR-0004, ADR-0006, ADR-0007 and ADR-0008 all stand unmodified,
and ADR-0004 decision 3's one-read property is preserved by construction. Three
prospective lifecycle copies collapse into one.

**Costs.** Every request pays one extra store-lock acquisition and thread dispatch,
contending with in-flight writes. A rebuild is a latency spike absorbed by every
concurrent request. Memory becomes the binding resource for a multi-collection service.
`SCHEMA_VERSION = 2` invalidates every existing collection for dense work.
`MetadataStoreProtocol` gains a member, so every in-test structural fake must implement
it — and because `mypy --strict` covers `tests` as well as `src`, those surface as
typecheck failures rather than runtime surprises.

**What this does NOT fix.**

- **A one-request staleness window remains**, and no design of this shape eliminates it:
  another process can commit between the marker read and the answer. The claim is
  *bounded* staleness — never more than one request behind, self-correcting — not
  linearizability. Reopen-per-request has the identical window for the identical reason.
- **The dense over-fetch cap is unchanged.** What changes is the window's duration:
  previously bounded by process lifetime, now by one generation.
- **Orphaned vectors are unchanged.** The marker tracks SQLite; an interrupted ingest
  leaves vectors under a document ID SQLite never recorded, and no bump describes them.
- **Cross-store atomicity is unchanged.**
- **Out-of-band edits to source files are invisible.** The marker tracks the store, not
  the filesystem; a same-length in-place edit still defeats `resolve_citation`'s
  length-only drift check.
- **Deleting a collection's files out of band while serving is not recoverable.** On
  POSIX the runtime keeps reading the unlinked inode, including its generation.
- **Half-dense collections remain undetected**, per the existing no-backfill limitation.
- **Adjacent gap flagged for ADR-0014:** `SQLiteMetadataStore` sets `foreign_keys` and
  `journal_mode` but not `busy_timeout`. Under WAL readers do not block on a writer, but
  two *writers* do, and SQLite's default busy timeout is 0 — so the loser fails
  immediately with a locked-database error surfacing as `StorageError`. Tolerable for a
  CLI, questionable for a service.

## References

- [ADR-0002](ADR-0002-index-persistence.md) — SQLite as durable truth, rebuild-at-open,
  the single-connection/one-lock model, and the staleness analysis whose
  silent-zero-results branch is this ADR's motivating defect.
- [ADR-0004](ADR-0004-embedding-identity-binding.md) — decision 3's one-read TOCTOU
  discipline; decision 4's `application_id`/`user_version` stamp this ADR bumps;
  decision 5's pre-1.0 rebuild-not-migrate rule.
- [ADR-0008](ADR-0008-dense-search-requires-a-dense-collection.md) — the per-mode
  refusal reading the dense-bound verdict each rebuild re-derives.
- [ADR-0012](ADR-0012-rerank-eval-stage-reorders-upstream-stage.md) — decision 2 defers
  per-request rerank to Phase 4; decision 6 above puts the reranker on the
  process-lifetime side.
- [ADR-0014](ADR-0014-read-only-service-surface-and-outbound-endpoint-safety.md) — the
  service surface that consumes this runtime.
- SPEC.md §2, §7, §8, §9.
- `scripts/measure_retriever_open.py` — the measurement method, and the script owed a
  mode that measures this design's own per-request cost.
- `src/groundkit/identity.py` — the placement precedent decision 8 follows, and the one
  property it cannot.
- `KNOWN_LIMITATIONS.md` — the entries this ADR explicitly does not close.

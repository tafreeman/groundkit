# ADR-0011 — A BM25-only ingest refuses a collection that has a dense side

- **Status:** Accepted (owner, 2026-08-15)
- **Date:** 2026-08-15
- **Deciders:** Andy Freeman (owner)

## Context

`KNOWN_LIMITATIONS.md` has recorded since Wave C that "a BM25-only indexer will
orphan a vector-bearing collection": an `Indexer` built without an embedder and
vector store still replaces, prunes and deletes documents in a collection whose
vectors an earlier dense run wrote. It has no vector-store handle, so those
vectors survive their documents.

That was tolerated on an explicit argument — the orphans are *loud*. Wave C's
read path fails closed on them: `Retriever.search` raises `RetrievalError` on a
dense hit whose document has no stored source. A loud residue was judged better
than a silent one, and the exposure was bounded, because reaching it required a
document to actually change. An unchanged corpus hash-matched, skipped, and
rewrote nothing.

[ADR-0009](ADR-0009-incremental-skip-key-is-a-processing-fingerprint.md) removed
that bound without anyone noticing. The skip key stopped being a content hash
and became a fingerprint over content, chunker and chunking configuration —
derived differently, so it matches nothing an earlier build stored. Every
document in every pre-existing collection therefore mismatches on the first run
after that change.

Review of PR #5 found the consequence. The chain is short and every link is in
the code as written:

1. A BM25-only `grk ingest` over a previously dense-ingested collection
   mismatches on every document, because the stored key predates ADR-0009.
2. `_persist_document` gates all vector work on `embedder is not None and
   vector_store is not None`, so on this path nothing is embedded and nothing is
   deleted — but `replace_document` still runs, installing a **fresh
   `document_id`** (`Document.document_id` defaults to a per-load `uuid4`).
3. The old vectors keep the old id. SQLite no longer references it. They are
   orphaned — all of them, in one command.
4. The same write stores the new fingerprint, so a later dense run hash-skips
   every one of those documents and **never re-embeds them**. The orphans are
   permanent; the collection cannot re-derive itself.
5. Dense and hybrid searches then fail closed on the first orphan that ranks
   into a candidate window, reporting an index inconsistency — which names the
   symptom and not the BM25-only ingest three steps upstream that caused it.

So the tolerated hazard changed shape: from "an edited document may strand its
vectors, loudly" to "one ordinary command strands an entire collection's dense
side, and the error surfaces later somewhere else." The loudness argument does
not survive that, because what is loud is the *eventual read failure*, not the
write that caused it.

## Decision

`Indexer._verify_identity` refuses a run when this indexer has no embedder and
the collection is manifest-bound, raising `ConfigurationError` naming the
remedy.

1. **The manifest is the test for "has a dense side."** It is written on a
   collection's first dense write (ADR-0004) and immutable thereafter, so its
   presence is exactly the condition under which a BM25-only writer cannot keep
   the two stores in step.
2. **The check runs before any work.** `_verify_identity` is already the first
   thing both `index_source` and `index_directory` do, for a reason this
   decision inherits verbatim: a misconfigured process must be stopped before it
   has destroyed part of a healthy collection, not after. A guard that fired
   part-way through would have orphaned whatever it had already rewritten.
3. **`ConfigurationError`, matching ADR-0008.** The symmetric refusal on the
   read side — a dense or hybrid search against a collection with no manifest —
   is a `ConfigurationError` naming the re-ingest remedy. This is the same class
   of caller mistake at the other boundary and gets the same treatment.
4. **The refusal is narrow.** A collection with no manifest was never
   dense-ingested, so a BM25-only run over it proceeds unchanged. That
   deliberately preserves the documented BM25-first-then-dense flow, where no
   manifest exists until the first dense write, and which has its own test.

This completes [ADR-0004](ADR-0004-embedding-identity-binding.md) decision 3's
ingest boundary rather than departing from it. That boundary was already meant
to be checked before any dense mutation; `_verify_identity` simply returned
immediately whenever there was no embedder — which is precisely the
configuration that cannot maintain the dense side.

## Alternatives considered

**Leave it, and document it harder.** Rejected. The existing entry was already
honest and already there, and it did not prevent the defect from being
introduced by an unrelated change to the skip key — because nothing connected
"the skip key changed derivation" to "the orphaning gate is the skip key". A
documented hazard that a routine change can silently escalate is not a
mitigation.

**Make the BM25-only path maintain the dense side.** Rejected: it cannot. There
is no vector-store handle by construction, and inventing one — opening the
collection's LanceDB directory by convention from inside the indexer — would put
store-location knowledge in a component that deliberately receives its stores by
injection, and would make a BM25-only install fail on a missing optional extra.

**Migrate the fingerprint in place instead, so nothing is rewritten.** Rejected
for the reason ADR-0009 gives for not versioning the key: the old and new keys
are not inter-convertible, since the fingerprint covers inputs the stored hash
never saw. Detecting "this hash looks like a pre-ADR-0009 content hash" is a
guess about an opaque digest.

**Warn instead of refusing.** Rejected. A warning on a command that is otherwise
reporting success is exactly how this would be missed, and the damage is done by
the time it is printed. SPEC.md §2 is fail-closed, and ADR-0008 already settled
that the symmetric case raises.

## Consequences

- A BM25-only `grk ingest` against a dense collection now fails fast with a
  message naming both remedies (supply the dense pair, or delete and re-ingest),
  instead of succeeding and corrupting the dense side.
- Collections built before ADR-0009 still need one dense re-ingest to adopt the
  new fingerprint. That was already true and is unchanged; what changes is that
  attempting it *without* the dense pair is now refused rather than silently
  destructive.
- `KNOWN_LIMITATIONS.md`'s BM25-only-orphan entry narrows to the case that
  remains reachable: an `Indexer` holding a stale handle, or a collection whose
  manifest was never written despite vectors existing. It is no longer reachable
  through the CLI.
- One more reason a collection must be deleted and rebuilt rather than
  half-migrated, consistent with ADR-0004 decision 5 (indexes are cheap to
  regenerate pre-1.0).

## References

- [ADR-0009](ADR-0009-incremental-skip-key-is-a-processing-fingerprint.md) — the
  fingerprint change that escalated this hazard from occasional to total.
- [ADR-0004](ADR-0004-embedding-identity-binding.md) — decision 3 (the ingest
  boundary this completes), decision 5 (rebuild is cheap pre-1.0), decision 6
  (delete-path reconciliation).
- [ADR-0008](ADR-0008-dense-search-requires-a-dense-collection.md) — the
  symmetric refusal on the read side, and the precedent for the error type.
- SPEC.md §2 (fail closed; no silent divergence between configuration and
  persisted state).

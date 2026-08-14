# ADR-0008 — A dense or hybrid search refuses a collection that has no vectors

- **Status:** Accepted (owner, 2026-08-14)
- **Date:** 2026-08-14
- **Deciders:** Andy Freeman (owner)

## Context

Phase 3 Wave C decided that opening a `Retriever` with a dense pair over a
collection that was only ever ingested BM25-only should succeed, and that the
dense side should then "behave as an honestly empty index, never an error".
That was recorded in code (`verify_dense_side_present` returns early when
`get_manifest()` is `None`), in `KNOWN_LIMITATIONS.md` (the no-backfill
limitation), and in a test class asserting exactly it.

The reasoning was sound as far as it went. An unbound manifest is a
*legitimate* collection state — it means "BM25-only", whose upgrade path is
the documented no-backfill limitation rather than a defect — and it is
genuinely distinguishable from the corrupt state `verify_dense_side_present`
does refuse (manifest bound, documents present, vector store empty).

What that reasoning missed is what the caller receives. Reviewing PR #4
produced the concrete sequence, and it is worse than "empty":

```
grk ingest ./docs                      # BM25-only collection
grk search "..." --mode hybrid         # returns results, scored 0.016
grk ingest ./docs --dense              # "0 vectors written" — all hash-skipped
```

The hybrid search returns hits scored `1/(60+1)` — RRF over one non-empty
list. The dense side contributed nothing, so the ranking *is* BM25's, and
`SearchResponse.metadata["stage"]` stamps it `"fusion"`. The caller is handed
lexical results labelled as fused, with no error and no warning. The obvious
remedy makes it worse: enabling `--dense` over that collection backfills
nothing, because the content-hash gate runs before embedding, so the
collection stays permanently vector-less and nothing reports that.

This repo has already ruled on this exact shape once. A review finding on
PR #3 closed an unrecognised `mode` falling through to the hybrid branch and
returning fused results stamped `"fusion"` — "a wrong answer presented as a
valid one", fixed by raising. The manifest-less case produces the same
artifact by a different route.

`--mode dense` is only marginally better: it returns zero results, which
reads as "nothing matched" when the truth is "there is no dense index".
That is the silent-absence class SPEC.md §2 fails closed against, and it is
the *same* symptom `verify_dense_side_present` already refuses for the
bound-but-empty case. Treating identical user-visible behaviour as an error
in one state and as normal in the other was an inconsistency, not a
distinction.

## Decision

**A `dense` or `hybrid` search against a collection with no embedding-identity
manifest raises `ConfigurationError`.**

1. **The check lives in `search()`, per mode — not in `open()`.** An unbound
   manifest remains a legitimate state to open a dense-paired `Retriever`
   over: a caller may open once and search `"bm25"` only, and failing at
   `open()` would break that for no benefit. It is the *mode* that cannot be
   answered, so the mode is where it fails. `Retriever.open()` records
   whether the collection is manifest-bound and enforces nothing.
2. **`ConfigurationError`, not `StorageError`.** This is a caller/state
   mismatch with an actionable remedy, not corruption. `StorageError` stays
   reserved for `verify_dense_side_present`'s lost-dense-side case, where
   vectors *did* exist and no longer do.
3. **The error names the no-backfill trap explicitly** and tells the caller
   to re-ingest into a fresh collection with `grk ingest --dense`. An error
   that merely says "no manifest" would send them to the remedy that silently
   does nothing.
4. **Wave C's contrary tests are replaced, not deleted.** The class asserting
   "never an error" now asserts the refusal, with its docstring recording
   that the behaviour was deliberately reversed and why.

## Alternatives considered

- **Leave it, and document the trap in the README.** This is what PR #4 did
  first, and it is insufficient on its own: documentation does not reach the
  caller holding a result set, and the failure is silent precisely when the
  user believes they are measuring hybrid retrieval. Grounded,
  citation-verifiable retrieval cannot ship a default-adjacent path whose
  failure mode is "returns plausible results of the wrong kind". The README
  guidance was kept — it is still the right place to explain how to build a
  dense index — but it is no longer the only defence.
- **Refuse at `open()` instead.** Simpler to implement and to reason about,
  and rejected because it makes an entirely valid usage — open with a dense
  pair, search `"bm25"` — fail for a mode the caller never used. It would
  also have made `grk search --mode bm25` fail whenever `--embed-*` flags
  were present, which is a usability regression with no correctness gain.
- **Warn instead of raising.** Rejected twice over. The CLI configures no
  logging handler, so a `logger.warning` is invisible to exactly the user in
  the trap; and a warning attached to a result set the caller then uses is
  the "warn and continue" pattern SPEC.md §2 rules out elsewhere in this
  codebase.
- **Auto-backfill the collection on first dense search.** Rejected: an
  implicit, unbounded embedding pass triggered by a read is a large, slow,
  possibly billable side effect from an operation the caller expects to be
  cheap — and it would write vectors into a collection whose embedding
  identity the caller never chose.
- **Make `--mode dense` return empty but refuse only `hybrid`.** Rejected as
  the inconsistency that created this in the first place. Both modes promise
  the caller vector retrieval; neither can deliver it, and only one of them
  being an error is a distinction the caller cannot see or predict.

## Consequences

- **This is a breaking change for any caller relying on the old behaviour.**
  Pre-1.0 (`0.1.0.dev0`), and the behaviour being broken is one that returned
  misleading results, so the break is the point. Callers who genuinely want
  lexical results should ask for `--mode bm25`, which is unchanged and
  remains the default (ADR-0007).
- `KNOWN_LIMITATIONS.md`'s no-backfill entry stays true and gains a sharper
  ending: the collection is still not backfilled, but attempting to *read*
  it densely now fails loudly instead of silently degrading.
- The check costs one `get_manifest()` query at `open()` and nothing per
  search. `open()` was already doing manifest work for the dense pair
  (ADR-0004 decision 3), so this is one additional cheap read on a path that
  already pays for BM25's O(corpus) rebuild.
- A collection that *was* dense-ingested and then had its documents replaced
  by a BM25-only indexer keeps its manifest, so it passes this check and is
  caught by the existing orphan and lost-dense-side guards instead. The three
  checks are complementary and none subsumes another.

## References

- [ADR-0004](ADR-0004-embedding-identity-binding.md) — the manifest this
  check reads; decision 3 fixes the boundaries where *identity* is verified,
  which this does not change.
- [ADR-0005](ADR-0005-fusion-and-rerank-scoring.md) — decision 6, why fused
  scores carry no threshold, and therefore why an empty dense side is
  invisible in a fused ranking.
- [ADR-0007](ADR-0007-default-retrieval-mode.md) — BM25 remains the default,
  so the refused modes are opt-in ones.
- `KNOWN_LIMITATIONS.md` — the no-backfill limitation this makes loud.
- SPEC.md §2 (fail closed; never a silent fallback).

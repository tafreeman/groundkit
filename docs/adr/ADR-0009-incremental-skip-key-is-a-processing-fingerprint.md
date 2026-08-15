# ADR-0009 — The incremental skip key is a processing fingerprint, not a content hash

- **Status:** Accepted (owner, 2026-08-14)
- **Date:** 2026-08-14
- **Deciders:** Andy Freeman (owner)

## Context

[ADR-0002](ADR-0002-index-persistence.md) decision 3 made incremental
re-ingest **content-hash gated**: `documents.content_hash` stores a SHA-256 of
the document's bytes, and `Indexer._process` skips a source whose stored hash
matches what it just loaded.

That gate answers "have these bytes changed?". The question it is actually
asked is "would re-processing this source produce different rows?" — and the
two diverge the moment anything other than the content decides what the rows
are. Chunking configuration does exactly that. Reviewing the Phase 3 branch
produced the sequence:

```
grk ingest ./docs                  # chunk_size=100 -> 6 chunks of <=100 chars
grk ingest ./docs                  # chunk_size=250 -> "1 unchanged", 0 written
```

The second run hash-matched and skipped. The collection kept its six
100-character chunks while every configured value said 250, no error was
raised, and nothing on disk recorded which configuration the stored chunks
were actually built under. It is also permanent: every later run makes the
same comparison and reaches the same answer, so the collection can never
re-derive itself. Swapping the `ChunkerProtocol` implementation has the same
effect for the same reason.

Two things make this worse than a stale cache. First, chunk boundaries are
the substrate the whole repo rests on — `Chunk.content` must equal
`document.content[start_offset:end_offset]`, citations resolve against those
offsets, and the eval harness pins `EVAL_CHUNKING_CONFIG` explicitly
(`evals/runner.py`) precisely because a boundary shift silently changes what
a retrieval result *means*. Second, the same gate is the dense path's
re-embed gate, so the stale chunks keep their stale vectors: a caller who
changes chunk size and re-ingests gets neither new chunks nor new embeddings,
and is told "unchanged".

Silent, permanent divergence between configuration and stored state is the
failure class SPEC.md §2 exists to exclude.

## Decision

**The incremental skip key is a fingerprint over every input the indexer
knows decides a document's stored chunks, not over its content alone.**

1. **Three inputs, hashed with NUL domain separation**
   (`indexer._processing_fingerprint`): the document content, the chunker's
   type name, and the chunking configuration serialized via
   `ChunkingConfig.model_dump_json()`. Serializing the model rather than
   listing fields means a field added to `ChunkingConfig` later joins the
   fingerprint automatically — the failure mode of an explicit field list is
   that someone adds a setting and forgets, which is this defect again.
2. **`None` normalizes to `ChunkingConfig()`.** `RecursiveChunker` resolves a
   missing config to exactly that, so the two spellings produce identical
   chunks and must fingerprint identically. Hashing them apart would
   re-index an entire corpus for a change that is not one — the opposite
   defect, and just as real.
3. **The stored column keeps its name and its type.** `content_hash` remains
   an opaque hex string in the `documents` table. No schema change, no
   `PRAGMA user_version` bump, and therefore no interaction with ADR-0004
   decision 4's manifest-capability stamp — which would otherwise declare
   every existing collection unusable for dense work.
4. **The chunker contributes its type name, not a richer identity.**
   `ChunkerProtocol` exposes no identity, and widening that seam to carry one
   is not justified by this defect. A type name catches the realistic case (a
   different chunker class) and misses only a chunker whose behaviour changes
   without its type changing, which is indistinguishable from a code upgrade.

## Alternatives considered

- **Persist the chunking configuration in its own column and compare it.**
  The most explicit option, and rejected on cost: it requires a schema change
  to `documents`, and `CREATE TABLE IF NOT EXISTS` does not migrate an
  existing table, so it forces a `SCHEMA_VERSION` bump. Under ADR-0004
  decision 5 that makes every pre-existing collection manifest-incapable and
  refused for dense work — a much larger blast radius than the defect, to
  store data that the hash already encodes. The fingerprint gets the same
  fail-closed behaviour with no migration.
- **Invalidate the whole collection when the configuration changes.**
  Requires knowing the previous configuration, which is the persisted-column
  option above with an extra step, and it throws away documents whose
  fingerprint is genuinely unchanged.
- **Leave it and document it in `KNOWN_LIMITATIONS.md`.** Rejected on the
  same grounds as ADR-0008: the divergence is invisible at the moment it
  matters, the run reports "unchanged" rather than anything a reader would
  investigate, and documentation does not reach a caller holding a result
  set. This repo has now made that call twice.
- **Include the embedding identity in the fingerprint too.** This would make
  enabling `--dense` over an existing collection re-index and therefore
  backfill it, closing the no-backfill limitation as a side effect. Rejected
  as a separate decision wearing this one's clothes: it changes what
  `--dense` costs on an existing collection from nothing to a full re-embed,
  it is the auto-backfill ADR-0008 explicitly declined, and ADR-0008's error
  message already directs callers to a fresh collection. The no-backfill
  limitation stands unchanged.

## Consequences

- **One full re-index the first time this ships against an existing
  collection.** Stored hashes were computed the old way, so nothing matches
  and every document is rebuilt. This is correct rather than merely
  tolerable: those chunks were produced under a configuration the store never
  recorded, so re-deriving them is the only way to know what they are. Cost
  is seconds, which is ADR-0004 decision 5's reasoning applied to chunks.
- A dense-enabled collection pays a re-embed on that first run, which against
  a hosted provider is billable. It is a one-off, it is bounded by corpus
  size, and it is the same work a caller would have had to do manually to
  correct the divergence.
- `documents.content_hash` no longer means "hash of the content". Every
  docstring naming it as the skip key says so; ADR-0002 decision 3 carries a
  forward pointer here rather than being rewritten.
- The gate remains a single mechanism shared by chunking and embedding
  (`indexer._process`), so incremental re-embedding stays a property of the
  one skip rather than a second mechanism that could drift from it.
- Nothing about the *ordering* invariant changes: dense writes still precede
  the SQLite commit, and the residue analysis in `KNOWN_LIMITATIONS.md` is
  unaffected.

## References

- [ADR-0002](ADR-0002-index-persistence.md) — decision 3, the content-hash
  gate this replaces; decisions 1, 2 and 4 are untouched.
- [ADR-0004](ADR-0004-embedding-identity-binding.md) — decision 4 (the schema
  stamp a new column would have forced) and decision 5 (indexes are cheap to
  regenerate pre-1.0), both of which shape the alternatives above.
- [ADR-0008](ADR-0008-dense-search-requires-a-dense-collection.md) — the
  no-backfill trap, deliberately left as-is here.
- SPEC.md §2 (fail closed; no silent divergence between configuration and
  persisted state).

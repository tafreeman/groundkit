# ADR-0006 — The dense seam returns `(Chunk, score)`, not `RetrievalResult`

- **Status:** Accepted (owner, 2026-08-13)
- **Date:** 2026-08-13
- **Deciders:** Andy Freeman (owner)

## Context

`VectorStoreProtocol.search` was declared in Phase 1 to return
`list[RetrievalResult]`, inherited from ARP's vector store shape (ADR-0001).
It had no implementation until Wave A of Phase 3, so the declaration was never
exercised — and building the first implementation surfaced that it conflicts
with a correction groundkit had already made on the lexical side.

`RetrievalResult` requires a `source`. Only `Document` carries one; `Chunk`
does not. `BM25Index.search` therefore returns `(Chunk, score)` pairs and
leaves the join to `retrieval/search.py`, where `Retriever.search` resolves
each hit against `get_document_sources()` and **fails closed** — raising
`RetrievalError` rather than emitting a citation it cannot verify. That
asymmetry is documented deliberately: BM25 doesn't build `RetrievalResult`
precisely because it cannot know a source.

A vector store has the same limitation and the same lack of a store handle, so
the Phase 1 signature left it only one way to satisfy the return type: read
`source` out of `chunk.metadata`. That is available — `ingestion/chunking.py`
seeds `metadata={"source": document.source, **document.metadata}` — and the
first implementation duly did it, failing closed when the key was absent.

The problem is that `chunk.metadata["source"]` is a **snapshot taken at chunk
time**, while `documents.source` in SQLite is the durable truth (ADR-0002)
that the lexical path already treats as authoritative. The two can disagree.
Re-ingest a document from a moved or renamed path and a dense hit cites the
stale path while a BM25 hit for that same document cites the current one —
in a single fused response, with no error raised on either side. Phase 3's
whole purpose is to return those two result sets together.

## Decision

**`VectorStoreProtocol.search` returns `list[tuple[Chunk, float]]`**, matching
`BM25Index.search` exactly. Resolving a chunk to its document's source, and
therefore constructing every `RetrievalResult`, belongs to
`retrieval/search.py` and happens against the metadata store alone.

Consequences that follow and are part of this decision:

1. **No store may read `chunk.metadata["source"]` to build a citation.** The
   key remains populated by the chunker (removing it is a separate change with
   its own migration cost, deliberately not bundled here) but is not load-
   bearing for retrieval.
2. Scores crossing the seam are still `>= 0.0` with a deterministic tie-break,
   so the producer-side clamp closing ADR-0001 hazard 2's class of defect
   stays where it is; `Retriever` is a backstop, not the enforcement point.
3. Both stores' `add`/`delete` semantics are unchanged. This ADR touches the
   read path only.

## Alternatives considered

- **Keep `RetrievalResult` and populate `metadata["source"]` reliably at
  ingest.** Rejected: it makes a per-chunk copy of a document-level fact
  load-bearing, which is a second source of truth for the same value —
  precisely the shape ADR-0002 and SPEC.md §2 rule out. It also cannot be
  made correct by care alone; the copy goes stale on any re-ingest at a new
  path, and nothing detects it.
- **Pass a source-resolver callback into the vector store.** Rejected: it
  hands the storage layer a retrieval-layer concern through a back channel,
  and every implementation would have to be trusted to call it. The seam
  boundary is the point.
- **Give the vector store a metadata-store handle.** Rejected: it inverts the
  dependency (index → retrieval), and makes every vector store implementation
  responsible for citation correctness rather than one place.
- **Leave the asymmetry and normalize at the fusion layer.** Rejected: fusion
  would consume two different types and have to unpick a `RetrievalResult`
  back into a chunk to rank it. The divergence would also survive as a live
  correctness bug in the dense-only path, not merely an inconvenience.

## Consequences

- Wave C's RRF consumes two lists of identical shape, and the source join
  happens once, after fusion, on the surviving results — fewer joins than
  resolving per-retriever, and one place where the fail-closed rule lives.
- `_to_retrieval_result` and the "chunk has no source in metadata" error path
  are deleted from `index/dense.py`, along with their tests. A vector store
  can no longer fail for a citation-shaped reason, because it no longer makes
  citations.
- A dense store is now usable with chunks that carry no metadata at all, which
  makes test fixtures simpler and removes a class of fixture-only failure.
- The Phase 1 protocol changed after being declared stable. That is the
  correct outcome — a Protocol with no implementation is a hypothesis, and
  this one was falsified the first time it met a real store. The signature
  parity conformance tests are what made the change safe to make.

## References

- [ADR-0001](ADR-0001-promote-vs-rewrite.md) — the ARP port whose vector-store
  shape this corrects; hazards 3 and 4 constrain the rest of the signature and
  are unaffected.
- [ADR-0002](ADR-0002-index-persistence.md) — SQLite as the durable truth for
  the document→source mapping.
- [ADR-0004](ADR-0004-embedding-identity-binding.md) — the sibling decision
  preventing the dense store from diverging from SQLite on embedding identity;
  this one prevents divergence on source.
- SPEC.md §2 (no in-process state shadowing persisted state) and §5.2
  (`RetrievalResult` score normalization guaranteed by producers).

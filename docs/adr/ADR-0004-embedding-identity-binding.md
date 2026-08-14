# ADR-0004 — Dense store integrity: embedding identity binding and delete-path safety

- **Status:** Accepted (owner, 2026-08-13)
- **Date:** 2026-08-13
- **Deciders:** Andy Freeman (owner)

## Context

Phase 3 writes the first dense vectors. Two ways the dense store can silently
diverge from SQLite — the durable truth per [ADR-0002](ADR-0002-index-persistence.md)
— have to be closed before that write happens, not after, and they are recorded
together because they are the same failure in two costumes: the dense side
holding something the metadata store does not agree with, with no error raised.

**Divergence 1 — semantic space.** SPEC.md §2 forbids cross-provider embedding
fallback outright, because "mixed semantic spaces corrupt an index silently"
(ARP KNOWN_LIMITATIONS §4.7 is the named cautionary precedent). Today that
principle is enforced only at the provider call: `EmbeddingConfig.dimensions`
is checked per response, and a width mismatch raises `EmbeddingError`
(`config.py:73`, `providers/embeddings.py`). Nothing enforces it across
*time*. The SQLite schema is `documents` + `chunks` only (`index/metadata.py`)
— it records no provider, no model name, no vector width, and no schema
version. Index a collection with `nomic-embed-text`, reopen it configured for
a different model, and every dense result is quietly wrong: no exception, no
log line, just degraded ranking that looks like a bad retriever rather than a
corrupt index.

Width-checking alone does not close this. The default is `nomic-embed-text` at
768 dimensions (`config.py:73-74`), and `all-mpnet-base-v2` is also 768 — two
mutually incomprehensible semantic spaces with identical arithmetic. Any check
that infers identity from vector width admits exactly the substitution it was
built to reject.

**Divergence 2 — deletion scope.** LanceDB filters and deletes through SQL
expression strings; its documentation "embraces the utilization of standard SQL
expressions as predicates for filtering operations", with examples of the form
`item IN ('foo', 'bar')`. ADR-0001 hazard 5 records ARP interpolating document
IDs directly into that expression. This is not a theoretical concern: the same
construction produced a real, filed defect in LangChain's LanceDB integration,
where building a delete predicate by string concatenation emitted
`id IN ('doc1,doc2')` instead of `id IN ('doc1','doc2')` — one combined string
where two identifiers were meant (langchain-community#190). That bug deletes
nothing; the inverse bug deletes too much. Either way SQLite and LanceDB stop
agreeing about what the collection contains, and `Retriever.open()` — which
rebuilds BM25 from SQLite and would now join it against a differently-populated
vector table — has no way to notice.

## Decision

1. **A collection manifest, single-row, in SQLite.** A `collection_manifest`
   table stores the embedding `provider`, `model_name`, `dimensions`, and a
   `created_at` timestamp. It is written on the first dense write and is
   thereafter immutable for the life of the collection. It lives in SQLite
   because ADR-0002 makes SQLite the durable truth, and because a manifest that
   can be separated from the data it describes is not a manifest.
2. **Identity is `(provider, model_name, dimensions)`, checked as a triple.**
   Not dimensions alone (insufficient, per Context), and not provider alone.
   All three must match or the collection does not open.
3. **Verified at every boundary that could introduce a second space** —
   `Retriever.open()` and any ingest that would write vectors — and a mismatch
   raises a new typed `IndexIdentityError` under `GroundkitError`. Never a
   re-embed, never a fallback, never a warning-and-continue. This is SPEC.md §2
   fail-closed applied one layer below where it currently lives.
4. **`PRAGMA application_id` and `PRAGMA user_version`** are set on store
   creation: `application_id` marks the file as groundkit's, `user_version`
   carries the schema version. SQLite reserves both for exactly this purpose —
   the user-version "is an integer that is available to applications to use
   however they want", and SQLite itself makes no use of it. Version bumps are
   applied inside a transaction with the version write as the last statement
   before commit, so a failed migration cannot leave a store claiming a version
   it does not have.
5. **Pre-1.0, an unversioned store is a rebuild, not a migration.** A store
   without `application_id`/`user_version` predates this ADR; `grk` reports it
   and refuses to open it for dense work rather than guessing. The repo is
   `0.1.0.dev0`, `evals/results/` is gitignored, and every index in existence is
   reproducible from `grk ingest` — a migration path here would be code written
   to preserve data that costs seconds to regenerate.
6. **Delete predicates are never built by string interpolation.** Document IDs
   reaching a LanceDB predicate are validated against the same strict identifier
   pattern used for collection names before use, and the predicate is
   constructed by a single helper with a regression test that feeds it a hostile
   ID containing quote characters. Deletion is verified by count, not assumed:
   the dense delete returns the number of vectors removed, and the caller
   reconciles it against the chunk count SQLite deleted.

## Alternatives considered

- **Sidecar manifest file** (`<collection>.manifest.json` beside the SQLite
  file). Rejected: it reintroduces the exact anti-pattern ADR-0002 and SPEC.md
  §2 ban — a second source of truth that can drift from, or be separated from,
  the store it describes. A copy, move, or partial backup silently decouples
  them, and the failure is invisible until a query returns nonsense.
- **Per-chunk embedding identity columns.** Rejected on two grounds: it is
  redundant storage for a value that is invariant across the collection, and
  more importantly it makes a mixed-space collection a *representable state*.
  The schema should make the corrupt condition impossible to express, not merely
  detectable after the fact.
- **Infer identity from the first stored vector's width.** Rejected: this is the
  768-vs-768 case above. It infers where it should assert, and admits the
  substitution it exists to prevent.
- **Automatic re-embed on mismatch.** Rejected: it converts a configuration
  error into an expensive silent success, which is precisely the "never a
  fallback" clause of SPEC.md §2. It also cannot be correct in general — the
  source documents may no longer exist at their recorded paths.
- **Delegate to LanceDB's own schema enforcement.** Rejected: LanceDB enforces
  vector width, which is the check already shown to be insufficient, and it has
  no notion of which model produced the vectors. It also inverts ADR-0002's
  truth ordering.
- **Escape quotes in delete predicates rather than validating IDs.** Rejected as
  the primary defence: escaping is a correctness patch applied at the point of
  greatest risk, and langchain-community#190 is evidence of how easily
  hand-built predicate strings go wrong even without an adversary. Validating
  the identifier shape first means a malformed ID cannot reach the predicate at
  all; escaping remains as defence in depth, not as the plan.

## Consequences

- The first dense write becomes a schema-touching operation. That cost is paid
  once, deliberately, before any vector exists — which is the entire reason this
  ADR blocks Wave A rather than following it.
- `Retriever.open()` gains a manifest read. Combined with the existing O(corpus)
  BM25 rebuild, this makes `open()` measurably more expensive; ADR-0002 already
  carries a revisit trigger for open cost, and Phase 3 should measure against it
  rather than let Phase 4's per-request service discover it.
- Changing embedding model becomes an explicit destroy-and-reingest, and `grk`
  must say so in the error rather than leaving the user to infer it. This
  matches the wider ecosystem: index dimensions generally cannot be changed
  after creation, and switching models means recreating the index.
- A collection is now self-describing, which Phase 4's `index_status` MCP tool
  can report and Phase 6's observability can label. That was not the motivation,
  but it is a real dividend.
- `IndexIdentityError` is a new public error type and belongs in the errors
  taxonomy documented for Phase 7.

## References

- SPEC.md §2 (fail closed; no cross-provider embedding fallback), §5.3
  (persistence layout), and [ADR-0002](ADR-0002-index-persistence.md) (SQLite as
  durable truth) — the in-repo constraints this ADR implements.
- [ADR-0001](ADR-0001-promote-vs-rewrite.md) hazard 5 — unescaped delete
  predicate, the ported defect decision 6 closes.
- [SQLite: PRAGMA user_version / application_id](https://www.sqlite.org/pragma.html#pragma_user_version)
  — the reserved, application-defined integers used in decision 4.
- [langchain-community issue #190 — "LanceDB delete method generates malformed
  SQL when deleting by IDs"](https://github.com/langchain-ai/langchain-community/issues/190)
  — a filed instance of hazard 5's construction: `id IN ('doc1,doc2')` where
  `id IN ('doc1','doc2')` was meant, caused by concatenating IDs into a single
  quoted string.
- [LanceDB — Metadata filtering](https://docs.lancedb.com/search/filtering) —
  documents SQL expression strings as the filter/delete predicate mechanism.
- [Milvus — dimension mismatch with Sentence Transformer embeddings](https://milvus.io/ai-quick-reference/why-do-i-see-a-dimension-mismatch-or-shape-error-when-using-embeddings-from-a-sentence-transformer-in-another-tool-or-network)
  — fixed-width models (384/768/1024) and the shape errors that follow a model
  swap; corroborates that width is a weak identity signal since models share it.

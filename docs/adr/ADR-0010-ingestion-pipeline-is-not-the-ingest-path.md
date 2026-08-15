# ADR-0010 — `IngestionPipeline` is a load→chunk utility, not the ingest path

- **Status:** Accepted (owner, 2026-08-14)
- **Date:** 2026-08-14
- **Deciders:** Andy Freeman (owner)

## Context

`groundkit.ingestion.IngestionPipeline` and `groundkit.indexer.Indexer` both
orchestrate "load, then chunk". The pipeline is exported from
`groundkit.ingestion.__init__`, is covered by `tests/test_ingestion_pipeline.py`,
and has **no production consumer**: nothing in `src/` constructs one. `Indexer`
imports `DEFAULT_MAX_CONCURRENT` and `discover_files` from `ingestion/pipeline.py`
and its own directory walk, by its docstring, "mirrors
`IngestionPipeline.ingest_directory`".

An external repository review (August 2026) read that shape as duplicated
orchestration and asked the question directly: is the public `IngestionPipeline`
a supported API to compose into `Indexer`, or should it be de-publicized until it
has a production consumer? It framed the answer as a choice between those two.

The framing has a false premise. Composing them is not a neutral refactor that
merely costs effort — it is the one option that breaks something:

`Indexer._process` loads a source, computes a **processing fingerprint**
(ADR-0009), compares it against the stored one, and `continue`s on a match
**before it chunks**. `IngestionPipeline.ingest()` takes no store, knows no
fingerprint, and always loads *and* chunks. Composing `ingest()` into `_process`
therefore either chunks every unchanged document on every run, or requires
splitting the pipeline into load-only and chunk-only halves — at which point what
remains is the two protocol calls `Indexer` already makes directly, wrapped.

The first cost is not merely CPU. ADR-0009 records that this one gate is *also*
the dense re-embed gate: an unchanged document is never embedded because it never
reaches chunking. Moving chunking ahead of the gate separates those two, and
re-embedding an unchanged corpus against a hosted provider is billable. The
property ADR-0009 exists to guarantee — that incremental re-embedding is a
consequence of the single skip rather than a second mechanism free to drift —
depends on nothing running before that comparison.

## Decision

`IngestionPipeline` stays public, stays tested, and is **not** composed into
`Indexer`. Neither branch of the review's question is taken.

1. It is a **standalone load→chunk utility**: for a caller who wants chunks from
   a file or directory with bounded concurrency and no persistence. That is a
   coherent thing to want, it is the shape ARP's ingestion had (ADR-0001), and it
   is why the class has tests despite having no internal caller.
2. `Indexer` is **the ingest path**. It is the only thing that may write to a
   collection, because incremental behavior, fingerprinting, manifest binding and
   dense lockstep all live there and all depend on ordering the pipeline does not
   express.
3. The shared surface between them is deliberately narrow and stays that way:
   `discover_files` and `DEFAULT_MAX_CONCURRENT`, both pure and both about
   *finding* files, neither about processing them.
4. Both modules state the boundary in their docstrings, so the next reader who
   notices the resemblance finds the reason before acting on it.

## Alternatives considered

**Compose `IngestionPipeline` into `Indexer`.** The review's preferred reading of
its own question. Rejected: it moves chunking ahead of the fingerprint gate,
which costs a re-chunk of every unchanged document per run and, on a dense
collection, a re-embed that ADR-0009 spent a defect fix to avoid. A variant that
splits the pipeline in half to preserve the ordering is not a composition — it
dissolves the pipeline into the protocol calls `Indexer` already makes, and
deletes a working public class to remove roughly ten lines of resemblance.

**De-publicize it (drop it from `ingestion.__init__`, or delete it).** Rejected on
evidence rather than principle: "no production consumer" is true and is not the
same as "no consumer". The class is the documented directory-scale load→chunk
entry point (SPEC.md §4), it is exercised, and removing a tested public utility to
answer a duplication question would trade a real capability for a cosmetic one.
Revisit if it is still unconsumed when the Phase 4 adapters land, since those are
the most likely consumer and their not using it would be the actual signal.

**Leave it undocumented and unanswered.** Rejected because the question has now
been asked twice — once by the review, once by the docstring that says `Indexer`
"mirrors" the pipeline. An unanswered resemblance invites exactly the merge that
breaks incremental re-embedding, and the breakage is silent: the collection stays
correct, the bill and the wall-clock go up, and no test fails.

## Consequences

- Two modules keep a visible resemblance that is not duplication. The cost is
  paid in docstrings, which is the cheapest place to pay it.
- `IngestionPipeline` remains public API surface with no internal caller, so its
  tests are the only thing holding its contract. That is acceptable while it is
  the documented directory-scale entry point, and is the reason its tests may not
  be deleted as "testing an unused class".
- Any future incremental behavior must land in `Indexer`, not the pipeline. A
  pipeline that grew a store reference would be `Indexer` under another name.
- The Phase 4 REST/MCP adapters must call `Indexer` for anything that writes.
  Reaching the pipeline directly from an adapter would produce chunks that never
  reach a collection.

## References

- [ADR-0009](ADR-0009-incremental-skip-key-is-a-processing-fingerprint.md) — the
  fingerprint gate whose position ahead of chunking this decision protects, and
  its dual role as the re-embed gate.
- [ADR-0001](ADR-0001-promote-vs-rewrite.md) — the promote decision that brought
  ARP's ingestion shape over, plus gap #2 (directory-scale ingestion) that
  `ingest_directory` exists to close.
- [ADR-0002](ADR-0002-index-persistence.md) — decision 3, the incremental gate
  ADR-0009 rewrote.
- SPEC.md §4 (directory-scale ingestion in v1 scope), §2 (no silent divergence
  between configuration and persisted state).

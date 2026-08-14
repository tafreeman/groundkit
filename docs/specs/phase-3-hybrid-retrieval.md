# Phase 3 — Hybrid retrieval + rerank

Feature spec for SPEC.md §9 Phase 3: *dense (LanceDB w/ metadata filtering),
RRF, optional cross-encoder (normalized scores); each with eval delta vs
baseline.*

Status: **planned**. Nothing here overrides SPEC.md; where this document makes
a decision that SPEC.md does not already contain, it names the ADR that must
land first.

## 1. What Phases 1–2 already provide

Phase 3 is unusually well-seated: the seams exist and are conformance-tested,
so this phase is mostly filling them in rather than reshaping anything.

| Asset | State | Phase 3 use |
|---|---|---|
| `index/protocols.py::VectorStoreProtocol` | defined, exact signature, no `**kwargs` | implement twice (LanceDB + in-memory) |
| `retrieval/protocols.py::RerankerProtocol` | defined, param named `query` | implement once |
| `RetrievalConfig.rrf_k` | present, default 60 | fusion constant |
| `EmbeddingConfig` + Ollama/OpenAI embedders | shipped Phase 1 | vector generation |
| `InMemoryEmbedder` | shipped, deterministic hash vectors | offline tests — **not** eval quality (§6) |
| `EvalReport.stages[].stage` | `Literal["bm25","dense","fusion","rerank"]` | append stages, no schema change |
| Delta semantics | derived at read time vs `stages[0]` | no stored-delta drift |
| `retrieval/*` in `core_subset` | glob | `fusion.py`/`rerank.py` auto-gated at 80% |

Two things are *not* seated and must be built before any vector is written:

- **`Indexer` has no embedder and no vector store.** It writes only to
  `MetadataStoreProtocol`. The dense write path does not exist.
- **The SQLite schema records no embedding identity and no schema version.**
  `documents` and `chunks` only. See §3.1 — this is the highest-consequence
  gap in the phase.

## 2. Done means

1. Dense retrieval over LanceDB with metadata filtering that provably filters.
2. RRF fusion of lexical + dense, pure and deterministic.
3. Optional local cross-encoder rerank with normalized non-negative scores.
4. ADR-0001 hazards 2, 3, and 5 each fixed *with a named regression test*
   (SPEC.md §1: the hazard list is a spec obligation, not advice).
5. An eval report containing `bm25` (baseline) + `dense` + `fusion` + `rerank`
   stages, with each stage's delta vs baseline reported — **including if a
   stage loses to BM25** (SPEC.md §6 baseline discipline).
6. All gates green: ruff, ruff-format, `mypy --strict`, pytest, both coverage
   gates, pip-audit, gitleaks.
7. SPEC.md §9 status updated, `KNOWN_LIMITATIONS.md` updated, ADRs accepted.

## 3. Decisions that need an ADR before code

[ADR-0004](../adr/ADR-0004-embedding-identity-binding.md) and
[ADR-0005](../adr/ADR-0005-fusion-and-rerank-scoring.md) are **Accepted**. The
subsections below state the problems they were written to settle; the ADRs hold
the decisions, alternatives, and citations.

A third decision was forced by Wave A rather than anticipated here:
[ADR-0006](../adr/ADR-0006-dense-seam-returns-chunk-score-pairs.md). Phase 1
declared `VectorStoreProtocol.search` as returning `RetrievalResult`, which
turned out to be unsatisfiable without reading `source` from an ingest-time
`chunk.metadata` snapshot — a copy that disagrees with `documents.source` after
any re-ingest at a new path, so dense and BM25 hits could cite different paths
for the same document in one fused response. The seam now returns
`(Chunk, score)`, matching BM25. This is the useful kind of surprise: a
Protocol with no implementation is a hypothesis, and Wave A was the first thing
to test it.

Wave B forced two implementation shapes the original plan did not spell out,
though neither is a new decision: ADR-0004 already settled *what* identity is
(decision 2) and *that* deletes must be reconciled (decision 6), and these are
how those decisions were realized in code, not departures from them. The
manifest seam (`write_manifest`/`verify_manifest`) takes the narrow
`EmbeddingIdentity` triple — `provider`, `model_name`, `dimensions` — rather
than a whole `EmbeddingConfig`, which also carries operational settings
unrelated to semantic-space identity and a `provider` `Literal` no
third-party embedder could satisfy. And `Indexer`'s dense write order —
chunk, embed, write the manifest, delete the previous document's vectors, add
the new ones, commit SQLite last — is itself an invariant, chosen so SQLite
is never left ahead of the dense store (see `KNOWN_LIMITATIONS.md` for the
failure mode this avoids). No ADR was skipped for either; both are
implementation shapes serving decisions ADR-0004 already made.

### 3.1 ADR-0004 — Embedding identity binding (blocking, do first)

**Problem.** SPEC.md §2 forbids cross-provider embedding fallback because
"mixed semantic spaces corrupt an index silently". The current store cannot
detect that condition: nothing persists which provider, model, or dimension
produced the vectors in a collection. Index a collection with
`nomic-embed-text` (768-d), reopen it configured for a different model, and
every dense result is silently garbage — the exact failure the principle
exists to prevent, one layer below where the principle is currently enforced.

**Decision to record.** A collection manifest — provider, model name,
dimensions, plus a schema version — persisted in SQLite, written on first
dense write, and verified on every `Retriever.open()` and every subsequent
ingest. Mismatch is a typed error (`ProviderNotConfiguredError` family / new
`IndexIdentityError`), never a re-embed and never a fallback.

Also settle here: whether adding a `schema_version` to an existing Phase 1/2
SQLite file is a migration or a rebuild. Rebuild is defensible while the repo
is pre-1.0 and `evals/results/` is gitignored anyway; say so explicitly rather
than leaving it implicit.

**Why blocking:** it changes the on-disk schema. Landing it after the first
vector is written turns a schema addition into a migration.

### 3.2 ADR-0005 — Fusion and rerank scoring

- RRF formula and the `rrf_k` default (60 is already in config — record *why*).
- Deterministic tie-break for equal fused ranks. BM25 determinism was pinned in
  Phase 2; fusion must not reintroduce order instability.
- Rerank score normalization: raw cross-encoder logits are unbounded and
  frequently negative, while `RetrievalResult.score` is `Field(ge=0.0)`. This
  is ADR-0001 hazard 2 verbatim. Record the chosen mapping (sigmoid vs
  min-max) and its consequence — sigmoid preserves ordering and bounds to
  (0,1) without needing the whole batch; min-max is batch-dependent and makes
  scores incomparable across queries.
- Model choice for the cross-encoder, and that it is a **local, non-LLM**
  model (SPEC.md §4 puts LLM-based reranking out of scope).

## 4. Wave plan

Sized to mirror Phase 1/2 commit granularity — each wave is a reviewable
changeset that leaves the tree green.

### Wave A — dense store, both paths

- `ADR-0004` accepted; collection manifest + `schema_version` in SQLite.
- `index/dense.py`: `LanceDBVectorStore` implementing `VectorStoreProtocol`.
- `InMemoryVectorStore` in the same seam. SPEC.md §5.3 requires metadata
  filtering "on both in-memory and LanceDB paths from the first dense-store
  commit, with regression tests on both paths" — both land together or neither.
- LanceDB added as an **optional extra**, not a base dependency.
- Hazard regression tests:
  - **Hazard 3 (silent filter drop):** assert filtering actually removes
    non-matching chunks, *and* that no call spelling silently no-ops — a
    misspelled filter key must raise, not be absorbed.
  - **Hazard 5 (unescaped delete predicate):** a document ID containing quote
    characters must not alter the delete expression. Parameterize or escape;
    test with a hostile ID.
  - Dimension/length mismatch on `add` → `StorageError`.

### Wave B — dense write path (**Landed**)

- `Indexer` accepts optional keyword-only `embedder` and `vector_store`;
  both absent leaves Phase 1 BM25-only behaviour unchanged and still
  tested, and exactly one supplied raises `ConfigurationError` at
  construction.
- `replace_document` parity delivered: deleting, replacing, renaming, or
  emptying a document deletes its vectors in the same logical operation.
- Incremental re-index re-embeds only changed documents — the existing
  content-hash gate short-circuits before chunking, extended rather than
  duplicated.
- Manifest verified on ingest and fails closed: `verify_manifest` runs
  before any load/chunk/embed/delete work, and `write_manifest` binds the
  collection on the first real dense write. `Retriever.open()`
  verification remains Wave C — see §3 and `KNOWN_LIMITATIONS.md`.
- `IndexReport` gained `vectors_written` and `vectors_deleted`.

### Wave C — dense retrieval + RRF fusion (**Landed**)

- `retrieval/fusion.py`: pure RRF, no I/O, deterministic tie-break.
- `Retriever` gains a dense path and a hybrid mode; `SearchResponse.metadata`
  reports the stage honestly (`"dense"` / `"fusion"`).
- Snapshot semantics extended to the dense side and documented in the class
  docstring alongside the existing BM25 wording — a stale retriever must fail
  closed the same way on both.
- Citation resolution unchanged: dense hits resolve through the same
  `citations.py` path, so a dense result is exactly as verifiable as a lexical
  one.
- **Inherited from Wave B — `Retriever.open()` manifest verification
  (closed).** ADR-0004 decision 3 names two boundaries; Wave B closed ingest
  and left this one, because a retriever with no dense read path cannot
  introduce a second semantic space and so had nothing to verify. Wave C gave
  it something to read and, in the same commit, added the check:
  `Retriever.open()` now verifies the collection's manifest against the
  supplied embedder's identity before the O(corpus) BM25 rebuild runs,
  whenever a dense pair is passed. A mismatch is `IndexIdentityError`. Both
  ADR-0004 decision-3 boundaries are now closed.
- **Inherited from Wave B — CLI wiring (closed, default off).** `grk ingest`
  previously constructed its `Indexer` without an embedder or vector store, so
  the command line wrote no vectors. It was deferred here deliberately rather
  than half-wired in Wave B: turning it on needed a provider flag and a
  running Ollama, and was only useful once `grk search` could read what it
  writes, so ingest and search got their flags together. Wave C adds
  `grk ingest --dense` and `grk search --mode {bm25,dense,hybrid}`, sharing
  `--embed-provider`/`--embed-model`/`--embed-dimensions`/`--embed-base-url`.
  Both remain opt-in: the default install, default commands, and CI need no
  Ollama, and `grk search` still defaults to `bm25` (Q1, below).

### Wave D — optional cross-encoder rerank

- `retrieval/rerank.py` behind `RerankerProtocol`, optional extra.
- **Hazard 2 regression test:** feed negative logits, assert no
  `ValidationError` and that ordering is preserved.
- Unconfigured reranker → typed error, never a silent passthrough (SPEC.md §2
  fail-closed).
- Heavy deps (torch/sentence-transformers) must stay out of the base install;
  CI's default job must not pull them.

### Wave E — eval deltas (the phase gate) (**Landed**)

Taken before Wave D deliberately. R1 names the eval delta as the phase's
largest scheduling risk, and Wave E depends on nothing rerank provides, so
building it first means Wave D reports into a harness that already works
rather than one being built under it.

- `run_eval` accepts an optional keyword-only `embedder`/`vector_store` pair
  (both or neither, `ConfigurationError` otherwise — matching `Indexer` and
  `Retriever`) and emits `bm25` → `dense` → `fusion` into one report. One
  index and **one retriever** serve every stage, so the stages differ by
  retrieval strategy alone; ground truth is resolved once per judgment and
  shared, so no two stages can disagree about what counted as relevant.
- Deltas are derived at read time by `evals/delta.py` and never stored.
  `StageDelta` is a return type, never a field of `EvalReport`.
- Per-stage latency percentiles come from each stage's own per-query
  timings — the existing `StageResult` fields, now populated per stage.
- **The honest-loss path is tested, in both directions.** A losing stage is
  reported with its real numbers, flagged `is_regression`, and printed with
  a `REGRESSION vs baseline` verdict. Verified per SPEC.md §8 by mutating the
  source twice — `is_regression` forced to `False`, and a filter that drops
  stages losing to baseline — and observing 6 and 10 test failures
  respectively, then restoring.
- **No noise threshold.** `is_regression` is a strict sign test. A tolerance
  band would be an invented number, and R2 says a real effect on this corpus
  can be smaller than any epsilon anyone would pick. `is_improvement` is
  tracked separately rather than as the negation, so a stage that gains one
  metric and loses another reports as `MIXED` instead of hiding half the
  result.
- `EVAL_GATED=1` gates a real-model run (`tests/test_eval_gated.py`,
  `.github/workflows/eval-gated.yml`: schedule + `workflow_dispatch` + an
  `eval-gated` PR label). Skipping cleanly is the *default* outcome; the gated
  tests assert the run reports honestly and deliberately **do not** assert
  that dense or fusion beats BM25, since a test that reddens on a loss would
  pressure the next person to grow the corpus until it passes.
- `RunConfig` gained two optional, defaulted fields — `embedding` (the
  ADR-0004 identity triple) and `rrf_k`, both `None`-meaning-absent. Without
  the first, two runs over identical golden data with *different* embedders
  agree on `corpus_hash` and `judgments_hash` while measuring two different
  semantic spaces: ADR-0004's silent-mixing failure one layer above the
  index. Additive-with-default rather than a `schema_version` bump; the
  reasoning, and the condition that would force a bump instead, are in
  `schema.py`'s module docstring.

### Wave F — docs and gates

- SPEC.md §9 Phase 3 → done with date; `KNOWN_LIMITATIONS.md` updated.
- ADR-0004/0005 accepted and listed in `docs/adr/index.md`.
- Decide whether `src/groundkit/index/dense.py` joins
  `[tool.groundkit.coverage].core_subset`. It is core retrieval, and the
  current subset list would otherwise exclude it while covering `bm25.py` —
  an asymmetry worth closing deliberately rather than by omission.
- No hardcoded metric numbers in any doc (SPEC.md §2).

## 5. Hazard obligations carried into this phase

ADR-0001's hazard list is a spec obligation: each named defect gets a fix and
a regression test **in the phase that ports it**. Phase 3 ports the three that
have been waiting:

| # | Defect | Where it lands | Test shape |
|---|---|---|---|
| 2 | Cross-encoder feeds raw negative logits into a `ge=0.0` contract | Wave D | negative logits → no crash, order preserved |
| 3 | `**kwargs` absorbs any metadata filter without error | Wave A | filter works; wrong key raises; no spelling no-ops |
| 5 | Document IDs interpolated into LanceDB's SQL-like delete expression | Wave A | hostile ID with quotes deletes exactly one document |

## 6. Risks and open questions

**R1 — The eval delta cannot be measured in normal CI.** `InMemoryEmbedder`
produces deterministic *hash-derived* vectors, which are semantically
meaningless. It is correct for exercising code paths and wrong for measuring
retrieval quality. A dense delta computed with it would be noise presented as
a number, which SPEC.md §2 ("real data only") forbids. Consequence: the
quality delta must come from a real-embedding run (Ollama, local or gated),
while normal CI runs only the deterministic structural assertions. The
`EVAL_GATED=1` mechanism in SPEC.md §6 already anticipates this — Phase 3 is
the phase that has to actually build it. **This is the single largest
scheduling risk in the phase.**

**Mechanism built in Wave E; measurement still outstanding.** `EVAL_GATED=1`
now exists as a pytest gate (`tests/test_eval_gated.py`, skipped by default
and by design) and a workflow (`.github/workflows/eval-gated.yml`). Normal
CI remains offline and runs only structural assertions on the dense path;
no test anywhere asserts a dense or fusion quality *value*. Recording the
embedder identity in `RunConfig.embedding` makes the distinction
machine-checkable rather than a convention a reader has to remember: a
report whose `embedding.provider` is `inmemory` is self-labelling as
structural, and the CLI stamps an explicit warning on it.

**Retired.** A gated run against the committed golden corpus, using
`nomic-embed-text` through local Ollama, has produced a real delta — see Q1
for what it says and what it does and does not settle. The risk R1 named was
that the phase would reach its gate with no way to measure quality; that is
now false in both senses, mechanism and measurement. Worth noting for
calibration: the same harness run with `InMemoryEmbedder` reported dense and
fusion as clear *regressions*, and the real model reported them as clear
improvements. The two disagree in direction, on the same corpus, through the
same code — which is precisely why SPEC.md §2 refuses to let the hash
embedder produce a quality number, and is the sharpest available evidence
that the refusal is load-bearing rather than ceremonial.

**R2 — The corpus may be too small for a meaningful delta.** The Phase 2
baseline runs over 10 documents / 84 chunks / 44 judgments. On a corpus that
size, dense-vs-BM25 differences may be within noise, and a "win" may not
survive corpus growth. Mitigations, in preference order: state the corpus size
alongside every delta; treat a loss or a wash as a legitimate reported outcome;
consider whether corpus growth belongs in this phase or a later one — do not
grow the corpus *in response to* a disappointing delta, which would be
fitting the benchmark to the result.

**R3 — Rerank dependency weight.** A cross-encoder implies torch. Keep it an
extra, keep it out of the default install, and confirm `pip-audit` still runs
against the exported requirements without dragging the ML stack into the base
audit surface.

**R4 — `Retriever.open()` cost compounds.** BM25 already rebuilds in memory at
open, O(corpus), accepted in ADR-0002 with a revisit trigger. Adding a LanceDB
open and a manifest check makes `open()` heavier. Worth measuring in this
phase and checking against ADR-0002's stated revisit trigger rather than
discovering it in Phase 4 when the service opens retrievers per request.

**Measured this wave.** `scripts/measure_retriever_open.py` is the method:
for a range of synthetic corpus sizes, it ingests once BM25-only and once
dense-enabled (the in-memory hash embedder — only vector-plumbing cost is
timed, not embedding quality), then times repeated fresh `Retriever.open()`
calls per configuration, composing the full store-open + LanceDB-connect +
manifest-verify + snapshot-query + BM25-rebuild path exactly as Phase 4's
per-request service would pay it. Qualitative conclusion: the dense
additions — manifest verification, the LanceDB connection, and the
open()-time document snapshot query — add overhead that stays near-constant
as corpus size grows, leaving the BM25 rebuild as the only O(corpus) term in
`open()`. ADR-0002's revisit trigger is therefore not met by this wave's
measurements. Per repo policy, no number from a run of that script is quoted
here or anywhere else — regenerate them by running the script.

**Q1 — Default retrieval mode after Phase 3.** Does `grk search` default to
hybrid, or stay BM25-only until a delta justifies the switch? **The quality
half is now measured; the decision is not yet made, and needs an ADR.**

A gated run against the committed golden corpus (`nomic-embed-text` via local
Ollama) produced the first real delta. Qualitatively — the values live in the
generated artifact, not here (SPEC.md §2): **both dense and fusion improved on
the baseline across recall@1, recall@5, MRR and nDCG@10, with no metric
regressing on either stage.** Fusion additionally improved recall@10, where
dense tied. So the retrieval-quality argument for hybrid is real, and this is
the outcome R2 warned might *not* appear.

That does not settle Q1 on its own, because two costs sit outside the quality
metrics and both cut against making hybrid the default:

- **Abstention is lost entirely.** BM25 abstained on every no-answer query;
  dense and fusion abstained on none. A vector search returns its `top_k`
  nearest neighbours regardless of distance, so abstention is structurally
  unreachable on those paths (see `KNOWN_LIMITATIONS.md`). Defaulting to
  hybrid would mean the default mode of a tool whose entire premise is
  grounded, citation-verifiable retrieval never says "I have nothing".
- **The default path would require a provider the default install lacks.**
  SPEC.md §10 makes "`grk` works end-to-end locally with zero cloud
  credentials" part of the v1 definition of done. Hybrid needs a running
  embedding model and costs roughly two orders of magnitude more latency per
  query than the lexical path. A default that fails without Ollama is a
  different product than the one §10 describes.

The honest reading is that the measurement licenses hybrid as the
*recommended* mode where a provider is configured, not as the unconditional
default. Committing to that is a user-visible behaviour change with a stated
tradeoff, which by repo convention is an ADR, not a spec edit — so Q1 stays
open pending one, now with data rather than for lack of it. Re-run the gate
before deciding: one run on a corpus this size is evidence, not proof.

**Q2 — Does the manifest live in SQLite or beside it? — Settled.**
[ADR-0004](../adr/ADR-0004-embedding-identity-binding.md) decision 1 puts it
in-band, single-row, in SQLite. A sidecar file was rejected as a second source
of truth that a copy or partial backup can silently decouple from the store it
describes.

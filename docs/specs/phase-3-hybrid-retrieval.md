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

### Wave B — dense write path

- `Indexer` accepts an optional embedder + vector store. Both absent ⇒ Phase 1
  BM25-only behaviour, unchanged and still tested.
- `replace_document` parity: deleting or replacing a document must delete its
  vectors in the same logical operation. A document row without its vectors is
  the Phase 1 `replace_document` hazard repeated one store over.
- Incremental re-index re-embeds only changed documents (content-hash gate
  already exists — extend, don't duplicate).
- Manifest verified on open and on ingest; mismatch fails closed.

### Wave C — dense retrieval + RRF fusion

- `retrieval/fusion.py`: pure RRF, no I/O, deterministic tie-break.
- `Retriever` gains a dense path and a hybrid mode; `SearchResponse.metadata`
  reports the stage honestly (`"dense"` / `"fusion"`).
- Snapshot semantics extended to the dense side and documented in the class
  docstring alongside the existing BM25 wording — a stale retriever must fail
  closed the same way on both.
- Citation resolution unchanged: dense hits resolve through the same
  `citations.py` path, so a dense result is exactly as verifiable as a lexical
  one.

### Wave D — optional cross-encoder rerank

- `retrieval/rerank.py` behind `RerankerProtocol`, optional extra.
- **Hazard 2 regression test:** feed negative logits, assert no
  `ValidationError` and that ordering is preserved.
- Unconfigured reranker → typed error, never a silent passthrough (SPEC.md §2
  fail-closed).
- Heavy deps (torch/sentence-transformers) must stay out of the base install;
  CI's default job must not pull them.

### Wave E — eval deltas (the phase gate)

- Runner emits multiple stages into one report; deltas derived vs `stages[0]`.
- Per-stage latency percentiles (SPEC.md §6 names BM25/dense/fusion/rerank
  explicitly — the schema already has the fields).
- The honest-loss path is tested: a stage that underperforms baseline is
  reported as such, not suppressed.
- `EVAL_GATED=1` workflow for real-model runs; skips cleanly when unconfigured.

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

**Q1 — Default retrieval mode after Phase 3.** Does `grk search` default to
hybrid, or stay BM25-only until a delta justifies the switch? Baseline
discipline argues for the latter: let the measured delta make the decision,
and record it.

**Q2 — Does the manifest live in SQLite or beside it?** ADR-0004 settles this;
SQLite is the durable truth per ADR-0002, which argues for in-band.

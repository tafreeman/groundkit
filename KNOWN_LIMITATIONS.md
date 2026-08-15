# Known limitations

Honest and current, per repo policy. Updated with each phase.

## Current state (Phase 3, all waves built)

Hybrid retrieval works end-to-end locally, behind opt-in flags: `grk
ingest --dense` embeds each chunk into a LanceDB vector store alongside the
existing SQLite write, and `grk search --mode {bm25,dense,hybrid}` reads
lexical, vector, or RRF-fused (ADR-0005) results back — every mode still
resolving to a citation-bearing result with character offsets. `grk eval
--dense` now runs the same three strategies over the golden corpus and
reports each stage's signed delta against the BM25 baseline. Everything
remains offline with no cloud credentials by default: no flag is required,
and `grk search` defaults to, and stays on, BM25 (Q1,
`docs/specs/phase-3-hybrid-retrieval.md`) — dense and hybrid are reachable
to opt into, not yet the default. Not yet built, arriving in their phases
per SPEC.md §9:

- **The dense and fusion quality delta has been measured once, locally, and
  is not yet a standing CI result.** `grk eval --dense` emits `bm25` →
  `dense` → `fusion` into one report, deltas are derived at read time
  (`evals/delta.py`), and a stage that loses to baseline is reported as a
  loss rather than suppressed. A gated run (`EVAL_GATED=1`,
  `nomic-embed-text` via local Ollama) found both dense and fusion improving
  on the baseline with no metric regressing — the values are in the generated
  artifact, never restated in docs (SPEC.md §2). Two caveats stand: that is a
  single run over a 10-document corpus, which R2
  (`docs/specs/phase-3-hybrid-retrieval.md`) warns is small enough that a
  result may not survive corpus growth; and `.github/workflows/eval-gated.yml`
  is `workflow_dispatch`-only during active development, so **nothing
  re-measures this automatically** — the result above is a point-in-time
  local measurement, and it will not be contradicted by CI if it stops being
  true. Re-enabling the schedule is noted in the workflow file and slated for
  Phase 7. The CI-default `InMemoryEmbedder` produces
  hash-derived vectors with no semantic signal and, on this same corpus,
  reports both stages as regressions instead — so a delta from it is noise
  with a sign, not a weak measurement. The CLI stamps an explicit warning on
  any such report and `RunConfig.embedding` records the provider, so an
  artifact self-labels which of the two it is.
- **`run_eval` requires a disposable vector store, and its cleanup has one
  gap.** An eval builds a throwaway index whose SQLite half lives in an OS
  temp directory deleted when the run ends. A caller-supplied
  `vector_store` is therefore treated as disposable too: it must be empty
  (`ConfigurationError` otherwise) and every vector the run writes is deleted
  on the way out, including when the run fails. The guard exists because
  vectors stranded in a live collection are not inert — `Retriever`'s orphan
  check fails that collection's dense searches closed the moment one ranks
  into the candidate window, with no visible connection to having run an
  eval. The residual gap: `Indexer` commits SQLite *last*, so a crash
  part-way through ingest can leave the in-flight document's vectors in a
  store SQLite never recorded, and the purge — which enumerates documents via
  SQLite — cannot see them. The emptiness guard bounds this to a store the
  caller already declared disposable, and the purge logs loudly on failure,
  but it is not a guarantee.
- **Hybrid cannot abstain on a no-answer query; dense does not abstain as
  configured.** BM25 returns nothing when no indexed chunk shares a term with
  the query, which is what makes the corpus's `no_answer` judgments measure
  abstention at all. A vector search has no equivalent intrinsic floor — it
  returns its `top_k` nearest neighbours regardless of distance — but the two
  dense-side modes differ, and the difference is worth stating precisely:
  - **Hybrid/fusion: structural.** ADR-0005 decision 6 excludes rank-derived
    fused scores from `score_threshold` altogether
    (`_resolve(..., apply_threshold=False)`), so no configuration makes a
    hybrid query abstain on relevance grounds.
  - **Dense: configurational.** The dense branch *does* apply
    `score_threshold`, and
    `test_impossible_score_threshold_zeroes_bm25_and_dense_but_not_hybrid`
    proves a high enough one returns zero dense hits. The default is `None`
    and the eval runs unthresholded, so the measured stage abstained on
    nothing — but the capability exists, and this is a missing defensible
    value rather than a missing mechanism.

  Either way `no_answer_abstained_count` is 0 for both stages as run, while
  the baseline abstains on every one, so abstention is not comparable across
  stages — read it per stage, never as a delta. See
  [ADR-0007](docs/adr/ADR-0007-default-retrieval-mode.md), where this is the
  reason the default stays BM25.
- **A dense/hybrid result list can still be shorter than `top_k` during the
  staleness window between an `open()` and the retriever's next reopen.**
  The dense path over-fetches to avoid this where it can: `_dense_candidates`
  (`retrieval/search.py`) widens its fetch, doubling from `top_k`, until
  either `top_k` hits survive the `open()`-time snapshot filter or the store
  is exhausted — so a hit dropped for belonging to a document ingested after
  `open()` *is* backfilled from a lower rank. The residual gap is the cap on
  that widening (`_MAX_SNAPSHOT_FETCH_ATTEMPTS`): if enough post-`open()`
  content outranks the eligible chunks to survive every widening attempt,
  the list is returned short, and a warning is logged rather than the
  truncation passing silently. Reopening the retriever clears the window.
  BM25 has no equivalent gap — the stale in-memory index simply has no
  representation of the new content at all.
- **The embedding-identity manifest is verified at both ADR-0004 decision-3
  boundaries.** Wave B closed the ingest boundary — `Indexer` calls
  `verify_manifest` before any load/chunk/embed/delete work, and
  `write_manifest` binds the collection on its first real dense write. Wave
  C closes the second: whenever a dense pair is supplied,
  `Retriever.open()` verifies the collection's manifest against the
  embedder's `(provider, model_name, dimensions)` triple *before* the
  O(corpus) BM25 rebuild does any work. A mismatch raises
  `IndexIdentityError`, never a re-embed and never a fallback. A collection
  with no manifest (never dense-ingested) verifies trivially — there is
  nothing yet for a mismatch to exist against. **That one verification is
  also the sole source of the dense-bound verdict** ADR-0008's per-mode
  refusal reads: `verify_manifest` returns the manifest it checked, rather
  than `open()` reading it a second time afterwards. Two reads admitted a
  race with a real silent-corruption outcome — an unbound collection passes
  the identity check trivially, and a concurrent dense ingest (another task,
  or another process, neither of which the store's `asyncio.Lock` spans)
  binding it to a different provider in between made the later read report
  "bound", so the retriever answered `dense` and `hybrid` by matching this
  embedder's query vectors against that provider's index. The residual
  window is biased closed: a collection bound *after* the read is treated as
  unbound and both modes are refused, which is the same snapshot rule that
  already governs everything else a retriever cannot see after `open()`.
- **Cross-store writes are not atomic.** SQLite and the vector store share
  no transaction. The write order — chunk, embed, write the manifest (once,
  before the first vector add), add the new vectors, delete the previous
  document's, commit SQLite last — is chosen so SQLite is never ahead of the
  dense store, and so the document is never left with *no* vectors at any
  point. A dense store *behind* SQLite is silent: the incremental skip key
  means that document is never retried and never appears in dense results. A
  dense store *ahead* of SQLite is detectable: `Retriever.search` already
  fails closed on a hit whose document has no stored source. The residue of
  an interrupted ingest is therefore possibly-orphaned vectors, not a
  silently missing document — but those orphans carry a document ID SQLite
  never recorded, so a later re-ingest cannot reclaim them by that ID.
  Recovering means deleting and rebuilding the collection, which is cheap
  pre-1.0 and consistent with ADR-0004 decision 5.
- **Adding before deleting trades a silent failure for a loud one; it does
  not eliminate the residue.** Deleting first was worse — it opened a window
  with no vectors at all, and a crash there followed by a content reversion
  (`git checkout`) left SQLite's stored fingerprint matching the restored bytes,
  so the document was hash-skipped forever and silently absent from dense
  results. Adding first closes that: the previous vectors survive, so the
  reverted content still resolves. What it costs is that the interrupted
  run's vectors stay behind under an uncommitted document ID, and a single
  such orphan makes `Retriever.search` fail closed for *any* query that
  ranks it — not just for that document. Prune sweeps iterate SQLite, so
  they cannot reach it. Fully closing this needs the document row committed
  before the dense write with its content hash withheld until after
  (a two-phase commit `MetadataStoreProtocol` does not currently expose);
  until then the recovery is a collection rebuild, as above.
- **Delete reconciliation warns, it does not fail.** ADR-0004 decision 6
  requires the caller to reconcile the vector-delete count against the
  chunk count SQLite deleted. A mismatch is logged and surfaced in
  `IndexReport`, not raised, because a collection first ingested BM25-only
  and later re-ingested with the dense path enabled legitimately deletes
  chunks and zero vectors — strict equality would fail a completely
  healthy upgrade.
- **Turning the dense path on does not backfill an existing collection.**
  The incremental skip gate runs before chunking and therefore before
  embedding — that is exactly what keeps an unchanged document from being
  re-embedded on every run. Its key is a fingerprint over content, chunker
  and chunking configuration (ADR-0009), and deliberately *not* over the
  embedding identity: including that would turn `--dense` on an existing
  collection into an implicit full re-embed, which is the auto-backfill
  ADR-0008 declined. The cost is that enabling an embedder and
  vector store over a collection already ingested BM25-only leaves every
  unchanged document without vectors: it is skipped, so it is never
  embedded. Only documents whose content actually changes afterwards gain
  vectors. The collection is then permanently half-dense until it is
  re-ingested from scratch. This is also the case the delete reconciliation
  above deliberately tolerates. Forcing a backfill means deleting the
  collection and re-ingesting it. **Reading it densely is no longer silent
  (ADR-0008):** a `dense` or `hybrid` search against a collection with no
  embedding-identity manifest raises `ConfigurationError` naming the
  re-ingest remedy, instead of returning zero results (`dense`) or BM25's
  ranking stamped `"stage": "fusion"` (`hybrid`). A collection that *is*
  manifest-bound but only partially embedded — documents that changed after
  the dense path was enabled — is still not detected, and remains the
  genuine half-dense case this entry describes.
- **A BM25-only indexer can no longer orphan a manifest-bound collection —
  it is refused (ADR-0011).** An `Indexer` constructed without an embedder or
  vector store has no vector store to delete from, so replacing or pruning a
  document in a collection whose vectors an earlier dense run wrote left those
  vectors surviving their documents. This was tolerated until 2026-08-15 on the
  grounds that the orphans are *loud* — Wave C's dense read path fails closed on
  them, `Retriever.search` raising `RetrievalError` on a hit whose document has
  no stored source, with regression tests for both the deleted-after-open and
  deleted-before-open cases.

  That argument stopped holding when ADR-0009 changed the skip key. A content
  hash matched on unchanged content, so reaching the hazard required a document
  to actually change; a fingerprint derived differently matches nothing an
  earlier build stored, so the *first* BM25-only ingest after that change
  rewrites every document and orphans the whole dense side in one ordinary
  command — permanently, because the same write stores the new fingerprint and a
  later dense run then hash-skips every one of them. What was loud is the
  eventual read failure, not the write that caused it, and that failure names an
  index inconsistency rather than the ingest three steps upstream.

  `Indexer._verify_identity` now refuses a run whose indexer has no embedder
  when the collection is manifest-bound, before any load, chunk or write, with
  a `ConfigurationError` naming both remedies. The residual hazard is what the
  manifest cannot see: a collection holding vectors whose manifest was never
  written, or an `Indexer` holding a store handle from before a manifest
  appeared. Neither is reachable through the CLI.
- **The CLI exposes the dense path, entirely opt-in.** `grk ingest --dense`
  embeds and writes vectors alongside the SQLite write; `grk search --mode
  {bm25,dense,hybrid}` reads them back (default `bm25`, and it stays there —
  Q1 is decided in ADR-0007). Both share `--embed-provider`/`--embed-model`/
  `--embed-dimensions`/`--embed-base-url`, which resolve to their defaults
  (`ollama`/`nomic-embed-text`/768/the local Ollama endpoint) only once a
  dense path is actually active — supplying any of them without `--dense`
  (ingest) or without a dense `--mode` (search) fails closed with
  `ConfigurationError` rather than being silently ignored.
  `--embed-provider inmemory` is the same offline test double described
  above: correct for exercising the CLI's dense wiring, never a quality
  measurement. LanceDB data for a collection lives at
  `<index-dir>/<collection>.lance`, beside `<collection>.sqlite3`. The
  default install, default commands, and CI need no Ollama and are
  unchanged by any of this.
- **The cross-encoder reranker is now called by the eval harness; `Retriever.search`
  and `grk search` still cannot reach it.** `run_eval` accepts an optional
  `reranker` and, when supplied, appends a `rerank` stage over the best
  upstream stage the run produced — `fusion` with a dense pair, `bm25`
  without — and `grk eval --rerank` wires it to the CLI. That is deliberate
  and narrow, not a partial rollout waiting to be finished: ADR-0012
  decision 2 keeps rerank out of `retrieval/search.py` entirely, so it is
  reachable only through `grk eval --rerank`, `SearchMode` stays the
  three-way `Literal` ADR-0007 already settled, and a reranked `grk search`
  remains genuinely out of scope for Phase 3.

  **A delta is now emitted and derivable, and reported honestly including
  when it loses (SPEC.md §6) — but which run produced it matters exactly as
  much as it does for the dense/fusion numbers above.** The default `uv run
  pytest` suite exercises the `rerank` stage's full wiring — `run_eval`, the
  CLI flags, `RunConfig`'s three `rerank_*` fields, `derive_rerank_attribution`
  — through a protocol-conformant stub reranker rather than a cross-encoder,
  because the `rerank` extra is not in the dev group (below) and CI must not
  pull torch. That stub's delta is the same trap `InMemoryEmbedder` carries
  for dense: structurally valid, semantically noise with a sign. A delta that
  means anything needs `RERANK_GATED=1` with `uv sync --extra rerank`
  (`tests/test_eval_rerank_gated.py`, the companion to `test_rerank_gated.py`
  that drives a real cross-encoder through `run_eval` itself rather than in
  isolation), and — like the Ollama eval gate above —
  `.github/workflows/rerank-gated.yml` is `workflow_dispatch`-only during
  active development, so **nothing re-measures this automatically**.

  **The rerank row's meaning is configuration-dependent, and the artifact
  says so rather than leaving it implicit.** A reranker reorders whichever
  stage was best available for the run, so two reports can agree on
  `corpus_hash`/`judgments_hash` and still describe two different
  experiments. `RunConfig.rerank_input`, `rerank_candidates`, and
  `rerank_model` are therefore recorded together or not at all — a validator
  enforces it — precisely so two reports are never silently incomparable in
  this one dimension.

  **The delta against `stages[0]` is not the reranker's own contribution when
  the input was `fusion`** — it sums fusion's gain over BM25 with rerank's
  gain over fusion into one number. `derive_rerank_attribution`
  (`evals/delta.py`) diffs the `rerank` stage against its own input stage
  instead, and the CLI prints both — but that diff is a clean isolation of
  the cross-encoder's own contribution only when the input is `bm25`. BM25
  scores independently of `top_k`, so the wider candidate fetch reranking
  uses only adds a superset of what the baseline stage already scored. RRF
  does not: it sums `1/(rrf_k + rank)` over the rankings a chunk is visible
  in at the fetched depth, so `fusion@50` is a different ranking function
  from `fusion@10`, not a deeper slice of it — a chunk absent entirely from
  the depth-10 fused list has ranked first at depth 50. A fusion-input
  attribution therefore measures the rerank pipeline at the wider candidate
  depth against fusion as reported, a real production comparison, and not
  the cross-encoder's isolated contribution (ADR-0012 Consequences).

  **A gated run** (`RERANK_GATED=1`, `cross-encoder/ms-marco-MiniLM-L-6-v2`
  reranking `bm25`, with `nomic-embed-text` via local Ollama for the dense
  pair) **found the BM25-input rerank stage improving on the baseline with no
  metric regressing** — the values are in the generated artifact, never
  restated in docs (SPEC.md §2), and that comparison is the clean one
  described above. **The fusion-input delta supports no conclusion**, for two
  independent reasons rather than one: the attribution is confounded exactly
  as described above, and separately, its movement sits within a small
  number of query flips on the answerable judgment set — inside the noise
  band R2 (`docs/specs/phase-3-hybrid-retrieval.md`) already warns about.
  Read it as neither a win nor a loss on this corpus, not as a weak
  positive.

  The same run exposed two further caveats:

  - `recall@10` reached its ceiling on the rerank stage, so on this corpus
    that metric can no longer discriminate a better reranker from this one.
  - The reported p99 latency is the sample maximum — `MetricSet`'s
    percentiles are nearest-rank over the same small per-judgment sample
    noted under Phase 2 caveats below — and it is dominated by the one-time
    cross-encoder model load, not steady-state inference. A reader comparing
    `latency_p99_ms` across stages without knowing that would read a
    once-per-process cost as a tail-latency regression in the reranker
    itself.

  **Candidate depth is over-fetched to `MAX_TOP_K`,** pinned rather than
  exposed as a CLI knob, because a depth equal to `top_k` would hand the
  reranker a set it can only permute — its `recall_at_10` would then equal
  the input stage's by arithmetic, not by measurement. Truncation to `top_k`
  still happens *after* reranking, so a reranker can promote a candidate the
  upstream stage ranked below the cut only if that candidate was inside the
  list it was handed; it cannot recover a document the upstream stage never
  retrieved. That remains the ceiling on any gain.

  Also still true: the `rerank` extra (`sentence-transformers`, which pulls
  in torch) stays out of the default install and out of the dev group, so
  the default suite never loads a model — only the pure surface (the
  sigmoid, ordering, fail-closed paths) is proved offline.

  **A related reproducibility trap: `uv run mypy` disagreed with itself
  depending on whether the `rerank` extra was installed.** With
  `sentence-transformers` (and therefore torch) resolved, the
  `type: ignore[import-not-found]` comments guarding the optional import in
  `retrieval/rerank.py` became unused and mypy failed on them; without the
  extra, mypy needed those comments and passed. Fixed by also listing
  `unused-ignore` on both codes, but the general shape is worth recording
  because the fix is local and the gap that let it happen is not: no CI job
  runs mypy with the extra installed — `rerank-gated.yml`, the one workflow
  that installs it, runs `pytest` only — so this class of
  environment-dependent typecheck failure is not caught automatically
  anywhere, and only surfaces to whoever happens to run `uv sync --extra
  rerank && uv run mypy` locally.
- **`score_threshold` does not apply to hybrid results.** ADR-0005
  decision 6: an RRF-fused score is a function of result-set size and
  retriever count, not a probability, and thresholding it would silently
  mean something different from thresholding BM25 or dense scores directly.
  `RetrievalConfig.score_threshold` therefore applies to `bm25` and `dense`
  mode results only, never to fused scores, and never to the pre-fusion
  candidate lists either (thresholding those would reintroduce the same
  threshold by a side door). Phase 3 ships hybrid search with no confidence
  cutoff at all. This is a deliberate deferral, not an oversight (ADR-0005
  Consequences).
- **Filtered dense search costs O(corpus).** A `metadata_filter` triggers a
  full-table over-fetch, then filters and truncates in Python, because
  metadata is stored as one opaque JSON blob rather than structured columns
  and pushing caller-controlled values into a LanceDB `WHERE` predicate is
  the hazard class ADR-0004 decision 6 exists to close. Unfiltered search
  stays a cheap top-k vector query. The alternative — silently returning
  fewer than `top_k` — is a defect, not a tradeoff, so the cost is accepted
  for now. Structured metadata columns would fix it and are not built.
- **Metadata filtering is equality-only.** A filter matches when every
  key/value pair is present and equal. No ranges, no negation, no nesting.
- **`index/dense.py` is now in the coverage `core_subset`, and it is a mixed
  file.** Wave F added it: it is scoring, the vector peer of the already-gated
  `bm25.py`, it holds both live ADR-0001 hazards, and `retrieval/fusion.py` was
  already gated by the `retrieval/*` glob, so leaving it out gated the combiner
  but not one of its inputs. The residual gap is inside the file rather than
  around it — roughly two-thirds shared helpers plus `InMemoryVectorStore`,
  one-third `LanceDBVectorStore` behind the optional `dense` extra, all under a
  single number. Well-covered LanceDB rows can therefore mask a thin in-memory
  path or the reverse, which is the offsetting the subset exists to prevent,
  admitted here at file granularity. What makes it safe is that `lancedb` is
  pinned in the dev group so CI genuinely exercises both halves — a convention,
  not an invariant. Splitting `LanceDBVectorStore` into its own module would
  remove the caveat and is deliberately not bundled into Wave F.
  `index/metadata.py` remains outside the subset by per-file reasoning, not
  oversight (`pyproject.toml` records why).
- **The two vector stores diverge on zero-magnitude vectors.** LanceDB's
  cosine search omits them from results entirely; `InMemoryVectorStore`
  returns them at score 0.0. Real embedding models do not emit zero vectors,
  so this is documented rather than forced into artificial parity — but the
  two paths are not byte-identical on that degenerate input.
- **Chunk metadata still carries an ingest-time `source` snapshot.**
  `ingestion/chunking.py` seeds `metadata["source"]`, duplicating a fact that
  `documents.source` owns durably. ADR-0006 makes retrieval ignore the copy,
  so it can no longer produce a wrong citation, but the stale-on-re-ingest
  copy is still written and still exposed in `RetrievalResult.metadata`.
  Removing it is a separate change with its own migration cost.
- PDF/HTML loaders and URL ingestion (with the SSRF guard) — v1 scope, not
  yet scheduled into a phase; the loader currently reads `.md`/`.markdown`/
  `.txt` only.
- REST API and MCP server (Phase 4); synthesis, query rewrite, redaction
  (Phase 5); IaC and OTel observability (Phase 6); docs site (Phase 7).
- BM25 rebuilds in memory at open — O(corpus) startup cost, accepted and
  bounded by ADR-0002's revisit trigger.
- Cloud-provider embedding-endpoint SSRF guard and outbound redirect policy
  (loopback/private/link-local/IPv6-mapped rejection) — lands with the
  service phase (Phase 4); Phase 1's OpenAI-compatible embedding provider
  accepts an operator-set `base_url` with no such validation today. See
  SECURITY.md.
- A UTF-8 BOM is not stripped at load (`utf-8`, not `utf-8-sig`), so a
  leading byte-order-mark character (U+FEFF) appears in the first chunk's
  content and in citations resolved from it. Cosmetic — offsets stay
  correct because the loader and the citation resolver use identical
  decoding.
- `resolve_citation` detects source drift only by length (`end_offset >
  len(text)`); a same-length in-place edit to a source file goes undetected
  and can resolve to silently wrong text.
- Total ingestion size is unbounded: individual files cap at 10 MiB, but
  nothing caps file count or aggregate bytes for a directory run.

## Phase 2 caveats

- The adversarial category does not test prompt-injection resistance. With
  no LLM in the retrieval path, "injected text must never surface as
  instructions" is untestable in Phase 2; that arrives with the Phase 5
  faithfulness judge. What Phase 2 asserts is fixture correctness — that
  documents referenced by adversarial judgments genuinely contain
  injection-styled text, and that retrieval returns it verbatim like any
  other content. A passing adversarial test is **not** a claim that
  groundkit resists prompt injection.
- nDCG's IDCG is inflated by boundary-straddled gold quotes. A single gold
  quote that spans a chunk boundary resolves to two relevant chunks, so
  IDCG assumes two ideal positions where a retriever that found the answer
  via either half alone cannot reach 1.0. Accepted because SPEC.md §6
  specifies `min(|gold|, 10)` with no span/chunk distinction. recall@k is
  unaffected — it is hit-rate, not set-recall.
- No-answer queries are keyword phrases, not questions. The tokenizer is a
  bare `\w+` split with no stopword removal, so a query phrased as a
  sentence carries words present in any English prose, scores above zero,
  and returns hits — abstention would be unobservable. Content-word-only
  queries make it measurable without inventing a score threshold. This
  makes no-answer queries shaped differently from the other categories;
  that asymmetry is deliberate.
- Baseline deltas are intra-run only. `evals/results/` is gitignored, so no
  historical artifact reliably exists to diff across runs. Later stages
  compare against the baseline stage inside the same report.
- `MetricSet`'s cutoffs are fixed at 1/5/10, so `run_eval` refuses a
  `top_k` below 10 (`MIN_EVAL_TOP_K`) rather than publishing an `@10` field
  computed from a shorter list. The floor used to live only in `cli.py`,
  which left a library caller free to emit a report whose `recall_at_10`
  was a `recall@1` under a `@10` name — a gold chunk at rank 7 scoring
  `0.000` or `1.000` from identical inputs depending on a cutoff nothing
  reading the field is obliged to cross-check. Evaluating a genuinely
  top-3 system is therefore not expressible here; that needs configurable
  cutoffs in the schema, not a smaller `top_k`.
- Latency percentiles are computed from a small sample (one measurement per
  judgment), so p95/p99 are order-of-magnitude indicators, not reliable
  tail estimates.
- Two byte-identical chunks still tie non-deterministically in BM25
  ordering — they are indistinguishable by any content-derived key. (The
  tie-break fix resolved the far more common case of *distinct* chunks
  tying.)

## Deliberately out of scope for v1 (will not be built)

- Multi-tenant auth — single-operator service; the shared-secret header on
  mutating routes is not a tenancy model.
- Distributed indexing — one node, file-based index.
- Fine-tuning of embedding or rerank models.
- Agent loops — this is a retrieval service, not an agent runtime.
- UI beyond the docs site.
- GraphRAG.
- Broad vector-DB support — LanceDB (dense) + SQLite (metadata) behind
  interfaces; pgvector is a designed-for extension point, not a v1 feature.
- A semantic key-value memory store — ARP's `memory.py` was dropped per
  ADR-0001: its in-process key map is non-durable even over a persistent
  store, the exact anti-pattern this repo bans.
- LLM-based reranking — rerank is a local, non-LLM cross-encoder only.

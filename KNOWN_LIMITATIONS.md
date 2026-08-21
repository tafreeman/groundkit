# Known limitations

Honest and current, per repo policy. Updated with each phase.

## Current state (Phases 3–5: hybrid retrieval, service surface, LLM boundary)

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
  now runs on a weekly cron (Mondays 06:00 UTC) alongside its label and
  manual triggers — re-enabled in Phase 7, as the workflow file's own note
  anticipated — so this **is** re-measured automatically now. The result
  quoted above is still the single run described, not yet superseded by a
  later one, but a regression would no longer go uncaught by CI. The
  CI-default `InMemoryEmbedder` produces
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
  [ADR-0007](https://github.com/tafreeman/groundkit/blob/main/docs/adr/ADR-0007-default-retrieval-mode.md),
  where this is the reason the default stays BM25.
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
  `.github/workflows/rerank-gated.yml` now runs on its own weekly cron
  (Mondays 06:30 UTC, offset from `eval-gated.yml`'s so the two heavy jobs do
  not contend for runners), alongside label and manual triggers, re-enabled
  in Phase 7 — so this **is** now re-measured automatically.

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
- PDF/HTML loaders — v1 scope, not yet scheduled into a phase; `FileLoader`
  still reads `.md`/`.markdown`/`.txt` only, so no ingest path can create a
  document from a local PDF or HTML file (see the ADR-0016 section below for
  the extractors that already exist to read one back). URL ingestion is no
  longer on this list: `grk ingest` accepts an `http(s)` URL directly, behind
  the SSRF guard described further down, and stores a verifiable local
  snapshot.
- BM25 rebuilds in memory at open — O(corpus) startup cost, accepted and
  bounded by ADR-0002's revisit trigger.
- **BM25 scoring is off the event loop and no longer linear in corpus size,
  but an unselective query still is.** `BM25Index.search` scores the union of
  the query terms' postings rather than every indexed chunk (GK-018, erratum
  to ADR-0002 decision 2), so its cost tracks the number of chunks holding a
  query term. That is a real bound for a selective query and no bound at all
  for an unselective one: a stopword present in every chunk has every chunk in
  its postings list, so a natural-language query still scores most of the
  corpus. There is no stopword list and no term-at-a-time early exit — both
  change which results come back, or need a score bound to be safe, and
  neither is built. Both call sites — the `bm25` branch and the `hybrid`
  branch's lexical half — still dispatch through `asyncio.to_thread`, as does
  `InMemoryVectorStore.search`'s equivalent scan, because the pure-Python
  scoring holds the GIL for most of its run and a concurrent caller still
  contends for it; what that buys is that the loop keeps turning. Pinned by
  `tests/test_retrieval.py::test_bm25_scoring_does_not_run_on_the_event_loop`
  and its dense counterpart in `tests/test_dense.py`, both asserting thread
  identity rather than timing, and by `tests/test_bm25_postings.py` for the
  candidate walk itself.
- **A writer that loses a lock race waits, then fails; nothing retries it.**
  `SQLiteMetadataStore` sets `PRAGMA busy_timeout` explicitly, to
  `metadata.BUSY_TIMEOUT_MS`. Past that window the operation raises
  `StorageError: database is locked` and no layer above retries, so an
  ordinary collision — an operator running `grk ingest` against a collection
  the service is writing to — surfaces as a 5xx that would have succeeded on
  a second attempt. The timeout was previously inherited, unnamed, from
  `sqlite3.connect`'s `timeout=5.0` default; setting it here changes no
  behaviour and makes the value this module's rather than the stdlib's.
- A UTF-8 BOM is not stripped at load (`utf-8`, not `utf-8-sig`), so a
  leading byte-order-mark character (U+FEFF) appears in the first chunk's
  content and in citations resolved from it. Cosmetic — offsets stay
  correct because the loader and the citation resolver use identical
  decoding.
- **`search` results are not verified against their sources; `fetch_chunk` is
  the step that verifies.** `search` returns the *indexed* text, because
  verifying per hit is one file read per result per query. `fetch_chunk`
  re-reads the cited span and byte-compares it to the indexed chunk, which does
  catch a same-length in-place edit, and reports `verified` / `drifted` /
  `unresolvable`. So a caller who quotes a `search` hit directly is quoting
  unverified text, and one who calls `fetch_chunk` first is not. That division
  is deliberate, and it is the thing to know about this surface.

  Read literally, `resolve_citation` alone only detects drift by length
  (`end_offset > len(text)`) — but it is a *resolver*, not the verifier.
  `verify_citation` and `fetch_chunk` both do the full comparison.

  **A stored checksum would not improve this and is deliberately not added.**
  Verification already compares the actual bytes, and a digest comparison is at
  best equal to a byte comparison and at worst admits collisions. A hash only
  helps when the expected content is unavailable, and it never is: `Chunk`
  requires `content == document.content[start_offset:end_offset]` and that
  content is persisted. A per-chunk hash would be redundant state that can
  disagree with its own input — the same reason `evals/schema.py` stores no
  delta and `index/bm25.py` keeps no on-disk form.

  What remains genuinely open is time-of-check/time-of-use: verification
  describes the file as of the read, and nothing stops an edit a millisecond
  later. No checksum closes that either.
- Total ingestion size is unbounded: individual files cap at 10 MiB, but
  nothing caps file count or aggregate bytes for a directory run.
- **A colliding `metadata["source"]` key on ingest is silently overwritten by
  the authoritative path, not rejected.** `RecursiveChunker` seeds
  `document.source` into every chunk's metadata *after* splatting
  `document.metadata`, so a `Document` whose own metadata carries a `source` key
  loses that value — last-write-wins, with the authoritative key winning, by
  design. This is deliberate: `source` is never reserved by contract (neither
  ingest loader auto-populates it), so treating a collision as a hard failure
  would turn a benign round-trip — read a chunk's metadata back, re-ingest it —
  into an error. The trade-off is that a caller who *intentionally* stores their
  own `source` key in `Document.metadata` has it clobbered with no warning;
  nothing currently surfaces that collision to the caller. Before this was
  fixed the polarity was reversed, and the caller's value silently won.
- **No authentication. The 127.0.0.1 bind is load-bearing, not a default.**
  Every Phase 4 response is unauthenticated by construction (ADR-0014
  decision 1): the shared-secret header SPEC.md §7 requires for mutating
  routes is not built, because the set of mutating operations is empty. The
  loopback bind is therefore what decides who can reach the service at all
  (decision 7) — `Host` validation (ADR-0024) decides which of the requests
  that do arrive are answered, and neither is authentication — and
  `--allow-remote-access` exposes an unauthenticated, content-bearing
  surface — document text and absolute source paths over `search`,
  collection topology over `index_status` and `list_collections` — to
  anyone who can reach the port.
- **The REST surface admits any `[::…]`-shaped `Host`.** Starlette's
  `TrustedHostMiddleware` strips the port by splitting on the *first* colon, so
  `[::1]:8765` and a bare `[::1]` both reduce to `"["` — which the allow-list
  must therefore contain for any IPv6 loopback client to be served at all, and
  which also matches other bracketed literals beginning `[::`. This is a real
  widening, and it is not reachable by the DNS rebinding it sits next to: a
  bracketed address literal is never the product of a DNS answer, so no
  attacker-controlled zone can cause a browser to send one. Closing it would
  mean this repo owning a hand-written `Host` matcher and keeping it correct
  against IPv6 spellings, ports, trailing dots and case (weighed and rejected in
  ADR-0024).
- **`Host` matching is case-sensitive and does not strip a trailing root dot.**
  Starlette compares with `==` and the MCP SDK with `in`, so `LOCALHOST`,
  `LocalHost:8765` and `localhost.` are refused (400 / 421). It fails closed, so
  this is a compatibility gap, never a widening. Left open deliberately: the
  trailing dot is enumerable and letter case is not, so covering half would read
  as normalisation while being none. The trigger is a real client that cannot be
  configured around it.
- **A hostname `--host` is resolved twice** — once to derive the `Host`
  allow-list, once by the ASGI server to bind — and the answers can differ. The
  dangerous direction (routable when derived, loopback when bound) needs control
  of DNS for the operator's own bind name *plus* `--allow-remote-access`; the
  other direction is an availability failure visible on the first request.
  Address literals, including the default bind, are never resolved and are
  unaffected.
- **No mutating operation over either transport.** No ingest, no delete, on
  REST or MCP. Deliberate (ADR-0014 decision 1), not an omission: SPEC.md
  §1.2's four tools — `search`, `fetch_chunk`, `list_collections`,
  `index_status` — are the complete named surface, and the CLI covers
  ingest at a strictly smaller surface: a local operator's typed argument,
  not any caller reaching the port.
- **A missing `chunk_id` returns 400, not 404.** `handle_fetch_chunk` raises
  `ConfigurationError` for it, and separating that case from an invalid
  collection name would need either a new exception type (forbidden by
  ADR-0014 decision 9) or message matching (forbidden for fragility) —
  neither is taken, so the asymmetry stands. A non-existent *collection*,
  by contrast, does return not-found: `check_collection` refuses it as a
  precondition before any handler runs, which is a boundary check rather
  than an exception path (ADR-0014 decision 3; see
  `src/groundkit/service/errors.py`).
- **FIXED 2026-08-18 — `chunk_count()` materialized the corpus to return an
  integer.** `index_status` read `len(await store.get_chunks())`, which
  selected every chunk's full text and rebuilt each as a re-validated `Chunk`
  model. It now runs an aggregate: `SQLiteMetadataStore.count_chunks()`, a
  concrete capability of the SQLite store rather than a new
  `MetadataStoreProtocol` member. That protocol stays exactly as wide as it
  was — it is held to exact signature parity by conformance tests, and
  widening it for a reporting convenience is the trade ADR-0012 decision 3
  refused for `model_name` — but declining to widen it never required using
  the slowest implementation behind it. `CollectionRuntime` holds a concrete
  store, so it can ask. Pinned by
  `tests/test_runtime.py::test_chunk_count_does_not_read_chunk_content`,
  which asserts on the SQL the store executes rather than on which method was
  called. The document half of the same call was missed on the first pass and
  closed the same way afterwards: `count_documents()` replaced
  `len(await runtime.get_document_sources())`, which built the whole
  `{document_id: source}` mapping one line above the chunk count to take its
  length.
- **`grk serve` bounds concurrent corpus-scale work; it does not rate-limit.**
  `search` and `index_status` acquire a semaphore admitting
  `service.tools.MAX_CONCURRENT_CORPUS_SCANS` at a time, so peak memory tracks
  that bound rather than arrival rate. It counts concurrent operations, not
  requests per caller per interval: a client that waits for each response is
  unbounded, nothing attributes load to a caller, and the surface is still
  unauthenticated. Waiters queue rather than being shed, so a sustained
  overload shows up as latency, not as refusals — a caller cannot distinguish
  a slow corpus from a busy server. Connection exhaustion is bounded
  separately and generously by `cli.SERVE_MAX_CONNECTIONS`; see `SECURITY.md`.
- **`--host ''` is classified by resolving the empty string, which is
  platform-dependent.** `asyncio.create_server` maps `host == ""` to all
  interfaces, so the bind is routable either way, but the classification is
  not: on Windows `getaddrinfo("", None)` returns the machine's routable
  addresses and the allow-list is correctly unrestricted, while on glibc the
  same call raises and the fail-closed branch produces a loopback-only list on
  an all-interfaces bind — refusing every legitimate client. Both directions
  are safe; it is an availability inconsistency, and it requires
  `--allow-remote-access` either way. Found by a post-merge security review.
- **The first version of that bound used the wrong mechanism, and every
  document describing it was wrong with it.** uvicorn's `limit_concurrency` was
  set to a small number and described as bounding in-flight requests. It trips
  on `len(connections) >= limit or len(tasks) >= limit` — connections
  server-wide, idle keep-alive included — and substitutes a 503 app *before*
  routing, so a trip answers every route, the Kubernetes liveness and readiness
  probes included. `grk serve` mounts a stateful MCP transport whose clients
  hold an SSE stream open for up to half an hour, so idle sessions alone could
  have 503-ed a server doing no work and restart-looped the single replica.
  Recorded rather than quietly corrected because the failure class is the one
  this repo keeps meeting: a control whose documented semantics and actual
  mechanism differ, where every test written against the documented semantics
  passes.
- **Read-only does not mean the process writes no bytes.** Opening a WAL
  database updates its `-shm`/`-wal` sidecars, and `SQLiteMetadataStore.open`
  runs `CREATE TABLE IF NOT EXISTS` and a best-effort chmod on every open
  (ADR-0014 decision 4). The read-only claim is scoped to durable document,
  chunk, and manifest state — enforced in Python, not by the filesystem:
  the process holds a read-write file handle throughout. Filesystem-level
  enforcement (a read-only mount) is deferred to Phase 6.
- **FIXED 2026-08-16 — two read-only operations wrote into unrelated
  databases.** `handle_list_collections` advertised every contained
  `*.sqlite3`, and `SQLiteMetadataStore.open` applies `_SCHEMA`
  unconditionally, so asking `index_status` for one of those names created
  `documents`, `chunks`, `collection_manifest` and `collection_state` inside a
  database that was not groundkit's. Both halves now go through
  `metadata.is_groundkit_store`, a read-only identity probe: listing skips a
  foreign file, and `open` refuses it with a `StorageError` before any schema
  is applied. Two details are worth keeping rather than deleting with the
  entry. The probe opens with **`mode=ro&immutable=1`, and both flags are
  load-bearing** — `mode=ro` alone makes SQLite create the `-shm`/`-wal`
  sidecars of a WAL database, so the guard would have performed the very write
  it exists to prevent (caught by
  `test_list_collections_opens_nothing`, which already asserted the directory
  is unchanged). And the probe keys on `application_id` **only**, never
  `user_version`, so a pre-v3 store stays readable — folding the version in
  would have locked out exactly the stores ADR-0016's narrow write-guard was
  written to keep working.
- **Reranking a hybrid search returns a ranking no `grk search --mode
  hybrid` produces.** RRF is not depth-invariant: it sums
  `1/(rrf_k + rank)` over the rankings a chunk is visible in at the fetched
  depth, so the wider candidate depth used for reranking fuses a different
  set, not a deeper slice of the same one — a chunk absent from the
  depth-10 fused list can rank first at depth 50 (ADR-0012 Consequences;
  ADR-0014 Consequences). The response records the input depth so a client
  can tell.
- **DNS rebinding is not closed.** Between `url_safety`'s resolution and
  httpx's own connect-time resolution, the answer can change — the window
  is two resolutions in one process, small, not zero. Closing it needs
  connections pinned to the validated address with the original host
  preserved in `Host` and SNI, which is not built (ADR-0014 Consequences).
- **The proxy bypass is not closed.** `trust_env=False` was rejected because
  it also disables `SSL_CERT_FILE` and `.netrc`, which operators
  legitimately need, to close a bypass in the same trust domain as
  `base_url` itself. The threat model is operator misconfiguration, not a
  hostile operator (ADR-0014 Consequences).

- **The keyed document read and the `COUNT(*)` aggregates are one optional
  capability, so a store implementing only half of it falls back to the whole
  table.** `DocumentRecordStoreProtocol` declares `get_document_record`,
  `get_document_records`, `count_documents` and `count_chunks` together, and
  `Retriever`'s `isinstance` check is all-or-nothing: a hypothetical store that
  could answer the keyed document read but not the counts would silently take
  the whole-table fallback on every search. That is correct but slow, and it is
  invisible — nothing logs the downgrade. Accepted because no store in this
  repo is in that position (all four are the same single SQL statement against
  the same two tables) and because splitting the protocol would make a caller
  test two capabilities to establish one fact. Revisit if a second real
  implementation ever appears.
- **Within one search, a document's record is read once and memoized.** A
  document deleted between two hits on it inside the same `search` call is
  therefore still resolved from the earlier read. This is strictly no worse
  than the behaviour it replaced — which resolved every hit against a single
  snapshot taken before the search began — and the fail-closed guarantee across
  searches is unaffected, but the read is "live per search", not "live per
  hit".

- **The staleness cache's rebuild cliff is now measurable, and is not fixed.**
  `index_status` reports `retriever_acquires`, `retriever_rebuilds`,
  `rebuild_seconds_total` and `last_rebuild_seconds` (ADR-0026), so the cache's
  hit rate can be read rather than inferred from latency. What an operator will
  see during a concurrent `grk ingest` is the defect itself: ADR-0013 bumps the
  generation once per commit and `grk ingest` commits once per document, so an
  ingest over N changed files publishes N generations, fails the cache's equality
  predicate N times, and costs N full `BM25Index.from_store` rebuilds — each
  serialized against the ingest writer on the metadata store's single lock, and
  each waited through rather than served stale by every concurrent request
  (ADR-0013 decision 5). For the duration of an ingest the fallback is the
  reopen-per-request baseline ADR-0013 rejected on measurement, and the
  contention runs both ways: the reads slow the ingest that is invalidating them.
  The incremental remedy — a monotonic per-document watermark, a
  `get_chunks_since`, and a `remove_document` on the lexical index, behind a
  `SCHEMA_VERSION` bump — is deliberately not built. `remove_document` is the
  hard half: a watermark cannot represent a row that is *gone*, and
  `BM25Index`'s postings map is keyed by position in its chunk list, so removing
  a chunk from the middle invalidates every position above it. Whatever replaces
  it must be score-identical to a full rebuild, including the insertion-order
  tie-break, or ADR-0002 decision 2's "pure function of the persisted chunk set"
  invariant becomes false — and that invariant is the guard against repeating
  ARP's `memory.py` `_key_map` drift.
- **The rebuild counters are process-local and reset without saying so.** They
  describe one `CollectionRuntime` object's history, not the collection's, and
  are deliberately not persisted (persisting them would need a schema bump, would
  put a write on an unauthenticated read path, and would either invalidate the
  cache it measures or break ADR-0013 decision 1's bump-inside-the-`_op` rule). A
  runtime evicted by `CollectionRegistry`'s LRU bound and later reopened starts
  from zero, and so does a restarted process, so a scrape taken immediately after
  either reports zeros meaning "no data" rather than "no rebuilds" — a real trap
  for anyone graphing them. Nothing aggregates across processes: a multi-process
  deployment must read each process's own `index_status`.

- **URL ingestion's writes are off the event loop; its buffering is not.**
  `UrlLoader`'s snapshot write and its HTML scratch write now dispatch through
  `asyncio.to_thread`, so neither blocks the loop for the length of a
  content-sized write. Two smaller things stay on the loop deliberately: the
  fetched body is held whole in memory before either write — the size cap, not
  streaming, is what bounds it, because ADR-0016 decision 4 requires refusing
  past `DEFAULT_MAX_BYTES` rather than truncating, and a truncated read would
  produce offsets into a partial document — and the scratch
  `TemporaryDirectory`'s create and cleanup are constant-cost syscalls that are
  not dispatched. What changed is the *stalling*, not the *cost*. Pinned by
  `tests/test_url_loader.py::TestUrlLoaderWritesOffTheEventLoop`, which asserts
  thread identity rather than timing, as its BM25 and dense counterparts above
  do.

- **`metadata_filter` is a seam with no caller, and that is a decision.**
  SPEC.md §5.3 requires metadata filtering on both dense stores from the
  first dense-store commit, and it exists: `VectorStoreProtocol.search`
  declares it, `InMemoryVectorStore` and `LanceDBVectorStore` both apply it
  through the single `_matches_filter` helper, and `tests/test_dense.py`
  covers both paths. Nothing in the product passes one — no CLI flag, no
  `SearchRequest` field, and `Retriever._dense_candidates` calls
  `vector_store.search(embedding, top_k=fetch)` and nothing else — so the
  enabled branch is the most expensive code in the dense path, sits inside a
  file the coverage `core_subset` gates, and is proved only by its own unit
  tests. Recorded here rather than left to be rediscovered as an oversight.

  Exposing it is not the parameter-forwarding it looks like. `index/bm25.py`
  has no filter at all, so a surface-level filter would apply in `dense`
  mode, have nothing to apply to in `bm25` mode (the default, ADR-0007), and
  in `hybrid` mode filter one candidate list while RRF fused the survivors
  with an unfiltered lexical list — excluded chunks re-entering the ranking
  through the other side by a depth-dependent amount (ADR-0005). A filter
  honoured by one of three modes and silently leaky in a second is ADR-0001
  hazard 3's symptom, plausible-looking unfiltered results, one layer above
  where that hazard was closed. Two further costs: the keys chunk metadata
  actually carries are `source` — the ingest-time copy ADR-0006 made
  retrieval *ignore* because it goes stale on re-ingest — plus
  `file_name`/`file_extension` from `FileLoader` and `content_type` from
  `UrlLoader`, nearly all of them derivable from the `documents` row that
  owns them durably; and a non-empty filter switches LanceDB from a top-k
  query to `count_rows` plus a full-table fetch, so a `filter` field on
  `SearchRequest` would give any caller of the read-only surface a switch
  that makes every dense query O(corpus), and would be the first
  unbounded-cardinality input on a surface whose other fields are all
  bounded scalars.

  **Trigger to wire it up:** a concrete request to restrict results to a
  subset of one collection that cannot be met by making that subset its own
  collection — collections, not filters, are this repo's partitioning
  mechanism (one SQLite store and one LanceDB table each, and
  `SearchRequest` already takes `collection`). When that request exists the
  filter belongs at `Retriever`'s document join, over `documents` columns
  (`source`, `source_class`, `extractor`), where all three modes honour it
  identically and the durable source is filtered rather than its
  chunk-metadata copy; `metadata_filter` then becomes a push-down
  optimization beneath it rather than the surface itself. That change needs
  its own ADR for the bounds and failure mode of a caller-supplied filter
  (key count, key and value length, allowed key charset, and whether an
  unknown key fails closed or matches nothing): ADR-0014 decision 6's schema
  test checks field *names*, so it would pass such a field without
  constraining any of it.

- **`O_NOFOLLOW` does not exist on Windows, so the snapshot write and read are unguarded there.** `utils/path_safety.py`'s `O_NOFOLLOW` degrades to `0` — a no-op in the flag mask — so nothing crashes and nothing changes on win32, and both regression tests (`tests/test_url_loader.py::TestUrlLoaderSnapshotWriteDoesNotFollowASymlink` and `tests/test_snapshot_integration.py::TestSnapshotReadDoesNotFollowASymlink`) skip rather than passing vacuously. CI runs Linux, where both guards are real.
- **`UrlLoader`'s timeout bounds the fetch, not everything a `load()` call does.** `timeout_seconds` wraps the HTTP exchange. The preceding `ensure_safe_endpoint` DNS resolution and the trailing snapshot write are outside it, so a pathological resolver or filesystem can still hold a `load()` past the configured bound. Both are bounded by other means (the resolver by the OS, the write by `max_bytes`), and neither holds a remote connection open, which is the resource the bound exists to protect.
- **`Chunk.content_hash` is recomputed on every access, deliberately.** It is a sort tie-break in `retrieval/fusion.py` and `index/dense.py`, so it is hashed once per candidate per query (`index/bm25.py` caches it once at build time instead). Caching it on the model with `functools.cached_property` would unfreeze it: pydantic's `__setattr__` special-cases `cached_property` before it consults `frozen`, so `chunk.content_hash = ...` would silently succeed and every later reader — both tie-breaks and the value `index/metadata.py` persists — would use a string unrelated to `content`. The cost is unmeasured and the correctness loss is not, so it stays uncached; `tests/test_contracts.py::TestChunk::test_content_hash_cannot_be_decoupled_from_content_by_assignment` pins the refusal.

## Loaders workstream (ADR-0016) — URL ingestion landed (wave 4); the PDF/HTML ingest-side loaders have not

ADR-0016 schedules PDF/HTML/URL support in four waves. **Waves 1, 2 and 4 have landed, and
wave 3 has landed except for its ingest-side `.pdf`/`.html` loaders** — so the honest summary
is narrower than it used to be: groundkit *records, enforces and can re-verify* a source
class, and can now *produce* a `snapshot`-class document from a URL, but still cannot produce
an `extracted`-class document from a local PDF or HTML file. This is item 1 of SPEC.md §4.1's
v0.1.0 scope amendment, narrowed to that one remaining gap.

- **The PDF/HTML extractors exist; the local-file loaders that would use them still do
  not.** `groundkit/extraction.py` ships `PdfExtractor` and `HtmlExtractor` behind the `pdf`
  and `html` extras, with deterministic pinned configuration (`extraction_mode="plain"`,
  `html.parser`) and an identity string derived from distribution metadata. `resolve_citation`
  looks a citation's recorded extractor up in `active_extractors()` and re-extracts on a
  match. What is missing is the `.pdf`/`.html` **file loader** reachable from `grk ingest`:
  that needs multi-loader dispatch, which
  `docs/specs/loaders-extracted-and-remote-sources.md` §9.6 declines to design rather than
  guess at. `FileLoader` still handles `.md`/`.markdown`/`.txt` only, so no shipped path turns
  a local PDF or HTML file into an `extracted`-class document, and the re-extraction path
  stays exercised by tests rather than by a live file ingest. URL ingestion (below) is a
  separate path: it already reuses `html_extractor()` at fetch time to strip tags from an
  HTML-shaped response, but the resulting document is `source_class="snapshot"` either way —
  the extractor identity is not recorded on it, and re-extraction never runs for it.
- **A credential in a URL query string is refused by name, and the list of names is not
  exhaustive.** `_reject_unsafe_url_shape` refused userinfo (`user:pass@host`) from the start
  because this loader records the URL verbatim as `Document.source`; a `?token=…` reaches
  exactly the same places (SQLite, every `RetrievalResult` and `Citation`, the ingest log) and
  was permitted until it was refused too. The check is a denylist of unambiguous parameter
  names (`utils/url_safety.CREDENTIAL_QUERY_PARAMS`), compared after folding case and
  `-`/`_`. **A credential under a name not on that list still gets persisted.** `code` and
  `state` are deliberately absent — they are OAuth credentials in one context and a country
  code or a US state in another, and refusing them would reject ordinary document URLs.
  Redacting instead of refusing is not available: `documents.source` is `TEXT UNIQUE NOT
  NULL`, and `sanitize_url` redacts every query value unconditionally, so it would collapse
  `?id=42` and `?id=43` onto one identity. Supply credentials out of band.
- **Snapshot cleanup is per-document; collection-level deletion is still owed.** ADR-0023
  binds a snapshot's lifetime to its `documents` row — it is removed when that document is
  skipped as unchanged, replaced, deleted or pruned — which closes the two defects that
  existed before it (a re-ingest of an unchanged URL orphaned a full copy of the fetched
  text on every run, and deleting a document left its text on disk indefinitely). Two gaps
  remain. **Deleting a whole collection is not implemented at all**, so the third artifact
  SPEC.md §7 now names (`<collection>.snapshots/`, alongside `.sqlite3` and `.lance`) has
  no code to delete it; removing a collection today is a manual `rm` of all three. And
  cleanup is **best-effort by design** — a snapshot that cannot be unlinked (permissions, a
  Windows share violation) is a logged warning, not a failed ingest, because the durable
  state is already correct by then. That leaves disk litter which the next ingest of the
  same source retries, and which nothing else sweeps: there is no `grk gc`.
  A backup must cover all three artifacts: a restore from the `.sqlite3` alone cannot
  verify any `snapshot` citation, because the text it would compare against lives in the
  sibling `.snapshots` directory. `docs/guides/deployment.md`'s backup-scope paragraph
  says so.
- **The extractor-identity check no longer always-refuses, and neither does a `snapshot`
  citation.** An `extracted` citation still resolves only if its recorded extractor identity
  is active in this build — `active_extractors()` naming the recorded identity and reporting
  the active set when a match is missing — but that registry is no longer permanently empty.
  Wave 3 landed `PdfExtractor`/`HtmlExtractor` behind the `pdf`/`html` extras, and both extras
  are mirrored into the `dev` dependency group, so the default development and test
  environment already exercises a real, non-empty registry rather than the constant-empty one
  this entry used to describe. A build installed with neither extra still refuses every
  `extracted` citation — correctly, and for the same reason as before — but that is now a fact
  about which extras were installed, not a permanent pre-wave-3 state (fail closed either way,
  ADR-0016 decision 2). `snapshot` citations no longer refuse at all, pending or otherwise:
  wave 4's `UrlLoader` writes a local snapshot at ingest time, and `resolve_citation` reads it
  back through `_resolve_snapshot` (`retrieval/citations.py`) instead of refusing.
- **A pre-v3 store is now refused for *writes*, which is broader than the earlier promise.**
  `SCHEMA_VERSION` went 2→3 (ADR-0016 adds `source_class`/`extractor` to `documents`). Because
  `CREATE TABLE IF NOT EXISTS` never adds a column to a table that already exists, a store
  created before this cannot hold a source class, so `upsert_document`, `replace_document` and
  `get_document_records` refuse with a `StorageError` naming delete-and-re-ingest as the
  remedy. This supersedes the narrower rule that a legacy store "keeps serving BM25-only
  indexing unchanged" — that promise was made before the columns existed. **Reads are
  deliberately untouched**: an existing collection stays openable and searchable, it just
  cannot be added to. Pre-1.0 the remedy is a rebuild, never a migration (ADR-0004 decision 5).
- **The join has one narrow fallback worth knowing about.** `Retriever` reads provenance
  through `DocumentRecordStoreProtocol` and, for a store that does not implement it, degrades
  to `("text", None)`. That is sound rather than fail-open *for the stores it can apply to* — a
  store with no way to report a source class never had one to drop, unlike the real defect this
  work closed (a store that had the value and discarded it). `SQLiteMetadataStore` implements
  the protocol and a conformance plus signature-parity test pins that. The residual is a
  hypothetical *second* real store forgetting the method; there is only one today.

## Phase 6 — every IaC path exercised; two verifications are narrower than their rows read

Phase 6 lands in two changes (`docs/specs/phase-6-iac-observability.md` §3). The
first is `infra/` plus its ADRs and docs and **changed nothing under `src/`**;
the second adds the OpenTelemetry instrumentation and the JSON log formatter.
Both have landed. On 2026-08-16 a real groundkit span was observed in Jaeger,
and later the same day a real `terraform apply` against a live AWS account
provisioned the instance, ingested a document, and served a real search over
an SSM tunnel, the full documented compose cold-start ran end to end, and the
Kubernetes sequence ran on a single-node cluster. **Every IaC path has now been
exercised at least once**, so what remains is not missing coverage but two
verifications whose scope is narrower than a green row reads, plus one span
site. `infra/README.md` is the status board and records the exact scope of what
was executed.

- **The Kubernetes run was single-node, so the `ReadWriteOnce` hazard the
  documented sequence guards against was never reproducible.** Docker Desktop
  (kind mode, v1.36.1, one node) ran the sequence verbatim on 2026-08-16 —
  scale-down steps included — through `apply -k`, corpus load, a completed
  ingest Job (43 files, 1299 chunks), Deployment 1/1 Ready and a
  citation-bearing search over `port-forward`. What a single node cannot do is
  produce a multi-attach failure, because there is no second node to attach the
  volume from. The manifests are therefore confirmed self-consistent and
  confirmed to work in the small case; their multi-node behaviour is unproven.
  This is the case `infra/README.md` explicitly warned would "pass in the small
  case and stall in the real one", so the row names the cluster kind rather than
  reading as a general pass.

- **The compose stack's loopback-only binding is verified host-side only, and
  the other half could not be run.** The documented cold-start sequence
  completed on 2026-08-16 — model pull, the ingest one-shot (43 files, 1299
  chunks, 1299 vectors through Ollama), `up -d`, and a citation-bearing
  `POST /v1/search` over `127.0.0.1:8765`. The binding itself was demonstrated
  by a differential through this host's own LAN interface address: a control
  container published on `0.0.0.0` answered on `10.0.0.16:8766` while the
  service published on `127.0.0.1` was refused on `10.0.0.16:8765`, with the
  host listener table showing the two bind addresses. **What was not done is
  the check ADR-0021 decision 1 actually names — a connection attempt from a
  different host.** It was attempted from a phone and could not complete: that
  device could not reach the `0.0.0.0` control port either, while this host
  reaches it fine, which places the fault in the network rather than in
  anything here. The only Wi-Fi available is a guest SSID with client
  isolation, so no device on it can reach this machine. Closing it needs a
  wired host or the non-guest network. Kept as a distinct gap because "the
  bind is correct, demonstrated locally" and "no external host was ever
  refused" are different claims, and only the first is currently true.

- **A groundkit span has now been observed in Jaeger (2026-08-16), for `ingest`
  `retrieve` and — since a later run the same day — `synthesize`.**
  `src/groundkit/telemetry.py` is the
  tracer accessor and the typed attribute helper (ADR-0022 decision 3),
  instrumenting `Indexer.index_source`, `Indexer.index_directory`,
  `Retriever.search` and `Synthesizer.synthesize`. `opentelemetry-api` is a base
  dependency (ADR-0022 decision 1), so every instrumentation site works with no
  `otel` extra installed — spans are simply non-recording. The verified run
  exercised a real `ingest` and `search` against the compose stack's collector,
  and a sweep of the exported payload confirmed the allowlist held: no query
  text, no source path, no document content. The `synthesize` span was then
  observed too, from a `docker compose run … grk answer` against the running
  stack: it carried `chat.model`, `chat.provider`, `duration_ms` and
  `result_count` and nothing else, and a sweep for the question text, the
  completion text, both citations' offsets, the corpus path and the source
  filename found none of them. **Two caveats keep that from being a
  self-contained stack test.** The chat model was the operator's *host* Ollama
  via `host.docker.internal`, because the stack's own `ollama` volume holds
  only the embedding model; and the first attempts, with a larger local model,
  timed out against `ChatConfig.timeout_seconds`' 60-second default, which no
  CLI flag can raise. Those timeouts were independently useful: the error-path
  span carried `otel.status_code=ERROR` and `groundkit.failure_kind=ChatError`
  and still leaked nothing, which is the harder half of the allowlist claim.
- **FIXED 2026-08-16 — a provider timeout reported no reason at all.** Both
  `providers/llm.py`'s `_raise_chat_error` and
  `providers/embeddings.py`'s `_raise_embedding_error` rendered
  `f"... failed: {detail}"` from `_scrub(str(exc), secret)`, and `str()` on
  several `httpx` timeout types is the empty string — so a 60-second
  `ReadTimeout` surfaced as `Chat request to <url> failed:` with nothing after
  the colon. Found while verifying the `synthesize` span, where it cost three
  runs to identify a plain timeout. A shared `_error_detail` helper now falls
  back to the exception's class name; the scrubbing is unchanged. Recorded
  rather than deleted for one reason worth keeping: **only the chat half was
  reported**, and the embedding half had the identical defect, which is why the
  fix went into a helper both call rather than into the reported call site.
  A chat provider timeout still cannot be raised past
  `ChatConfig.timeout_seconds`' 60-second default from the CLI — no flag
  exposes it.
- **Installing the `otel` extra and setting `OTEL_*` does not, on its own, make
  a span record — and this was shipped wrong before it was caught.** The
  variables in ADR-0022 decision 2 are read by
  `opentelemetry.sdk._configuration`, which runs under the
  `opentelemetry-instrument` launcher and *not* on import, so absent an explicit
  `trace.set_tracer_provider` call the API keeps returning a
  `ProxyTracerProvider` whose spans are non-recording. The first version of this
  change had all three span sites correct, the allowlist enforced, and a green
  suite — and exported nothing at all, silently. `telemetry.configure_tracing()`
  is the fix, called from the CLI entry point;
  `tests/test_telemetry.py::TestConfigureTracing` is the regression test, and
  ADR-0022 decision 1 carries an amendment recording the trap. The general
  lesson is the one SPEC.md §1.4 already encodes: a unit suite cannot verify an
  export path, and only running the stack found this.
- **`Indexer.run` never existed, and ADR-0022 decision 5 originally named it —
  the ADR now carries an erratum rather than a silent fix.** The real public
  ingest entry points are `Indexer.index_source` and `Indexer.index_directory`
  (both `async`, `src/groundkit/indexer.py`), and each gets its own span
  rather than one shared "ingest" span — attributed with the document and
  chunk counts from the `IndexReport` each returns. A child span per file, for
  per-file latency, is a named follow-up and not part of this change: it would
  need its own decision about span semantics under `index_directory`'s
  concurrent fan-out.
- **All three of SPEC.md §3's span sites were Phase 6's to instrument,
  `synthesize` included — this entry used to defer it and no longer does.**
  The deferral was written while Phase 5 was under construction alongside
  this phase, when `providers/synthesis.py` genuinely did not exist. Phase 5
  then landed (SPEC.md §9: done 2026-08-15) and this branch merged `main`, so
  the seam — `async Synthesizer.synthesize(query, results)` — was present and
  settled by the time this change was written. The premise expired without
  the text changing, which is the failure mode to notice: a deferral
  justified by "it does not exist yet" has a shelf life, and nothing about
  merging the thing into existence makes the record update itself.

  The residual worth knowing is not a missing span but a sharper allowlist
  obligation: a synthesis span sits closer to prompt text, completion text and
  citation spans than any other site, and none of those may become a span
  attribute. Model identity, result count, latency and a typed failure code may.
- **The read-only-mount deferral is discharged in part, and the part is worth
  stating precisely.** The "read-only does not mean the process writes no bytes"
  entry above deferred filesystem-level enforcement to this phase. What that
  buys is the **corpus** mount: `/data/corpus` is a read-only mount in every
  deployment surface, so the documents each citation resolves against are
  unwritable by the serving process, enforced by the kernel rather than scoped
  in Python. `/data/index` remains read-write and no mount flag changes that
  while ADR-0002 keeps the store in WAL mode — `SQLiteMetadataStore.open` writes
  its sidecars, runs `CREATE TABLE IF NOT EXISTS`, and chmods, on every open. So
  read-only is now true of the corpus and still false of the index (ADR-0021
  decision 2). A read-only *open path* in `index/metadata.py` would close the
  rest and is a separate `src/` change with its own ADR.
- **A container cannot enforce the loopback bind, so the image is not safe to
  run with the obvious short command.** Inside a container a process bound to
  `127.0.0.1` is reachable from nothing, so the image binds `0.0.0.0` with
  `--allow-remote-access` and the guarantee moves to the publish boundary:
  host-loopback publish in compose, a ClusterIP Service in Kubernetes, no
  ingress rules at all in Terraform (ADR-0021 decision 1, ADR-0020 decision 2).
  `docker run -p 8765:8765` and `--network host` both publish an
  unauthenticated, content-bearing surface on every interface of the host, and
  the image cannot tell those cases from the safe one. What was a refusal in
  Phase 4 is a warning in a container.
- **The Kubernetes probes target a real operation because there is no health
  endpoint.** `GET /v1/collections` proves the process is serving HTTP and the
  index directory is listable; it does **not** prove any collection is usable,
  because `handle_list_collections` returns `[]` for a missing or empty index
  directory rather than failing. A pod with an unmounted volume therefore
  reports ready. Adding `/healthz` means widening ADR-0014 decision 2's
  route-parity exclusion set — a security-relevant `src/` change deliberately
  not taken in a change that touches no `src/` file
  (`docs/specs/phase-6-iac-observability.md` §4.2, Q1).
- **The Kubernetes in-cluster boundary rests on a NetworkPolicy, which can be
  silently inert.** A `ClusterIP` Service closes the cluster's edge and nothing
  else — every pod in every namespace can dial one directly — so
  `infra/k8s/networkpolicy.yaml` (default-deny ingress, empty `podSelector`) is
  what actually closes in-cluster reachability to this unauthenticated,
  content-bearing surface. On a cluster whose CNI does not enforce
  NetworkPolicy, the API server accepts the object, reports no status, emits no
  warning, and every pod can still reach the service. Enforcement has to be
  confirmed against the cluster; no manifest can assert it. The same policy may
  also break `kubectl port-forward` on a CNI that blocks node-to-pod traffic —
  most permit it, since kubelet probes would otherwise fail, but that is a CNI
  behaviour rather than a Kubernetes guarantee. compose and the Terraform
  module rest on a kernel-level socket bind instead and are contingent on
  nothing.
- **The base kustomization pins `sortOptions: order: fifo`, and that is
  load-bearing.** Kustomize's legacy sort reorders by kind and has no entry for
  `NetworkPolicy`, so it rendered *after* the Deployment — meaning a first
  `apply -k` to a shared cluster could have the Deployment controller create a
  reachable pod before the default-deny policy was submitted. For an
  unauthenticated, content-bearing service that window is the whole exposure.
  FIFO makes declaration order the applied order. Verified by rendering:
  `kubectl kustomize infra/k8s` now emits `NetworkPolicy` second, ahead of
  `Deployment`. Not gated — a render-order assertion is parked on
  `chore/infra-ci-checks-parked`.
- **The Kubernetes manifests do not solve getting documents onto the volume.**
  `pod-corpus-loader.yaml` is a `kubectl cp` target and a recipe, not a
  mechanism. A real deployment substitutes whatever it already uses.
- **Three Kubernetes sequencing rules are not enforceable by a manifest.** The
  `ReadWriteOnce` claim admits one pod, so the Deployment must be scaled to zero
  before either one-shot runs, or they sit Pending on a multi-attach error — on
  a multi-node cluster only, which means the wrong order passes on a laptop
  cluster and stalls in production. The image must be set in **both**
  `k8s/kustomization.yaml` and `k8s/ingest/kustomization.yaml`, because a
  kustomize `images:` transformer reaches only its own kustomization's
  resources; setting one leaves the other on the unpublished placeholder, and
  setting them differently has an ingest and the serve reading it disagree about
  the code that built the index. And the ingest Job must be **deleted before it
  is re-applied**: a Job's pod template is immutable, so `apply` over a
  completed one is accepted, creates no pod, and leaves a following
  `wait --for=condition=complete` to return immediately against the *previous*
  run — so within the hour before `ttlSecondsAfterFinished` collects it,
  re-ingesting after copying new documents reports success having done nothing.
  All three are documented in `infra/README.md` and in the manifests themselves;
  none is checked by anything.
- **`create_ssm_vpc_endpoints` creates `ssm` and `ssmmessages` only.**
  `ec2messages` is the legacy channel — the SSM agent on the pinned AL2023 image
  uses the other two — and it is not offered in every region. That is more than
  a tidiness point: `data.aws_vpc_endpoint_service` errors on a service its
  region does not offer, and a failing data source fails the entire `plan`, so
  an optional endpoint nothing used could break the whole module. Add it back
  through `ssm_vpc_endpoint_services` if you run an older agent, in a region you
  have confirmed offers it.
- **`embedding_base_url` is escaped differently for the systemd unit than for
  the ingest helper, and that asymmetry is deliberate.** systemd reads `%` in
  `ExecStart` as the start of a unit specifier, so `%20` is not a space but an
  invalid specifier and `systemctl enable --now` refuses the whole unit —
  ending bootstrap with no service. The unit therefore doubles `%` to `%%` and
  the helper does not, because a shell has no such rule and doubling it there
  would pass a different URL to the ingest. Both still carry the same URL; only
  one has to say so in systemd's escaping. An explicit port is separately
  validated to 1-65535, since port 0 otherwise produces a real egress rule
  permitting nothing usable.
- **The Terraform module's data volume is prepared by a retrying unit, because
  the attachment can arrive after cloud-init has finished.**
  `aws_volume_attachment` cannot be requested until the instance resource
  completes, by which time cloud-init is already running. A slow attach used to
  exhaust a bounded inline wait and end provisioning permanently, while
  Terraform went on to attach the volume and report a clean apply — an
  apparently applied deployment with no service and nothing that would retry.
  Storage prep is now `groundkit-storage.service`, a `oneshot` with
  `Restart=on-failure` that starts `groundkit.service` on success. It also
  resolves the volume under three names, since a Xen instance renames the
  requested `/dev/sdf` to `/dev/xvdf` and has neither the NVMe by-id link nor
  the requested name.
- **`data_volume_type` accepts only `gp3` and `gp2`, which is narrower than
  "any block device type".** `io1`/`io2` require an `iops` argument this module
  does not set and `st1`/`sc1` have a 125 GiB minimum the 20 GiB default
  violates, so both fail at apply for a value the docs used to invite.
  Supporting them means exposing and validating type-dependent size and IOPS.
- **The Terraform module needs outbound HTTPS at boot, so a private subnet needs
  NAT.** Bootstrap installs docker from Amazon Linux's CDN and pulls the
  container image. `create_ssm_vpc_endpoints` carries the Session Manager
  control channel only: set on an otherwise egress-free subnet it produces an
  instance an operator can open a session to and no service running on it,
  because `set -e` ended bootstrap at `dnf install`. A genuinely egress-free
  deployment needs a prebaked AMI, or `ecr.api`/`ecr.dkr` interface endpoints
  plus the `s3` gateway endpoint *and* a VPC-reachable package mirror — none of
  which this module builds (ADR-0020 decision 3).
- **The Terraform module's embedding egress is derived, and two forms of
  endpoint it cannot derive fail at `plan` rather than being supported.** The
  standing egress rule is TCP 443; `embedding_base_url` adds a rule for its own
  port when that is not 443, which is the documented case, since Ollama listens
  on 11434. A host given as a **DNS name** cannot be turned into a CIDR at plan
  time and needs `embedding_egress_cidr` supplied alongside it. An **IPv6**
  endpoint is not supported at any port: the module writes `cidr_ipv4` rules
  only, the standing HTTPS rule included, so "443 is already covered" is a
  statement about IPv4 and nothing else. Both refuse loudly, which is the
  intended trade — the alternative is an embed call that times out at the
  security group on a deployment that applied cleanly.
- **Three Terraform inputs are constrained to character classes because they are
  written into files on the instance.** `container_image` and
  `embedding_base_url` reach both the generated ingest helper and the systemd
  unit; `collection` reaches the helper. They are single-quoted there, and the
  classes — which exclude the single quote, `$`, backticks, backslashes and
  whitespace — are what make that quoting a property rather than a convention.
  `collection` mirrors `index/metadata.py`'s own pattern and `.`/`..` rejection,
  so a name groundkit would refuse cannot reach an instance. A legitimate value
  outside those classes is rejected at `plan`; `&` is deliberately still allowed.
- **The Terraform deployment is x86_64 and `instance_type` is validated against
  it.** The AMI filter selects `al2023-ami-2023.*-x86_64` and the container image
  is built by a plain `docker build` on an amd64 runner, so it carries one
  manifest. A Graviton instance type is refused at `plan` rather than by EC2 at
  launch, and arm64 support is a multi-architecture image build rather than a
  different value for that variable. A Graviton family the pattern does not
  recognise falls through to EC2's own launch rejection, which is loud.

  The conditions that produce those refusals are resource `precondition`
  blocks, and **they have never executed.** `terraform validate` does not
  evaluate a precondition and `terraform console` cannot reach one, so nothing
  offline can. CI checks the state they reject rather than the rejection
  itself — see `infra/terraform/aws-ec2/README.md`'s status table, where that
  row is deliberately separate from the rows CI earned. The `instance_type`
  refusal is a variable `validation` instead, which *is* reachable offline and
  is checked directly.
- **No groundkit image is published to any registry**, so the image reference in
  `deployment.yaml` is a placeholder that will not pull, and the Terraform
  module's `container_image` has no default.
- **The Terraform module creates no backups.** `prevent_destroy` on the data
  volume means the index survives instance replacement and nothing more; there
  is no snapshot schedule and no DLM policy. SPEC.md §7 names backup scope,
  retention and deletion behaviour as product decisions owed before any
  deployment that is not a single user's local machine, and ADR-0020 decision 4
  settles exactly one of them. It is also a single instance: every AMI or
  instance-type change is downtime.
- **The Terraform path has now been proved against a real AWS account
  (2026-08-16); compose and Kubernetes were verified separately the same day
  (see the entries above).** SPEC.md §1.4 requires each
  path be verified with the date recorded and SPEC.md §2 forbids recording a
  date no run produced, so some rows are still empty. The machine this tree
  was originally written on had a Docker CLI with **no running daemon**, a
  `kubectl` with **no cluster context**, and no cloud credential — nothing was
  built, pulled, applied or planned there. What *was* executed and observed at
  that point: the compose file parses and interpolates, the Kubernetes base
  renders and every manifest parses, the Terraform module passes `fmt -check`
  and `validate` against AWS provider 5.100.0 and 6.60.0, and the user-data
  template renders to a script that passes `bash -n`. The `infra` CI job covers
  what that machine could not — on this change's first run the image built, ran
  as uid 10001 under a read-only root, and all six pinned third-party tags
  resolved — and it gates all of that on every pull request from now on. It also
  evaluates the two things this module *derives* from string inputs, rather
  than only proving they parse: the ECR-registry match that decides an IAM
  attachment and a `docker login`, and the egress rule the embedding endpoint
  needs, both checked with `terraform console`, which resolves locals and
  `templatefile()` without a provider credential.

  A later session had a real AWS account available and closed the last
  Terraform row: `terraform plan` against a real personal-sandbox account in
  `us-east-1` produced 9 resources to add with a real AMI resolved and the
  ECR-detection logic correctly matched a real private-ECR image reference;
  `terraform apply` created them; bootstrap installed docker, authenticated to
  the private ECR repo and pulled the image, and mounted the data volume via
  the retrying storage-prep unit; `groundkit-ingest` indexed a planted
  document; and a real `POST /v1/search` over an SSM port-forward tunnel
  returned the correct citation-bearing result. `terraform destroy` ran in the
  same session. One finding worth naming here rather than only in
  `infra/README.md`: the account's default VPC ships as all-public with no NAT
  gateway, which does not satisfy this module's "private subnet with NAT"
  network prerequisite — `associate_public_ip_address` is pinned `false`
  unconditionally, so a public-subnet-with-no-NAT deployment would apply
  cleanly and then fail bootstrap's `dnf install` under `set -e`. A NAT
  gateway and a dedicated route table had to be created outside the module as
  a prerequisite, which is a real operational cost of this module's documented
  network requirement, not a defect it introduced. Full scope — what this run
  did and did not exercise (BM25-only, no SSM VPC endpoints, one account, one
  region) — is in `infra/README.md`.

  Compose and Kubernetes are no longer unrun: the entries above record a
  completed `compose up` cold start and a completed `apply -k` sequence, both
  from 2026-08-16, with their own narrower caveats stated there.

## Phase 5 caveats (the LLM boundary)

- **The redaction pass exists and is wired at exactly one boundary: cloud
  chat egress.** `build_chat` wraps every `openai_compatible` chat provider
  in `RedactingChat` with no operator opt-out (ADR-0017), constructing a
  fresh `Redactor` per call — a long-lived one would let `restore()` expand
  a token from one request into a value captured in another, a
  cross-request disclosure the mitigation itself would manufacture
  (regression-tested). The default pattern floor covers structurally
  recognizable values only: emails, E.164/US phone shapes, IPv4, long
  secret-shaped tokens. **No person-name pattern ships** — free-text name
  detection by regex is unreliable enough that shipping one under the label
  "redacts names" would be a false promise; SPEC.md §2's "names → tokens"
  is met through *configured* patterns, which is what its own "configurable
  patterns" clause provides. Two recorded residuals: the **embedding
  boundary is not redacted** — a deliberate SPEC §2 deviation (ADR-0017
  decision 5: order-dependent tokens are unstable across ingest/search
  processes, a vector over redacted text stops describing the stored chunk,
  and `CollectionManifest` records nothing about redaction, so a mixed
  collection would pass identity verification); and text that already
  contains a literal token-shaped substring (`[EMAIL_1]`) is
  indistinguishable from a real token once redaction runs, so `restore()`
  corrupts it (tested, documented, unfixed).
- **The faithfulness judge is advisory and uncalibrated.** Verdicts gate
  nothing and the exit code never depends on them; malformed or incoherent
  model output is a `JudgeError`, never coerced. The calibration procedure
  required before gating could ever be proposed is documented in
  `providers/judge.py`'s module docstring; no human-labeled verdict set exists
  yet, and normal CI never runs the judge.
- **Synthesis quality is unmeasured by the default `pytest` suite, but it is
  no longer true that nothing re-measures it automatically.** `grk eval
  --synthesis` runs the planted-marker citation-echo check (SPEC.md §2)
  against a real chat provider and writes its own artifact
  (`evals/results/echo-latest.json`); there is deliberately no offline double
  for it — an echo number from a scripted chat would be noise presented as a
  measurement. `eval-gated.yml` (see above) now runs `grk eval --dense
  --synthesis --judge` on the same weekly cron as the dense/fusion gate, so
  an echo/judge result is produced on a schedule rather than only when an
  operator happens to run it locally — it still never runs in `ci.yml`
  itself, since that job must stay offline and credential-free and this one
  needs a live chat provider.

  **`EvalReport.synthesis` now has a producer** — it was "structure without a
  producer" until `grk eval --judge` landed (2026-08-16), which requires
  `--synthesis`, synthesizes over the golden corpus against the run's best
  stage, and folds the judge tallies into the main artifact. One limit
  survives that change and is the reason this entry is not simply deleted:
  the judge is **advisory only — it exits 0 and gates nothing** (SPEC.md §6)
  until calibrated against human labels, so a falling faithfulness tally
  fails no build even on the now-automatic weekly run.
- **`grk answer` is CLI-only; synthesis is off the service surface
  (ADR-0019).** Synthesis is a read, but it adds cost amplification and
  egress amplification that a loopback bind does not bound. Named
  consequence: agentic-evalkit can grade groundkit's *retrieval* through
  the HTTP/MCP `ExecutionTarget` boundary, but not its generative half.
- **Prompt-injection defense at the synthesis boundary is structural
  only.** Retrieved content passes through `sanitize_content`
  (`providers/context_assembly.py`, ported from ARP per ADR-0001) before
  entering a prompt: delimiter-tag forgery is neutralized, control
  characters stripped, lines quote-prefixed. Instruction-like phrasing
  survives verbatim inside the quoting — this raises the bar against naive
  smuggling and is not a guarantee a model treats the content as inert.
  The ported `TokenBudgetAssembler`/`frame_content` envelope is available
  but unwired; only `sanitize_content` is live.
- **Citation-marker parsing is deliberately naive.** Every `[n]` in a
  completion counts as a marker, including bracketed numbers inside echoed
  source text — the distinction is not reliably recoverable from text
  alone (documented in `providers/synthesis.py`). Redaction tokens cannot
  collide with markers: the marker regex is digits-only and token
  categories always carry letters (regression-tested against ADR-0018's
  recorded hazard).
- **Query rewrite is reachable only through `grk answer --rewrite` and no
  eval number measures it.** Enabling it changes retrieval inputs, so its
  effect on retrieval quality is unquantified; a rewrite failure is a
  typed error, never a silent fallback to the original query.

## Phase 2 caveats

- The adversarial category does not test prompt-injection resistance. With
  no LLM in the retrieval path, "injected text must never surface as
  instructions" was untestable in Phase 2. Phase 5 added the structural
  sanitization pass and the advisory judge described above, but neither
  turns this category into a resistance test — that remains an unclaimed
  property. What Phase 2 asserts is fixture correctness — that
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

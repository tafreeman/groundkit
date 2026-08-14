# Known limitations

Honest and current, per repo policy. Updated with each phase.

## Current state (Phase 3, Wave E)

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

- **The dense and fusion quality delta is measurable but not yet measured.**
  Wave E built the harness: `grk eval --dense` emits `bm25` → `dense` →
  `fusion` into one report, deltas are derived at read time
  (`evals/delta.py`), and a stage that loses to baseline is reported as a
  loss rather than suppressed. What has not happened is a run that produces
  a *meaningful* number. The CI-default `InMemoryEmbedder` produces
  hash-derived vectors with no semantic signal, so the delta it yields is
  noise presented as a number (SPEC.md §2) — the CLI stamps an explicit
  warning on any such report, and `RunConfig.embedding` records the provider
  so the artifact self-labels. A real delta needs `EVAL_GATED=1` against a
  live embedding model (`.github/workflows/eval-gated.yml`). Until that has
  run, no claim about dense or hybrid retrieval quality is supported, and Q1
  stays open.
- **A dense or fusion stage cannot abstain on a no-answer query.** BM25
  returns nothing when no indexed chunk shares a term with the query, which
  is what makes the corpus's `no_answer` judgments measure abstention at
  all. A vector search has no equivalent floor: it returns its `top_k`
  nearest neighbours regardless of how far away they are, and
  `score_threshold` defaults to `None` (and is excluded from fused scores
  entirely, ADR-0005 decision 6). So `no_answer_abstained_count` is
  structurally 0 for the dense and fusion stages while the baseline can
  abstain on every one. This is a real property of the retrieval strategies,
  not an artifact of the test embedder, and it means abstention is not
  comparable across stages — read it per stage, never as a delta.
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
  nothing yet for a mismatch to exist against.
- **Cross-store writes are not atomic.** SQLite and the vector store share
  no transaction. The write order — chunk, embed, write the manifest (once,
  before the first vector add), add the new vectors, delete the previous
  document's, commit SQLite last — is chosen so SQLite is never ahead of the
  dense store, and so the document is never left with *no* vectors at any
  point. A dense store *behind* SQLite is silent: the content-hash skip key
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
  (`git checkout`) left SQLite's `content_hash` matching the restored bytes,
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
  The incremental content-hash gate runs before chunking and therefore
  before embedding — that is exactly what keeps an unchanged document from
  being re-embedded on every run. The cost is that enabling an embedder and
  vector store over a collection already ingested BM25-only leaves every
  unchanged document without vectors: it is skipped, so it is never
  embedded. Only documents whose content actually changes afterwards gain
  vectors. The collection is then permanently half-dense until it is
  re-ingested from scratch, and nothing reports that state. This is also
  the case the delete reconciliation above deliberately tolerates. Forcing
  a backfill means deleting the collection and re-ingesting it.
- **A BM25-only indexer will orphan a vector-bearing collection.** The
  inverse of the above: an `Indexer` constructed without an embedder or
  vector store still happily replaces, prunes, and deletes documents in a
  collection whose vectors were written by an earlier dense-enabled run.
  It has no vector store to delete from, so those vectors survive their
  documents. The orphans are loud, not silent: Wave C's dense read path
  fails closed on them — `Retriever.search` (dense and hybrid modes)
  raises `RetrievalError` on a hit whose document has no stored source,
  with regression tests for both the deleted-after-open and
  deleted-before-open cases. Nothing prevents the situation being created
  in the first place, because the store carries no record that dense
  writes ever happened beyond the manifest itself.
- **The CLI exposes the dense path, entirely opt-in.** `grk ingest --dense`
  embeds and writes vectors alongside the SQLite write; `grk search --mode
  {bm25,dense,hybrid}` reads them back (default `bm25`, unchanged by this
  wave — Q1 stays open). Both share `--embed-provider`/`--embed-model`/
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
- **`index/dense.py` is not in the coverage `core_subset`.** It sits at 100%
  today, but the gate that would keep it there covers `retrieval/*`,
  `ingestion/chunking.py`, and `index/bm25.py` only. Adding it is a Wave F
  decision (see `docs/specs/phase-3-hybrid-retrieval.md`).
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

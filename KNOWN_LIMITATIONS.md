# Known limitations

Honest and current, per repo policy. Updated with each phase.

## Current state (Phase 3, Wave A)

BM25-only retrieval works end-to-end locally: `grk ingest` (file or
directory, incremental by content hash), `grk search` (citation-bearing
results with character offsets), and `grk eval` (retrieval-quality harness
against the committed golden corpus, BM25-only baseline) — all against a
persisted SQLite index that survives restarts, all offline with no cloud
credentials. Not yet built, arriving in their phases per SPEC.md §9:

- **Dense retrieval exists but is not reachable.** Wave A landed the vector
  stores (`InMemoryVectorStore`, `LanceDBVectorStore`) and the ADR-0004
  collection manifest, but nothing wires them into `Indexer` (Wave B) or
  `Retriever` (Wave C). `grk ingest` writes no vectors and `grk search` is
  still BM25-only. Hybrid fusion and rerank are Waves C and D.
- **The embedding-identity manifest is enforceable but unenforced.** ADR-0004
  decision 3 requires verification at `Retriever.open()` and at any ingest
  that writes vectors. The store-side machinery (`write_manifest`,
  `verify_manifest`) is implemented and tested; no caller invokes it yet, so
  today nothing actually stops a model swap. That wiring is Wave B, and until
  it lands the guarantee is theoretical.
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

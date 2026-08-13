# Known limitations

Honest and current, per repo policy. Updated with each phase.

## Current state (Phase 2)

BM25-only retrieval works end-to-end locally: `grk ingest` (file or
directory, incremental by content hash), `grk search` (citation-bearing
results with character offsets), and `grk eval` (retrieval-quality harness
against the committed golden corpus, BM25-only baseline) — all against a
persisted SQLite index that survives restarts, all offline with no cloud
credentials. Not yet built, arriving in their phases per SPEC.md §9:

- Dense retrieval, hybrid fusion, rerank, and metadata filtering at search
  time (Phase 3) — embedding providers exist and are tested, but nothing
  consumes them yet.
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

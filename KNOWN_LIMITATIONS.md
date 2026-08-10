# Known limitations

Honest and current, per repo policy. Updated with each phase.

## Current state (Phase 1)

BM25-only retrieval works end-to-end locally: `grk ingest` (file or
directory, incremental by content hash) and `grk search` (citation-bearing
results with character offsets) against a persisted SQLite index that
survives restarts. Not yet built, arriving in their phases per SPEC.md §9:

- Retrieval-quality eval harness and golden corpus (Phase 2).
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

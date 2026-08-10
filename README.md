# groundkit

Grounded, citation-verifiable hybrid retrieval: a persisted BM25 + dense index,
a named MCP server, and a retrieval eval harness — fully local by default.

> **Status: pre-alpha (Phase 0).** The spec, ADR-0001, and this skeleton exist;
> no retrieval code has landed yet. See [SPEC.md](SPEC.md) for what is being
> built and in what order, and [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) for
> what is deliberately out of scope.

## What this will be

- **Hybrid retrieval** — BM25 + dense embeddings + reciprocal-rank fusion +
  optional local cross-encoder rerank, over an index that survives restarts.
- **A real MCP server** — stdio + streamable HTTP, exposing `search`,
  `fetch_chunk`, `list_collections`, `index_status`; installable and
  connectable from Claude Desktop/Code.
- **A retrieval eval harness** — labeled golden corpus with recall@k, MRR and
  nDCG@k computed by deterministic, unit-tested code; BM25-only is the baseline
  every feature must beat (or the report says it didn't).
- **Local-first** — Ollama embeddings and a file-based index by default; cloud
  providers are opt-in and sit behind a redaction boundary.

Deterministic core, LLM at the boundary: no LLM runs in the retrieval path.

## Development

```bash
uv sync --group dev
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest
```

## Provenance

groundkit is a standalone successor to the RAG library inside
[agentic-runtime-platform](https://github.com/tafreeman/agentic-runtime-platform),
built to close its verified production gaps (persistence, directory-scale
ingestion, metadata filtering, retrieval-quality evals, service/MCP surface,
IaC). The per-module promote-vs-rewrite decision is recorded in
[ADR-0001](docs/adr/ADR-0001-promote-vs-rewrite.md).

**Portfolio composition:** groundkit may consume
[executionkit](https://github.com/tafreeman/executionkit) for LLM call
patterns at the synthesis boundary, and is gradable by
[agentic-evalkit](https://github.com/tafreeman/agentic-evalkit) via its
HTTP/MCP `ExecutionTarget` boundary. It imports the internals of neither
repo.

## License

MIT

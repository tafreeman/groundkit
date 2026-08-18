# Architecture

groundkit is a retrieval service built around one structural commitment, from
which most of the rest follows: **the core is deterministic, and the LLM sits
at the boundary.** Ingestion, chunking, indexing, scoring, fusion, and citation
resolution are pure, typed, tested code. No model runs in the retrieval path.

The corollary is that retrieval quality is *measurable* rather than
impressionistic — which is why the eval harness landed in Phase 2, before
hybrid retrieval and rerank, instead of after them.

## Components

```
ingestion/          loaders (path-contained), chunking (offset-preserving)
index/              bm25 (persisted), dense (LanceDB), metadata (SQLite)
retrieval/          search orchestration, RRF fusion, rerank, citations
providers/          embeddings (Ollama | OpenAI-compat), redaction,
                    optional query-rewrite/synthesis
service/            FastAPI api, MCP server
evals/ (package)    deterministic metrics engine
cli                 grk
```

Every component boundary is a `typing.Protocol` seam, and every seam has a
conformance test. That is a structural fix for a specific failure: the
predecessor project drifted its implementations away from their interfaces
because nothing checked, and
[ADR-0001](../adr/ADR-0001-promote-vs-rewrite.md) hazards 3–4 record what that
cost.

## The principles, and what enforces each

An architecture principle with nothing enforcing it is a preference. Each of
SPEC.md §2's non-negotiables has a mechanism:

| Principle | What enforces it |
|---|---|
| Deterministic core, LLM at the boundary | No model import anywhere under `retrieval/`; the rerank model import is deferred and optional. See [The LLM boundary](llm-boundary.md). |
| Citations are verifiable | Every result carries document ID, chunk ID and char offsets; `retrieval.verify_citation` re-reads the source and checks them. |
| Anonymization at the LLM boundary | Enforced on cloud **chat** egress (`build_chat` wraps it in `RedactingChat`, no opt-out); deliberately absent on the embedding boundary (ADR-0017 decision 5). The boundary is documented in full [here](llm-boundary.md), including exactly what the pattern pass does and does not cover. |
| Fail closed | Frozen Pydantic models with `extra="forbid"`; unconfigured provider raises a typed error; no cross-provider embedding fallback, ever. |
| Real data only | No metric value is written into any document by policy. Numbers come from generated eval artifacts or live badges. |
| No in-process state shadowing persisted state | SQLite is the durable truth; the BM25 index is rebuilt at open ([ADR-0002](../adr/ADR-0002-index-persistence.md)). |

## Data model

Frozen Pydantic v2 models with `extra="forbid"`: `Document`, `Chunk` (carrying
`content_hash`, `document_id`, `chunk_index`, and char offsets `start`/`end`),
`RetrievalResult`, and `Citation`. Confidence thresholds are named constants
rather than inline literals. See the [contracts reference](../reference/contracts.md).

## Persistence

One index directory per collection: a LanceDB table for vectors and chunk
refs, SQLite for documents, chunks, offsets, and the ingest state that makes
re-indexing incremental. Content-hash dedup on write.

Two consequences worth knowing before you deploy anything:

- **SQLite here is content-bearing data, not just an index.** It holds your
  document text. Deletion behaviour, file permissions, backup scope and
  retention are product decisions, not operational details to settle later.
- **Deleting a collection means deleting both files** — its `.sqlite3` *and*
  its `.lance` sibling. Neither is inferable from the other's absence, and
  [ADR-0011](../adr/ADR-0011-bm25-only-ingest-refuses-a-dense-collection.md)
  covers the failure mode where they disagree.

## Where the decisions are recorded

The [ADR index](../adr/index.md) is the authoritative list. The ones that
explain the most about how retrieval behaves:

- [ADR-0002](../adr/ADR-0002-index-persistence.md) — why SQLite is the truth
  and BM25 rebuilds at open.
- [ADR-0005](../adr/ADR-0005-fusion-and-rerank-scoring.md) — rank-based RRF
  and why rerank scores are sigmoid-normalized.
- [ADR-0007](../adr/ADR-0007-default-retrieval-mode.md) — why BM25 is still
  the default even though hybrid measures better.
- [ADR-0008](../adr/ADR-0008-dense-search-requires-a-dense-collection.md) —
  why a hybrid search against a vector-less collection is an error rather
  than a silent fallback.

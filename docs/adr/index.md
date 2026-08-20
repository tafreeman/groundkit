# Architecture Decision Records

Convention for this repo: `ADR-NNNN-<slug>.md`, four-digit zero-padded, one
decision per record. ADRs record irreversible decisions with the alternatives
that were considered; deviating from SPEC.md requires an ADR proposal first.

Grouped by theme below so you can find the record that governs the part of
the codebase you're touching, rather than scanning every title in commit order.
Within a group, order is ADR number (roughly chronological).

## Foundational

What groundkit is, what it took from ARP, and how its two durable stores
(SQLite, the eval corpus) are shaped.

| ADR | Title | Status |
| --- | ----- | ------ |
| [ADR-0001](ADR-0001-promote-vs-rewrite.md) | Promote vs rewrite: what groundkit takes from ARP's RAG library | Accepted |
| [ADR-0002](ADR-0002-index-persistence.md) | Index persistence: SQLite truth, BM25 rebuild-at-open | Accepted |
| [ADR-0003](ADR-0003-eval-corpus-and-metrics.md) | Eval corpus and metrics: quote-anchored judgments, hit-rate recall, threshold-free abstention, JSONL | Accepted |

## Dense retrieval & embedding integrity

The vector store, what binds it to an embedding identity, and where a
dense-shaped request is refused outright rather than silently degraded.

| ADR | Title | Status |
| --- | ----- | ------ |
| [ADR-0004](ADR-0004-embedding-identity-binding.md) | Dense store integrity: embedding identity binding and delete-path safety | Accepted |
| [ADR-0006](ADR-0006-dense-seam-returns-chunk-score-pairs.md) | The dense seam returns `(Chunk, score)`, not `RetrievalResult` | Accepted |
| [ADR-0007](ADR-0007-default-retrieval-mode.md) | Hybrid is recommended where configured; BM25 stays the default | Accepted |
| [ADR-0008](ADR-0008-dense-search-requires-a-dense-collection.md) | A dense or hybrid search refuses a collection that has no vectors | Accepted |
| [ADR-0011](ADR-0011-bm25-only-ingest-refuses-a-dense-collection.md) | A BM25-only ingest refuses a collection that has a dense side | Accepted |

## Hybrid scoring & rerank

How BM25 and dense results are combined, and how a cross-encoder reorders
the result.

| ADR | Title | Status |
| --- | ----- | ------ |
| [ADR-0005](ADR-0005-fusion-and-rerank-scoring.md) | Hybrid scoring: rank-based RRF fusion and sigmoid-normalized rerank | Accepted |
| [ADR-0012](ADR-0012-rerank-eval-stage-reorders-upstream-stage.md) | The rerank eval stage reorders the best available upstream stage | Accepted |

## Ingestion & incremental indexing

What counts as "unchanged" on a re-ingest, the boundary between the
load→chunk utility and the actual ingest path, the cached runtime that
tracks whether a collection is stale, and what that cache costs while an
ingest is invalidating it.

| ADR | Title | Status |
| --- | ----- | ------ |
| [ADR-0009](ADR-0009-incremental-skip-key-is-a-processing-fingerprint.md) | The incremental skip key is a processing fingerprint, not a content hash | Accepted |
| [ADR-0010](ADR-0010-ingestion-pipeline-is-not-the-ingest-path.md) | `IngestionPipeline` is a load→chunk utility, not the ingest path | Accepted |
| [ADR-0013](ADR-0013-collection-runtime-persisted-staleness-marker.md) | A cached `CollectionRuntime` whose validity is a marker persisted in SQLite | Accepted |
| [ADR-0026](ADR-0026-measure-the-rebuild-cliff-before-rebuilding-incrementally.md) | Measure the rebuild cliff before rebuilding incrementally, and re-defer ADR-0002's persisted postings against a trigger the measurement can satisfy | Accepted |

## Extracted & remote sources

PDF/HTML content extraction and URL ingestion: how a citation stays
verifiable when the source isn't a local text file, and what happens to a
fetched snapshot over a document's lifetime.

| ADR | Title | Status |
| --- | ----- | ------ |
| [ADR-0016](ADR-0016-citation-verifiability-for-extracted-and-remote-sources.md) | Citation verifiability for extracted and remote sources | Accepted |
| [ADR-0023](ADR-0023-snapshot-lifecycle-is-bound-to-the-document-row.md) | A snapshot exists iff its document row references it; the `Indexer` deletes one at skip/replace/delete, best-effort and never fatal | Accepted |

## Service surface

The read-only REST/MCP surface and why its dependencies are base
requirements rather than an extra.

| ADR | Title | Status |
| --- | ----- | ------ |
| [ADR-0014](ADR-0014-read-only-service-surface-and-outbound-endpoint-safety.md) | Phase 4 is a read-only service surface, and outbound endpoints are guarded | Accepted |
| [ADR-0015](ADR-0015-service-dependencies-are-base-not-an-extra.md) | Service dependencies are base requirements, not an optional extra | Accepted |
| [ADR-0024](ADR-0024-host-header-validation-on-both-transports.md) | The loopback bind is not a boundary against a browser, so `Host` is validated on both transports | Accepted |
| [ADR-0025](ADR-0025-library-constructors-default-host-validation-on.md) | The library constructors default `Host` validation on, not off | Accepted |

## LLM boundary

Where text is allowed to leave the process, how a synthesized answer is
validated rather than trusted, and why synthesis stays off the service
surface.

| ADR | Title | Status |
| --- | ----- | ------ |
| [ADR-0017](ADR-0017-chat-seam-and-redaction-boundary.md) | One narrow chat seam, implemented directly, with redaction wrapped around it | Accepted |
| [ADR-0018](ADR-0018-llm-output-is-validated-never-trusted.md) | LLM output is validated, never trusted: cited synthesis, the echo check, and an advisory judge | Accepted |
| [ADR-0019](ADR-0019-grk-answer-and-no-synthesis-on-the-service-surface.md) | Synthesis lands as `grk answer`, and does not reach the service surface | Accepted |

## Infrastructure & observability

Deployment target, container hardening, and the tracing/logging dependency
shape.

| ADR | Title | Status |
| --- | ----- | ------ |
| [ADR-0020](ADR-0020-terraform-target-single-host-with-block-storage.md) | The Terraform module targets one AWS host with attached block storage, reachable only through SSM | Accepted |
| [ADR-0021](ADR-0021-container-exposure-and-filesystem-hardening.md) | In a container the loopback guarantee moves outward, and the read-only claim splits in two | Accepted |
| [ADR-0022](ADR-0022-observability-dependency-shape-and-span-attribute-allowlist.md) | The OTel API is a base dependency, the SDK is an extra, and span attributes are an allowlist | Accepted |

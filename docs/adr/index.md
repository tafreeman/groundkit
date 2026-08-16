# Architecture Decision Records

Convention for this repo: `ADR-NNNN-<slug>.md`, four-digit zero-padded, one
decision per record. ADRs record irreversible decisions with the alternatives
that were considered; deviating from SPEC.md requires an ADR proposal first.

| ADR | Title | Status |
| --- | ----- | ------ |
| [ADR-0001](ADR-0001-promote-vs-rewrite.md) | Promote vs rewrite: what groundkit takes from ARP's RAG library | Accepted |
| [ADR-0002](ADR-0002-index-persistence.md) | Index persistence: SQLite truth, BM25 rebuild-at-open | Accepted |
| [ADR-0003](ADR-0003-eval-corpus-and-metrics.md) | Eval corpus and metrics: quote-anchored judgments, hit-rate recall, threshold-free abstention, JSONL | Accepted |
| [ADR-0004](ADR-0004-embedding-identity-binding.md) | Dense store integrity: embedding identity binding and delete-path safety | Accepted |
| [ADR-0005](ADR-0005-fusion-and-rerank-scoring.md) | Hybrid scoring: rank-based RRF fusion and sigmoid-normalized rerank | Accepted |
| [ADR-0006](ADR-0006-dense-seam-returns-chunk-score-pairs.md) | The dense seam returns `(Chunk, score)`, not `RetrievalResult` | Accepted |
| [ADR-0007](ADR-0007-default-retrieval-mode.md) | Hybrid is recommended where configured; BM25 stays the default | Accepted |
| [ADR-0008](ADR-0008-dense-search-requires-a-dense-collection.md) | A dense or hybrid search refuses a collection that has no vectors | Accepted |
| [ADR-0009](ADR-0009-incremental-skip-key-is-a-processing-fingerprint.md) | The incremental skip key is a processing fingerprint, not a content hash | Accepted |
| [ADR-0010](ADR-0010-ingestion-pipeline-is-not-the-ingest-path.md) | `IngestionPipeline` is a load→chunk utility, not the ingest path | Accepted |
| [ADR-0011](ADR-0011-bm25-only-ingest-refuses-a-dense-collection.md) | A BM25-only ingest refuses a collection that has a dense side | Accepted |
| [ADR-0012](ADR-0012-rerank-eval-stage-reorders-upstream-stage.md) | The rerank eval stage reorders the best available upstream stage | Accepted |
| [ADR-0013](ADR-0013-collection-runtime-persisted-staleness-marker.md) | A cached `CollectionRuntime` whose validity is a marker persisted in SQLite | Accepted |
| [ADR-0014](ADR-0014-read-only-service-surface-and-outbound-endpoint-safety.md) | Phase 4 is a read-only service surface, and outbound endpoints are guarded | Accepted |
| [ADR-0015](ADR-0015-service-dependencies-are-base-not-an-extra.md) | Service dependencies are base requirements, not an optional extra | Accepted |
| [ADR-0016](ADR-0016-citation-verifiability-for-extracted-and-remote-sources.md) | Citation verifiability for extracted and remote sources | Accepted |
| [ADR-0020](ADR-0020-terraform-target-single-host-with-block-storage.md) | The Terraform module targets one AWS host with attached block storage, reachable only through SSM | Accepted |
| [ADR-0021](ADR-0021-container-exposure-and-filesystem-hardening.md) | In a container the loopback guarantee moves outward, and the read-only claim splits in two | Accepted |
| [ADR-0022](ADR-0022-observability-dependency-shape-and-span-attribute-allowlist.md) | The OTel API is a base dependency, the SDK is an extra, and span attributes are an allowlist | Accepted |
| [ADR-0017](ADR-0017-chat-seam-and-redaction-boundary.md) | One narrow chat seam, implemented directly, with redaction wrapped around it | Accepted |
| [ADR-0018](ADR-0018-llm-output-is-validated-never-trusted.md) | LLM output is validated, never trusted: cited synthesis, the echo check, and an advisory judge | Accepted |
| [ADR-0019](ADR-0019-grk-answer-and-no-synthesis-on-the-service-surface.md) | Synthesis lands as `grk answer`, and does not reach the service surface | Accepted |

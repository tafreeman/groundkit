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

# Changelog

All notable changes to groundkit are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

No metric value appears in this file. Retrieval-quality numbers come from
generated eval artifacts, never from prose (SPEC.md §2), so a release note
here says which stages a run reports and where the report lands — never what
it scored.

## [Unreleased]

### Security

- `Host` is now validated on both service transports, closing inbound DNS
  rebinding against `grk serve` (ADR-0024, amending ADR-0014 decision 7's threat
  model). The `127.0.0.1` bind was the service's only access control, and it is
  not a boundary against a browser: a page on any site the victim visits can
  re-point its own hostname at `127.0.0.1`, after which the browser treats the
  service as same-origin — no CORS preflight, response fully readable — while
  the connection genuinely arrives from loopback. Both surfaces now enforce one
  allow-list derived at serve time from the bind address: Starlette's
  `TrustedHostMiddleware` on the REST app (which also covers the mounted `/mcp`
  sub-path) and the MCP SDK's `TransportSecuritySettings` on the
  streamable-HTTP transport, which the SDK defaults *off* for backwards
  compatibility and which the lower-level API groundkit uses does not
  auto-enable. The MCP transport additionally refuses any cross-origin
  `Origin`; an absent `Origin`, which is what a non-browser client sends, is
  allowed.
- The allow-list is decided by whether the bind is *routable*, not by whether it
  was typed as a loopback literal, and it names the address actually bound. A
  hostname is resolved at serve time and fails closed when it does not resolve;
  address literals are never resolved. `--allow-remote-access` widens both
  allow-lists together and logs a warning naming the address — once the port is
  routable, `Host` validation stops nobody who could not already connect
  directly, and a restrictive list would break every reverse-proxy and
  overlay-network deployment (ADR-0024 decision 3). The container default
  `--host 0.0.0.0 --allow-remote-access` (ADR-0021) is unaffected.

### Changed

- `grk serve` bounds concurrent corpus-scale work. `search` and `index_status`
  acquire a semaphore admitting `service.tools.MAX_CONCURRENT_CORPUS_SCANS` at
  a time, shared across both transports because they share the handlers. Every
  operation on this unauthenticated surface is O(corpus), so arrival rate alone
  previously decided peak memory — an OOMKill on the single-replica Kubernetes
  deployment was traffic, not corpus growth, while the manifest's own comment
  said otherwise. Waiters queue rather than being shed: a waiter holds a
  connection but not a corpus-scale working set. This is a concurrency bound,
  not rate limiting; `SECURITY.md` states the difference.
- uvicorn is separately started with `limit_concurrency =
  cli.SERVE_MAX_CONNECTIONS`, a generous backstop against connection
  exhaustion. It is named for connections because that is what uvicorn counts —
  server-wide, idle keep-alive included — and because it substitutes a 503 app
  before routing, so a trip answers every route including the Kubernetes
  probes. It cannot express "bound the expensive work", and is no longer
  described as though it could.
- `SQLiteMetadataStore` sets `PRAGMA busy_timeout` explicitly, to
  `metadata.BUSY_TIMEOUT_MS`. The value it had was already five seconds,
  inherited from `sqlite3.connect`'s `timeout=5.0` default rather than chosen;
  nothing named it, stated it, or tested it. Behaviour is unchanged and the
  value now belongs to the module, guarded by a test that forces the connect
  default to 0 and requires the pragma to hold.

### Fixed

- BM25 scoring no longer runs on the event loop. `BM25Index.search` scores the
  whole corpus in pure Python and was called inline from `Retriever.search`'s
  `async def`, so on `grk serve` — one uvicorn worker, one loop — a single
  query stalled every other in-flight request, `index_status` and `fetch_chunk`
  included. Both call sites now dispatch through `asyncio.to_thread`, as does
  `InMemoryVectorStore.search`, which was `async def` with no `await` in its
  body. This moves the stall, not the cost: the scan is still linear in corpus
  size.
- `index_status` no longer reads the entire corpus to count it.
  `CollectionRuntime.chunk_count` materialized every chunk's full text as
  re-validated models to return one integer; it now runs an aggregate through
  `SQLiteMetadataStore.count_chunks`. `MetadataStoreProtocol` is unchanged.
- `rerank_by_logits` now preserves a result's `source_class` and `extractor`
  through cross-encoder reranking instead of silently reverting them to the
  contract defaults (`"text"` / `None`). Previously every reranked citation from
  a URL-ingested (`snapshot`) or extracted (`extracted`) document lost its
  provenance with no error, routing it into the plain read-and-slice citation
  path ADR-0016 exists to keep such content out of. Reachable from every REST
  and MCP search request, which expose `rerank`.
- Caller-supplied `Document.metadata["source"]` no longer silently overwrites
  the authoritative `document.source` in emitted chunk metadata. `RecursiveChunker`
  now seeds `source` *last* in the merge, so it wins a key collision instead of
  losing to it — previously a colliding key made `metadata_filter={"source": ...}`
  on dense search return zero or wrong results with nothing raised.

### Documentation

- `config.IndexConfig.index_dir` and `index.protocols.VectorStoreProtocol` no
  longer claim dense vector storage "arrives in Phase 3"; both now name
  `InMemoryVectorStore` and `LanceDBVectorStore`, which have existed since
  2026-08-15. Both render on the published docs site via mkdocstrings, so the
  site was telling a reader that a shipped capability did not exist.
- `SPEC.md` §9, `README.md`, the docs-site homepage, the installation guide and
  the MCP-clients guide no longer state that the release has not happened.
  `pip install groundkit` is now the primary install path, and the PyPI version,
  Python versions and License badges are present as live endpoints.

## [0.1.0] - 2026-08-18

First release. Everything below is initial, so this entry describes the
surface as shipped rather than a diff against an earlier version.

### Retrieval

- Persisted index, one directory per collection: SQLite as the durable truth
  for documents and chunks, with a LanceDB table for vectors. The BM25 index
  holds no on-disk form of its own and is rebuilt from the persisted chunk set
  at open, so nothing in memory can drift out of sync with what is on disk.
- Character-offset citations on every result. A citation is verified by
  re-reading the span from its source and byte-comparing it, not by asserting
  it — `groundkit.retrieval.verify_citation`.
- `Citation` and `RetrievalResult` reject an inverted span (`end_offset <=
  start_offset`) at construction, matching the ordering check `Chunk` has
  enforced since Phase 1. `RetrievalResult`'s `content`-length arithmetic is
  deliberately not held to the same check as `Chunk`'s, since a downstream
  rendering (e.g. a provenance envelope) may wrap the original span's content
  without changing its offsets.
- `bm25`, `dense`, and `hybrid` retrieval modes; hybrid fuses with reciprocal
  rank. BM25 is the default and stays the default: unlike the dense-side
  modes it abstains when no indexed chunk shares a term with the query.
- Optional local cross-encoder rerank behind the `rerank` extra. Non-LLM,
  and reported in the eval harness as its own stage with a measured delta.
- Incremental ingestion with content-hash dedup; unchanged files are skipped.
- Embedding identity is bound to a collection on its first dense write, and
  verified at both the ingest and retrieval boundaries. A mismatch is a typed
  error — never a re-embed, and never a cross-provider fallback.
- `Document.source_class` and `.extractor` are persisted and propagated onto
  `RetrievalResult` and `Citation`, and resolving a citation dispatches on the
  recorded class. Behind the `pdf` and `html` extras, an `extracted`-class
  citation is re-verified by re-running the same extractor that produced it
  and slicing the result, rather than a plain read-and-slice of the source.
- `grk ingest` accepts an http(s) URL alongside a path. `UrlLoader` fetches it
  behind the same SSRF guard as cloud-provider endpoints — redirects refused
  rather than followed, an oversized body refused rather than truncated,
  private endpoints refused unconditionally — and writes the fetched text to
  a per-collection snapshot file. A `snapshot`-class citation resolves by
  reading that file back, never by re-fetching, because a re-fetch is a
  different observation at a different time and cannot verify anything. A URL
  carrying a credential in its userinfo or in a credential-shaped query
  parameter (`?token=`, `?api-key=`, ...) is refused before the fetch rather
  than stored, since the URL is kept verbatim as `Document.source` and would
  otherwise be re-served in every citation built from it.

### Service surface

- `grk serve` runs the FastAPI REST surface and the MCP streamable-HTTP
  transport over one runtime; `grk serve-mcp` runs the MCP stdio transport for
  Claude Desktop and Claude Code.
- Four read-only operations, generated from one registry so the two transports
  cannot diverge: `search`, `fetch_chunk`, `list_collections`, `index_status`.
- Binds `127.0.0.1` by default and refuses a non-loopback bind without an
  explicit acknowledgement. **There is no authentication of any kind**, so the
  address the service is reachable at is its entire access control.

### LLM boundary

- Optional query rewrite and cited synthesis, both skippable, both behind
  interfaces. No LLM runs anywhere in the retrieval path.
- Synthesis may cite only retrieved spans. An out-of-set citation is an error,
  never repaired. Empty citations are a genuine abstention, not a failure.
- A redaction pass wraps cloud chat egress with no operator opt-out. The
  embedding boundary is a deliberate, documented exception — see
  `docs/architecture/llm-boundary.md`.
- An advisory faithfulness judge (`grk answer --judge`, `grk eval --judge`).
  **Advisory only: it exits 0 and gates nothing** until calibrated against
  human labels.
- Synthesis is CLI-only and deliberately absent from the service surface.

### Eval harness

- `grk eval` scores a committed golden corpus with deterministic, unit-tested
  metrics: recall@k (hit-rate at k=1,5,10), MRR, and nDCG@10, plus per-stage
  latency percentiles.
- BM25-only is always the intra-run baseline, and every later stage reports
  its signed delta against it in the same report — including when it loses.
  Deltas are derived at read time rather than stored, so they cannot disagree
  with their inputs.
- Gold spans resolve against corpus text before any indexing runs, failing
  closed on a missing or ambiguous quote.
- `InMemoryEmbedder` produces hash-derived vectors. It exercises code paths
  offline and is wrong for measuring quality; a report produced with it is
  labelled as such.

### Observability

- OpenTelemetry spans on ingest, retrieve, and synthesize. `opentelemetry-api`
  is a base dependency so instrumentation needs no guarded imports; the SDK
  and OTLP exporters live in the `otel` extra. With neither configured, every
  span is non-recording.
- Span attributes are an allowlist enforced by a typed-keyword helper with no
  `**kwargs`, so query text, document content, citation spans, absolute source
  paths, and metadata-filter values cannot become attributes by construction.
- Structured JSON logs opt-in via `GROUNDKIT_LOG_FORMAT=json`; human-readable
  stays the default for a terminal.

### Infrastructure

- Multi-stage non-root image (uid 10001), ready for a read-only root
  filesystem; a compose stack with Ollama, an OTel collector and Jaeger;
  Kubernetes manifests including a default-deny NetworkPolicy and the one-shot
  ingest Job; and a Terraform module for a single EC2 host with block storage
  and no inbound path, reached over SSM port forwarding.
- Every path has been exercised at least once, with the scope of each run
  recorded in `infra/README.md` — including which runs were narrower than
  their row reads.

### Known gaps in this release

Named here because they are v1 scope that did not ship, not oversights.
`KNOWN_LIMITATIONS.md` is the full and current record.

- **PDF/HTML ingestion is not reachable from `grk ingest`.** The deterministic
  extractors and resolve-time citation re-verification exist behind the `pdf`
  and `html` extras, but the ingest-side loaders need multi-loader dispatch
  that is deliberately undesigned.
- **No HTML eval report.** `grk eval` writes JSON only.
- No authentication, no rate limiting, and no multi-tenancy on the service
  surface.

[Unreleased]: https://github.com/tafreeman/groundkit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/tafreeman/groundkit/releases/tag/v0.1.0

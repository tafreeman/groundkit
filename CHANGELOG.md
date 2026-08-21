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

- The URL loader's snapshot write no longer follows a symbolic link. `ensure_within_base` resolves symlinks, so a snapshot path that is *already* a link out of `<collection>.snapshots/` was refused — but the check and the write are two syscalls, and anything able to create a file in that directory could win the gap between them and have the write land wherever the link pointed. The write is now `O_NOFOLLOW`, and only that errno becomes an `IngestionError`; every other `OSError` propagates unchanged. The window was narrow (`document_id` is an unguessable `uuid4`, and citation resolution returns nothing unless the bytes still match) and the flag costs nothing. One limit is stated rather than implied: Windows has no `O_NOFOLLOW`, so the write is unchanged there. The *read* side carried the same gap until the entry below closed it.
- Citation resolution's snapshot read no longer follows a symbolic link, closing the half of the pair above that the write-side fix left open — and the more exploitable half. `_resolve_snapshot` ran `ensure_within_base` and then `Path.read_text`: two syscalls with nothing between them, so anything able to create a file in `<collection>.snapshots/` in that window had citation resolution read, and return over `fetch_chunk` to an unauthenticated service caller, whatever the link pointed at. The write side could corrupt a file; the read side exfiltrates one. The read is now `O_NOFOLLOW`, refusing with `verdict="unresolvable"`, and only that errno is reinterpreted — every other `OSError` reaches the existing handler unchanged. `O_NOFOLLOW`, `O_BINARY` and the errno set that classifies a symlink refusal moved to `utils/path_safety.py` and are now shared by both sides rather than defined once per module, because two copies of "which errno means the final component was a link" can drift apart. Windows still has no such flag, so the read is unchanged there and its regression test skips.

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

- `src/groundkit/extraction.py` is now inside the SPEC.md §8 coverage core
  subset, and `src/groundkit/snapshots.py` is argued out of it. Both sit at
  the package root where no glob in the table catches them, and neither had a
  note in either direction — an omission the table's own convention forbids as
  firmly as an unargued entry. `extraction.py` is the re-derivation half of
  citation verification: `resolve_citation`'s `extracted` branch decides
  verified-versus-drifted entirely by what the recorded extractor returns on a
  second pass, so gating `retrieval/citations.py` through the glob while
  leaving `extraction.py` out gated the comparison but not the text being
  compared. `snapshots.py` is two one-expression path functions with no I/O and
  no branch, and every step that decides a snapshot citation's verdict lives in
  the already-gated `_resolve_snapshot`; the whole-package gate covers it, as
  with `telemetry.py`. `README.md`'s enumerated subset list follows the table.

- `FaithfulnessJudge` and `FaithfulnessVerdict` moved from `groundkit.evals.judge`
  to `groundkit.providers.judge` (GK-021). They are `ChatProtocol` consumers
  exactly like `Synthesizer` and `QueryRewriter`, and they now sit beside them.
  The move is what stops the eval harness being a runtime dependency:
  `groundkit.answer` — the `grk answer` composition root, no part of the
  harness — imported the eval package to reach the judge, which made
  `groundkit.evals` undroppable from a plain library install and would have
  forced a standing exemption in the structural guard below. Import path only:
  the class, its verdict schema, its prompt template and its advisory, ungated
  semantics are unchanged, and `grk answer --judge` / `grk eval --judge` behave
  identically.
- A second structural guard,
  `tests/test_deterministic_core.py::test_runtime_surface_does_not_import_the_eval_harness`,
  AST-scans every module under `src/groundkit` for an import of
  `groundkit.evals` and fails on any hit. `cli.py` is the single exemption and
  the only defensible one, since it hosts `grk eval`. The eval harness ships in
  the wheel today, so this guards the *option* of making it an extra rather than
  a property already relied on. The deterministic-core scan in the same module
  now bars `groundkit.providers.judge` by name — it used to bar the judge for
  free via the `groundkit.evals` prefix, and stopped the moment the judge became
  a provider.
- `groundkit.answer` no longer claims, unqualified, that "every collaborator is
  injected". Every collaborator *instance* is injected; the *types* are
  concrete — only the search collaborator is a structural Protocol, while the
  synthesizer, rewriter and judge are the named classes the module imports. The
  claim is narrowed rather than made true by adding indirection, because all
  three are themselves `ChatProtocol` consumers and the substitution a caller
  actually wants (a different model, a scripted fake, a redacting wrapper) is
  reached through that seam.

- `BM25Index` builds the postings list ADR-0002 said it built. `index_chunks`
  now records a `term -> [chunk index]` map alongside the document-frequency
  counter, and `search` scores the union of the query terms' postings instead
  of every indexed chunk, so query cost tracks the number of chunks holding a
  query term rather than corpus size. The result list is unchanged —
  identical, not approximately so: a chunk holding no query term already
  scored exactly zero and was already discarded by the existing filter, and
  the candidate union is walked in ascending chunk index, so even the
  insertion-order fallback between two byte-identical chunks is preserved.
  Nothing is persisted and nothing shadows SQLite, so the O(corpus)
  rebuild-at-open cost is untouched and ADR-0002's deferred "persisted BM25
  postings tables in SQLite" alternative stays deferred — this changes what a
  query costs, not what an open costs. Recorded as an erratum on ADR-0002
  decision 2, which named three structures where the code had two. Pinned by
  `tests/test_bm25_postings.py`, which compares `search` against a
  transcription of the pre-change whole-corpus loop over the golden corpus —
  byte for byte, chunk identity and exact float scores, ties included — for
  every query in `evals/judgments.jsonl`, and which sabotages one posting to
  show that comparison can fail.

- The read paths that needed at most `top_k` document IDs now ask for them by
  key instead of materializing the `documents` table. `Retriever.search` read
  every stored row — one validated model per row — on **every** query, before
  the mode branch even ran; `fetch_chunk` did the same for the single document
  its already-keyed chunk read had just named, and did it before establishing
  the chunk existed at all. Both now use a new keyed
  `get_document_record(document_id)` on the optional
  `DocumentRecordStoreProtocol`, which also gains the `COUNT(*)` aggregates
  that previously existed only on the concrete SQLite store. Query cost is now
  proportional to the results rather than to the corpus.
- Two consequences worth naming, both pinned by tests asserting on the SQL the
  store executes rather than on which method was called: a search whose hits
  all fall below `score_threshold` now performs no document read at all, since
  the join happens per surviving hit after the filter; and a `fetch_chunk`
  naming an unknown chunk id — the cheapest possible rejection on an
  unauthenticated read-only surface — no longer costs a corpus-proportional
  scan.
- The document read stays **live** per search and is deliberately not cached at
  `open()`. That is what makes a hit against a document deleted after `open()`
  fail closed instead of resolving to a citation nothing can verify, so the
  cheaper fix was the wrong one. Within a single search the lookup memoizes per
  distinct document ID, which is no less live than the single whole-table
  snapshot it replaces. A store that does not implement the optional capability
  keeps working through the previous whole-table path, now taken lazily.
- Every hand-built metadata-store double in the suite derives from one shared
  base (`tests/metadata_store_doubles.py`), itself checked against both store
  protocols by `assert_signature_parity`. Test-maintenance cost was the
  recorded reason for keeping capabilities off the required protocol and for
  the `isinstance` fork in the retrieval hot path; widening a protocol is now
  one edit in one file rather than one per double.

- `grk search` and `grk answer` open their collection through `CollectionRuntime`
  instead of calling `Retriever.open` themselves, carrying out ADR-0013 decision 7,
  which had been recorded and not implemented. Four read lifecycles become two — the
  runtime's and the eval runner's, whose exemption that decision already argued.
  Behaviour is unchanged for every current collection: the runtime's cache cannot hit
  in a one-shot process, so it is still one open, one rebuild, one close. What changes
  is that the CLI and the service surface now share that code, so the default test
  suite exercises the open path a long-lived server runs on — the benefit the ADR
  claimed and did not deliver. Two visible edges: a collection predating ADR-0013's
  schema now emits the runtime's "freshness cannot be asserted" warning on stderr
  (never stdout, so `--json` is unaffected), and the dense vector store is opened by a
  factory during the runtime's rebuild rather than eagerly by the command — which makes
  structural an ordering the previous code held by convention, since a collection name
  is validated by `SQLiteMetadataStore.open` before `<index-dir>/<collection>.lance`
  can be derived from it.

- `UrlLoader` bounds a fetch in wall-clock time. `timeout_seconds` (default 30 seconds, the same default and the same strictly-positive invariant as `EmbeddingConfig.timeout_seconds`) is enforced with `asyncio.timeout` around the whole exchange — connect, status, headers and every body read together. httpx's own `timeout=` is per operation, so a server that answers just inside the read timeout, indefinitely, never trips it: the bound the caller thought it set was not the bound it got. The bound covers the fetch only, not the preceding address check or the snapshot write.
- `IngestionPipeline.ingest_directory` no longer returns while work it started is still running. A file that fails still raises, still deterministically (path-sorted order, first error in argument order), but only once every dispatched file has settled. `asyncio.gather` without `return_exceptions=True` hands the first exception to the awaiting frame the moment it is raised and neither cancels nor awaits the siblings already in flight; `IngestionPipeline` is exported public API, so a third-party host that caught the error inherited loads and chunkings it could not see, await, or cancel. There is no torn write here — this module never touches a collection (ADR-0010) — so the fix is about supervision, not durability, unlike the identically-shaped one in `indexer.py`.

### Fixed

- A snapshot whose content has CRLF line endings now resolves to the span that was actually indexed. `UrlLoader` decodes the response body and writes it byte-for-byte, so a source served with CRLF keeps it in both `Document.content` and the snapshot on disk — but the read used `Path.read_text`, whose default is universal-newline mode, turning every CRLF back into a bare LF. The text came back one character shorter per line break, so every offset past the first one was wrong: `fetch_chunk` returned a shifted span, or reported `drifted` on a snapshot that had not drifted. The read now decodes the bytes itself. Found while closing the symlink gap above, not reported separately — the two share the one line of code that changed.

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

- Neither of `UrlLoader`'s content-sized writes runs on the event loop. The
  snapshot write at the end of `load` and the scratch write inside
  `_extract_html` each handle up to `max_bytes` of fetched body, and were the
  only content-sized filesystem I/O left in the package called inline from an
  `async def`; both now dispatch through `asyncio.to_thread`, as every
  comparable site already did. This was bounded in practice only because URL
  ingestion is CLI-only and nothing else is scheduled on that loop — a property
  of the current caller, not of this module, which is exactly the code a
  service-side ingest tool would reuse verbatim. `_write_snapshot` stays
  synchronous deliberately: the caller dispatches it, and tests keep calling it
  directly to exercise the containment check without a fetch.
- `run_eval` computes `judgments_hash` off the event loop, as it already
  computed `corpus_hash` one statement earlier. The comment above the pair
  claimed a symmetry ("both reads") that was true of their `OSError` handling
  and false of their dispatch — the judgments file was read whole and hashed on
  the loop. Both are now dispatched, and the comment says which half of "both"
  it means.

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

- `scripts/measure_retriever_open.py` now prices the whole query path, not just
  `Retriever.open()`. Four selectable sections (`--sections`): the existing open
  timings; `BM25Index.search` and `Retriever.search` per query, each reported
  beside the measured candidate fraction that bounds what a postings list could
  skip; the full-table document read every search performs, against a keyed
  single-row read of the same table and the rows-materialized-versus-`top_k`
  amplification; and a warm `CollectionRuntime.acquire()` against a rebuild,
  followed by an ingest window timing the same commits with and without a
  concurrent acquire loop. Queries are derived from the corpus's own measured
  term frequencies rather than written in, so none can quietly stop being rare
  or common when the generator changes. Everything runs offline with no
  credentials. This closes the mode ADR-0013 recorded the script as owed — "a
  mode that times a warm acquire against a rebuild so this claim is measured
  rather than asserted" — and supplies the measurement ADR-0002 names as the
  trigger for reconsidering persisted postings. As before, no number it produces
  is restated in any doc.

- ADR-0026 records why measurement precedes the incremental-rebuild work and
  re-defers ADR-0002's persisted-BM25-postings alternative against a **new**
  trigger. The old trigger — "rebuild-at-open time is *measured* to be a problem
  for a real corpus size" — was half-unmeetable rather than merely unmet:
  `scripts/measure_retriever_open.py` settled the per-open cost for ADR-0013, but
  whether that cost is a *problem* depends on how often an open happens, and no
  instrument in the repo could observe that. The new trigger asks for a recorded
  reading of the counters above from a real corpus, quoted in the ADR that acts
  on it, and states which of the two competing remedies each shape of reading
  selects. ADR-0013 is unmodified: its per-commit bump, stamp ordering,
  single-flight and wait-rather-than-serve-stale rules all stand as written.

### Added

- `index_status` now reports what the ADR-0013 staleness cache actually *does*,
  not only that it is switched on: `retriever_acquires`, `retriever_rebuilds`,
  `rebuild_seconds_total` and `last_rebuild_seconds` (ADR-0026). `cache_enabled:
  true` was equally compatible with a cache hitting on every request and with one
  that had not hit since the process started, and the only external symptom of
  the difference was latency — which on this path has several other causes (an
  O(corpus) aggregate, a cold page cache, a dense probe, a reranker). The hit
  fraction is derived by the reader as `1 - retriever_rebuilds /
  retriever_acquires` and is deliberately not stored, following the rule the eval
  harness applies to a stage delta: a ratio kept beside its own inputs is
  redundant state that can disagree with them. Nothing new is disclosed — a
  rebuild happens exactly when the generation moved, and `generation` was already
  on this response.
- `CollectionRuntime.rebuild_stats()` returns those counters as a frozen
  `RebuildStats` snapshot. Rebuilds are counted at entry rather than at
  publication and timed in a `finally`, so a rebuild that raises is still charged
  for the lock it held and the waiters it blocked; the natural implementation —
  incrementing beside the `self._cached` assignment — reports a runtime whose
  every rebuild fails as one that never rebuilds at all, with a perfect hit rate,
  while it takes the rebuild lock and does an O(corpus) read on every request.
  `handle_index_status` builds no retriever, so reading the meter cannot move it,
  and a test asserts two consecutive calls leave `retriever_acquires` unchanged.

### Removed

- `requirements-audit.txt` is no longer committed; it is generated and gitignored. `ci.yml`'s `audit` job and `release-gates.yml`'s supply-chain step both re-export it from `uv.lock` immediately before `pip-audit` reads it, so the tracked copy was never the file audited — a second rendering of the lockfile that nothing read and that had already drifted from it once. `uv.lock` remains the tracked source of truth.
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

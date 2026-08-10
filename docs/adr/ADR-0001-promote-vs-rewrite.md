# ADR-0001 — Promote vs rewrite: what groundkit takes from ARP's RAG library

- **Status:** Proposed (awaiting owner review before Phase 1)
- **Date:** 2026-08-10
- **Deciders:** Andy Freeman (owner) · Phase-0 inventory: 9-agent fan-out over
  `agentic-runtime-platform` (ARP) at branch `relock_scope`, 2026-08-10

## Context

groundkit exists to close the verified production gaps of the RAG library
inside ARP (`agentic-workflows-v2/agentic_v2/rag/`, 17 modules, ~3.9k LOC).
The library is well-typed, Protocol-seamed, and unusually well-tested in
places — but it is an *unwired subsystem*: a persistence-capable LanceDB store
exists and passes tests, yet zero non-test callers reference it; the only live
entry point (the `agentic rag` CLI) hardcodes in-memory components in
module-level globals that reset every process.

Every module was read in full by an assigned inventory agent; every claimed
gap was adversarially audited against the whole repo. This ADR records the
per-module port decision. "Promote" means the code is copied into groundkit
nearly verbatim (groundkit never imports `agentic_v2`); "adapt" means copied
with named, targeted changes; "rewrite" means new implementation informed by
the original; "drop" means not carried into v1.

## Decision drivers

1. groundkit is standalone: no `agentic_v2` import, ever (the only outside-rag
   couplings found are `AgenticError` — a 2-line base class — `utils/path_safety`,
   `tools/base.BaseTool`, and `integrations/{base,tracing}`; all cheap to sever).
2. Local-first: Ollama by default, cloud opt-in behind a redaction boundary.
3. Fail closed: no silent fallback, no coerced output, no cross-provider
   semantic-space mixing.
4. Persistence and metadata filtering are v1 requirements, not extensions.
5. Known defects are fixed at port time, never copied.

## Verified starting point (gap audit — all 8 claims confirmed)

| # | Claimed gap | Verdict | Key evidence |
|---|---|---|---|
| 1 | No persistence across processes | **Confirmed** | CLI holds RAG state in module globals (`cli/helpers.py:283-284`) and hardcodes `InMemoryEmbedder`/`InMemoryVectorStore` (`:301-337`). `LanceDBVectorStore` persistence exists and is tested (`test_vectorstore_lancedb.py:124`) but has zero non-test callers — capability defined, never wired. |
| 2 | No directory-scale ingestion | **Confirmed** | Both loaders raise `IngestionError("Not a file")` on directories (`loaders.py:63-64,124-125`); no directory-walking code exists; CLI help text advertising `--source ./docs` is aspirational (`cli/rag_commands.py:51,66`). |
| 3 | LanceDB metadata filtering unimplemented | **Confirmed** | `search`'s third param is `_metadata_filter`, never read in the body (`vectorstore.py:323-367`); keyword call raises `TypeError`, positional call is silently dropped (`factory.py:196-200`). |
| 4 | Protocol naming mismatches | **Confirmed** | `NoOpReranker.rerank(_query, …)` vs protocol's `query` (`reranking.py:34` vs `protocols.py:129`); `_metadata_filter` vs `metadata_filter` (above). Neither covered by `test_protocol_conformance.py`. |
| 5 | No retrieval-quality eval metrics | **Confirmed** | ADR-035 self-admits it (`:402-407`); repo-wide grep for recall@/MRR/nDCG finds only prose. The eval package's one RAG rubric is an LLM judge grading answer behavior, not ranking quality. |
| 6 | No retrieval service API | **Confirmed** | Server mounts nine routers, none RAG-related (`server/app.py:158-168`). |
| 7 | No MCP server surface | **Confirmed** | ARP ships an MCP *client* stack only; zero `mcp.server` implementations repo-wide. |
| 8 | No retrieval IaC | **Confirmed** | compose/Dockerfile/Bicep all target the workflow platform; zero rag/lancedb/retrieval hits in any infra file; no k8s or Terraform exist at all. |

## Per-module decisions

| ARP module (LOC) | Decision | groundkit destination | Required changes at port time |
|---|---|---|---|
| `contracts.py` (103) | **Promote** | `contracts.py` | Replace `is_high_confidence`'s hardcoded `0.7` with a named constant/config. Keep frozen + `extra="forbid"` everywhere. |
| `protocols.py` (147) | **Promote** | per-package protocols | Port verbatim; add the missing conformance tests so implementations can never drift from the protocol again (gap #4 root cause). |
| `ingestion.py` (86) | **Promote** | `ingestion/` | Sever `AgenticError` ancestry (local error root). Add a directory/batch entry point (`ingest_many`, bounded concurrency) as new code — single-source `ingest()` alone cannot meet the directory-scale requirement. |
| `retrieval.py` (332) | **Promote** | `retrieval/` (+ `index/bm25.py`) | Port BM25/RRF/HybridRetriever near-verbatim (zero deps, well-tested). New code: persist the BM25 index — ARP's is in-memory only. Fix the silent no-op `score_threshold=0.0` semantics or document them. |
| `context_assembly.py` (239) | **Promote** | `providers/` (Phase 5) | Port near-verbatim — the `<retrieved_context>` framing/sanitization is verified by 16 tests and matches groundkit's trust-boundary requirement. |
| `memory.py` (195) | **Drop (v1)** | — | Clean code, but a semantic KV memory store is an ARP-agent feature outside groundkit's v1 scope. Its `_key_map` is also non-durable even over a persistent store — the exact class of bug groundkit exists to kill. Revisit post-v1 if a use case appears. |
| `chunking.py` (178) | **Adapt** | `ingestion/chunking.py` | **Fix the reproduced infinite loop** in `_hard_split` (`chunking.py:165-177`): separator-free text (base64, long URLs, minified code) with `chunk_overlap > 0` (default 64) hangs ingestion. Make `start` monotonic; add the regression test ARP never had. Do not implement the `semantic` strategy Literal until a chunker exists (config/implementation trap in ARP). |
| `loaders.py` (145) | **Adapt** | `ingestion/loaders.py` | Copy `utils/path_safety.py` wholesale (65-line stdlib module, zero coupling) and keep `allowed_base_dir` containment. Collapse the ~90%-duplicated Markdown/Text loaders into one parametrized class before adding PDF/HTML. Fix empty-file logging inconsistency; drop dead `**kwargs`; add a size cap. Add the path-traversal test ARP never wrote. |
| `config.py` (112) | **Adapt** | `config.py` | Keep frozen/`extra="forbid"` + real-invariant validators (overlap < size; `db_path` required for lancedb). Reshape enums/defaults: Ollama-first, hybrid params (RRF k, BM25 k1/b), configurable Ollama endpoint (frozen out in ARP). Unknown keys remain startup failures. |
| `errors.py` (34) | **Adapt** | `errors.py` | Local `GroundkitError(Exception)` root — no ported hierarchy. Keep the taxonomy; wire the never-raised `ChunkingError`/`RetrievalError` to real raise sites (e.g. the fixed hard-split guard) or drop them. |
| `vectorstore.py` (406) | **Adapt** | `index/dense.py` (+ test double) | Port `InMemoryVectorStore` as the dev/test double. Port `LanceDBVectorStore`'s shape (cosine, `asyncio.to_thread`, content-hash dedup, tested persistence) but: rename `_metadata_filter` → `metadata_filter` and **implement filtering**; **parameterize/escape the delete predicate** (currently an unescaped f-string); add the filter regression tests ARP lacks. |
| `reranking.py` (203) | **Adapt** | `retrieval/rerank.py` | Fix `NoOpReranker._query` → `query`. **Normalize cross-encoder scores (sigmoid) before constructing results** — raw MS-MARCO logits are commonly negative and violate the `ge=0.0` score contract: a crash on real models, untested in ARP. Declare sentence-transformers as a proper optional extra. Drop `LLMReranker` — groundkit's rerank is non-LLM local-model only, per spec. |
| `embeddings.py` (724) | **Adapt (patterns), rewrite (providers)** | `providers/embeddings.py` | Keep the verified patterns: lazy-import seam ("this is the seam tests replace"), typed fail-loud errors, dimension checks, credential scrubbing — and **also scrub `__cause__`** (KNOWN_LIMITATIONS §4.10 leak). Replace LiteLLM with direct Ollama + OpenAI-compatible HTTP implementations per spec (local-first, fewer deps). **No cross-provider fallback chain** — mixed semantic spaces (§4.7) are the opposite of fail-closed; an unconfigured/failed provider is a typed error. Keep `InMemoryEmbedder` strictly as a labeled test double (§4.9: hash expansion, zero semantic signal — no factory path may return it in production). |
| `factory.py` (312) | **Adapt** | `factory.py` | Keep the builder shape and the fake-injection test strategy. Delete the Voyage→OpenAI→local fallback chain and both signature workarounds (positional-only reranker call, `type: ignore[return-value]`) — the underlying bugs are fixed instead. |
| `tools.py` (302) | **Adapt** | `service/mcp_server.py` | The execute() orchestration (validate → retrieve → trace → assemble → serialize; and the ingest sequence) plus validation bounds (`_MAX_TOP_K=50`, `_MAX_QUERY_LEN=4096`) port well. Drop `BaseTool` inheritance entirely (drags in ARP's approval-gate/engine machinery); re-express as MCP SDK tool handlers with Pydantic input models. Add tests for the validation branches and failure paths ARP left uncovered. |
| `tracing.py` (245) | **Adapt** | `observability.py` | Keep `RAGTracer`'s emit/span API shape (`perf_counter_ns`, injected adapter). Replace ARP's adapter zoo with a minimal local protocol — and emit real OTel spans (ARP's ADR-035 admits it never does). Fix the asymmetry: `ingest_span` gets the same error-capture behavior as `query_span`. |
| `__init__.py` (123) | **Rewrite** | `__init__.py` | ARP's facade re-exports 15 modules including agent-runtime glue. Write fresh, exporting only what groundkit actually ships. |

## Port-time hazard list (defects that must not survive the port)

1. **Infinite loop** — `RecursiveChunker._hard_split`, reproduced by execution,
   on separator-free text with overlap > 0 (`chunking.py:165-177`). Untested in ARP.
2. **Negative-score crash** — `CrossEncoderReranker` feeds raw logits into a
   `Field(ge=0.0)` contract (`reranking.py:119-128` vs `contracts.py:75`).
3. **Silent filter drop / TypeError** — `_metadata_filter` (`vectorstore.py:323-329`).
4. **Keyword-call TypeError on default reranker** — `_query` (`reranking.py:34`).
5. **Unescaped delete predicate** — document IDs interpolated into LanceDB's
   SQL-like delete expression.
6. **Credential leak via `__cause__`** — scrubbed message, unscrubbed chained
   exception (`embeddings.py:719-724`; KNOWN_LIMITATIONS §4.10).
7. **Non-durable key map over a durable store** — `memory.py` `_key_map`
   (dropped from v1, but the pattern is banned generally: no in-process state
   may shadow persisted state).

## Patterns deliberately carried forward

LanceDB choice (with ARP's evaluation rubric); three-stage hybrid retrieval
(dense + BM25 concurrently, RRF k=60, optional cross-encoder); frozen Pydantic
contracts at every boundary; genuinely-optional heavy deps via lazy-import
seams; fail-loud dimension/provider checks; call-time-only credential reads
with scrubbing; `<retrieved_context>` defensive framing (documented as a
visible boundary, not a proof); `allowed_base_dir` containment; Protocol seams
per component; fake-injection over `importorskip` so optional-path code is
executed in CI, not skipped. ARP's `rag-extra-tests` lesson is adopted
inverted: no `continue-on-error` on any job that is the sole proof of a
backend — such jobs are blocking from day one.

## Alternatives considered

- **Depend on ARP as a library.** Rejected: pulls the agent runtime into a
  standalone service; the RAG surface is unexported/unwired in ARP; portfolio
  composition rules make groundkit a peer, not a consumer.
- **Fork the ARP repo.** Rejected: carries 17 modules plus platform history
  when v1 needs roughly half, and inherits the defect list above wholesale.
- **Clean-room rewrite.** Rejected: the contracts, protocols, BM25/RRF core,
  framing/sanitization and embedding seams are verified by ~5.3k lines of
  behavioral tests; rewriting them re-derives solved problems and shed
  test evidence for no benefit.

## Consequences

Positive: v1 starts from tested code for its deterministic core; every known
ARP defect has a named fix and a regression test obligation; groundkit carries
no `agentic_v2` coupling. Negative: ported code must be re-reviewed under
groundkit's stricter gates (mypy --strict repo-wide, no `continue-on-error`);
divergence from ARP is permanent — fixes here do not flow back automatically
(out of scope by design; ARP may cherry-pick).

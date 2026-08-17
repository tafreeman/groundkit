# groundkit — SPEC v1

Living spec. No feature code lands before its section exists here. Deviations
require an ADR proposal first ([ADR index](docs/adr/index.md)). Phase status
lives in §9; sessions start by reading this file, the ADR index, and open
phase status.

## 1. Mission

A standalone, production-grade retrieval service:

1. **Hybrid retrieval** — BM25 + dense embeddings + reciprocal-rank fusion +
   optional local cross-encoder rerank — over a **persisted index** that
   survives restarts.
2. A **named MCP server** (stdio + streamable HTTP): `search`, `fetch_chunk`,
   `list_collections`, `index_status`. Real and installable, not a demo.
3. A **retrieval eval harness**: labeled golden corpus; recall@k (k=1,5,10),
   MRR, nDCG@10; runnable offline in CI.
4. **Real IaC**: Dockerfile, compose, Kubernetes manifests, Terraform module —
   each verified to work, with the verification date recorded.
5. **Fully local by default**: Ollama embeddings, file-based index; cloud
   providers opt-in; air-gap friendly.

Provenance: standalone successor to ARP's `agentic_v2.rag`;
[ADR-0001](docs/adr/ADR-0001-promote-vs-rewrite.md) records the per-module
promote/adapt/rewrite/drop decision, the confirmed gap audit, and the
port-time hazard list. The hazards in ADR-0001 are spec obligations: each
named defect gets a fix and a regression test in the phase that ports it.

## 2. Non-negotiable architecture principles

- **Deterministic core, LLM at the boundary.** Ingestion, chunking, indexing,
  scoring, fusion, and citation resolution are pure, typed, fully-tested code.
  No LLM in the retrieval path. LLM use is confined to optional query rewrite
  and optional synthesis, both behind interfaces, both skippable.
- **Citations are verifiable.** Every returned passage carries document ID,
  chunk ID, and character offsets resolvable to source. Synthesis may cite
  only retrieved spans; the eval harness includes a planted-marker check for
  citation echo.
- **Anonymization at the LLM boundary.** A redaction pass runs before any text
  leaves the process for a cloud provider: names → tokens, with configurable
  patterns. Local mode sends nothing anywhere. The boundary is documented
  explicitly (architecture docs, Phase 7) — where text can leave the process,
  what is redacted, and what is not.
- **Fail closed.** Unconfigured provider → typed error. Malformed structured
  output → schema rejection, never coercion. Unknown config keys → startup
  failure. No cross-provider embedding fallback, ever: mixed semantic spaces
  corrupt an index silently (ARP KNOWN_LIMITATIONS §4.7 is the cautionary
  precedent).
- **Real data only.** No hardcoded metric numbers in docs. Numbers come from
  generated eval artifacts or dynamic badges, or are omitted.
- **No in-process state shadowing persisted state.** Anything the index knows
  must be recoverable from disk (ADR-0001 hazard 7).

## 3. Repo identity & conventions

- Python 3.11+; `pyproject.toml` single config source; `uv` for env/lock.
- Gates: `ruff check`, `ruff format --check`, `mypy --strict`, pytest; ruff
  line-length 100; CI additionally runs pip-audit and gitleaks. No CI job that
  is the sole proof of a backend may be `continue-on-error`.
- Layout: `src/groundkit/` package · `tests/` mirroring modules · `docs/`
  (MkDocs, strict) · `docs/adr/` (`ADR-NNNN-<slug>.md`) · `docs/specs/`
  (feature specs as they arrive) · `evals/` (corpus data + `judgments.jsonl`
  + generated `results/`, gitignored — the runner *code* is the
  `groundkit.evals` package under `src/groundkit/`, §5.1, invoked via
  `grk eval`) · `infra/` (created in Phase 6, not before — empty IaC dirs are
  decoration).
- Storage: LanceDB (dense vectors, file-based) + SQLite (document/chunk
  metadata), both behind interfaces so pgvector can be added later; pgvector
  is not built in v1.
- Embeddings: provider interface with **direct** Ollama (default) and
  OpenAI-compatible implementations — no LiteLLM (ADR-0001: patterns ported,
  provider layer rewritten). BM25: pure-Python, ported from ARP's tested
  implementation — decided in ADR-0001 (vs `rank-bm25`); the Phase 1 ADR
  covers the persistence design around it only.
- Observability: OpenTelemetry spans on ingest/retrieve/synthesize (ARP otel
  collector-config conventions); structured JSON logs with request ID,
  latency, result counts, typed failure codes. Never log document content or
  queries at info level.
- Portfolio composition (documented in README): may consume `executionkit`
  for LLM call patterns at the synthesis boundary; gradable by
  `agentic-evalkit` via its HTTP/MCP `ExecutionTarget` boundary; imports the
  internals of neither.
- Docs site: MkDocs → GitHub Pages. Tokens: ember `#d97757`, accent-2
  `#e8a285`, bg `#08080c`, cream `#ececde`, success `#4ade80`; JetBrains Mono
  + DM Serif Display.

## 4. Scope (v1)

**In:** file/directory/URL ingestion (md, txt, pdf, html); configurable
chunking (size/overlap + structure-aware markdown); hybrid retrieval with RRF;
optional non-LLM local cross-encoder rerank; persisted index with incremental
re-index; MCP server (four tools above); FastAPI REST mirroring the MCP tools;
eval harness; CLI (`grk ingest | search | eval | serve | serve-mcp`); IaC.

**Out (recorded in KNOWN_LIMITATIONS.md, not built):** multi-tenant auth,
distributed indexing, fine-tuning, agent loops, UI beyond the docs site,
GraphRAG, additional vector DBs, semantic KV memory store (ARP's `memory.py`,
dropped per ADR-0001), LLM-based reranking.

## 5. Architecture

### 5.1 Components

```
ingestion/          loaders (path-contained), chunking (offset-preserving)
index/              bm25 (persisted), dense (LanceDB), metadata (SQLite)
retrieval/          search orchestration, RRF fusion, rerank, citations
providers/          embeddings (Ollama | OpenAI-compat), redaction,
                    optional query-rewrite/synthesis (Phase 5)
service/            FastAPI api, MCP server
evals/ (package)    deterministic metrics engine
cli                 grk
```

Component boundaries are `typing.Protocol` seams (ported from ARP), each with
a conformance test — the missing-conformance-test root cause of ARP's
signature drift (ADR-0001 hazards 3–4) is closed structurally.

### 5.2 Data model (contracts)

Frozen Pydantic v2 models, `extra="forbid"`: `Document`, `Chunk` (with
`content_hash`, `document_id`, `chunk_index`, char offsets `start`/`end`),
`RetrievalResult` (score ≥ 0 with normalization guaranteed by producers),
`Citation` (document ID + chunk ID + offsets). Confidence thresholds are named
constants, not inline literals.

### 5.3 Persistence layout

One index directory per collection: LanceDB table (vectors + chunk refs),
SQLite (documents, chunks, offsets, ingest state for incremental re-index,
BM25 statistics or serialized index — design decision recorded in the Phase 1
ADR). Content-hash dedup on write. Metadata filtering is implemented on both
in-memory and LanceDB paths from the first dense-store commit, with regression
tests on both paths.

## 6. Eval harness

- Golden corpus: ≥ 8 documents, ≥ 40 labeled query→relevant-chunk judgments,
  committed. Includes ambiguous, no-answer, and adversarial cases
  (prompt-injection text in documents must never surface as instructions).
- Metrics: recall@k (1,5,10), MRR, nDCG@10 — deterministic code with its own
  unit tests. Latency percentiles per stage (BM25 / dense / fusion / rerank).
- Corpus-integrity tests in normal no-network CI: schema validity, unique IDs,
  category coverage, size floor asserted **in the test** (the test is the
  authoritative number, not the README).
- Faithfulness judge (synthesis mode): LLM-as-judge, schema-validated verdict,
  injectable model call so unit tests never touch the network. **Advisory
  only — exits 0, gates nothing** — until calibrated against human labels;
  the calibration procedure required to ever make it gate is documented.
- Baseline discipline: BM25-only is the baseline. Every retrieval feature
  reports its delta vs baseline in the generated report; a feature that does
  not beat baseline is reported as such.
- Results: `evals/results/latest.json` (gitignored) + generated HTML report.
  A gated workflow (`EVAL_GATED=1`) runs real-model paths on schedule/label
  and skips cleanly when unconfigured.

## 7. Security & hygiene

- `.env.example` only; pre-commit runs ruff, mypy, gitleaks.
- SSRF guard on URL ingestion and on cloud-provider endpoint URLs
  (loopback/private/link-local, incl. IPv6-mapped spellings) with
  `redirect: error` semantics outbound. The local Ollama endpoint is the one
  named, deliberate exception — it is loopback by design; the guard scopes
  around it, not over it.
- pip-audit in CI; direct deps pinned with bounds; uv lockfile parity check.
- Mutating REST routes **and mutating MCP operations**: shared-secret header,
  constant-time compare, disabled when the secret is unset; binds 127.0.0.1 by
  default. The rule is per *operation*, not per transport — an MCP tool that
  ingests, deletes, or re-indexes is the same mutation a REST route would be,
  and Phase 4 ships both surfaces over one runtime. Naming only REST here was
  an omission, not a decision to leave MCP open.
- SQLite is content-bearing data, not just an index. A collection store holds
  document text, so deletion behavior, file permissions, backup scope, and
  retention are product decisions owed before any deployment that is not a
  single user's local machine — not operational details to settle afterwards.
  Deleting a collection means deleting its `.sqlite3` *and* its `.lance`
  sibling; neither is inferable from the other's absence.
- Loader path containment via ported `path_safety` (`allowed_base_dir`), with
  the traversal test ARP never wrote.
- Credential scrubbing covers exception messages **and** `__cause__` chains
  (ADR-0001 hazard 6).
- SECURITY.md with an honest operational-scope statement.

## 8. Engineering process

Spec-driven (this file); ADRs for irreversible decisions with alternatives.
Harness before features: eval harness + golden corpus land immediately after
the deterministic core skeleton, before hybrid/rerank/synthesis; every
subsequent feature lands with its eval delta. Tests are the gate: coverage
floor 80% on the core subset (retrieval, chunking, scoring, citation
resolution — subset defined in `pyproject.toml` when Phase 1 lands, optional
providers excluded, README states the subset honestly). `mypy --strict`
clean. A failing gate blocks the phase. Conventional commits, small
changesets; each phase ends with CI green and docs updated in the same change.

**A regression test is not accepted on the strength of passing.** Every test
written to close a named defect — an ADR-0001 hazard, a review finding, a bug
fix — must be run against the unfixed code and observed to *fail*, then run
against the fix and observed to pass. Both directions get reported. A test
that passes before the fix is testing something other than the defect, and a
green suite is evidence of nothing until that has been ruled out. This matters
most for exactly the defects this repo keeps finding: crash windows,
cancellation paths, and cross-run reproducibility are unreachable on a
non-failing path, so they are invisible to coverage and to a passing suite
alike. Revert the source (not the test), run, restore, run.

## 9. Phase plan & status

| Phase | Deliverable | "Done" means | Status |
|---|---|---|---|
| 0 | Inventory & spec | ADR-0001 + SPEC v1 + skeleton, gates green, owner review | done 2026-08-10 |
| 1 | Deterministic core | ingest→chunk→BM25→embedding interface→persisted index→citation-resolving retrieval; unit tests; coverage gate on; chunker loop + loader fixes in with regression tests | done 2026-08-10 |
| 2 | Eval harness | golden corpus + metrics engine + BM25 baseline report as reference artifact | done 2026-08-11 |
| 3 | Hybrid + rerank | dense (LanceDB w/ metadata filtering), RRF, optional cross-encoder (normalized scores); each with eval delta vs baseline | done 2026-08-15 |
| 4 | Service + MCP | FastAPI + MCP server + CLI; `grk ingest ./docs && grk serve-mcp` connectable from Claude Desktop/Code with documented client config | done 2026-08-15 |
| 5 | Boundary features | optional query rewrite + cited synthesis; redaction pass (names → tokens, configurable patterns); advisory faithfulness judge | done 2026-08-15 (redaction covers cloud **chat** egress only — the embedding boundary is a recorded deviation, ADR-0017; no gated synthesis workflow yet) |
| 6 | IaC + observability | multi-stage non-root Dockerfile; compose (service+Ollama+collector+Jaeger); k8s (deployment, service, PVC, probes); Terraform module for one concrete provider; OTel verified end-to-end in compose | done 2026-08-16 — `infra/` landed 2026-08-15 (ADR-0020/0021/0022, `docs/specs/phase-6-iac-observability.md`); OTel instrumentation and JSON logs implemented (Phase 6 change 2: `telemetry.py`, spans on `Indexer.index_source`/`Indexer.index_directory`, `Retriever.search`, `Synthesizer.synthesize`) and verified in compose 2026-08-16: `ingest` and `retrieve` spans observed in Jaeger with the ADR-0022 attribute allowlist holding (no query text, source path or document content in the exported payload), plus the collector→Jaeger leg and the Terraform security-group preconditions. Also 2026-08-16: `terraform plan`/`apply` against a real AWS account (`us-east-1`, personal sandbox) — instance provisioned, image pulled from a real private ECR repo, a document ingested, and a real search served over an SSM port-forward tunnel; `terraform destroy` ran in the same session. Also 2026-08-16: the full documented compose cold-start — `ollama-pull`, the `ingest` one-shot (43 files, 1299 chunks, 1299 vectors), `up -d`, and `GET /v1/collections` plus a citation-bearing `POST /v1/search` over the `127.0.0.1:8765` publish; the loopback-only binding was demonstrated host-side (a `0.0.0.0` control port answered on this host's LAN address while `:8765` was refused on the same address), and the from-another-host leg was attempted and **could not complete** because the only available Wi-Fi is a guest SSID with client isolation. And 2026-08-16: the documented Kubernetes sequence verbatim on a **single-node** Docker Desktop (kind mode, v1.36.1) — `apply -k`, corpus load, ingest Job complete (43 files, 1299 chunks), Deployment 1/1 Ready, and a citation-bearing search over `kubectl port-forward`. **Every IaC path in this row has now been exercised at least once.** Two qualifiers remain rather than gaps in coverage: the multi-node `ReadWriteOnce` path a single node cannot produce, and the from-another-host half of the compose bind check, which could not run because the only Wi-Fi available is a guest SSID with client isolation. And 2026-08-16: the **`synthesize` span in a real trace**, completing SPEC.md §3's three-site list — a `docker compose run … grk answer` against the running four-service stack returned a cited answer over 2 BM25 results and produced `groundkit.synthesize.synthesize` in Jaeger carrying only `chat.model`, `chat.provider`, `duration_ms` and `result_count`; a sweep of the exported payload for the question text, the completion text, both citations' offsets, the corpus path and the source filename found none of them, and an error-path span from an earlier timed-out attempt leaked nothing either. Scope: the chat model was the operator's host Ollama rather than the stack's own (which holds only the embedding model). **Every span site and every IaC path in this row has now been exercised.** What remains is not unverified work but two limits of the available hardware, accepted and recorded rather than left open: the multi-node `ReadWriteOnce` path a single node cannot produce, and the from-another-host half of the compose bind check, blocked by a guest SSID with client isolation. `infra/README.md` is the status board and records the exact scope of each run |
| 7 | Docs + release | MkDocs site, README live badges only, MIT, v0.1.0 tag, PyPI publish workflow | machinery done 2026-08-15; **Phases 4–6 are now closed, so the release is unblocked** — version bumped to `0.1.0` on 2026-08-16 in both `pyproject.toml` and `groundkit.__version__`; PyPI pending publisher registered 2026-08-16 (trusted publishing has no token path, so this had to precede the first upload); what remains is the tag and the published GitHub release itself |

Phase 7 ran out of order, deliberately and partially. Everything in it that
does not describe Phases 4–6 has landed: the MkDocs site (strict build, gated
in CI and as a release gate), the LLM-boundary document §2 assigns to this
phase, live-badge-only README, the PyPI trusted-publishing workflow, the
release-gate suite that blocks it, and the re-enabled `eval-gated` /
`rerank-gated` schedules those workflows reserved for "end of development".

What has **not** happened is the release itself. No v0.1.0 tag, no GitHub
release, nothing published to PyPI. The reason that paragraph used to give —
that the MCP server, the service surface and the IaC did not exist, so a 0.1.0
on PyPI would be a version number that could not be withdrawn and reused — has
expired: Phases 4, 5 and 6 are all closed. The version was bumped off
`0.1.0.dev0` on 2026-08-16, in both `pyproject.toml` and
`groundkit.__version__` (two independent declarations that nothing but
`tests/test_smoke.py::test_version_matches_pyproject` holds together — the
release gate's parity step reads `pyproject.toml` alone).

The PyPI *pending publisher* — required because `publish.yml` is
**trusted-publishing only**, with no token input wired, against a project that
does not exist on PyPI yet — was registered by the owner on 2026-08-16. What
remains is the release event itself: the publish workflow is inert until a
GitHub release is published, and the tag it carries must match
`pyproject.toml`'s version or the parity gate refuses it.

The docs site describes only what is built, and states the gaps where they
fall. Phase 4's service/MCP pages and Phase 5's LLM-boundary update have
landed — the egress inventory now documents the redaction pass where it
actually runs (cloud chat, ADR-0017), not the embedding row this paragraph
originally predicted. Still owed with Phase 6: the IaC verification dates.

## 10. Definition of done (v1)

CI green (lint, types, tests, corpus-integrity, docs strict build, pip-audit,
secret scan) · coverage gate met on the defined core subset · `grk` works
end-to-end locally with zero cloud credentials · MCP server connects from a
real client with documented config · eval report reproducible from a clean
clone in ≤ 2 commands · each IaC path verified once with the verification
date recorded · KNOWN_LIMITATIONS.md honest and current · no number in any
doc that wasn't generated.

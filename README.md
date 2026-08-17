# groundkit

<!-- Badges are live endpoints only, never hand-written values (SPEC.md §2:
     "Real data only ... numbers come from generated eval artifacts or dynamic
     badges, or are omitted"). A badge showing a stale value is the failure the
     policy exists to prevent, so nothing here is a literal.

     Only badges that resolve to real state TODAY are present. Three that a
     released project would carry are deliberately absent until they have
     something true to report, rather than shipped rendering "not found":

       PyPI version / Python versions   — need the first published release
                                          (.github/workflows/publish.yml).
       License                          — shields' github/license endpoint
                                          needs a public repo; a hand-written
                                          MIT badge would be exactly the
                                          literal this policy excludes. The
                                          LICENSE file and the docs site's
                                          license page carry it meanwhile.

     Add each one when the thing it reports becomes true. -->

[![ci](https://github.com/tafreeman/groundkit/actions/workflows/ci.yml/badge.svg)](https://github.com/tafreeman/groundkit/actions/workflows/ci.yml)
[![docs](https://github.com/tafreeman/groundkit/actions/workflows/docs.yml/badge.svg)](https://tafreeman.github.io/groundkit/)

Grounded, citation-verifiable hybrid retrieval: a persisted BM25 + dense index,
a named MCP server, and a retrieval eval harness — fully local by default.

**Documentation: <https://tafreeman.github.io/groundkit/>**

> **Status: Phases 0–6 done; the v0.1.0 release itself is the only thing
> left.** BM25 retrieval, a persisted index, citation-bearing search, and a
> retrieval eval harness work end-to-end locally with no cloud credentials —
> see the Quickstart below. Dense and hybrid (RRF) retrieval work too, opt-in
> behind `--dense` / `--mode` and requiring a local embedding provider. A
> local cross-encoder reranker is available behind the optional `rerank`
> extra and is wired into the eval harness, so it reports a measured delta
> like every other retrieval feature ([ADR-0012](docs/adr/ADR-0012-rerank-eval-stage-reorders-upstream-stage.md));
> it is not part of `grk search`.
>
> Since then: the **MCP server and REST API** ship over one runtime
> (`grk serve`, `grk serve-mcp`), the **LLM boundary** is built — optional
> query rewrite, cited synthesis that may cite only retrieved spans, a
> redaction pass on cloud chat egress with no operator opt-out, and an
> advisory faithfulness judge — and the **IaC** is real and exercised
> (Dockerfile, compose with an OTel collector and Jaeger, Kubernetes
> manifests, and a Terraform module that has been applied and destroyed
> against a live account). OpenTelemetry spans cover ingest, retrieve and
> synthesize.
>
> **groundkit is not on PyPI yet.** The publish workflow and its blocking
> release gates exist and the version is at `0.1.0`; what remains is the tag
> and the published release. Two v1 scope items are deliberately unbuilt and
> named as such — **PDF/HTML ingestion and URL ingestion**, whose extractors
> and citation re-verification landed but whose ingest-side loaders did not
> (see [SPEC.md](SPEC.md) §4). See
> [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) for everything deliberately
> out of scope or presently broken — it is honest and current, including
> about defects.

## What this is

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
Where text can and cannot leave the process is written down in full —
[docs/architecture/llm-boundary.md](docs/architecture/llm-boundary.md). The
redaction pass named above now exists and wraps **cloud chat egress with no
operator opt-out** ([ADR-0017](docs/adr/ADR-0017-chat-seam-and-redaction-boundary.md));
that document also records what it does *not* cover — the embedding boundary
is a deliberate, named exception, so read it before pointing an embedding
provider at a cloud endpoint.

## Quickstart

```bash
uv sync
uv run grk ingest ./docs
uv run grk search "your query" --json
```

Ingestion is incremental (unchanged files are skipped by content hash), the
index persists under `.groundkit/` and survives restarts, and every result
carries a citation — source path plus character offsets — that
`groundkit.retrieval.verify_citation` can check against the source file.
No cloud credentials are required for any of this.

## Eval harness

```bash
uv run grk eval
```

Runs the retrieval-quality harness against the committed golden corpus and
judgment set — `evals/corpus/` plus `evals/judgments.jsonl`, authored per
the contract in [evals/README.md](evals/README.md) — and writes a full
report to `evals/results/latest.json` (gitignored, regenerated per run,
never committed). Fully offline and credential-free: the harness builds a
throwaway index over the corpus and scores it with the same deterministic
BM25 retrieval path `grk search` uses. BM25-only is the baseline every
later retrieval feature (hybrid, rerank) reports its delta against, in the
same report.

## Which retrieval mode should I use?

`grk search` defaults to `--mode bm25`, and that is a deliberate decision
rather than an unfinished one — see
[ADR-0007](docs/adr/ADR-0007-default-retrieval-mode.md).

Measured against the golden corpus with a real embedding model, **`--mode
hybrid` beat the BM25 baseline on every retrieval-quality metric**, with no
metric regressing. Reproduce it yourself rather than taking that on trust —
no metric value is written into this README by policy, and the run writes a
full report:

```bash
uv run grk eval --dense --embed-model nomic-embed-text
```

### Building a dense index — start a fresh collection

**`--mode hybrid` only works against a collection that was ingested with
`--dense`.** You cannot add the dense path to an existing BM25-only
collection:

```bash
uv run grk ingest ./docs                       # BM25-only collection
uv run grk search "your query" --mode hybrid   # error: no embedding-identity manifest
uv run grk ingest ./docs --dense               # "0 vectors written" — all hash-skipped
```

Ingestion is incremental by content hash, and that check runs *before*
embedding — which is what stops unchanged documents being re-embedded on
every run, and is also why turning `--dense` on later backfills nothing.
Only documents whose content changes afterwards ever gain vectors, so the
collection stays permanently vector-less and the second command above
reports `0 vectors written` rather than fixing anything.

The search **fails loudly** rather than answering ([ADR-0008](docs/adr/ADR-0008-dense-search-requires-a-dense-collection.md)).
Before that, it returned BM25's ranking stamped `"stage": "fusion"` — lexical
results labelled as hybrid, with no error — which is the failure mode this
whole section exists to prevent. `--mode bm25` is unaffected, and a
dense-paired retriever may still search `bm25`; only the modes that need
vectors are refused.

Do this instead — a dense collection from the first ingest:

```bash
ollama pull nomic-embed-text                       # once; any embedding model works
uv run grk ingest ./docs --collection dense --dense
uv run grk search "your query" --collection dense --mode hybrid
```

To convert a collection you already have, delete it and re-ingest: remove
`.groundkit/<collection>.sqlite3` and `.groundkit/<collection>.lance`, then
run the `--dense` ingest above. There is no in-place upgrade — see
[KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) for why, and for the inverse
hazard (a BM25-only ingest over a dense collection orphans its vectors).

### The tradeoff

**Hybrid is recommended wherever you have an embedding provider configured
and can accept two costs**, both of which sit outside those quality metrics:

- **It cannot abstain.** BM25 returns nothing when no indexed chunk shares a
  term with your query. Hybrid always returns its top-k, however irrelevant,
  because fused scores are rank-derived and no score threshold applies to
  them (ADR-0005 decision 6). For a question your corpus cannot answer, BM25
  says nothing and hybrid answers confidently. `--mode dense` *does* honour
  `score_threshold`, but no defensible default value has been measured yet.
- **It needs a running embedding provider** (Ollama by default) and costs
  substantially more latency per query. BM25 needs neither, which is why it
  remains the default: `grk` works end-to-end with zero cloud credentials
  and no model server, and the default path is not permitted to break that.

If you are indexing content you will ask open-ended questions of and can
tolerate a confident answer to an unanswerable one, use hybrid. If you need
"I found nothing" to be a possible answer, stay on BM25.

## Development

```bash
uv sync --group dev
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest --cov && uv run coverage report
```

CI enforces an 80% coverage floor twice, so neither gate can hide the other:
once on the whole package, and again on the SPEC.md §8 core subset —
`retrieval/` (retrieval + citation resolution), `ingestion/chunking.py`
(chunking), `index/bm25.py` (lexical scoring), and `index/dense.py` (vector
scoring). The core subset is the literal list in `pyproject.toml`'s
`[tool.groundkit.coverage]` table; optional providers (e.g.
`providers/embeddings.py`) are excluded from it. That table also records the
one caveat this subset carries — `index/dense.py` is a mixed file, and gating
it wholesale admits inside one file the offsetting the subset exists to
prevent.

To build the documentation site:

```bash
uv sync --group docs --extra dense
uv run mkdocs serve            # or: uv run mkdocs build --strict
```

CI builds it with `--strict`, which promotes MkDocs warnings to errors: a
broken relative link, a `#fragment` no heading produces, or a page under
`docs/` that was never added to the nav all fail the build.

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

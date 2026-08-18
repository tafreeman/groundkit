# groundkit

If you point an AI assistant at a folder of documents and ask it questions,
it will answer — confidently — whether or not the documents actually
support that answer. It might summarize them faithfully. It might also
invent a detail, or attribute a claim to a document that doesn't contain
it, and nothing about how the answer sounds tells you which one happened.

groundkit is a retrieval system built to close that specific gap: finding
the right passage, and proving it is the right passage. Every result names
the exact source file and character range it came from — a **citation** —
and groundkit re-reads that file and checks the text is actually there
before returning the result, rather than trusting its own pointer. Getting
a pointer that precise starts with splitting every document into small,
bounded passages (**chunks**) before anything is indexed or searched.

**Who this documentation is for:** engineers and technical evaluators
looking at a retrieval component to sit underneath an AI assistant, a
support tool, or a search feature — including by connecting an assistant
like Claude to it directly over **MCP** (Model Context Protocol — the
standard AI assistants use to call external tools). If you want something
you can point at a folder of Markdown/text files or a URL and get back
verifiable, checkable search results today, groundkit already does that. If
you need PDF/HTML ingestion or a hosted service, neither is built yet — see
[Known limitations](limitations.md).

## What it does today

Grounded, citation-verifiable hybrid retrieval: a persisted BM25 + dense
index, a named MCP server, and a retrieval eval harness — fully local by
default. Term by term:

- **Hybrid retrieval** — combines **BM25** (keyword search: scores
  documents by the words they literally share with the query) with **dense
  embeddings** (matching by meaning, so a query for "car" can match a
  document that only says "automobile"), merged by **reciprocal-rank fusion
  (RRF)**, over an index that survives restarts, with an optional local
  **cross-encoder reranker** — a slower, more accurate second pass over the
  top few results.
- **Verifiable citations** — every passage carries document ID, chunk ID and
  character offsets that resolve back to source, and a function that checks
  them.
- **A retrieval eval harness** — a fixed set of test questions with
  known-correct answers, scored against a labelled golden corpus by
  deterministic, unit-tested metric code. BM25-only is the baseline every
  feature reports a delta against, including when the delta is negative.
- **Local by default** — Ollama embeddings and a file-based index. `grk` works
  end to end with zero cloud credentials.

## Status

!!! note "All seven phases done; v0.1.0 released 2026-08-18"

    The deterministic core, the persisted index, hybrid retrieval, the eval
    harness, the LLM boundary (query rewrite, synthesis, redaction), the MCP
    server, the REST API, and the IaC are all built and gated, and the release
    itself has shipped — `pip install groundkit` works. The phase plan in
    [SPEC.md](https://github.com/tafreeman/groundkit/blob/main/SPEC.md) §9 is
    the authoritative status, and
    [Known limitations](limitations.md) is the authoritative list of what
    does not work.

## What makes it different from a RAG demo

Three things, all of which are unglamorous and all of which are the point.

**The core is deterministic.** No model runs in the retrieval path. Scoring,
fusion and citation resolution are pure typed functions, so retrieval quality
is a measurement rather than an impression.

**The harness came before the features.** The eval harness and golden corpus
landed in Phase 2, ahead of hybrid retrieval and rerank, so every retrieval
feature since has arrived with a measured delta against the BM25 baseline
instead of an argument.

**It fails closed.** Unknown config key → startup failure. Unconfigured
provider → typed error. Hybrid search against a collection with no vectors →
a refusal, not a silent fall back to lexical results
([ADR-0008](adr/ADR-0008-dense-search-requires-a-dense-collection.md)).
There is no cross-provider embedding fallback anywhere, because mixing
semantic spaces corrupts an index quietly.

## Start here

<div class="grid cards" markdown>

- **[Installation](getting-started/installation.md)** — install, and what the
  optional extras cost you.
- **[Quickstart](getting-started/quickstart.md)** — ingest a directory and
  search it, in three commands.
- **[MCP clients](guides/mcp-clients.md)** — connect Claude Desktop or Claude
  Code to a running collection.
- **[Retrieval modes](guides/retrieval-modes.md)** — which mode to use, and
  the one thing hybrid cannot do.
- **[The LLM boundary](architecture/llm-boundary.md)** — exactly where text
  can leave the process, and what is redacted.

</div>

## No numbers here

You will not find a recall or nDCG figure (standard retrieval-quality
metrics) written into this site. SPEC.md §2 permits a number in a document
only when it comes from a generated eval artifact or a live badge — a
hand-copied metric is a number that was true once. The [eval harness
guide](guides/evals.md) shows how to generate the report yourself; it takes
two commands from a clean clone.

## Provenance

groundkit is a standalone successor to the RAG library inside
[agentic-runtime-platform](https://github.com/tafreeman/agentic-runtime-platform),
built to close its verified production gaps. The per-module
promote-vs-rewrite decision, the gap audit, and the port-time hazard list are
recorded in [ADR-0001](adr/ADR-0001-promote-vs-rewrite.md) — and each named
hazard is a spec obligation with a fix and a regression test in the phase that
ported it.

It composes with the rest of the portfolio without importing into it: it may
consume [executionkit](https://github.com/tafreeman/executionkit) for LLM call
patterns at the synthesis boundary, and it is gradable by
[agentic-evalkit](https://github.com/tafreeman/agentic-evalkit) through that
project's HTTP/MCP `ExecutionTarget` boundary.

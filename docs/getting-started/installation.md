# Installation

This page is for anyone putting groundkit on their own machine for the first
time — to try it, evaluate it, or develop against it. You do not need to know
anything about search or retrieval systems going in; the terms that matter
are explained the first time each one comes up. By the end, you will have a
working local copy that can index and search a folder of documents, and you
will know which optional pieces (if any) you need for what you're trying to
do — the base install needs no accounts, no API keys, and no other service
running.

groundkit targets Python 3.11+ and uses [uv](https://docs.astral.sh/uv/) — a
Python package and environment manager — for environments and locking.

## From PyPI

```bash
pip install groundkit
```

That gives you the base install: BM25 retrieval (keyword search — it ranks
results by how many words they literally have in common with your query,
the same idea a library card catalog or Ctrl+F uses), the persisted index
(the index is saved to disk and survives a restart — nothing lives only in
memory), citation-resolving search (every result points at an exact,
checkable location in a source file, not just a paraphrase of it), and the
`grk` CLI. It pulls groundkit's runtime dependencies — `pydantic`, `httpx`,
`fastapi`, `uvicorn`, `mcp` (the SDK for MCP, the Model Context Protocol — a
standard way for AI assistants such as Claude to call external tools; see
[MCP clients](../guides/mcp-clients.md)), and `opentelemetry-api` (hooks for
optional tracing, inert until you configure a destination for the traces) —
and needs no credentials.

## From a clone

Use a clone instead of PyPI if you want to develop against groundkit itself
(see [Development install](#development-install) below) or run the eval
harness against the golden corpus, which ships in the repo rather than the
package:

```bash
git clone https://github.com/tafreeman/groundkit
cd groundkit
uv sync
```

That gives you the same base install: BM25 retrieval (keyword search — it ranks
results by how many words they literally have in common with your query,
the same idea a library card catalog or Ctrl+F uses), the persisted index
(the index is saved to disk and survives a restart — nothing lives only in
memory), citation-resolving search (every result points at an exact,
checkable location in a source file, not just a paraphrase of it), and the
eval harness (a repeatable way to prove a change to search actually made
results better, by scoring it against a fixed set of questions with known
correct answers — see [Eval harness](../guides/evals.md)). It pulls
groundkit's runtime dependencies — `pydantic`, `httpx`, `fastapi`, `uvicorn`,
`mcp` (the SDK for MCP, the Model Context Protocol — a standard way for AI
assistants such as Claude to call external tools; see
[MCP clients](../guides/mcp-clients.md)), and `opentelemetry-api` (hooks for
optional tracing, inert until you configure a destination for the traces) —
and needs no credentials.

## Extras

Five extras — optional add-on packages you install only if you need the
capability they unlock — exist because their cost is real enough that a
default install should not pay it. The two below back retrieval; `otel` (the
tracing SDK and OTLP exporters that actually ship trace data somewhere — see
[Deployment](../guides/deployment.md)) and `pdf`/`html` (the parsing
libraries behind PDF/HTML citation verification — note that `grk ingest`
does not yet read `.pdf` or `.html` files directly, see
[Known limitations](../limitations.md); these extras cover what happens once
such a document has been extracted by some other path — see
[Extracted and remote source loaders](../specs/loaders-extracted-and-remote-sources.md))
are documented where they're used.

=== "Dense retrieval"

    ```bash
    uv sync --extra dense
    ```

    Adds LanceDB, the vector store behind `--mode dense` and `--mode
    hybrid` — dense retrieval (also called "embeddings-based" search)
    matches by meaning instead of exact words, so a query for "car" can find
    a passage about "automobile"; hybrid combines dense with BM25 into a
    single ranking. Each chunk (a passage of a document, small enough to
    cite precisely) has to be converted into that meaning-representation
    ahead of time, which is why this extra needs an embedding provider too
    — see below. [Retrieval modes](../guides/retrieval-modes.md) covers the
    full tradeoff between the modes.

=== "Cross-encoder rerank"

    ```bash
    uv sync --extra rerank
    ```

    Adds `sentence-transformers`, and through it torch: a multi-gigabyte
    install. It enables the optional reranker — a slower, more accurate
    second pass that re-scores just the top few search results before they
    are returned. This is why the extra is not mirrored into the dev group
    and why CI's default job never installs it — the rerank backend is
    proved by a separate gated workflow instead.

## An embedding provider

Dense and hybrid retrieval need embeddings. The default provider is a local
Ollama instance:

```bash
ollama pull nomic-embed-text
```

groundkit talks to `http://127.0.0.1:11434` unless you configure otherwise.
Nothing leaves the machine on this path.

An OpenAI-compatible endpoint is supported as an opt-in alternative. Before
you enable it, read [The LLM boundary](../architecture/llm-boundary.md):
your full document text and every query go to that endpoint. groundkit's
redaction pass (`providers/redaction.py`) is scoped to chat egress only — the
embedding boundary deliberately has no redaction pass in front of it. That is
a recorded deviation
([ADR-0017](../adr/ADR-0017-chat-seam-and-redaction-boundary.md) decision 5),
not work still owed.

## Development install

```bash
uv sync --group dev
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest --cov && uv run coverage report
```

CI enforces the 80% coverage floor twice — once on the whole package and again
on the SPEC.md §8 core subset — so a well-covered peripheral module cannot
offset an under-tested core one. The core subset is the literal list in
`pyproject.toml`'s `[tool.groundkit.coverage]` table.

To build the docs site locally:

```bash
uv sync --group docs --extra dense
uv run mkdocs serve
```

The `dense` extra is not optional for the docs build: the API reference imports
groundkit to render it, and a missing optional dependency shows up as a broken
page. CI builds with `--strict`, so a broken link, a missing anchor, or a page
absent from the nav fails the build.

## Next

With the base install in place, [Quickstart](quickstart.md) takes you from a
clone to a working search in three commands.

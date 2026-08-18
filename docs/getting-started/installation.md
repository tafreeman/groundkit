# Installation

groundkit targets Python 3.11+ and uses [uv](https://docs.astral.sh/uv/) for
environments and locking.

## From a clone

```bash
git clone https://github.com/tafreeman/groundkit
cd groundkit
uv sync
```

That gives you the base install: BM25 retrieval, the persisted index,
citation-resolving search, and the eval harness. It pulls groundkit's runtime
dependencies — `pydantic`, `httpx`, `fastapi`, `uvicorn`, `mcp`, and
`opentelemetry-api` — and needs no credentials.

## Extras

Five extras exist because their cost is real enough that a default install
should not pay it. The two below back retrieval; `otel` (the tracing SDK and
OTLP exporters — see [Deployment](../guides/deployment.md)) and `pdf`/`html`
(the extraction libraries behind PDF/HTML citation verification — see
[Extracted and remote source loaders](../specs/loaders-extracted-and-remote-sources.md))
are documented where they're used.

=== "Dense retrieval"

    ```bash
    uv sync --extra dense
    ```

    Adds LanceDB, the vector store behind `--mode dense` and `--mode hybrid`.
    You will also need an embedding provider — see below.

=== "Cross-encoder rerank"

    ```bash
    uv sync --extra rerank
    ```

    Adds `sentence-transformers`, and through it torch: a multi-gigabyte
    install. This is why the extra is not mirrored into the dev group and why
    CI's default job never installs it — the rerank backend is proved by a
    separate gated workflow instead.

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

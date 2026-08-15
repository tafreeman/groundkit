# Quickstart

Three commands, no credentials, no model server.

```bash
uv sync
uv run grk ingest ./docs
uv run grk search "citation offsets" --json
```

## What just happened

`grk ingest` walked the directory, loaded every supported file, chunked it
while preserving character offsets, and wrote documents, chunks and BM25
statistics into a persisted collection under `.groundkit/`. Ingestion is
incremental: re-run it and unchanged files are skipped by content hash.

`grk search` rebuilt the index from disk, scored your query with BM25, and
returned results that each carry a citation — source path plus character
offsets. The index survives restarts; nothing is held only in memory.

## Verifying a citation

The offsets are not decoration. They resolve:

```python
from groundkit.retrieval import verify_citation
```

`verify_citation` re-reads the source file and confirms the cited span still
contains what the citation claims. That is what "citations are verifiable"
means here — a property you can check, not an assurance.

## Adding dense and hybrid retrieval

Hybrid needs vectors, and vectors have to be there from the first ingest.
**You cannot add the dense path to a collection that was built BM25-only:**

```bash
ollama pull nomic-embed-text
uv run grk ingest ./docs --collection dense --dense
uv run grk search "citation offsets" --collection dense --mode hybrid
```

The reason a `--dense` ingest over an existing BM25 collection backfills
nothing — and reports `0 vectors written` rather than fixing anything — is
that the content-hash skip runs *before* embedding. [Retrieval
modes](../guides/retrieval-modes.md) covers this, and
[ADR-0008](../adr/ADR-0008-dense-search-requires-a-dense-collection.md)
records why the mismatched search is a loud error instead of a quiet fallback.

## Running the eval harness

```bash
uv run grk eval
```

Scores the committed golden corpus with the same deterministic BM25 path
`grk search` uses and writes a full report to `evals/results/latest.json`.
Offline, credential-free, and reproducible from a clean clone. See the
[eval harness guide](../guides/evals.md).

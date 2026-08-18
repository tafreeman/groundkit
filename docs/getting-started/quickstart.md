# Quickstart

This page is for anyone who wants to see groundkit actually search something,
right now, without reading how it works first. It picks up right after
[Installation](installation.md). No background in search or retrieval is
assumed. By the end, you will have indexed a real folder of files and run a
search against it whose results you can check against the source yourself —
plus pointers to two optional next steps: search-by-meaning, and an automated
way to check whether a change made search better or worse.

Three commands, no credentials, no model server.

```bash
uv sync
uv run grk ingest ./docs
uv run grk search "citation offsets" --json
```

## What just happened

`grk ingest` walked the directory, loaded every supported file, split each
one into chunks (passages short enough to point at precisely — long enough
to make sense on their own) while preserving each chunk's exact character
offsets within the original file, and wrote documents, chunks and BM25
statistics into a persisted collection under `.groundkit/`. BM25 is the
keyword-matching algorithm behind the default search: it ranks a chunk by how
many words it literally has in common with your query, the same underlying
idea as a Ctrl+F search, just scored rather than a yes/no match. Ingestion is
incremental: re-run it and unchanged files are skipped by content hash.

`grk search` rebuilt the index from disk, scored your query with BM25, and
returned results that each carry a citation — source path plus character
offsets, pointing at the exact span of the source file the result came from,
not a paraphrase of it. The index survives restarts; nothing is held only in
memory.

## Verifying a citation

The offsets are not decoration. They resolve:

```python
from groundkit.retrieval import verify_citation
```

`verify_citation` re-reads the source file and confirms the cited span still
contains what the citation claims. That is what "citations are verifiable"
means here — a property you can check, not an assurance.

## Adding dense and hybrid retrieval

BM25 matches literal words, so it cannot connect a query about "cars" to a
chunk about "automobiles" — dense retrieval closes that gap by matching on
meaning instead. Meaning is represented as a vector (a list of numbers
produced by an embedding model), and comparing vectors is how a dense search
finds a chunk that means the same thing without sharing any words. Hybrid
runs both BM25 and dense search and merges the two ranked lists into one, so
you get literal-word matches and meaning-based matches together.

Hybrid needs those vectors, and they have to be there from the first ingest.
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

An eval harness is a fixed set of questions with known-correct answers,
scored automatically — it's how you prove a change to search actually made
results better, rather than just different. groundkit's is built in:

```bash
uv run grk eval
```

Scores the committed golden corpus (that fixed question set, checked into the
repo) with the same deterministic BM25 path `grk search` uses, and writes a
full report to `evals/results/latest.json`. Offline, credential-free, and
reproducible from a clean clone. See the [eval harness guide](../guides/evals.md).

## Next

[Retrieval modes](../guides/retrieval-modes.md) walks through when hybrid is
worth its extra cost and when it isn't. [MCP clients](../guides/mcp-clients.md)
connects this same index to Claude Desktop or Claude Code so an AI assistant
can search it directly.

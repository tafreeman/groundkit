# Retrieval modes

This page is for anyone who has to pick how groundkit searches — including if
you will never type the command yourself and just need to understand the
tradeoff someone else is choosing on your behalf. groundkit can find results
three ways: by matching the literal words in a query, by matching what the
query *means*, or by combining both. Which one is right depends on what you're
indexing and on whether "I don't know" needs to be an answer groundkit is
allowed to give. By the end of this page you'll be able to make that call
yourself, without needing to run anything.

`grk search` takes `--mode bm25` (the default), `--mode dense`, or
`--mode hybrid`.

## The short answer

Use **hybrid** if you have an embedding provider configured (a running
service that turns text into a searchable representation of its meaning —
setup cost below) and your users ask open-ended questions. Stay on **BM25**
if "I found nothing" needs to be a possible answer.

| Mode | What it is | What it buys you | What it costs | Can it say "I don't know"? |
|---|---|---|---|---|
| **BM25** (the default) | Keyword matching — ranks a chunk by how many words it literally shares with your query, the scored version of a Ctrl+F search | Works instantly, everywhere, on anything you can already install | Nothing beyond the base install — no extra setup, no model download | **Yes.** Returns nothing when no indexed chunk shares a term with the query |
| **Dense** | Meaning-based matching — text is converted to a vector (a list of numbers standing in for its meaning) by an embedding model, so a query about "cars" can find a chunk about "automobiles" | Finds relevant results that don't share any words with the query | An embedding provider must be running (a local model download via Ollama, or a cloud endpoint), and your documents must be re-ingested with `--dense` — see below | In principle, yes — but no safe default cutoff has been measured yet, so it ships off |
| **Hybrid** | BM25 and dense search run together and their two ranked lists are merged into one (a technique called reciprocal rank fusion, or RRF) | Combines literal-word and meaning-based matches; measured better than BM25 alone on every retrieval-quality metric | Same setup as dense, plus noticeably more time per query | **No.** Always returns its top results, however irrelevant — see the warning below |

Measured against the golden corpus (a small, fixed set of documents and
questions with known-correct answers — see the
[eval harness guide](evals.md)) with a real embedding model, hybrid beat
the BM25 baseline on every retrieval-quality metric, with no metric
regressing. No figure for that is written here on purpose — SPEC.md §2 allows
a number in a document only when it is generated or live, and this one is
reproducible in a single command:

```bash
uv run grk eval --dense --embed-model nomic-embed-text
```

## Why BM25 is still the default

Because the default path is not permitted to require a model server.
`grk` works end to end with zero cloud credentials and nothing running
locally, and BM25 is what makes that true. Hybrid being *better* on the
metrics does not make it a safe default when it cannot run at all on a machine
without Ollama. [ADR-0007](../adr/ADR-0007-default-retrieval-mode.md) records
the decision.

## The tradeoff the metrics do not show

Two costs sit outside every quality number:

!!! warning "Hybrid cannot abstain"

    BM25 returns nothing when no indexed chunk shares a term with your query.
    Hybrid always returns its top-k, however irrelevant, because fused scores
    are rank-derived and no score threshold applies to them
    ([ADR-0005](../adr/ADR-0005-fusion-and-rerank-scoring.md) decision 6).

    Ask a question your corpus cannot answer and BM25 says nothing while
    hybrid answers confidently. `--mode dense` *does* honour
    `score_threshold` (a minimum relevance score below which a mode returns
    nothing at all — the mechanism that lets a mode say "I don't know"), but
    no defensible default value for it has been measured yet, so it is off.

**Hybrid needs a running embedding provider** and costs substantially more
latency per query. BM25 needs neither.

## Building a dense collection

A collection is a named, self-contained index — `--collection NAME` lets you
keep more than one side by side, each searched independently. This is the
part that catches people, so it is worth being blunt about: **`--mode hybrid`
only works against a collection ingested with `--dense`, and you cannot
upgrade a collection in place.**

```bash
uv run grk ingest ./docs                       # BM25-only collection
uv run grk search "your query" --mode hybrid   # error: no embedding-identity manifest
uv run grk ingest ./docs --dense               # "0 vectors written" — all hash-skipped
```

Ingestion is incremental by content hash, and that check runs *before*
embedding — which is exactly what stops unchanged documents being re-embedded
on every run, and is also why turning `--dense` on afterwards backfills
nothing. Only documents whose content changes later ever gain vectors, so the
collection stays permanently vector-less.

Do this instead, from the first ingest:

```bash
ollama pull nomic-embed-text                       # any embedding model works
uv run grk ingest ./docs --collection dense --dense
uv run grk search "your query" --collection dense --mode hybrid
```

To convert a collection you already have, delete it and re-ingest: remove both
`.groundkit/<collection>.sqlite3` **and** `.groundkit/<collection>.lance`, then
run the `--dense` ingest above. Deleting one without the other leaves a
collection whose two halves disagree.

### Why the mismatch is an error

The search fails loudly rather than answering
([ADR-0008](../adr/ADR-0008-dense-search-requires-a-dense-collection.md)).
Before that change it returned BM25's ranking stamped `"stage": "fusion"` —
lexical results labelled as hybrid, with no error and no way for a caller to
tell. That silent mislabelling is the failure mode the refusal exists to
prevent.

`--mode bm25` is unaffected, and a dense-paired retriever may still search
`bm25`; only the modes that actually need vectors are refused. The inverse
hazard — a BM25-only ingest over a dense collection, which orphans its vectors
— is refused too, per
[ADR-0011](../adr/ADR-0011-bm25-only-ingest-refuses-a-dense-collection.md).

## Rerank

A reranker is an optional, slower, more accurate second pass that re-scores
just the top few search results before they're returned — it does not
replace BM25, dense, or hybrid, it refines whichever one already ran. The
optional local cross-encoder reranker reorders the best available upstream
stage ([ADR-0012](../adr/ADR-0012-rerank-eval-stage-reorders-upstream-stage.md)).
It runs entirely in-process — no text is transmitted anywhere to rerank it —
behind the `rerank` extra, whose install cost is covered in
[Installation](../getting-started/installation.md).

Raw cross-encoder logits are unbounded and frequently negative, so groundkit
sigmoid-normalizes them before they reach a `RetrievalResult`, whose score
contract is non-negative. That normalization is
[ADR-0005](../adr/ADR-0005-fusion-and-rerank-scoring.md) decision 4, and it
exists because ADR-0001 hazard 2 is the case where it was missing.

## Next

[Eval harness](evals.md) shows how the hybrid-beats-BM25 numbers referenced
above are produced, and how to reproduce them on your own corpus.
[Deployment](deployment.md) covers running any of these modes as a service
rather than from a terminal.

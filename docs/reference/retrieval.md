# Retrieval

Search orchestration, reciprocal-rank fusion, cross-encoder rerank, and
citation resolution. This is the deterministic core: no LLM runs on any path
in this package.

## Search

::: groundkit.retrieval.search

## Fusion

::: groundkit.retrieval.fusion

## Rerank

`rerank_by_logits` re-scores results but must not re-derive them: each
reranked `RetrievalResult` carries forward the `source_class` and
`extractor` of the result it replaces, rather than reverting to the
contract's `"text"`/`None` defaults. That is what keeps citation
verification (see [Citations](#citations) below) correct for a snapshot or
extracted document after a rerank pass, not only before one.

::: groundkit.retrieval.rerank

## Citations

::: groundkit.retrieval.citations

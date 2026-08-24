# SciFact validation — 2026-08-21

First external validation of groundkit's retrieval against a published baseline.
Commit `88fe570`, BM25-only, no embedding provider.

Two notes on reading the p-values below. They are **achieved significance
levels** from a paired bootstrap over the observed deltas (Smucker, Allan &
Carterette 2007), not null-centred p-values, and every `significant` /
`indistinguishable` verdict in this document is decided by whether the 95%
confidence interval excludes zero -- never by the p-value. Cells that
originally read `0.0000` now read `< 0.0002`: a Monte Carlo estimate over
10,000 resamples cannot resolve a probability below `1/B`, so the generating
scripts apply the standard `(r+1)/(B+1)` finite-sample correction and can no
longer print an exact zero. No effect size, interval or verdict changed.

## Headline: the BM25 implementation is correct

| configuration | chunks | nDCG@10 | MRR | R@10 | R@1 |
|---|---|---|---|---|---|
| whole document (chunk_size 12000) | 5,183 | **0.6646** | 0.638 | 0.782 | 0.525 |
| chunk ~= median doc (1500) | 9,549 | 0.6507 | 0.617 | 0.784 | 0.507 |
| groundkit default (512/64) | 25,028 | 0.5960 | 0.564 | 0.734 | 0.455 |

**BEIR published BM25 on SciFact: nDCG@10 = 0.665.** Indexing whole documents
reproduces it to 0.0004. That is the validation: groundkit's Okapi BM25, its
tokenizer and its scoring are sound, confirmed against an external reference
rather than against its own corpus.

k1/b barely move it — 0.658 to 0.665 across Anserini (0.9/0.4), Lucene (1.2/0.75)
and groundkit's own default (1.5/0.75) — so neither parameter tuning nor the
missing stemming/stopwords (groundkit tokenizes with `re.findall(r"\w+", lower)`,
BEIR's baseline used Elasticsearch's English analyzer) explains any real part of
the gap.

## The entire gap is chunking: −0.069 nDCG@10 (−10.4% relative)

SciFact documents are abstracts, median 1,425 characters. Splitting them at 512/64
produces ~4.8 fragments per document and costs a tenth of the score, monotonically
with granularity. Nothing is gained in exchange — the documents were already
smaller than most context windows.

The actionable form: **a chunker that leaves short documents whole would recover
0.069 here for free.** This corroborates, on a real collection with an external
reference, the golden-corpus finding that chunk size dominates architecture.

## Method note that matters

BEIR ranks documents; groundkit ranks chunks. Scored natively, `grk eval` reports
nDCG@10 = 0.371 on this set — *not comparable to 0.665*, because IDCG is computed
over ~5 gold chunks per query, so finding the right document and returning one
chunk of it is scored as if four were missed. `scifact_doclevel.py` collapses the
chunk ranking to first-seen distinct documents before scoring, which is the only
way the two numbers mean the same thing. At whole-document chunking the collapse
is the identity, and the result lands on the published value — which validates the
scorer and the retriever together.

`MAX_TOP_K = 50` caps the candidate pool. At 50 chunks/query the default config
yields ~37.5 distinct documents, enough for a top-10, but a chunk-ranking system
structurally needs a deeper pool than a document-ranking one to produce the same
top-10 documents.

## Reproduce

```bash
uv run --no-sync python evals/beir/adapt_beir.py --src <dir>/scifact --out <dir>/scifact-gk
```

```bash
uv run --no-sync python evals/beir/scifact_sweep.py <dir>/scifact-gk
```

Data: BEIR SciFact, 5,183 documents / 300 test queries / 339 qrels, all binary
relevance — which is why groundkit's binary nDCG is directly comparable and no
graded-relevance work was needed.

---

# Chunker fix — landed 2026-08-21

The 0.069 gap turned out to have a specific, fixable cause rather than being an
inherent cost of chunking.

## The defect

`RecursiveChunker._merge_parts` flushed the accumulated run whenever the *incoming*
part would overflow `chunk_size`. When that incoming part is oversized **on its own**
it gets recursively re-split at a finer separator regardless — so flushing first buys
no smaller output, it only guarantees the accumulated run is emitted alone.

For `heading\n\nlong body` — the shape of essentially every markdown document, and of
every SciFact record (`title\n\ntext`) — the heading is its own part, the body
overflows, and the bare heading became a chunk.

Measured on SciFact before the fix: **21.8% of all chunks under 128 characters**,
median chunk 363 against a 512 target, p10 = 80.

## The fix and what it bought

When the incoming part overflows on its own, fold the accumulated run into it and let
`_flush`'s existing recursion place that text at the head of the first sub-chunk.

| | chunks | <128 chars | nDCG@10 | MRR | R@10 | R@1 |
|---|---|---|---|---|---|---|
| before | 25,028 | 21.8% | 0.5960 | 0.564 | 0.734 | 0.455 |
| **after** | **20,219** | **2.2%** | **0.6229** | **0.591** | **0.757** | **0.492** |
| whole-doc ceiling | 5,183 | — | 0.6646 | 0.638 | 0.782 | 0.525 |

Paired bootstrap, 10,000 resamples, n = 300 — **every metric significant**:

| metric | delta | 95% CI | p |
|---|---|---|---|
| nDCG@10 | +0.0269 | +0.0109 … +0.0438 | 0.0004 |
| MRR | +0.0270 | +0.0090 … +0.0460 | 0.0036 |
| recall@1 | +0.0372 | +0.0094 … +0.0661 | 0.0078 |
| recall@10 | +0.0236 | +0.0028 … +0.0453 | 0.0240 |

It recovers ~39% of the gap to the whole-document ceiling **and produces 19% fewer
chunks** — less to store, embed and hold in RAM. Better and cheaper, which is rare
enough to be worth stating plainly.

Regression test: `tests/test_chunking.py::TestOversizedNeighborDoesNotOrphanShortLeadingPart`,
shown to fail against the reverted source and pass against the fix (SPEC.md §8).
Full gate green afterwards: 1,671 passed / 27 skipped, 96.95% total coverage,
97.54% core subset, ruff and mypy clean.

## What the remaining gap is, and why it should probably stay

Whole-document indexing still leads by 0.042. That is **not** a further bug to fix:
SciFact's qrels are document-level, so a metric computed over documents structurally
favours indexing documents whole. Chunking exists to make citation spans precise, and
this benchmark cannot see that benefit — it can only see the cost. Closing the rest of
the gap by abandoning chunking would be fitting the metric, not improving retrieval.

A span-level benchmark (LegalBench-RAG, whose ground truth is `(file, character range)`)
would measure the other direction and is the right next instrument.

## Golden-corpus impact

`grk eval` on the 10-document golden corpus moved 84 → 71 chunks, nDCG@10 0.586 → 0.601,
recall@1 0.333 → 0.389, MRR 0.503 → 0.553, recall@10 0.889 → 0.806. Mixed, and at n = 36
none of it is significant (the 95% half-width there is ±0.08–0.12). The n = 300 SciFact
result above is the authoritative one.

---

# Dense and fusion on SciFact — 2026-08-21

Ollama `nomic-embed-text` (768d), LanceDB, document-level scoring, n = 300.
The 512/64 rows below use the **post-fix** chunker (20,219 chunks).

| chunking | stage | nDCG@10 | MRR | R@10 | R@1 | p50 |
|---|---|---|---|---|---|---|
| 512/64 | bm25 | 0.6229 | 0.591 | 0.757 | 0.492 | 151 ms |
| 512/64 | **dense** | **0.7173** | 0.686 | 0.842 | 0.574 | 512 ms |
| 512/64 | fusion | 0.7151 | 0.684 | **0.850** | 0.571 | 657 ms |
| whole doc | bm25 | 0.6646 | 0.638 | 0.782 | 0.525 | 61 ms |
| whole doc | dense | 0.7033 | 0.667 | 0.846 | 0.554 | 512 ms |
| whole doc | fusion | 0.7087 | 0.679 | 0.835 | 0.564 | 552 ms |

Paired bootstrap, 10,000 resamples, nDCG@10, post-fix chunker:

| contrast | delta | 95% CI | p | |
|---|---|---|---|---|
| dense vs bm25 | +0.0944 | +0.0536 … +0.1350 | < 0.0002 | **significant** |
| fusion vs bm25 | +0.0922 | +0.0650 … +0.1201 | < 0.0002 | **significant** |
| fusion vs dense | −0.0022 | −0.0307 … +0.0256 | 0.8622 | not significant |

## Three findings

**Dense decisively beats BM25 — +0.094 nDCG@10, p < 0.0001.** On the 84-chunk golden
corpus this advantage sat inside the noise band; at n = 300 on a real collection it is
unambiguous. `nomic-embed-text` at 0.717 also beats the dense baselines BEIR originally
published (~0.644 for GenQ/TAS-B), which is the expected sanity check — those are
2021-era models.

**Fusion buys nothing over dense alone here** (−0.0022, p = 0.86) while costing ~145 ms
per query. It is not *harmful* — the earlier −0.012 reading at the pre-fix chunker was
noise, not a regression — it is simply a wash on this collection. Fusion does hold the
best R@10 (0.850), untested for significance. The default being hybrid is a latency cost
with no measured quality return on SciFact.

**BM25 and dense want opposite chunk granularities.** BM25 prefers whole documents
(0.6646 vs 0.6229 — correct document-level term statistics); dense prefers chunks
(0.7173 vs 0.7033 — embeddings dilute over long text). One `ChunkingConfig` serves both
stages, so tuning chunk size for either detunes the other. The chunker fix above gained
BM25 +0.027 and dense +0.0008 — it repaired a lexical defect, and dense never cared.

## Dense latency: 87% is LanceDB, and it is a constant

At 20,219 vectors: query embed 66 ms, **LanceDB search 446 ms**, total 512 ms.

It does not grow with corpus size — 5,183 vectors and 25,028 vectors both give ~512 ms
total — so this is fixed per-query overhead, not scan cost. `create_index` never appears
in `index/dense.py`, so there is no ANN index, but the absence is not what is costing
the time; a brute-force scan of 20k × 768 floats would show up as growth, and there is
none. The good news is that dense latency will not degrade as the corpus grows. The bad
news is 446 ms of constant overhead per query, which is the single largest latency item
in the system and is currently unexplained.

---

# Embedding model bake-off — 2026-08-21

Five Ollama models over the same SciFact corpus, identical chunking (512/64), scored
document-level over 300 queries. One persisted store per model (ADR-0004 forbids mixing
embedding identities in one collection).

| model | params | dims | nDCG@10 | MRR | R@10 | R@1 | embed | vectors |
|---|---|---|---|---|---|---|---|---|
| qwen3-embedding | 7.6B | 4096 | **0.7372** | 0.706 | 0.857 | 0.600 | 126.3m | 1636MB |
| nomic-embed-text | 137M | 768 | 0.7173 | 0.686 | 0.842 | 0.574 | **28.2m** | 1361MB |
| mxbai-embed-large | 334M | 1024 | 0.7080 | 0.687 | 0.818 | 0.567 | 43.7m | 1386MB |
| snowflake-arctic-embed2 | 568M | 1024 | 0.6836 | 0.658 | 0.796 | 0.554 | 51.7m | 1386MB |
| bge-m3 | 567M | 1024 | 0.6518 | 0.620 | 0.785 | 0.502 | 45.9m | 1386MB |

Paired bootstrap, 10,000 resamples, vs `nomic-embed-text`:

| model | delta | 95% CI | p | |
|---|---|---|---|---|
| qwen3-embedding | +0.0199 | −0.0094 … +0.0503 | 0.1846 | **not significant** |
| mxbai-embed-large | −0.0093 | −0.0349 … +0.0170 | 0.4836 | not significant |
| snowflake-arctic-embed2 | −0.0337 | −0.0573 … −0.0106 | 0.0044 | significant |
| bge-m3 | −0.0655 | −0.0926 … −0.0396 | < 0.0002 | significant |

## The finding

**A 137M model is statistically tied with the 7.6B model that tops the open-source MTEB
leaderboard.** qwen3 scores highest in absolute terms but its confidence interval crosses
zero — at n=300 this collection cannot separate them. It costs **4.5x the ingest time**
(126 vs 28 min), **5.3x the vector width**, and runs at 100% of this machine's memory
bandwidth roofline while nomic sits at 16% of its own.

Three models (qwen3, nomic, mxbai) are mutually indistinguishable. Two are significantly
worse — and both of those, snowflake-arctic-embed**2** and bge-m3, are the *multilingual*
models. On English scientific abstracts their multilingual capacity is spent on a problem
this corpus does not pose. That is the concrete reason an MTEB composite (56+ tasks,
100+ languages) fails to predict single-domain performance.

**Practical recommendation: keep `nomic-embed-text`.** It is tied with the best available
and 4.5x cheaper to ingest.

## Methodological caveat, measured not assumed

Ollama applies **no prefix template** to any of these models (all are bare `{{ .Prompt }}`),
yet nomic, mxbai and snowflake all document asymmetric query/document prefixes. The bake-off
therefore ran three of five models outside their documented configuration. Measured impact
on a 799-doc / 100-query subset:

| model | no prefix | documented prefix |
|---|---|---|
| nomic-embed-text | 0.8639 | 0.8529 |
| mxbai-embed-large | 0.8596 | 0.8593 |
| snowflake-arctic-embed2 | 0.8140 | **0.8298** |

Immaterial for nomic and mxbai. **Snowflake is the exception: its documented `query: `
prefix is worth +0.0158**, and it was the one model that lost significantly in the
bake-off. A quiet-machine re-check reproduced its no-prefix 0.8140 exactly, confirming
that quality is deterministic and the earlier contended measurement was sound.

So the bake-off understates snowflake by roughly 0.016. Even corrected it stays below
nomic on this subset (0.8298 vs 0.8639), so the *ranking* holds and snowflake genuinely
underperforms rather than merely being misconfigured. But **whether its full-run deficit
(-0.0337, p=0.0044) survives the correction is untested** — a +0.016 shift would leave
roughly -0.018, which may well fall inside the confidence interval. Re-running snowflake
over the full 300 queries with its prefix is the honest way to close this; it does not
change the practical recommendation either way.

**Timings above are contaminated**: wave 1's four models embedded concurrently, so their
embed minutes and query latencies are inflated relative to a solo run (nomic solo measured
512 ms, not 918 ms). Quality numbers are unaffected — retrieval is deterministic given a
fixed index.

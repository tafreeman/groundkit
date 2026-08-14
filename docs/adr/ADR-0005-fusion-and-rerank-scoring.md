# ADR-0005 — Hybrid scoring: rank-based RRF fusion and sigmoid-normalized rerank

- **Status:** Accepted (owner, 2026-08-13)
- **Date:** 2026-08-13
- **Deciders:** Andy Freeman (owner)

## Context

Phase 3 combines two retrievers whose scores share no scale, then optionally
reorders the result with a third model whose scores share a scale with neither.

- BM25 scores are unbounded positive, corpus-dependent, and not comparable
  across queries.
- Cosine similarity is bounded `[-1, 1]`, and its distribution depends on the
  embedding model.
- Cross-encoder relevance scores are raw logits — unbounded and routinely
  **negative**.

`RetrievalResult.score` is `Field(ge=0.0)` (`contracts.py`), and SPEC.md §5.2
requires "score ≥ 0 with normalization guaranteed by producers". ADR-0001
hazard 2 records ARP violating exactly this: `CrossEncoderReranker` fed raw
logits straight into the constrained field and crashed. The published
sentence-transformers behaviour makes the magnitude concrete — MS MARCO
cross-encoders return values like `8.607138` and `-4.3200774` for a single
query/document batch. Any design that lets a raw logit reach the contract is
one negative score away from a `ValidationError` in the retrieval path.

Phase 2 also pinned determinism deliberately: BM25 ordering is stable and
`EVAL_CHUNKING_CONFIG` fixes chunk boundaries so two runs are comparable.
Fusion must not reintroduce order instability through unspecified tie-breaking,
or the eval harness stops being able to attribute a delta to the feature under
test.

## Decision

1. **Fuse by rank, not by score — Reciprocal Rank Fusion.** Implemented in
   `retrieval/fusion.py` as pure code, using the original formulation:

   > `RRFscore(d ∈ D) = Σ_{r∈R} 1 / (k + r(d))`

   This is chosen specifically because it needs no score comparability. The
   alternative family the source paper benchmarks against, CombMNZ, "requires
   for each `r` a corresponding scoring function `s_r : D → R`" — that is, score
   fusion is *defined* in terms of scores that can be meaningfully summed. BM25
   and cosine cannot be. RRF consumes only the permutation.

2. **`k` defaults to 60, and the default is documented as weakly-held.**
   `RetrievalConfig.rrf_k` already carries 60. The value comes from the original
   paper, where "k = 60 was fixed during a pilot investigation and not altered
   during subsequent validation", and their pilot "indicated that k = 60 was
   near-optimal, but that the choice was not critical". Elasticsearch ships the
   same constant as its `rrf` retriever's `rank_constant` default. So: 60 is
   well-precedented and poorly-theorised, and this ADR records it as a default
   to measure against, not a tuned value. Any change to it is a baseline change
   under SPEC.md §6 discipline and must be reported as such.

3. **Tie-breaking is total, explicit, and content-derived.** Equal RRF scores
   break by ascending `(content_hash, chunk_index)`. The comparator must never
   depend on dict or set iteration order, and fusion over the same inputs must
   produce a byte-identical ordering across runs and across Python versions —
   asserted by test, in the same spirit as Phase 2's BM25 determinism pinning.

   This decision originally named `chunk_id`, which contradicted the word
   "content-derived" in its own first sentence: `chunk_id` is
   `uuid.uuid4().hex` (`contracts.py`), regenerated on every ingest. Because
   it is persisted, ordering was stable within a given index and across
   process restarts — but the same corpus re-ingested produced a different
   fusion ranking wherever scores tied, which is exactly the "byte-identical
   across runs" property this decision exists to guarantee. RRF ties are not
   rare: scores are rank-derived, so two chunks reached at the same rank in
   one list each collide exactly.

   `content_hash` alone is not a total order — duplicate content shares a
   hash, which is why `index/bm25.py` documents falling back to insertion
   order. `chunk_index` breaks that remaining tie, leaving only genuinely
   identical content at the same position in different documents unordered,
   where the choice is unobservable in the results. Both components are
   content-derived and survive re-ingest, which `chunk_id` does not, and the
   pair now agrees with the tie-break `index/bm25.py` and `index/dense.py`
   already use rather than deliberately diverging from it.

4. **Cross-encoder logits are normalized with a sigmoid, never min-max.**
   `sigmoid: ℝ → (0, 1)` is total, so no logit — however negative — can violate
   `ge=0.0`; that is what closes hazard 2, structurally rather than by clamping.
   It is also **monotonic, so it does not change the ranking**, which is the
   property that matters for a reranker: sentence-transformers documents
   precisely this, noting that loading a `CrossEncoder` with
   `activation_fn=torch.nn.Sigmoid()` yields scores between 0 and 1 and "This
   does not affect the ranking", and that `nn.Sigmoid()` is already the library
   default when `num_labels=1`. groundkit therefore sets the activation
   explicitly rather than inheriting it, so the guarantee is ours and is
   asserted in our tests, not a library default that could change.

5. **RRF output needs no clamping.** With `k > 0` (enforced by
   `RetrievalConfig`) and ranks starting at 1, every term `1/(k + r(d))` is
   strictly positive, so fused scores satisfy `ge=0.0` by construction. No
   defensive `max(0.0, …)` is added — a clamp there would mask a bug rather than
   prevent one.

6. **Fused and reranked scores are documented as intra-response only.** An RRF
   score is a function of the result-set size and the number of contributing
   retrievers; a sigmoid-mapped logit is comparable across queries only for a
   fixed model. Neither is a probability and neither may be compared across
   configurations. `RetrievalConfig.score_threshold` is consequently **not**
   applied to fused scores in Phase 3 — thresholding a rank-derived quantity
   would silently mean something different from thresholding BM25, and SPEC.md
   §6 already fixes abstention as "returned zero results", with no threshold
   anywhere in the harness (ADR-0003 decision 3).

## Alternatives considered

- **Normalize both score lists, then take a weighted sum (convex combination).**
  Rejected: it needs a normalization scheme (min-max over what window?) *and* a
  weight, both corpus- and query-dependent, and tuning either against a
  10-document golden corpus would fit noise. RRF's appeal is explicitly that it
  "requires no tuning". This is a genuine trade — a well-tuned weighted sum can
  beat RRF on a corpus it was tuned for — and the plan's baseline discipline
  leaves the door open to revisit it with a measured delta rather than an
  assertion.
- **Learned fusion / learning-to-rank.** Rejected for v1: it requires training
  data groundkit does not have, and puts a trained model in the retrieval path
  that SPEC.md §2 wants kept deterministic. Notably the source paper found RRF
  "consistently equaled or bettered other methods we tried" and beat the LTR
  methods whose investigation prompted the work, so the unsupervised baseline is
  not an obviously weak starting point.
- **Min-max normalization of cross-encoder scores.** Rejected on three counts:
  it is batch-dependent, so the same document scores differently depending on
  what it was ranked alongside; it makes scores incomparable across queries in a
  way sigmoid does not; and it degenerates on a single-result batch (zero range).
- **Discard rerank scores and keep only the reordering.** Rejected: every
  `RetrievalResult` must carry a score by contract, and Phase 5's cited
  synthesis needs a magnitude, not just an order.
- **Clamp negative logits to zero.** Rejected: it destroys the ordering
  information among all negatively-scored documents, collapsing them to a tie,
  and it satisfies the contract while corrupting the result — the worst
  combination.

## Consequences

- Fusion is pure and dependency-free, so `retrieval/fusion.py` falls under the
  core coverage subset automatically via the `retrieval/*` glob in
  `[tool.groundkit.coverage].core_subset`. It must be tested to that standard
  from its first commit.
- Rerank is optional and its model is a heavyweight extra; the base install and
  the default CI job must not pull it in.
- `rrf_k` becomes a baseline-affecting parameter. It belongs in the eval
  report's `RunConfig` alongside the chunking values Phase 2 pinned, for exactly
  the reason those were pinned: a changed constant with an unchanged corpus hash
  would make two incomparable runs look comparable.
- Declining to threshold fused scores means Phase 3 ships without a confidence
  cutoff on hybrid results. That is a deliberate deferral, not an oversight, and
  belongs in `KNOWN_LIMITATIONS.md`.
- Because sigmoid is monotonic, the rerank stage's *ranking* metrics (recall@k,
  MRR, nDCG) are unaffected by the normalization choice — so the eval delta
  measures the reranker, not the squashing function. That is the point.

## References

- Gordon V. Cormack, Charles L. A. Clarke, Stefan Büttcher, "Reciprocal Rank
  Fusion outperforms Condorcet and individual Rank Learning Methods", SIGIR
  2009, pp. 758–759. [PDF](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)
  · [ACM DL](https://dl.acm.org/doi/10.1145/1571941.1572114) — source of the
  formula, of `k = 60` and its pilot-investigation provenance, of the
  "near-optimal, but … not critical" finding, and of the CombMNZ contrast that
  motivates rank-based fusion.
- [Elasticsearch — Reciprocal rank fusion](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)
  — production corroboration: same formula, `rank_constant` defaulting to 60,
  and the rationale that "the different relevance indicators do not have to be
  related to each other".
- [sentence-transformers — CrossEncoder usage](https://sbert.net/docs/cross_encoder/usage/usage.html)
  and [MS MARCO cross-encoders](https://www.sbert.net/docs/pretrained-models/ce-msmarco.html)
  — MS MARCO models return logits rather than 0–1 scores; sigmoid activation
  maps them to `[0, 1]` and "does not affect the ranking".
- [sentence-transformers — CrossEncoder API](https://sbert.net/docs/package_reference/cross_encoder/cross_encoder.html)
  — `activation_fn` defaulting to `nn.Sigmoid()` when `num_labels=1`.
- [ADR-0001](ADR-0001-promote-vs-rewrite.md) hazard 2 — raw logits into a
  `ge=0.0` contract, the ported defect decision 4 closes.
- [ADR-0003](ADR-0003-eval-corpus-and-metrics.md) decision 3 — threshold-free
  abstention, which decision 6 preserves.

# ADR-0012 — The rerank eval stage reorders the best available upstream stage

- **Status:** Accepted (owner, 2026-08-15)
- **Date:** 2026-08-15
- **Deciders:** Andy Freeman (owner)

## Context

SPEC.md §9 requires every Phase 3 retrieval feature to report an eval delta
against the BM25 baseline. Dense and fusion did, from Wave E. Rerank did not:
`run_eval` accepted no reranker and never appended the `rerank` stage, even
though `rerank` was already a legal `StageName` (`evals/schema.py`) and Wave D
had already landed `retrieval/rerank.py` — `CrossEncoderReranker` behind
`RerankerProtocol`, with ADR-0005 decision 4's sigmoid closing ADR-0001
hazard 2. So rerank was **unmeasured**: nothing in the repo supported a claim
that it improved retrieval on this corpus, and Phase 3 could not be called
done on that basis alone.

The reason this sat open through Wave D rather than being wired in alongside
it is that a reranker doesn't fit the shape the other two stages fit. Dense
and fusion are each a `SearchMode` — a way of producing a ranked candidate
list from a query. A reranker consumes a candidate list that already exists
and returns it reordered (`RerankerProtocol.rerank(query, results, *,
top_k) -> results`). It has no query-independent way to produce results of
its own, so it has no stage of its own to be measured "as" — measuring it
requires deciding what it reorders.

The Phase 3 spec (Wave D) named this as two open decisions, neither to be
settled by omission:

1. Which stage does rerank rerank?
2. Does rerank reach `Retriever.search`?

## Decision

**Decision 1 — the rerank stage reorders the best upstream stage the run
produced:** `fusion` when the run was given a dense pair, `bm25` otherwise.
`_planned_stages` (`evals/runner.py`) reads this off the plan it already
built for the dense stages rather than re-deriving it from the embedder/
vector-store pair — so the stage named in the artifact is, by construction,
the stage the report actually contains, not a second independent answer to
the same question that could disagree with the first.

This was chosen over the alternative of always reranking BM25, and it
carries a real cost that alternative did not:

- Its meaning depends on run configuration. Two reports can agree on
  `corpus_hash`, `judgments_hash`, and every other `RunConfig` field while
  their `rerank` rows describe different experiments — one reranking BM25,
  the other reranking fusion. That is why `RunConfig.rerank_input` is
  mandatory, not documentation: without it, the incomparability between two
  such reports is silent, which is the one outcome this ADR refuses.
- Its delta against `stages[0]` is **not** the reranker's contribution when
  the input was fusion — it sums fusion's gain over BM25 with rerank's gain
  over fusion, one number with nothing separating the two effects.
  `derive_rerank_attribution` (`evals/delta.py`) closes this by diffing the
  `rerank` stage against its own input stage rather than against `stages[0]`,
  and `_print_eval_summary` (`cli.py`) prints both: the baseline delta every
  other stage gets, and the attribution, labelled by `baseline_stage` so a
  renderer needs no special-casing to know which is which.
- It needs a running embedding provider **and** the rerank extra to produce
  the number that matters on a dense run, where always-reranking-BM25 would
  have needed only the extra.

The reason it was still chosen over always-BM25: it measures the pipeline
that would actually be deployed — a caller running hybrid search plus rerank
gets a number about that combination, not about a configuration nobody runs.

**Decision 1a — candidates are over-fetched to `MAX_TOP_K` (50) and
truncated back to `top_k` after reranking.** `_evaluate_judgment` asks the
retriever for `rerank_candidates` results when a reranker is set, not
`top_k`, and the reranker truncates back down after reordering
(`reranker.rerank(query, results, top_k=top_k)`). The sharp point: reranking
truncates *after* reordering, so a candidate depth equal to `top_k` hands the
model a set it can only permute — every `@k` metric at the full cutoff is
then pinned to the input stage's by arithmetic rather than measured, and a
reader seeing `recall@10 +0.000` would read a finding where there was none.
`_print_rerank_provenance` warns exactly on this condition
(`rerank_candidates == top_k`) so the artifact cannot present an arithmetic
inevitability as a result without a flag attached.

`MAX_TOP_K` is chosen as the default depth because it is the retriever's own
structural ceiling (`retrieval/search.py`), not a tuned multiplier picked
against this corpus. It is not exposed as a CLI flag — `grk eval --rerank`
always passes `rerank_candidates=MAX_TOP_K` — so two CLI runs cannot silently
differ on the one dimension that decides whether their `@10` metrics were
free to move at all. `run_eval` keeps `rerank_candidates` as a library
parameter for callers who need a different depth, and `RunConfig
.rerank_candidates` records whatever value actually produced the report,
CLI or not.

**Decision 2 — rerank does NOT reach `Retriever.search`; there is no fourth
`SearchMode`.** No production code changed to accommodate it:
`retrieval/search.py` is untouched, and the eval runner asks the retriever
for the *input* stage's results in the input stage's existing mode, then
reranks what comes back as a step the runner itself performs. Three reasons,
independent of each other:

- The three existing `SearchMode` values name how candidates are
  *produced* — BM25 postings, a dense vector query, RRF fusion of the two.
  A reranker consumes candidates a mode already produced; folding it into
  the same literal would make one `SearchMode` value mean two different
  kinds of thing (a production strategy, and a post-processing step over
  one).
- ADR-0007 turns the default-mode question on abstention: hybrid stays
  recommended-not-default because nothing today makes it abstain on a
  no-answer query. Rerank is a permutation of an existing result list plus a
  truncation — it cannot make a stage abstain that did not already abstain,
  so it answers none of ADR-0007's question and has no business inside the
  literal that question is about.
- `grk search --mode rerank` on a default install could only ever raise
  `RerankerNotConfiguredError` — the rerank extra is deliberately absent
  from the base install and the dev group (`pyproject.toml`). A mode that
  fails by default on every install that doesn't opt into a multi-gigabyte
  extra is a different kind of default than the other three.

This is deferred to Phase 4, where a service boundary makes "rerank this
request" a per-request option on an HTTP or MCP call rather than a mode
baked into `SearchMode`. ADR-0007 is untouched by this decision — its
default stays BM25, for its own reasons, unaffected by anything rerank does.

**Decision 3 — the artifact records `rerank_input`, `rerank_candidates` and
`rerank_model`, all three or none.** `RunConfig`'s model validator
(`_validate_rerank_fields`, `evals/schema.py`) enforces this directly: a
`present` count that is neither 0 nor 3 raises. `rerank_model` extends
ADR-0004's silent-mixing argument one layer above the index — two
cross-encoders are two different measurements of the same stage name, in
exactly the way two embedding models are two different measurements of
`dense`.

`rerank_model` is read via `getattr(reranker, "model_name", None)`
(`_reranker_identity`, `evals/runner.py`), **not** by widening
`RerankerProtocol` with a `model_name` member. That protocol encodes
ADR-0001 hazard 4 and is held to exact signature parity by
`tests/test_protocol_conformance.py`
(`TestRerankerProtocolConformance.test_cross_encoder_reranker_conforms`);
changing a seam to satisfy a reporting need is the wrong trade, and every
other protocol in this repo is held to the same rule. A test double with no
`model_name` attribute falls back to `type(reranker).__name__`, which makes a
stub run self-labelling in the artifact — `"_StubReranker"` where a real run
would show `"cross-encoder/ms-marco-MiniLM-L-6-v2"` — exactly as
`embedding.provider == "inmemory"` self-labels the dense stages.

Half a rerank record is unrepresentable on purpose: a `rerank` stage whose
`RunConfig.rerank_input` is `None` is precisely the silently-incomparable
artifact this ADR exists to prevent, and a `rerank_input` naming a stage with
no `rerank` stage present would claim a measurement the run never made.
Neither reads as valid.

**Decision 4 — the stage's latency includes the rerank call**, timed
separately in `_evaluate_judgment` and *added* to the retrieval latency
rather than replacing it or being tracked apart from it. The cross-encoder
is the dominant cost of the stage — by a wide margin, given it is a
multi-gigabyte model doing a forward pass over `rerank_candidates` pairs per
query — and it is exactly what a reader weighs the quality gain against.
Reporting retrieval time alone would publish the model as very nearly free,
which is the opposite of true and would show up as a suspiciously flat
`latency_p50_delta_ms` on a stage that is, in reality, the slowest one in the
report.

## Alternatives considered

- **Always rerank BM25, regardless of what else the run measured.**
  Rejected: it is simpler and configuration-independent, but it does not
  measure the pipeline anyone would actually deploy. A caller running
  `--mode hybrid` in production and consulting a `rerank` row that reordered
  plain BM25 is reading a number about a configuration they don't run. The
  chosen alternative accepts the configuration-dependence cost explicitly
  (via `rerank_input`) rather than avoiding it by measuring the wrong thing.
- **Truncate candidates to `top_k` before reranking, matching what
  `Retriever.search` would return by default.** Rejected: reranking then has
  nothing to reorder beyond a fixed set, and every `@k` metric at the report
  cutoff becomes an arithmetic identity with the input stage rather than a
  measurement. `MAX_TOP_K` was chosen precisely to keep every metric free to
  move in both directions.
- **A fourth `SearchMode` (`"rerank"`) wired into `Retriever.search`.**
  Rejected per decision 2 — it overloads what a `SearchMode` value means,
  it answers no part of ADR-0007's abstention question, and it makes
  `grk search`'s default path capable of raising `RerankerNotConfiguredError`
  for a mode most installs cannot satisfy.
- **Widen `RerankerProtocol` with a `model_name` member so the identity read
  is type-checked rather than `getattr`.** Rejected per decision 3: the
  protocol is a signature-parity-tested seam encoding a named hazard, and an
  artifact's reporting need is not a reason to change a contract two
  unrelated implementations (`CrossEncoderReranker` and any future backend)
  would then both have to satisfy.
- **Report rerank latency as retrieval time only, with the rerank call timed
  and logged separately but not summed into `latency_p50_ms`.** Rejected per
  decision 4: it would make the stage's headline latency number describe a
  different, cheaper thing than what a caller enabling `--rerank` actually
  pays for every query.

## Consequences

- The measured delta this ADR unblocks is meaningful only under
  `RERANK_GATED=1 uv run pytest tests/test_rerank_gated.py` with `uv sync
  --extra rerank` installed — a stub reranker's delta (an unconfigured
  `_StubReranker` or similar test double) is noise with a sign, the same
  trap `InMemoryEmbedder` sets for the dense stages (SPEC.md §2). A `rerank`
  row produced without a real cross-encoder is structurally valid and
  semantically meaningless, and `rerank_model` is what lets a reader tell
  the two apart without being told out of band.
- torch stays out of the default install and the dev group. `rerank` remains
  an optional `pyproject.toml` extra exactly as it was after Wave D; nothing
  about wiring in the eval stage changes that boundary.
- `.github/workflows/rerank-gated.yml` stays `workflow_dispatch`-only during
  active development, matching `eval-gated.yml`'s rationale: every run pulls
  a multi-gigabyte model, the gate is run locally far more often than a
  schedule would add value during this phase, and a cron firing against a
  moving branch produces results nobody reads. That means this ADR's
  measured delta **re-measures nothing automatically** — it is only as
  current as the last manual dispatch.
- A reranker cannot recover a document the upstream stage never retrieved.
  Truncation to `top_k` happens after reranking, so the reranker can only
  promote a candidate that was already inside the `rerank_candidates`-wide
  list it was handed — the ceiling on any rerank gain is set by the upstream
  stage's own recall at that candidate depth, not by anything the
  cross-encoder does.

## References

- [ADR-0005](ADR-0005-fusion-and-rerank-scoring.md) — decision 4, the
  sigmoid normalization this ADR's stage reports numbers through; this ADR
  does not restate or revisit that scoring decision.
- [ADR-0007](ADR-0007-default-retrieval-mode.md) — decision 3's abstention
  reopening condition is untouched by rerank per decision 2 above: a
  permutation-plus-truncation cannot make a non-abstaining stage abstain.
- [ADR-0004](ADR-0004-embedding-identity-binding.md) — the silent-mixing
  argument decision 3 reuses one layer up: two cross-encoders are two
  different measurements, exactly as two embedding models are.
- [ADR-0001](ADR-0001-promote-vs-rewrite.md) hazard 2 (raw logits into a
  `ge=0.0` contract, closed by ADR-0005) and hazard 4 (the seam this ADR
  declines to widen for a reporting convenience).
- [Phase 3 spec](../specs/phase-3-hybrid-retrieval.md) — Wave D's two open
  decisions this ADR settles, and Wave E's eval harness this stage extends.
- SPEC.md §2 (fail closed; no invented numbers; a hash-derived or
  unconfigured-backend metric is noise, not a measurement), §3 (no job that
  is the sole proof of a backend may be `continue-on-error`), §9 (every
  Phase 3 retrieval feature reports an eval delta against the BM25
  baseline).
- `KNOWN_LIMITATIONS.md` — the rerank-unmeasured entry this ADR's decision
  closes.

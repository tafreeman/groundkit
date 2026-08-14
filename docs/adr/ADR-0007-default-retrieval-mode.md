# ADR-0007 — Hybrid is recommended where configured; BM25 stays the default

- **Status:** Proposed (owner review pending)
- **Date:** 2026-08-14
- **Deciders:** Andy Freeman (owner)

## Context

This settles Q1 in [the Phase 3 spec](../specs/phase-3-hybrid-retrieval.md):
after hybrid retrieval works, does `grk search` default to it, or stay
BM25-only? Q1 was deliberately held open through Waves A–D because SPEC.md §6
makes the measured delta the decider, and committing to a default before a
measurement existed would have been choosing by assertion.

The measurement now exists. A gated run (`EVAL_GATED=1`, `nomic-embed-text`
via local Ollama) over the committed golden corpus found **dense and fusion
both improving on the BM25 baseline with no metric regressing**; fusion
additionally improved recall@10, where dense tied. Values are not restated
here — they live in the generated artifact (SPEC.md §2). Reproduce it:

```bash
uv run grk eval --dense --embed-model nomic-embed-text
```

which writes the full report to `evals/results/latest.json` (gitignored,
regenerated per run). The gated test suite covers the same path via
`EVAL_GATED=1 uv run pytest tests/test_eval_gated.py`, and
`.github/workflows/eval-gated.yml` runs both on demand.

Two facts constrain what may be concluded from that.

**First, the run is one run over a small corpus.** R2 in the Phase 3 spec
records the golden corpus as small enough that a result may not survive corpus
growth. A single measurement is evidence, not proof.

**Second — and decisive — the quality metrics do not measure everything the
default is responsible for.** Two costs sit entirely outside them:

1. **Hybrid cannot abstain at all; dense does not abstain as configured.**
   BM25 abstained on every no-answer judgment; the dense and fusion stages
   abstained on none. The two reach that outcome by different routes, and
   collapsing them would overstate the case:

   - **Hybrid is structural.** ADR-0005 decision 6 excludes rank-derived
     fused scores from `score_threshold` entirely, so *no* configuration of
     the current code makes a hybrid query return nothing on relevance
     grounds. `Retriever.search` calls `_resolve(..., apply_threshold=False)`
     on that branch, and
     `test_impossible_score_threshold_zeroes_bm25_and_dense_but_not_hybrid`
     pins it with a threshold no producer score could clear.
   - **Dense is configurational, not structural.** The dense branch *does*
     apply `score_threshold` (`apply_threshold=True`), and the same test
     proves a high enough threshold zeroes a dense result set. What is
     missing is a *principled value*: the default is `None`, the eval runs
     unthresholded, and nothing in the harness measures abstention quality,
     so any number picked today would be invented — which is why
     `StageResult` refuses a score threshold in the first place.

   The mode under consideration as the default is hybrid, so the structural
   half is the one that binds. The dense half is a gap in configuration and
   evidence rather than in capability, and is recorded separately so a future
   reader does not "fix" a limitation the code does not have.
2. **The default path would require a provider the default install lacks.**
   SPEC.md §10 makes "`grk` works end-to-end locally with zero cloud
   credentials" a v1 definition-of-done criterion. Hybrid needs a running
   embedding model, adds roughly two orders of magnitude of per-query latency,
   and pulls `lancedb` — an optional extra by ADR-0004/Wave A, deliberately
   not a base dependency. A default that fails without Ollama is a different
   product from the one §10 describes.

The tension is real and should be named plainly: on the axis Phase 3 set out
to measure, hybrid won. The argument below is not that the measurement is
wrong or weak — it is that the default is accountable to properties the
measurement does not cover.

## Decision

**`grk search` continues to default to `--mode bm25`. Hybrid is documented as
the recommended mode wherever an embedding provider is configured.**

Specifically:

1. **No behavioural change ships with this ADR.** `SearchMode` keeps its
   `"bm25"` default in `retrieval/search.py` and `cli.py`. This ADR records a
   decision that was previously implicit-pending-data as explicit-and-decided,
   so the next reader does not reopen it from the eval numbers alone.
2. **The recommendation is documented, not defaulted.** README and CLI help
   state that hybrid measured better on the golden corpus *and* that it
   abstains on nothing, so a caller choosing it is choosing that tradeoff
   knowingly. A recommendation a user opts into is honest; a default they
   inherit is not, when the cost is the tool's ability to say it found
   nothing.
3. **Abstention is the reopening condition, not corpus size.** Concretely,
   this decision is revisited when **hybrid** can abstain — a fusion-level
   rule that is not an invented threshold, since ADR-0005 decision 6 rules
   out applying an absolute cutoff to fused scores. A defensible
   `score_threshold` value for the dense path is a related but separate
   prerequisite: the dense mechanism already exists, so what that needs is
   evidence, not code. Both are design and measurement problems about
   abstention specifically. A larger corpus or a bigger quality delta
   resolves neither, and must not be mistaken for doing so.
4. **A second reopening condition, independent of the first:** if a future
   phase makes an embedding provider a base dependency rather than an extra,
   objection 2 lapses and only objection 1 remains. Recorded so the two are
   not conflated later.

## Alternatives considered

- **Default to hybrid on the strength of the measured delta.** Rejected on
  abstention. groundkit's stated purpose is grounded, citation-verifiable
  retrieval; a default mode that returns its `top_k` nearest neighbours for a
  question the corpus cannot answer produces confident, well-cited, wrong
  results — the exact failure the no-answer judgments exist to detect. It
  would also break SPEC.md §10 by making the default path require Ollama.
- **Default to hybrid when a provider is configured, BM25 otherwise.** The
  most tempting option, and rejected the most deliberately: it makes the
  retrieval strategy — and therefore the results, the scores, and whether
  abstention is possible at all — depend on ambient environment rather than
  on an explicit flag. Two users running the same command against the same
  index would get different answers with nothing in the output explaining
  why. That is the "silently different behaviour from ambient config" shape
  SPEC.md §2 rules out elsewhere, and it is worse here than a wrong default,
  because it is not reproducible.
- **Give hybrid an abstention rule, then default to it.** Rejected *for now*,
  and it is the right long-term direction (see decision 3). Note what is and
  is not missing: the dense path already applies `score_threshold`, so the
  mechanism exists there and only a defensible value is absent. Hybrid is the
  harder case — ADR-0005 decision 6 withholds thresholding from fused scores
  deliberately, because an absolute cutoff against a rank-derived quantity
  measures nothing, so hybrid needs a genuinely different rule rather than
  the dense one extended. Rejected here because either would have to be
  invented today: nothing in the current eval measures abstention quality,
  and `StageResult` already refuses a score threshold for exactly this reason
  — "any fixed cutoff would report noise from an arbitrary number rather than
  a real measurement". Picking one to justify a default reverses the order
  SPEC.md §6 requires.
- **Leave Q1 open until more corpus data exists.** Rejected: Q1 has been open
  since Wave A pending a measurement, that measurement now exists, and the
  blocking objection is a design gap rather than a data gap. Leaving it open
  would misrepresent *why* it is unresolved and invite the wrong fix — growing
  the corpus until the number is convincing, which R2 explicitly warns against.

## Consequences

- **The Phase 3 eval delta is a real result that changes no default.** That is
  an acceptable and honest outcome, and worth stating because the opposite
  expectation is natural: "harness before features" (SPEC.md §8) exists so
  features report their delta, not so a positive delta automatically ships.
- **`no_answer_abstained_count` becomes the metric to watch.** It is already
  reported per stage and deliberately excluded from the derived delta
  (`evals/delta.py`) because abstention is not comparable across stages. This
  ADR makes it the gating property for a future default change, so it moves
  from an observation to a requirement.
- **`grk search --mode hybrid` stays a first-class, documented path**, not a
  hedge. The measurement supports recommending it, and Wave F's docs say so
  with the tradeoff attached.
- **Phase 4's service inherits the same default**, and inherits this reasoning
  with it: an HTTP or MCP caller gets BM25 unless it asks otherwise, so the
  MCP surface cannot silently become non-abstaining.
- **This ADR will look over-cautious if abstention is later solved cheaply.**
  Recorded deliberately: the cost of being wrong in this direction is a slower
  default; the cost of being wrong in the other is a retrieval tool that
  confidently cites something for every question ever asked of it.

## References

- [Phase 3 spec](../specs/phase-3-hybrid-retrieval.md) — Q1 (this decision),
  R1 (the measurement mechanism), R2 (corpus size caveat).
- [ADR-0005](ADR-0005-fusion-and-rerank-scoring.md) — decision 6 excludes
  fused scores from `score_threshold`, which is why fusion cannot abstain by
  thresholding today.
- [ADR-0004](ADR-0004-embedding-identity-binding.md) — LanceDB and the
  embedding providers as optional extras, not base dependencies.
- `KNOWN_LIMITATIONS.md` — the dense/fusion non-abstention entry.
- SPEC.md §2 (fail closed; no invented numbers), §6 (baseline discipline; the
  measured delta decides), §10 (zero cloud credentials as a v1 done criterion).

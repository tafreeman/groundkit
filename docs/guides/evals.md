# Eval harness

This page is for anyone who needs to know whether a change to groundkit's
search actually made results better — or who is deciding whether to trust a
claim that it did. An eval harness (short for "evaluation harness") is a
fixed, checked-in set of test questions with known-correct answers, scored
automatically; it is what turns "I think this is better" into a number
anyone can reproduce and check. By the end of this page you will know what
groundkit measures, how to regenerate any number it reports yourself, and
which of its checks are advisory only — meaning they report a result but
must never be mistaken for a pass/fail gate.

Retrieval quality is a measurement here, not a claim. The harness landed in
Phase 2 — before hybrid retrieval and before rerank — so that every retrieval
feature since has arrived with a delta (the measured difference against a
baseline result, positive or negative) rather than an argument for itself.

```bash
uv run grk eval
```

Offline, credential-free, and reproducible from a clean clone in two commands.
It builds a throwaway index over the committed golden corpus, scores it with
the same deterministic BM25 path `grk search` uses, and writes a full report
to `evals/results/latest.json`.

That path is gitignored. Reports are regenerated, never committed — a report
in git is a number that was true once (SPEC.md §2).

## What is measured

- **recall@k** at k = 1, 5, 10 — of the top *k* results returned, did at
  least one of them contain a correct answer?
- **MRR** (Mean Reciprocal Rank) — on average, how near the top of the
  results list did the first correct answer land?
- **nDCG@10** — a single score that rewards a correct result more for
  ranking higher, not just for being present somewhere in the top 10.
- **Latency percentiles per stage** — how long a query takes, stage by
  stage: BM25 (keyword matching), dense (meaning-based matching), fusion
  (merging the two), rerank (the optional accuracy pass over the top few).

The metric implementations are deterministic code with their own unit tests.
[ADR-0003](../adr/ADR-0003-eval-corpus-and-metrics.md) records the corpus and
metric design: quote-anchored judgments, hit-rate recall, threshold-free
abstention, JSONL.

## The corpus

`evals/corpus/` plus `evals/judgments.jsonl`, both committed, authored against
the contract in `evals/README.md`. It includes ambiguous cases, no-answer
cases, and adversarial ones — prompt-injection text (planted wording that
tries to trick an AI reader into treating a document's content as an
instruction to follow, rather than text to search over) planted in documents
that must never surface as instructions.

Corpus integrity is checked by the normal offline CI job: schema validity,
unique IDs, category coverage, and a size floor. **The floor is asserted in
the test, not in a document** — the test is the authoritative number, so prose
that drifts cannot weaken the gate.

## Baseline discipline

BM25-only is the baseline. Every retrieval feature reports its delta against
it in the generated report, and a feature that does not beat baseline is
reported as not beating baseline. That is the whole discipline: the report is
allowed to say the feature did not help.

To measure the dense and hybrid paths you need a real embedding model:

```bash
uv run grk eval --dense --embed-model nomic-embed-text
```

!!! warning "The in-memory embedder is a labelled test double"

    `InMemoryEmbedder` is a test double — a stand-in used in place of the
    real embedding model so retrieval plumbing can be exercised
    deterministically offline. It hash-expands text into vectors and has
    **zero semantic signal**: it cannot tell "car" from "banana," only that
    they're different strings. **A retrieval-quality number produced with it
    is noise formatted as a number** — not a smaller or rougher measurement
    of quality, but not a measurement of quality at all — which is why the
    runner warns on it and the CLI stamps a caveat onto any report generated
    with it.

## Rerank, synthesis, and the judge

Three more flags extend the same report:

- **`--rerank`** adds a cross-encoder reranking stage on top of whichever
  retrieval stage precedes it, scored the same way `--dense` is: as a delta
  against the BM25 baseline within one report.
- **`--synthesis`** runs the planted-marker citation-echo check (SPEC.md
  §2): a known snippet is planted in a source document, a chat model is
  asked a question that should cite it, and the check confirms the model's
  citation actually echoes that planted text rather than a plausible-looking
  but wrong one. This runs against a real chat provider and writes its own
  artifact (`evals/results/echo-latest.json`) — there is deliberately no
  offline double for it, because an echo number from a scripted provider
  would be noise presented as a measurement, exactly like a hash-derived
  dense score.
- **`--judge`** (requires `--synthesis`) synthesizes an answer for every
  golden-corpus query against the run's best available retrieval stage and
  runs the advisory faithfulness judge — a second model call that grades
  whether the answer's claims are actually backed by its citations, detailed
  further down this page — over each one, folding the outcome counts into
  the report's `synthesis` field.

```bash
uv run grk eval --rerank --rerank-model <cross-encoder-model>
uv run grk eval --synthesis --judge --chat-model <chat-model>
```

## The gated workflows

Two paths cannot be proved by an offline job — one that runs automatically on
every change, using only what is already on the CI machine — and each
instead has its own **gated** workflow: one that runs on a weekly schedule
and on an opt-in PR label, deliberately not on every change, because of what
it costs to run (below):

| Workflow | Proves | Why it is separate |
|---|---|---|
| `eval-gated.yml` | The real-model dense/fusion delta, and — now provisioning a chat model from the same Ollama service — the `--synthesis --judge` echo check and faithfulness judge | Needs a running Ollama, and pulls both an embedding model and a chat model |
| `rerank-gated.yml` | That a real cross-encoder emits raw unbounded logits — the premise the sigmoid exists for | Pulls torch, a multi-gigabyte install CI's default job must never carry |

Neither is `continue-on-error`. SPEC.md §3 forbids that for any job that is
the sole proof of a backend, and each of these is exactly that. `rerank-gated`
additionally fails if its tests *skip* — a gate that silently skips is
indistinguishable from one that passed.

## The faithfulness judge

"LLM-as-judge" means using a second AI model to grade the first one's answer
instead of a human reviewing it — it scales in a way human review does not,
but the grader can itself be wrong, which is precisely why the result below
is advisory rather than something that can block a change. Synthesis mode
(Phase 5) ships an LLM-as-judge faithfulness check — does the answer's
content actually match what its citations say? — with a schema-validated
verdict and an injectable model call, so unit tests never touch the network.
`grk eval --judge` runs it across the whole golden corpus; `grk answer
--judge` runs the identical judge over one query outside the eval harness.

**It is advisory only — it exits 0 and gates nothing — until it has been
calibrated against human labels.** The calibration procedure required to
ever let it gate is documented as part of that work. An uncalibrated judge
that blocks a merge is a coin flip with authority.

## Next

[Retrieval modes](retrieval-modes.md) is where the numbers this harness
produces actually get used to decide something, including the hybrid-vs-BM25
result referenced above. [Installation](../getting-started/installation.md)
covers the `dense` and `rerank` extras that `--dense` and `--rerank` need.

# Eval harness

Retrieval quality is a measurement here, not a claim. The harness landed in
Phase 2 — before hybrid retrieval and before rerank — so that every retrieval
feature since has arrived with a delta against a baseline rather than an
argument for itself.

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

- **recall@k** at k = 1, 5, 10
- **MRR**
- **nDCG@10**
- **Latency percentiles per stage** — BM25, dense, fusion, rerank

The metric implementations are deterministic code with their own unit tests.
[ADR-0003](../adr/ADR-0003-eval-corpus-and-metrics.md) records the corpus and
metric design: quote-anchored judgments, hit-rate recall, threshold-free
abstention, JSONL.

## The corpus

`evals/corpus/` plus `evals/judgments.jsonl`, both committed, authored against
the contract in `evals/README.md`. It includes ambiguous cases, no-answer
cases, and adversarial ones — prompt-injection text planted in documents that
must never surface as instructions.

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

    `InMemoryEmbedder` hash-expands text into vectors and has **zero semantic
    signal**. It exists so retrieval plumbing can be exercised deterministically
    offline. A retrieval-quality number produced with it is noise formatted as
    a number, which is why the runner warns on it and the CLI stamps a caveat
    onto any report generated with it.

## Rerank, synthesis, and the judge

Three more flags extend the same report:

- **`--rerank`** adds a cross-encoder reranking stage on top of whichever
  retrieval stage precedes it, scored the same way `--dense` is: as a delta
  against the BM25 baseline within one report.
- **`--synthesis`** runs the planted-marker citation-echo check (SPEC.md §2)
  against a real chat provider and writes its own artifact
  (`evals/results/echo-latest.json`) — there is deliberately no offline
  double for it, because an echo number from a scripted provider would be
  noise presented as a measurement, exactly like a hash-derived dense score.
- **`--judge`** (requires `--synthesis`) synthesizes an answer for every
  golden-corpus query against the run's best available retrieval stage and
  runs the advisory faithfulness judge over each one, folding the outcome
  counts into the report's `synthesis` field.

```bash
uv run grk eval --rerank --rerank-model <cross-encoder-model>
uv run grk eval --synthesis --judge --chat-model <chat-model>
```

## The gated workflows

Two paths cannot be proved by an offline job, and each has its own workflow
that runs weekly and on an opt-in PR label:

| Workflow | Proves | Why it is separate |
|---|---|---|
| `eval-gated.yml` | The real-model dense/fusion delta, and — now provisioning a chat model from the same Ollama service — the `--synthesis --judge` echo check and faithfulness judge | Needs a running Ollama, and pulls both an embedding model and a chat model |
| `rerank-gated.yml` | That a real cross-encoder emits raw unbounded logits — the premise the sigmoid exists for | Pulls torch, a multi-gigabyte install CI's default job must never carry |

Neither is `continue-on-error`. SPEC.md §3 forbids that for any job that is
the sole proof of a backend, and each of these is exactly that. `rerank-gated`
additionally fails if its tests *skip* — a gate that silently skips is
indistinguishable from one that passed.

## The faithfulness judge

Synthesis mode (Phase 5) ships an LLM-as-judge faithfulness check with a
schema-validated verdict and an injectable model call, so unit tests never
touch the network. `grk eval --judge` runs it across the whole golden corpus;
`grk answer --judge` runs the identical judge over one query outside the eval
harness.

It is **advisory only — it exits 0 and gates nothing** — until it has been
calibrated against human labels. The calibration procedure required to ever
let it gate is documented as part of that work. An uncalibrated judge that
blocks a merge is a coin flip with authority.

# Eval harness

This page is for anyone who needs to know whether a change to groundkit's
search actually made results better — or who is deciding whether to trust a
claim that it did. An eval harness (short for "evaluation harness") is a
fixed, checked-in set of test questions with known-correct answers, scored
automatically; it is what turns "I think this is better" into a number
anyone can reproduce and check. By the end of this page you will know what
groundkit measures, how to regenerate any number it reports yourself, how far
one of those numbers can honestly be carried, and which of its checks are
advisory only — meaning they report a result but must never be mistaken for a
pass/fail gate.

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

Two directories in the repository do hold committed artifacts —
`evals/perf/` and `evals/beir/` — and they are a narrow, argued exception
rather than a hole in that rule. Each is dated evidence for one specific
argument: a performance baseline a refactor is judged against, and a single
external-benchmark validation run. For those, comparing across runs is the
whole point of the artifact rather than the hazard, and evidence that exists
only on one machine is not evidence. Neither is a live report, and no number
from either is quoted on this site or in the README — they are read by
opening the file, beside the commit and corpus hash it was generated from.

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

## Running the harness over another corpus

The golden corpus is the default, not the only thing `grk eval` can score:
`--corpus-dir` and `--judgments` point it at any directory of documents plus
a judgments file that satisfies the contract in `evals/README.md`.

`groundkit.evals.beir` converts a **BEIR** dataset — the standard public
collection of retrieval benchmarks, each shipping documents, queries and
relevance labels — into exactly that shape:

```python
from pathlib import Path

from groundkit.evals.beir import adapt_beir_dataset

report = adapt_beir_dataset(Path("./scifact"), Path("./scifact-gk"))
```

It writes `<output>/corpus/` and `<output>/judgments.jsonl`, which `grk eval`
then scores through the same deterministic path the golden corpus goes
through:

```bash
uv run grk eval \
  --corpus-dir ./scifact-gk/corpus \
  --judgments ./scifact-gk/judgments.jsonl
```

The adapter downloads nothing — you supply a BEIR directory you already have —
and it is a library function with no `grk` verb, deliberately: adapting a
dataset *writes* a corpus, and the CLI's eval surface reads one.

Three mismatches it resolves in the open rather than papering over:

- **BEIR relevance is per document; a groundkit judgment is a verbatim
  quote.** Each adapted gold quote is set to the document's *entire* text, so
  the harness resolves it at offset zero and counts every chunk of that
  document relevant. That is document-level relevance expressed in the span
  vocabulary — the quote really is a verbatim substring, simply the maximal
  one — not an invention of span annotations BEIR does not provide.
- **BEIR has no judgment categories**, so every adapted row is written
  `normal` and the report's category breakdown collapses to a single bucket.
  The golden corpus's no-answer, ambiguous and adversarial coverage does not
  come along with the dataset; `no_answer` would need queries with no relevant
  document, and BEIR's qrels assert relevance on every row.
- **A BEIR corpus is an untrusted third-party download.** Every document id,
  query id and split name is validated as a single path component before it
  reaches a path join, each written path is containment-checked against the
  output directory as a second independent barrier, and the whole dataset is
  validated *before* a byte is written — so a dataset the harness would later
  reject leaves nothing behind. Adapting into a corpus directory that already
  holds files is refused rather than silently unioning two datasets into one
  corpus scored against judgments that never mentioned half of it.

!!! warning "A chunk-level score is not a document-level score"

    BEIR's published numbers rank **documents**. `grk eval` ranks **chunks**,
    and on an adapted set it computes the ideal ranking over every gold chunk
    — so retrieving the right document and returning one chunk of it is scored
    as though the rest were missed. The resulting figure is a valid
    measurement of groundkit against itself across configurations, and it is
    **not** comparable to a published BEIR number. Making the two mean the
    same thing requires collapsing the chunk ranking to first-seen distinct
    documents before scoring, which the harness deliberately does not do for
    you: it would be a second scoring mode whose output looks identical to the
    first.

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

## Is a delta real?

A delta is the difference between two point estimates over one fixed set of
queries. Over a few dozen of them, a small delta and "which queries happen to
be in the set" are not distinguishable by looking at the number, and the
report does not claim otherwise — it reports the difference it measured.

`groundkit.evals.significance` is what separates the two. It runs a **paired
bootstrap**: resample the queries with replacement, recompute the
candidate-minus-baseline difference on each resample, and read an interval off
the resulting distribution.

```python
from groundkit.evals.significance import compare_report_stages

result = compare_report_stages(report, baseline="bm25", candidate="fusion", metric="ndcg_at_10")
result.significant  # does the 95% confidence interval exclude zero?
```

Four properties of it are decisions rather than implementation details:

- **The unit of resampling is a query, not an individual retrieved hit**, and
  pairing is checked by query id — a missing or reordered query raises rather
  than quietly becoming an unpaired comparison.
- **`significant` is decided by the confidence interval, never by the
  p-value.** The two are computed independently, and the interval is the
  verdict.
- **The p-value is an achieved significance level**, in the Smucker/Allan/
  Carterette sense: the resamples are drawn from the observed deltas, not from
  a null-centred distribution, so it is not a null-hypothesis p-value and is
  not described as one. It carries a `(r + 1) / (B + 1)` finite-sample
  correction and therefore has a floor — no finite resampling can support a
  reported zero, and printing one would be a precision claim the method cannot
  make.
- **A result is reproducible and order-independent.** The generator is seeded
  and built per call, so a comparison does not depend on what was compared
  before it.

`compare_report_stages` reads two stages out of one eval artifact and refuses
a report containing two stages of the same name, since such a comparison
cannot say which of them it measured. `paired_bootstrap` takes two
`{query_id: score}` mappings directly, for comparing systems that are not two
stages of one report.

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
[Eval harness reference](../reference/evals.md) documents the two modules on
this page that are called from Python rather than through `grk eval`.

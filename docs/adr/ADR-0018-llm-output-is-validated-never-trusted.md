# ADR-0018 — LLM output is validated, never trusted: cited synthesis, the echo check, and an advisory judge

- **Status:** Accepted
- **Date:** 2026-08-15
- **Deciders:** Andy Freeman (owner)

## Context

groundkit's product claim is that citations are verifiable: every returned
passage carries a document id, a chunk id, and character offsets resolvable to
source, and `retrieval/citations.py` resolves one by re-reading the span and
comparing (SPEC.md §2, §5.2). That claim survives Phase 4 because nothing in the
system generates text — every string a caller sees came out of a file.

Phase 5 breaks that property on purpose. A synthesized answer is text no
document contains, and SPEC.md §2 bounds it with one sentence: *"Synthesis may
cite only retrieved spans; the eval harness includes a planted-marker check for
citation echo."* SPEC.md §6 adds the judge: LLM-as-judge, schema-validated
verdict, injectable model call so unit tests never touch the network, **advisory
only — exits 0, gates nothing — until calibrated against human labels**, with
the calibration procedure documented.

Those three obligations look separate and are one decision. Each is a rule about
the same thing: an LLM's output is an *input to validation*, never a result. The
alternative in each case is the same temptation in a different costume — accept a
citation the model asserts, repair a malformed answer, threshold an uncalibrated
verdict — and each would produce something that looks exactly like a working
system.

Two prior decisions constrain the shape. `EvalReport.schema_version` is
`Literal[1]` and `StageResult.stage` is the four-member retrieval `Literal` Phase
3 settled; `evals/schema.py`'s docstring licenses additive-with-default fields
only while `evals/results/` stays gitignored, and says the next field after that
stops being true is a version bump. And `RunConfig` is documented as *"a record
of the retrieval settings in effect"* — synthesis is not one.

## Decision

### 1. The model cites integers; groundkit constructs the citation

The synthesis prompt presents retrieved results as a numbered source list and
asks for `[n]` markers. `Synthesizer` resolves each `n` to `results[n-1]` and
builds the `Citation` from that `RetrievalResult`'s own fields.

**The model never supplies a document id, a source path, or an offset.** It
supplies an integer into a list groundkit built. That is what makes "synthesis
may cite only retrieved spans" structurally true rather than a model promise: a
citation to something that was not retrieved is not a bad answer, it is
unrepresentable. Every emitted `Citation` therefore resolves through the same
`retrieval/citations.py` path a `grk search` hit does, and is exactly as
verifiable.

Marker parsing is anchored to digits only, `\[(\d+)\]`. This is not a style
choice: the redaction token format is `[<UPPERCASE_NAME>_<n>]` and
`RedactingChat.restore()` (ADR-0017 decision 4) runs *before* markers are parsed,
so a restored answer can contain both spellings, and a looser pattern would read
`[PERSON_1]` as citation 1 of something. A regression test feeds an answer
containing a redaction-shaped token and asserts it is not parsed as a marker;
per SPEC.md §8 it must be shown to fail against the looser pattern first.

### 2. Every violation is a rejection, once — no repair, no retry

`SynthesisError` on an empty result set and on an out-of-range `[n]` (an
answer with no markers at all is *not* a violation — see the abstention
paragraph below). `QueryRewriteError` on a blank rewrite.
`JudgeError` on a verdict that fails to parse or fails schema validation. One
attempt each.

A repair pass ("your citation was out of range, try again") is the coercion
SPEC.md §2 bans wearing a helpful expression: it converts a model that cannot
follow the contract into one that appears to, and makes latency and cost
unbounded in a way nothing measures. Falling back to the original query on a
blank rewrite is the same defect `RerankerNotConfiguredError` exists to prevent —
a rewriter that quietly returns what it was given is indistinguishable from one
that worked, so a misconfigured run publishes rewrite-enabled numbers that are
really the plain query's.

Model-side abstention is representable, and it is the empty tuple: an answer
whose completion carries **no** markers is a valid `SynthesizedAnswer` with
`citations == ()` — the model saying "the sources do not answer this" in prose
is an honest outcome, not a contract violation. The citation contract is
set-membership over what *is* cited, never an obligation to cite. There is
deliberately no `abstained: bool` alongside it (settled, closing the question
the first draft of this ADR left open): a boolean that must always equal
`len(citations) == 0` is stored derived state that can disagree with its own
input — the same redundancy this repo already refused when it declined a
citation checksum and declined storing eval deltas. The upstream refusal still
stands separately: an empty *result set* is a `SynthesisError` before the
model is ever called, because synthesis with nothing citable is not an
abstention, it is an invocation error.

### 3. The verdict is a boolean and prose. There is no score

`FaithfulnessVerdict(faithful: bool, unsupported_claims: tuple[str, ...],
reasoning: str)`, frozen, `extra="forbid"`, parsed from the model's JSON and
validated — never coerced, never partially accepted.

A 0–1 faithfulness score from an uncalibrated judge is precisely the invented
number SPEC.md §2 forbids, and it is worse than a label because a number invites
a threshold, a threshold invites a gate, and the gate would rest on nothing. The
verdict says only what it can support: whether unsupported claims were found, and
which. `unsupported_claims` is the field that makes a verdict actionable to a
human; `faithful` is the field that must not be aggregated into an average and
reported as a quality metric.

### 4. The planted-marker echo check is two-sided, runs over its own corpus, and writes its own artifact

The check builds a **synthetic marker corpus** in a temp directory. Markers are
high-entropy and generated per run (`GK-ECHO-<hex>`), so no model can have
memorized them and no committed file changes. Three markers per case (one more
than this ADR's first draft, which presented only the positive and left "cites
a real but wrong source" unrepresentable): a **positive** document whose
answer-bearing passage carries a marker, a lexically similar **decoy** document
carrying a different marker — both presented in the context — and an
**absent** marker belonging to no document at all, never presented anywhere.

- **Positive side.** The answer's resolved citations, re-read from source through
  `retrieval/citations.py`, must contain the marker of the passage the answer
  cites — and citing the presented decoy instead is scored as a wrong-source
  failure, not a pass. This is what distinguishes a citation that points at
  real source text from one that points at a plausible span.
- **Negative side.** The absent marker may appear in neither the answer text
  nor any citation. A hit means prompt assembly leaked, or the model produced
  content it was not given.

It writes `evals/results/echo-latest.json` with its own `schema_version` rather
than nesting inside `EvalReport`. The reason is not tidiness: markers must be
generated per run, so planting them in `evals/corpus/` would change `corpus_hash`
on every run and pollute the retrieval baseline every later phase reports its
delta against. The check therefore runs over a *different corpus*, and nesting
its results inside a report keyed on the golden corpus's hash would assert a
shared provenance that does not exist.

**The checker is proved offline; the model is not.** A scripted chat double can
be scripted to leak, so the default suite proves the negative side actually
catches a leak — shown to fail first against a checker that does not look, per
SPEC.md §8. What a real model does is measured only under `SYNTHESIS_GATED=1`.

### 5. The judge is advisory, and this is the procedure that could ever change that

The judge exits 0. There is no `--fail-on-unfaithful`, no non-zero exit path
derived from a verdict, and no CI job that can redden on one. A verdict is a
second model's opinion; until it has been measured against human labels it is not
evidence about faithfulness, and reading it as a quality signal is the misuse the
advisory label exists to prevent.

Making it gate requires all of the following, in order, and a superseding ADR:

1. **A committed human-labelled set.** Answers labelled against the same rubric
   the judge is given, by a human, stored in its own file with its own schema and
   its own integrity test — the golden corpus's discipline applied to labels.
2. **Agreement measured by deterministic code**, in the leaf style of
   `evals/metrics.py`: agreement rate and per-class precision and recall against
   the human labels, computed by pure functions with their own unit tests, with
   no LLM anywhere in the measurement.
3. **A pre-registered threshold.** The agreement level that would license gating
   is written down *before* the measurement is run. A threshold chosen after
   seeing the number is a description of that number.
4. **Pinned judge identity and pinned prompt.** The verdict artifact records the
   judge's `(provider, model_name)` and the hash of the prompt template. Changing
   either invalidates the calibration, for ADR-0004's reason one layer up: two
   judges are two different instruments, and so are two prompts.
5. **A disagreement analysis, not just a number.** Which class the judge fails
   on decides what a gate would actually enforce. A judge with high agreement
   that fails only on the unfaithful class would gate nothing while looking
   calibrated.

Two things explicitly do **not** advance this: enlarging the label set until
agreement passes, and re-running the judge until a better number appears. Both
are the reflex Phase 3's R2 already names — fitting the benchmark to the result —
and they are named here so that the first person who wants a gate finds the
refusal already written.

### 6. Synthesis results are one additive field on `EvalReport`, and the licence's precondition becomes a test

`EvalReport` gains `synthesis: SynthesisReport | None = None`.
`schema_version` stays `Literal[1]`. `StageResult.stage` is untouched — synthesis
is not a ranking, produces no `@k` metrics, and has no delta against a BM25
baseline, so making it a stage would put a row into a table whose every column is
undefined for it. `RunConfig` is untouched too: it records retrieval settings,
and the synthesis and judge identities belong to `SynthesisReport`.

`SynthesisReport` carries the input stage, both model identities, the prompt
hashes, whether redaction was applied, per-query outcome counts — a three-way
`answered` / `abstained` / `rejected` split, because decision 2 makes an
empty-citations abstention a distinct, valid outcome that collapsing into
either neighbour would misreport — and the verdict counts. A per-query
`SynthesisError` is recorded as `rejected` rather than raised out of the run —
discarding an answer is not coercion, and a scripted-model run reporting many
rejections is information.

This is the **third** reliance on `evals/schema.py`'s additive-with-default
licence, whose stated precondition is that no artifact outlives the code that
reads it, which holds only because `evals/results/` is gitignored. Phase 5 stops
remembering that and starts checking it: a test asserts `evals/results/` is still
ignored by `.gitignore`. When that assertion fails, the next field is a
`schema_version` bump, and the test is what says so instead of a reader
remembering a docstring.

## Alternatives considered

- **Let the model emit chunk ids or offsets directly.** Rejected per decision 1:
  it makes an out-of-set citation representable, so the guarantee degrades from
  structural to checked — and a check can be relaxed by a later contributor who
  finds it inconvenient.
- **One repair round trip on a malformed answer.** Rejected per decision 2. It is
  the difference between "this model can produce cited answers" and "this model
  can produce cited answers when told twice", and only the artifact would know
  which.
- **Fall back to the original query when a rewrite is blank.** Rejected: it makes
  a rewrite-enabled run indistinguishable from a rewrite-disabled one, which is
  the exact failure `RerankerNotConfiguredError` was created for.
- **A numeric faithfulness score, or a three-class label.** Rejected per decision
  3: a score is an uncalibrated number, and a middle class ("partially
  supported") is a bucket whose boundary nothing defines, which would be the same
  invented threshold with a name instead of a value.
- **Plant markers in the golden corpus** so the echo check rides along with
  `grk eval`. Rejected per decision 4: per-run markers change `corpus_hash` on
  every run, and committed markers can be memorized and stop testing anything.
- **Nest the echo result inside `EvalReport`.** Rejected: it would claim a shared
  `corpus_hash` provenance the two runs do not have.
- **Add a `synthesis` stage to `StageResult.stage`.** Rejected per decision 6 —
  every column of a stage row is undefined for synthesis, and the baseline-delta
  machinery would produce a number by subtracting things that do not exist.
- **Bump `schema_version` to 2.** Rejected for now, and the rejection is
  conditional rather than permanent: the licence still holds, and decision 6
  converts its precondition from a convention into an assertion so the next
  author is told rather than trusted.
- **Ship the judge as a gate from the start**, on the reasoning that an
  unfaithful answer is a real defect. Rejected: SPEC.md §6 says advisory, and a
  gate whose sensitivity nobody has measured fails builds for reasons nobody can
  reproduce — after which it is disabled, which is worse than never having had it.

## Consequences

- Every citation groundkit emits — retrieved or synthesized — resolves through
  one code path, so the verifiability claim needs no asterisk for synthesis.
- `grk answer` fails rather than degrades in several ordinary situations:
  no results, a model that will not cite, a model whose JSON is malformed. That
  is the intended behaviour and it will read as brittleness to a first-time user;
  the docs have to say why rather than apologize.
- The judge produces labels nobody may aggregate into a quality claim until
  decision 5's procedure has been run. Nothing in the code prevents a reader from
  doing it anyway, which is why the constraint is written in the artifact's
  vocabulary (model identity, prompt hash) rather than only in prose.
- The default test suite proves the *validators*, never the *model*. That is a
  stronger position than it sounds — the validators are the security property —
  but it means a green suite says nothing about answer quality, and
  `KNOWN_LIMITATIONS.md` says so.
- Two artifacts now exist under `evals/results/`, with two schemas and two
  reasons to exist. A reader correlating them has to know they describe two
  corpora; the echo artifact's own schema states it.
- The marker/token collision (decision 1) is a hazard created by composing two
  independently reasonable formats. It is closed by one regex and one regression
  test, and it is recorded because the next format added at this boundary will
  have the same question to answer.

## References

- SPEC.md §2 (cite only retrieved spans; planted-marker echo check; schema
  rejection, never coercion; real data only), §6 (the judge's advisory status and
  the calibration obligation), §8 (a regression test must be shown to fail first).
- [ADR-0003](ADR-0003-eval-corpus-and-metrics.md) — the corpus and judgment
  discipline decision 5's label set is asked to match.
- [ADR-0004](ADR-0004-embedding-identity-binding.md) — the identity argument
  decision 5 applies to judges and prompts.
- [ADR-0012](ADR-0012-rerank-eval-stage-reorders-upstream-stage.md) — the
  configuration-dependent-row problem `SynthesisReport`'s recorded identities
  answer for synthesis.
- [ADR-0017](ADR-0017-chat-seam-and-redaction-boundary.md) — the seam these rules
  validate the output of, and the `restore()` ordering behind decision 1's regex.
- `src/groundkit/evals/schema.py` — the additive-with-default licence and its
  precondition.
- `docs/specs/phase-5-boundary-features.md` §6, §8, §11.

# Phase 5 — Boundary features

Feature spec for SPEC.md §9 Phase 5: *optional query rewrite + cited synthesis;
redaction pass (names → tokens, configurable patterns); advisory faithfulness
judge.*

Status: **planned**. Nothing here overrides SPEC.md; where this document makes a
decision SPEC.md does not already contain, it names the ADR that carries it.
Three ADRs are proposed —
[ADR-0017](../adr/ADR-0017-chat-seam-and-redaction-boundary.md),
[ADR-0018](../adr/ADR-0018-llm-output-is-validated-never-trusted.md),
[ADR-0019](../adr/ADR-0019-grk-answer-and-no-synthesis-on-the-service-surface.md).

Phase 5 is the first phase in which groundkit sends a prompt to a generative
model. Everything before it — ingestion, chunking, indexing, BM25, dense, RRF,
rerank, citation resolution, the whole service surface — is deterministic typed
code, and stays that way. The organizing constraint is therefore not "add an
LLM" but **keep the LLM at the edge of a system that already works without it**:
every Phase 5 feature is opt-in, every one of them is skippable, and no existing
command changes behaviour when none of them is used.

## 1. Goals

1. A single chat seam, `ChatProtocol`, with direct Ollama and OpenAI-compatible
   implementations and a scripted test double — the exact shape
   `providers/embeddings.py` already established for embeddings.
2. Optional query rewrite and optional cited synthesis, both behind that seam,
   both reachable only from a new `grk answer` command.
3. A redaction pass that runs on the chat boundary before any text reaches a
   cloud provider, so that
   `docs/architecture/llm-boundary.md`'s standing warning — *"the redaction pass
   is not implemented"* — stops being true (SPEC.md §9 names that page's update
   as an obligation of this phase).
4. An advisory faithfulness judge: schema-validated verdict, injected model
   call, **exits 0 and gates nothing**, with the calibration procedure that
   could ever change that written down before anyone wants it.
5. A planted-marker citation-echo check (SPEC.md §2), deterministic enough to be
   proved offline and honest enough to mean something against a real model.

## 2. Non-goals

- **No LLM anywhere in the retrieval path.** `Retriever.search` is untouched.
  There is no fourth `SearchMode`, no rewrite flag on `grk search`, and no
  synthesis in `retrieval/`.
- **No synthesis, rewrite or judge on the REST or MCP surface** (ADR-0019).
- **No retry, no repair, no second attempt.** A malformed or ungrounded model
  output is rejected once (ADR-0018).
- **No agent loop, no tool use, no streaming.** One request, one string back.
- **No numeric faithfulness score.** The judge returns a boolean and prose, not
  a number nothing calibrated (SPEC.md §2).
- **No multi-query expansion.** Rewrite produces one query. Fanning out to *N*
  rewritten queries and fusing their rankings is a change to RRF's inputs, which
  makes it a retrieval-architecture change rather than a boundary feature.
- **No spend cap or token accounting.** Recorded in `KNOWN_LIMITATIONS.md`
  rather than half-built.

## 3. The seam

`ChatProtocol` (`src/groundkit/providers/protocols.py`), alongside
`EmbeddingProtocol` and shaped like it:

```python
@runtime_checkable
class ChatProtocol(Protocol):
    @property
    def provider(self) -> str: ...

    @property
    def model_name(self) -> str: ...

    async def complete(self, prompt: str, *, system: str | None = None) -> str: ...
```

`provider` and `model_name` are on the seam for the reason
`EmbeddingProtocol.dimensions` is: a report that records which model produced a
verdict must read that identity off the object that produced it, not off a
config someone could pass that disagrees with it (ADR-0004's argument, one layer
up).

`complete` is string-in, string-out and non-streaming. Two consequences follow
from that narrowness and are stated here rather than discovered later:

- **Structured output is groundkit's problem, not the provider's.** No
  `response_format`, no server-side JSON schema. Every structured reply is
  parsed and validated on this side of the seam and *rejected* when it does not
  fit (SPEC.md §2: schema rejection, never coercion). A provider-side schema is
  a yield improvement at best and can never be the validation.
- **Truncation is not visible at the seam.** There is no `finish_reason`, so a
  reply cut off at the token limit arrives as an ordinary string. What catches
  it is the validation above it: a truncated JSON verdict fails to parse
  (`JudgeError`), and a truncated answer that loses its citation markers fails
  the uncited-answer rule (`SynthesisError`). A truncation that removes only the
  last sentence of an otherwise well-formed cited answer is **not** caught, and
  that residue belongs in `KNOWN_LIMITATIONS.md`.

Implementations (`src/groundkit/providers/llm.py`), taking constructor keyword
arguments rather than a config object:

| Class | Endpoint | Credential | Notes |
|---|---|---|---|
| `OllamaChat` | `{base_url}/api/generate` | none | local default; the SPEC.md §7 private-endpoint exception applies, scoped the same way `OllamaEmbedder` scopes it |
| `OpenAICompatChat` | `{base_url}/v1/chat/completions` | `os.environ[api_key_env]`, read at call time | never redacted-optional — see §5 |
| `ScriptedChatProvider(script)` | none | none | replies in order, **fails closed on exhaustion** |

`ScriptedChatProvider` is this phase's `InMemoryEmbedder`: correct for
exercising code paths offline and deterministically, and **wrong for measuring
anything**. Every number it produces is structural. It fails closed when the
script runs out rather than repeating or returning empty, because a double-call
bug that silently reused the last reply would make a broken pipeline look like a
working one.

The two HTTP providers reuse, rather than re-implement, the guards
`providers/embeddings.py` already carries: `validate_endpoint_shape` at
construction and `ensure_safe_endpoint` per request
(`utils/url_safety.py`, ADR-0014 decision 10), the `_allow_private_endpoint`
class attribute for Ollama, credential scrubbing over exception messages **and**
the `__cause__`/`__context__` chains (ADR-0001 hazard 6), and `_sanitize_url`'s
userinfo redaction. Duplicating any of them is how the hazard-6 fix decays on
its second copy; ADR-0017 decision 3 records the promotion.

**Only one new Protocol seam is introduced.** Query rewrite, synthesis and the
judge are concrete classes taking an injected `ChatProtocol`, not Protocols of
their own. A Protocol earns its keep when there are two implementations or a
declared extension point; there is exactly one synthesizer, and the seam tests
need to replace is the chat call itself. Three decorative Protocols would add
three conformance tests that assert a class matches an interface written from
it.

`tests/test_protocol_conformance.py` gains `TestChatProtocolConformance` with
`assert_signature_parity` over `OllamaChat`, `OpenAICompatChat`,
`ScriptedChatProvider` **and `RedactingChat`** — the decorator most of all, since
a wrapper that drifts from the seam it wraps is ADR-0001 hazard 4 with an extra
step.

## 4. Errors

`errors.py` gains five types under the existing local root. Each is a rejection;
none is a fallback.

| Type | Parent | Raised when |
|---|---|---|
| `ChatError` | `GroundkitError` | provider or transport failure at the chat boundary |
| `ChatProviderNotConfiguredError` | `ChatError` | provider unconfigured, credential env var unset, model absent. Never substitutes another provider (SPEC.md §2) |
| `QueryRewriteError` | `GroundkitError` | rewrite returned blank, or longer than `MAX_QUERY_LEN` |
| `SynthesisError` | `GroundkitError` | empty result set, out-of-range `[n]` marker, or an answer with no citations |
| `JudgeError` | `GroundkitError` | verdict JSON failed to parse or failed schema validation |

`QueryRewriteError` on a blank rewrite is worth its own sentence, because
"silently use the original query" is the tempting alternative and it is the same
defect `RerankerNotConfiguredError` exists to prevent: a rewriter that quietly
returns what it was given is indistinguishable from one that worked, so a
misconfigured run reports rewrite-enabled numbers that are really the plain
query's.

## 5. Redaction, and where it hooks in

`providers/redaction.py` is boundary-agnostic by design —
`RedactionPattern`/`RedactionConfig`/`RedactionResult`/`Redactor`, pure,
deterministic, no LLM, no network. Phase 5 supplies the boundary.

**The hook is a decorator on the seam, not a step inside each feature.**

```python
class RedactingChat:  # satisfies ChatProtocol
    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        redactor = Redactor(self._config)  # a FRESH instance per call
        ...  # redact prompt + system, call inner, restore output
```

Three properties make this the right placement, and all three fail under the
alternatives:

1. **It covers every generative path by construction.** Rewrite, synthesis and
   the judge all reach the network through `ChatProtocol.complete` and nothing
   else, so wrapping the seam covers three features with one wrapper and cannot
   be forgotten when a fourth arrives. Per-feature redaction would be three
   copies of one security control.
2. **A fresh `Redactor` per call is load-bearing, not tidiness.** Tokens are
   stable per distinct value *within an instance*. A wrapper holding one
   long-lived redactor accumulates a mapping across calls, and `restore()` would
   then be able to expand a token appearing in call B's output into a value
   captured during call A — a cross-request disclosure created by the mitigation
   itself. Per-call construction makes that unrepresentable.
3. **Citations cannot be corrupted by redaction.** `Citation` objects are built
   from groundkit's own `RetrievalResult`s (§6), never from model text, so a
   redacted prompt changes what the model *sees* and never what a citation
   *points at*. This is what lets redaction be applied without weakening the
   product claim.

**Marker/token collision — a named hazard with a named test.** The pinned token
format is `[<UPPERCASE_NAME>_<n>]` and the pinned citation-marker format is
`[n]`. `RedactingChat.restore()` runs *before* `Synthesizer` parses markers, so
a restored answer can contain both spellings. Marker parsing is therefore
anchored to digits only — `\[(\d+)\]` — so `[PERSON_1]` can never be read as
citation 1 of anything. `tests/test_synthesis.py` owes a regression test that
feeds an answer containing a redaction-shaped token and asserts it is not parsed
as a marker; per SPEC.md §8 it must be shown to fail against a looser pattern
first.

**What is covered and what is not.** Redaction covers the chat boundary. It does
**not** cover the embedding boundary, and that is a deliberate deviation from
SPEC.md §2's "before any text leaves the process for a cloud provider" recorded
in ADR-0017 decision 5 with its three reasons. The practical consequence, owed
to `KNOWN_LIMITATIONS.md` and to `docs/architecture/llm-boundary.md`: an
operator on `openai_compatible` *embeddings* still ships their corpus in the
clear, and the egress inventory's row 2 keeps saying **No** — with a reason
beside it now, rather than a gap.

Redaction is also a **mitigation, not a guarantee**: it finds what its patterns
describe, misses what they do not, and over-redacts where they are broad. The
docs must say so in those words rather than implying a corpus is safe once the
pass is on.

## 6. Cited synthesis

`Synthesizer(chat, *, prompt_template).synthesize(query, results)` →
`SynthesizedAnswer(answer: str, citations: tuple[Citation, ...])`.

The prompt presents the retrieved results as a numbered source list and asks for
`[n]` markers. **The model never supplies a document id, a source path, or an
offset.** It supplies an integer, groundkit resolves that integer to
`results[n-1]`, and the `Citation` is constructed from that `RetrievalResult`'s
own fields. That is the whole mechanism by which "synthesis may cite only
retrieved spans" (SPEC.md §2) becomes structurally true instead of a model
promise: a citation to something that was not retrieved is not a bad answer, it
is an unrepresentable one.

Fail-closed rules, all rejections:

- **Empty `results`** → `SynthesisError`. Nothing to ground against, so nothing
  to answer with. Abstention lives here, upstream of the model.
- **`[n]` out of range** → `SynthesisError`.
- **No markers at all** → a valid abstention: `SynthesizedAnswer` with
  `citations == ()`. The citation contract is set-membership over what *is*
  cited, not an obligation to cite — a model honestly saying "the sources do
  not answer this" is an outcome, not a violation. (This supersedes this
  document's first draft, which rejected uncited answers; Q2 below records
  the resolution.)
- One attempt. No repair pass, no re-prompt, no "fix your citations" round trip.

Defensive framing (`<retrieved_context>`, promoted from ARP's
`context_assembly.py` per ADR-0001) wraps the source list. ADR-0001 already
records what it is: *a documented, visible boundary, not a proof*. Phase 5 is
the first phase where the corpus's adversarial category — injection-styled text,
planted on purpose — actually reaches a model prompt, and nothing here defends
against prompt injection. That belongs in `KNOWN_LIMITATIONS.md` in plain words.

## 7. Query rewrite

`QueryRewriter(chat, *, prompt_template).rewrite(query) -> str`. Blank output is
`QueryRewriteError`; so is a rewrite exceeding `MAX_QUERY_LEN`
(`retrieval/search.py`), because a rewrite the retriever would reject is a
failure that should surface at the rewriter rather than three frames later.

Rewrite is reachable from `grk answer --rewrite` **only**. It is deliberately
not offered on `grk search`, and the reason is not caution: `SearchResponse` is
returned *unchanged* by the REST and MCP surfaces (ADR-0014 decision 8), and
`SearchResponse.query` has exactly one field for what was searched. A rewrite
makes "the query" two different strings, and whichever one that field carries is
wrong for some caller. `grk answer`'s report has room for both (§9); a
`SearchResponse` does not.

## 8. The advisory judge

`FaithfulnessJudge(chat, *, prompt_template).judge(*, query, answer, sources)` →
`FaithfulnessVerdict(faithful: bool, unsupported_claims: tuple[str, ...],
reasoning: str)`, in `src/groundkit/evals/judge.py`. Malformed or
schema-invalid JSON is `JudgeError` — parsed and validated with Pydantic
(`extra="forbid"`, frozen), never coerced, never partially accepted.

**Boolean and prose, no score.** A 0–1 faithfulness number from an uncalibrated
judge is precisely the invented number SPEC.md §2 forbids, and it is worse than
a label because it invites thresholding. The verdict says what it can support:
whether it found unsupported claims, and which.

**Advisory means advisory.** The judge exits 0. There is no
`--fail-on-unfaithful`, no non-zero exit path, and no CI job that reddens on a
verdict. ADR-0018 decision 5 records the calibration procedure that would have
to be completed — and the superseding ADR that would have to be written — before
any of that could change.

## 9. CLI surface

**`grk answer` is a new command** (ADR-0019). SPEC.md §4's verb set —
`ingest | search | eval | serve | serve-mcp` — predates this phase, so adding a
verb is a deviation and carries an ADR. The alternative, a `--synthesize` flag on
`grk search`, was rejected for the reason §7 gives about `SearchResponse`: the
flag would make the flagship read command emit two different shapes and make an
offline, deterministic command able to call out to a model.

```
grk answer QUERY [--index-dir DIR] [--collection NAME] [--top-k N]
                 [--mode {bm25,dense,hybrid}] [--embed-* ...]
                 [--rewrite] [--judge] [--json]
                 [--chat-provider {ollama,openai_compatible,scripted}]
                 [--chat-model NAME] [--chat-base-url URL]
```

`grk eval` gains `--synthesis` and `--judge` (flags on an existing command, so
no deviation), sharing the same `--chat-*` flags.

Flag discipline follows the rule `ingest`/`search`/`eval` already enforce: a
flag configuring a path the run will not take is a mistake to name, not one to
ignore. `--chat-*` without a chat path, or `--judge` without `--synthesis` on
`grk eval`, is a `ConfigurationError`.

`grk answer --json` emits an `AnswerReport` — a frozen model carrying `query`,
`rewritten_query` (nullable), the `SearchResponse`, the `SynthesizedAnswer`, and
the `FaithfulnessVerdict` (nullable). It lives in a new
`src/groundkit/answer.py` alongside the composition function that produces it,
for `runtime.py`'s reason: a composition root belongs outside whichever package
happened to need it first, so composing `providers.synthesis` with `evals.judge`
creates no dependency edge between them. Text output prints the answer with its
`[n]` markers intact followed by a numbered citation list, so what the reader
sees on the console carries the same pointers the JSON does.

## 10. Configuration

`config.py` gains `ChatConfig` and `resolve_chat_config`, exact peers of
`EmbeddingConfig` and `resolve_embedding_config` — same frozen/`extra="forbid"`
models, same defaulting from a fresh instance rather than a second copy of the
field defaults, same `ValidationError` → `ConfigurationError` translation at the
one construction site.

```python
class ChatConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["ollama", "openai_compatible", "scripted"] = "ollama"
    model_name: str = DEFAULT_CHAT_MODEL
    base_url: str = DEFAULT_OLLAMA_BASE_URL
    api_key_env: str = "GROUNDKIT_OPENAI_API_KEY"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=1024, gt=0)
    timeout_seconds: float = Field(default=60.0, gt=0)
```

**`api_key_env` is a variable name, never a value** (SPEC.md §7). The key is
read from the environment at call time, exactly as
`OpenAICompatibleEmbedder._resolve_api_key` does, and the
`ChatProviderNotConfiguredError` message names the variable and never its
contents.

`temperature` defaults to 0.0. That is not a reproducibility guarantee — no
hosted model offers one — and no doc may claim it is.

Two placement notes, both deliberate:

- **The provider classes take keyword arguments; `build_chat(config)` in
  `providers/llm.py` is the translation layer.** That is the same shape
  `build_embedder(config)` already has, and `providers → config` is an edge that
  already exists, so it introduces no cycle. What *would* introduce one is
  `config.py` importing anything from `providers`, which is why the redaction
  and chat configs are reached through factories rather than composed into
  `GroundkitConfig`.
- **`GroundkitConfig` gains nothing.** It is already unused by the CLI, which
  builds `RetrievalConfig()` and `EmbeddingConfig` directly; extending it would
  add fields nothing reads and force the import cycle above.

Redaction configuration (`RedactionConfig`) stays where the module owns it, in
`providers/redaction.py`. The default pattern set is a floor, not a policy —
see **Q3**.

## 11. Eval integration

Two things, kept separate, because they run over two different corpora.

**Synthesis and judge results are one additive field on the existing artifact.**
`EvalReport` gains `synthesis: SynthesisReport | None = None`, and
`schema_version` **stays `Literal[1]`**. `evals/schema.py`'s module docstring
already licenses additive-with-default and states the live precondition —
artifacts never outlive the code that reads them, because `evals/results/` is
gitignored. This is the third reliance on that licence, so Phase 5 stops
*remembering* the precondition and starts *checking* it: a test asserts
`evals/results/` is still ignored by `.gitignore`. The day that stops being true,
the next field is a version bump, and the test is what says so.

`SynthesisReport` carries the input stage, the synthesis and judge model
identities, the prompt-template hashes, whether redaction was applied, per-query
outcome counts (`answered` / `rejected`), and the verdict counts. Nothing lands
in `RunConfig`: that model is documented as *"a record of the retrieval settings
in effect"*, and synthesis is not a retrieval setting. Nothing lands in
`StageResult` either — `stage` stays the four-member `Literal` Phase 3 settled,
because synthesis is not a ranking and has no delta against a BM25 baseline.

A per-query `SynthesisError` is recorded as `rejected`, not raised out of the
run. Discarding an answer is not coercion; the run reports how many were
discarded, and a scripted-model run reporting a high rejection count is
information, not a failure.

**The planted-marker echo check is its own artifact, over its own corpus.**
Markers must be high-entropy and generated per run (so no model can have
memorized them), which means planting them in `evals/corpus/` would change
`corpus_hash` on every run and pollute the retrieval baseline. The check
therefore builds a synthetic marker corpus in a temp directory and writes
`evals/results/echo-latest.json` with its own `schema_version`. Nesting it in
`EvalReport` would imply it shares that report's `corpus_hash`, which it does
not. ADR-0018 decision 4 records the design; §12 of that ADR records the two
sides it checks.

**Gating.** The default suite drives everything through `ScriptedChatProvider`:
the plumbing, every failure path, the marker-collision regression, and the echo
checker's own ability to catch a scripted leak. It proves the *checker*, never a
*model*. A real measurement needs `SYNTHESIS_GATED=1` with a chat model pulled
into local Ollama, in `tests/test_synthesis_gated.py` and
`.github/workflows/synthesis-gated.yml` — its own gate rather than an extension
of `EVAL_GATED`, because a chat-model pull is a different and much larger cost
than an embedding model's and folding them together would slow an existing gate
to measure something unrelated to it. `workflow_dispatch`-only during active
development, matching the two existing gates and carrying the same reinstatement
note. It is the sole proof of that backend, so it is not `continue-on-error`
(SPEC.md §3).

## 12. Coverage

`providers/llm.py`, `providers/redaction.py`, `providers/query_rewrite.py`,
`providers/synthesis.py`, `evals/judge.py` and `answer.py` stay **outside** the
`[tool.groundkit.coverage].core_subset`. SPEC.md §8 defines that subset as
retrieval, chunking, scoring and citation resolution, and ADR-0014 decision 12
already refused to widen it to mean "anything important" — a subset that means
that stops measuring anything.

The near-miss is `providers/synthesis.py`, whose marker resolution looks like
citation resolution. It is not: it is a bounds-checked lookup into a list its
caller supplied, while `retrieval/citations.py` re-reads bytes from disk. All
six modules stay under the whole-package gate, and the security-load-bearing
behaviour is carried by the *named* tests in §5, §6 and §8 — a stronger control,
since a coverage number cannot tell an enforcement test from a decorative one.

## 13. Module placement

| Module | Contents |
|---|---|
| `src/groundkit/providers/protocols.py` | `ChatProtocol` |
| `src/groundkit/providers/llm.py` | `OllamaChat`, `OpenAICompatChat`, `ScriptedChatProvider`, `RedactingChat`, `build_chat` |
| `src/groundkit/providers/redaction.py` | `RedactionPattern`, `RedactionConfig`, `RedactionResult`, `Redactor`, default patterns |
| `src/groundkit/providers/query_rewrite.py` | `QueryRewriter` |
| `src/groundkit/providers/synthesis.py` | `Synthesizer`, `SynthesizedAnswer`, `<retrieved_context>` framing |
| `src/groundkit/evals/judge.py` | `FaithfulnessJudge`, `FaithfulnessVerdict` |
| `src/groundkit/evals/schema.py` | `SynthesisReport`, `EvalReport.synthesis` |
| `src/groundkit/evals/echo.py` | marker corpus, the two-sided echo check, its artifact |
| `src/groundkit/answer.py` | `AnswerReport` and the answer composition function |
| `src/groundkit/config.py` | `ChatConfig`, `resolve_chat_config`, `DEFAULT_CHAT_MODEL` |
| `src/groundkit/cli.py` | `grk answer`; `--synthesis`/`--judge` on `grk eval` |

`RedactingChat`, `answer.py`, `evals/echo.py` and the `SynthesisReport` schema
addition are the pieces the current implementation partition does not name an
owner for. They are listed here so the gap is visible rather than discovered at
integration.

## 14. Done means

1. `grk answer` works end-to-end against local Ollama, producing an answer whose
   every citation resolves through `retrieval/citations.py` to real source text.
2. `grk answer` with `--chat-provider openai_compatible` cannot send an
   unredacted prompt — enforced by `build_chat`, proved by a test, not by review.
3. Every fail-closed rule in §4/§6/§7 has a test, and each one is shown to fail
   against unfixed source first (SPEC.md §8). The marker/token collision test and
   the echo checker's leak-detection test are the two that matter most, because
   both pass trivially against a wrong implementation.
4. The judge runs, validates, and gates nothing; no CI job can redden on a
   verdict.
5. `EvalReport` parses at `schema_version` 1 with and without the new field, and
   the gitignore precondition is asserted rather than assumed.
6. All gates green: ruff, ruff-format, `mypy --strict`, pytest, both coverage
   gates, strict docs build, pip-audit, gitleaks.
7. `docs/architecture/llm-boundary.md` no longer says the redaction pass does not
   exist; its egress inventory gains the three generative rows and keeps row 2's
   **No** with ADR-0017 decision 5 beside it. SPEC.md §9 Phase 5 → done with a
   date; `KNOWN_LIMITATIONS.md` updated; the three ADRs accepted and listed.

## 15. Risks

**R1 — Every synthesis and judge number in normal CI is structural.** This is
Phase 3's R1 with a sharper edge: a hash-derived recall number at least looks
like a metric, whereas a scripted judge's `faithful: true` looks like an
*assessment*. `SynthesisReport` records the model identity so an artifact
self-labels which it is, and the CLI stamps the same explicit warning it stamps
for `InMemoryEmbedder`.

**R2 — The judge is an LLM grading an LLM, uncalibrated.** Until ADR-0018
decision 5's procedure has been run, a verdict is a second model's opinion and
is not evidence about faithfulness. It is advisory in the strong sense: reading
it as a quality signal is the misuse the advisory label exists to prevent.

**R3 — Prompt injection is not defended against, and Phase 5 is when it starts
mattering.** The golden corpus contains injection-styled text on purpose
(`INJECTION_MARKERS`, the adversarial category), and this is the first phase in
which that text reaches a model prompt. `<retrieved_context>` framing is a
visible boundary, not a proof — ADR-0001's own words. The adversarial judgments
still test retrieval, not resistance.

**R4 — Redaction is pattern-based.** It will miss and it will over-redact. It
raises the cost of a disclosure; it does not prevent one.

**R5 — Truncation is partly invisible** (§3), and there is no spend cap (§2).
`grk answer` is the first command that can be slow and can cost money, and
nothing bounds either.

**R6 — Two egress paths now exist and only one is redacted.** An operator who
reads "redaction landed" and then points `--embed-provider openai_compatible` at
a hosted endpoint has an unredacted corpus egress. The docs have to say this
where that operator will read it, not only here.

## 16. Open questions

**Q1 — Which model is `DEFAULT_CHAT_MODEL`? RESOLVED:** `llama3.2:3b`
(`config.DEFAULT_CHAT_MODEL`). The tag is explicit rather than a floating
alias so two runs under the default exercised the same weights. An absent
model is a `ChatProviderNotConfiguredError` naming the `ollama pull` remedy,
never a fallback.

**Q2 — Does `SynthesizedAnswer` need an `abstained` field? RESOLVED: no, and
the premise changed.** Zero markers is a *valid* abstention — `citations ==
()` — not a `SynthesisError` (§6 as amended). With that, an `abstained`
boolean would be stored state permanently equal to `len(citations) == 0`:
redundant derived data that can disagree with its own input, the same shape
this repo refused for citation checksums and stored eval deltas.

**Q3 — What belongs in the default redaction pattern set? RESOLVED:** the
shipped `DEFAULT_PATTERNS` floor — emails, E.164 and US phone shapes, IPv4,
long secret-shaped tokens; deliberately no person-name regex (a false promise
under the label "redacts names"; names are operator-configured patterns). The
refusal stands as proposed: `build_chat` fails closed on a cloud chat provider
with an explicitly empty pattern set, because "redaction enabled, nothing
configured" is the unredacted path wearing a redaction label.

**Q4 — Should query rewrite ever reach `grk search` or the service surface?**
Deferred with a named trigger rather than left open: it reopens when
`SearchResponse` can express two queries without breaking the shape ADR-0014
decision 8 keeps identical between `grk search --json` and the REST route — not
when a rewrite measurement looks good.

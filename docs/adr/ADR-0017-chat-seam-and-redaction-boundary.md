# ADR-0017 — One narrow chat seam, implemented directly, with redaction wrapped around it

- **Status:** Accepted
- **Date:** 2026-08-15
- **Deciders:** Andy Freeman (owner)

## Context

Phase 5 is the first phase in which groundkit sends a prompt to a generative
model (SPEC.md §9). Three features need it — optional query rewrite, cited
synthesis, and the advisory faithfulness judge — and SPEC.md §2 confines LLM use
to exactly those three, behind interfaces, all skippable.

Two constraints arrive already decided and are not reopened here. SPEC.md §3
requires **direct** provider implementations and names LiteLLM as rejected;
ADR-0001 recorded that decision for embeddings and rewrote the provider layer
accordingly. SPEC.md §7 requires the SSRF guard on cloud-provider endpoint URLs
with Ollama's loopback as the one named exception, and ADR-0014 decision 10 built
`utils/url_safety.py` to satisfy it for embeddings.

What is genuinely open is the shape of the seam, and one thing SPEC.md §3 leaves
as a *permission* rather than an instruction: groundkit "**may** consume
`executionkit` for LLM call patterns at the synthesis boundary." A permission is
not a decision, and leaving it unexercised by omission would be the same
scope-by-omission ADR-0014 decision 1 exists to prevent.

The other open question is where anonymization attaches. SPEC.md §2: *"A
redaction pass runs before any text leaves the process for a cloud provider."*
`providers/redaction.py` is deliberately boundary-agnostic — a `Redactor` over a
`RedactionConfig`, pure and deterministic, with no opinion about who calls it.
Something has to have that opinion.

## Decision

### 1. The seam is one non-streaming, string-in/string-out coroutine

`ChatProtocol` (`providers/protocols.py`) declares `provider`, `model_name`, and
`async def complete(self, prompt: str, *, system: str | None = None) -> str`.
No streaming, no tool calls, no message list, no `response_format`, no request
or response model.

The narrowness is the point. Every structured reply groundkit consumes — the
judge's verdict, the synthesizer's markers — is parsed and validated on *this*
side of the seam and rejected when it does not fit (SPEC.md §2: schema
rejection, never coercion). A provider-side JSON-schema parameter would improve
yield and could never be the validation, so putting one on the seam would add a
knob whose only honest description is "makes the thing we still have to check
more likely to pass."

`provider` and `model_name` sit on the seam for `EmbeddingProtocol.dimensions`'
reason: a report recording which model produced a verdict must read that
identity off the object that produced it, not off a config a caller could pass
that disagrees with it. That is ADR-0004's argument one layer above the index.

**The cost, stated rather than discovered:** there is no `finish_reason`, so a
reply truncated at the token limit arrives as an ordinary string. The validation
above the seam catches most of it — truncated JSON fails to parse, and an answer
truncated past its last `[n]` marker fails the uncited-answer rule — but a
truncation that removes only a trailing sentence from an otherwise well-formed
cited answer is not detected. That residue is recorded in
`KNOWN_LIMITATIONS.md`; it is not closed by this ADR.

### 2. Two real providers and one scripted double, direct HTTP, no LiteLLM

`OllamaChat` (`{base_url}/api/generate`), `OpenAICompatChat`
(`{base_url}/v1/chat/completions`, `Authorization: Bearer` from
`os.environ[api_key_env]` read at call time), and `ScriptedChatProvider`, which
replies from a queued script and **fails closed on exhaustion**.

The scripted provider is this phase's `InMemoryEmbedder` and inherits its rule:
it exercises code paths and produces nothing that is a measurement. Failing
closed rather than repeating its last reply is deliberate — a double-call bug
served a stale-but-plausible reply would make a broken pipeline test green.

There is no cross-provider fallback. An unconfigured provider is
`ChatProviderNotConfiguredError`, never a substitution, for the reason SPEC.md §2
gives about embeddings: a silent substitution produces plausible output from a
system the operator did not choose.

### 3. Outbound safety and credential scrubbing are promoted, not copied

`_post_json`'s catch-and-return-as-value pattern, `_scrub`, `_sanitize_url`'s
userinfo redaction, the chain-severing `raise ... from None` discipline, the
`validate_endpoint_shape` construction check, the per-request
`ensure_safe_endpoint` address check, and the `_allow_private_endpoint`
`ClassVar` all already exist in `providers/embeddings.py` and
`utils/url_safety.py`. The chat providers **share** them through a promoted
module rather than reimplementing them.

This follows ADR-0014 decision 6's precedent — `_resolve_embedding_config` was
*promoted* out of `cli.py` rather than imported privately or copied — and it
closes a specific decay path: ADR-0001 hazard 6 was a credential leaking through
an unscrubbed `__cause__`, and the natural way to reintroduce it is to write the
scrubbing a second time and get one component wrong, exactly as `_sanitize_url`
originally got `netloc` wrong.

### 4. Redaction hooks in as a decorator on the seam, with a fresh `Redactor` per call

`RedactingChat` satisfies `ChatProtocol`, wraps another `ChatProtocol`, redacts
`prompt` and `system` on the way in, and restores on the way out.

Three properties decide the placement:

1. **Coverage by construction.** Rewrite, synthesis and the judge reach the
   network through `complete` and nowhere else, so one wrapper covers all three
   and covers a fourth feature nobody has written yet. Per-feature redaction is
   three copies of one security control, and the fourth feature is the one that
   forgets.
2. **A fresh `Redactor` per call, not per wrapper.** Tokens are stable per
   distinct value *within an instance*. A long-lived redactor accumulates a
   mapping across calls, and `restore()` could then expand a token appearing in
   call B's output into a value captured during call A — a cross-request
   disclosure manufactured by the mitigation. Per-call construction makes it
   unrepresentable rather than unlikely.
3. **Citations are unreachable from redacted text.** `Citation` objects are
   built from groundkit's own `RetrievalResult`s (ADR-0018), never parsed out of
   model output, so redaction changes what the model sees and never what a
   citation points at. That is what allows redaction to be applied at full
   strength without weakening the product claim.

**`build_chat` decides, and the operator does not.** A cloud chat provider is
returned wrapped; there is no `--no-redaction` escape, because a lenient mode is
exactly what SPEC.md §2 forbids. A local Ollama provider is returned unwrapped
by default — local mode sends nothing anywhere — with redaction available opt-in
for exercising patterns. A test asserts the factory's return type per provider,
so "the cloud path is redacted" is a property rather than a review outcome.

### 5. The redaction boundary covers the chat seam only — the embedding boundary stays unredacted

This is a **deviation from SPEC.md §2's "any text ... for a cloud provider"**,
and it is recorded as a deviation rather than smoothed over. Three independent
reasons, any one of which is sufficient:

1. **The token mapping is not stable across processes.** Tokens are
   `[<UPPERCASE_NAME>_<n>]` with `n` assigned by first-seen order within one
   `Redactor`. Ingest and search are different processes seeing different text
   in different orders, so the same name becomes `[PERSON_1]` in one and
   `[PERSON_4]` in the other. Redacted-at-ingest vectors and
   redacted-at-query vectors would then describe different token spaces — a
   silently split semantic space, which is ADR-0004's failure class arriving
   through the mitigation instead of through a model swap.
2. **The vector would stop describing the stored chunk.** `Chunk.content` must
   remain the verbatim source substring or citations stop resolving, so a
   redacted embedding describes text that is not the text the citation points
   at. Retrieval quality then degrades in a direction nothing measures.
3. **The manifest cannot see it.** `CollectionManifest` binds
   `(provider, model_name, dimensions)`. It records nothing about redaction, so a
   collection embedded with redaction on and searched with it off passes
   identity verification and returns quiet nonsense — a *new* instance of exactly
   the hazard ADR-0004 was written to close.

The consequence is honest and unpleasant, and both `KNOWN_LIMITATIONS.md` and
`docs/architecture/llm-boundary.md` must carry it in plain words: **an operator
using `openai_compatible` embeddings still sends their corpus to a third party
in the clear.** The egress inventory's row 2 keeps saying **No** — now with a
reason next to it rather than a gap.

Reopening this needs a `Redactor` whose tokens are derived from the matched
value rather than from encounter order, plus a redaction marker in the
collection manifest so a mixed collection is refused. That is a change to a
pinned API and its own ADR.

### 6. groundkit does not depend on `executionkit` in v1

SPEC.md §3's permission is exercised as **no**.

The need at this boundary is one non-streaming HTTP call with a timeout, an
endpoint guard, credential scrubbing, and strict parsing — all four of which
already exist in this repo for embeddings and are shared by decision 3.
executionkit's value is in call *patterns* — consensus, chain-of-thought,
retries, provider tracking — and Phase 5 deliberately wants none of them:
ADR-0018 makes a single attempt with no repair and no retry, so a retry
framework at this seam would be capability pointed at a rule.

Against that near-zero benefit sits a real cost the portfolio has already paid
twice. ARP imports `executionkit.patterns.base._TrackedProvider`, a private,
unexported symbol with no consumer contract test; renaming it is a refactor EK
is entitled to make with no semver signal, and it breaks the consumer at import
time. Version pins across that boundary carry a matching silent failure: a pin
of `>=0.3.0,<0.4.0` means a 0.4.0 release stops the consumer exercising the path
at all, loudly to nobody. Taking a dependency on a sibling repo also creates
two-repo release coupling, which the portfolio hub already demonstrates by
turning red whenever one half moves without the other.

groundkit's stated portfolio position is that it "imports the internals of
neither" EK nor EVK. This decision keeps that literally true at the one boundary
where SPEC.md left a door open, and it takes the same shape ADR-0001 already
chose: **patterns are free to borrow, provider code is written here.**

## Alternatives considered

- **A `ChatRequest`/`ChatCompletion` model pair on the seam**, carrying
  temperature, token limits and a `finish_reason`. Rejected for v1: it widens the
  seam that every implementation and every conformance test must match in order
  to surface one field (`finish_reason`) whose absence is already partly covered
  by validation above the seam. Worth revisiting if truncation proves to be a
  real failure mode in practice rather than a theoretical one.
- **LiteLLM.** Rejected by SPEC.md §3 and ADR-0001 already; restated here only so
  the chat boundary is not treated as a fresh question. It also inverts the
  fail-closed posture, since its value proposition is smoothing over provider
  differences that this repo wants to see.
- **Official vendor SDKs (`openai`, `anthropic`).** Rejected: two SDKs to keep
  pinned and audited, each with its own auth and error surface, to replace one
  `httpx` POST whose OpenAI-compatible shape this repo already implements once
  for embeddings. It would also fork the credential-scrubbing and endpoint-guard
  path that decision 3 exists to keep single.
- **Depend on `executionkit`** — decision 6.
- **Redact inside each feature** (rewrite, synthesis, judge). Rejected per
  decision 4: three copies of one control, and the copy that gets forgotten is
  the one written last.
- **One long-lived `Redactor` per wrapper**, for stable tokens across a session.
  Rejected per decision 4: it creates a cross-call restoration path, which is a
  disclosure the mitigation would have introduced.
- **Redact the embedding path too**, satisfying SPEC.md §2 literally. Rejected
  per decision 5 — it would trade a disclosure risk for silent index corruption,
  which this repo consistently rates as the worse failure.
- **An opt-out flag for redaction on the cloud provider.** Rejected: there is no
  lenient mode, and a flag that turns a security control off is the control's
  most-used setting.

## Consequences

- Adding a generative feature costs one class and no new seam, and it is
  redacted, endpoint-guarded and credential-scrubbed by virtue of taking a
  `ChatProtocol` it did not construct.
- `RedactingChat` must itself pass signature-parity conformance. A decorator that
  drifts from the seam it wraps is ADR-0001 hazard 4 with an extra frame, and
  `isinstance` would not see it.
- Truncated output is only partly detectable (decision 1), and `KNOWN_LIMITATIONS.md`
  says so rather than the docs implying a completeness guarantee.
- The redaction claim is bounded twice over: to the chat boundary (decision 5),
  and to what the configured patterns actually match. Redaction raises the cost
  of a disclosure; it does not prevent one, and no doc may say otherwise.
- Promoting the HTTP and scrubbing helpers (decision 3) touches
  `providers/embeddings.py`, so the Phase 5 diff includes a refactor of a Phase 1
  module. A reviewer should expect it; the alternative is a second copy of the
  hazard-6 fix.
- Declining executionkit means any future need for a call pattern it already
  solves is re-solved here. That is accepted: the patterns Phase 5 needs are one
  POST and one strict parse.

## References

- SPEC.md §2 (LLM at the boundary, redaction, fail closed), §3 (direct
  providers, no LiteLLM; the executionkit permission), §7 (SSRF guard, Ollama
  exception, credentials as env-var *names*), §9 (Phase 5).
- [ADR-0001](ADR-0001-promote-vs-rewrite.md) — the provider-layer rewrite
  precedent and hazard 6 (`__cause__` scrubbing), hazard 4 (signature drift).
- [ADR-0004](ADR-0004-embedding-identity-binding.md) — the silently-mixed-space
  failure decision 5 declines to reintroduce.
- [ADR-0014](ADR-0014-read-only-service-surface-and-outbound-endpoint-safety.md)
  — decision 6 (promote, don't copy), decision 10 (`utils/url_safety.py`,
  `_allow_private_endpoint`).
- `docs/architecture/llm-boundary.md` — the egress inventory this decision
  updates, including the row 2 that keeps saying **No**.
- `docs/specs/phase-5-boundary-features.md` §3, §5, §10.

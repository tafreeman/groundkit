# ADR-0019 — Synthesis lands as `grk answer`, and does not reach the service surface

- **Status:** Accepted
- **Date:** 2026-08-15
- **Deciders:** Andy Freeman (owner)

## Context

SPEC.md §4 lists the CLI's verbs: `grk ingest | search | eval | serve |
serve-mcp`. That list is v1 scope, written before Phase 5 existed, and SPEC.md
§9's Phase 5 row independently commits the project to "optional query rewrite +
cited synthesis". The two are not in conflict about *whether* synthesis ships,
only about *where it is reachable*, and SPEC.md's opening line makes the
resolution procedural: deviating from it requires an ADR.

Two Phase 4 properties constrain the answer.

**`SearchResponse` is returned unchanged by both service transports** (ADR-0014
decision 8), so that a client parsing `grk search --json` and one parsing the
REST route parse the same shape. That property is load-bearing — it is the reason
Phase 4 has no parallel DTO layer — and anything that changes what
`grk search` emits changes what the service emits.

**Phase 4 has no authentication of any kind** (ADR-0014 decision 1). The
shared-secret header SPEC.md §7 requires for mutating operations was not built,
because the set of mutating operations is empty; the loopback bind is therefore
the service's only access control (decision 7), and SECURITY.md and
`KNOWN_LIMITATIONS.md` both say so in those words. The residual exposure Phase 4
accepted is **disclosure**, bounded by that bind.

## Decision

### 1. `grk answer` is a new command, and this is a recorded deviation from SPEC.md §4

```
grk answer QUERY [--index-dir] [--collection] [--top-k] [--mode {bm25,dense,hybrid}]
                 [--embed-* ...] [--rewrite] [--judge] [--json]
                 [--chat-provider {ollama,openai_compatible}]
                 [--chat-model] [--chat-base-url] [--chat-api-key-env]
```

It emits an `AnswerReport`: the original query, the rewritten query when
`--rewrite` was used, the answer text with its citations, the retrieved
results the answer is auditable against (a flat `results` tuple rather than
the whole `SearchResponse` — the response's own `query`/`top_k` echo would
re-import exactly the one-query-field ambiguity decision 2 exists to keep
out), and the `FaithfulnessVerdict` when `--judge` was used.

A separate verb, rather than a flag, because the two commands differ in every
dimension that matters to a caller. `grk search` is deterministic, offline,
free, and returns a ranked list; `grk answer` calls a model, can be slow, can
cost money, can fail for reasons `grk search` has no vocabulary for
(`ChatProviderNotConfiguredError`, `SynthesisError`), and returns prose with
pointers. A flag that switches a command between those two contracts is a flag
that changes what the command *is*.

### 2. `grk search` gains no synthesis and no rewrite flag

The decisive reason is structural rather than cautious. `SearchResponse` has one
`query` field, and a rewrite makes "the query" two different strings — the one
the user typed and the one the index was searched with. Whichever the field
carries is wrong for some caller, and because decision 8 of ADR-0014 returns that
model *unchanged* over REST and MCP, the ambiguity would not stay in the CLI: it
would land in the service surface's contract without anyone deciding to put it
there.

`AnswerReport` has room for both strings, which is the whole reason it exists.

### 3. `grk eval` gains `--synthesis`; an eval-side `--judge` waits for its producer

A flag on an existing command, so no deviation. It follows the fail-closed flag
rule `ingest`, `search` and `eval` already enforce: `--chat-*` without
`--synthesis` is a `ConfigurationError` rather than a silently ignored flag,
because a flag configuring a path the run will not take lets someone believe
they measured something the run never did.

What `--synthesis` runs today is the planted-marker echo check (decision 4),
writing its own artifact. An eval-side `--judge` — judge tallies over
golden-corpus answers, the producer that would populate
`EvalReport.synthesis` (decision 6) — is deliberately not built yet:
`SynthesisReport` is structure awaiting that producer, recorded as such in
`KNOWN_LIMITATIONS.md`. The judge is reachable today through `grk answer
--judge`, where its verdict is advisory and never changes the exit code
(ADR-0018 decision 5) — a rule that will bind the eval flag equally when it
lands.

### 4. No Phase 5 operation reaches REST or MCP, and the service package is barred from importing one

`TOOLS` keeps its four members. No `synthesize` tool, no `rewrite` request field,
no per-request `judge` option. `ToolSpec.side_effect` stays the one-member
`Literal`, untouched.

Synthesis is a *read* — it changes no document, chunk or manifest state — so
ADR-0014's read-only enforcement would not have blocked it. The refusal is a
separate judgment, and it is about what Phase 4's threat model actually assumed.
Phase 4 accepted **disclosure to anyone who can reach the port**, bounded by the
loopback bind and by the fact that a request could only cause groundkit to read
its own files. A synthesis route breaks that in two directions at once:

- **Cost amplification.** An unauthenticated request would trigger a billable
  outbound model call. That is a denial-of-wallet primitive, and it is a
  qualitatively different exposure from disclosure — an operator can survive
  someone reading a local corpus over loopback far more easily than an invoice.
- **Egress amplification.** With `--chat-provider openai_compatible`, an
  unauthenticated caller would cause the operator's *corpus text* to be sent to a
  third party. Phase 4's residual was "someone on this host can read what I
  indexed". This would be "someone on this host can make me publish what I
  indexed."

Neither is closed by the loopback bind, because the bind bounds *who can ask*,
not *what asking costs*.

The refusal is made structural rather than left to review: ADR-0014 decision 2
check 5's import scan over the `service` package — which already fails on any
import of `groundkit.indexer`, `groundkit.ingestion.loaders` or
`groundkit.ingestion.pipeline` — is extended to fail on
`groundkit.providers.llm`, `groundkit.providers.query_rewrite`,
`groundkit.providers.synthesis`, `groundkit.evals.judge` and
`groundkit.answer`. As with the ingest scan, it fires three steps upstream of the
route that would have exposed the capability.

### 5. The reopening trigger, named so it is not re-argued from scratch

A later phase may expose synthesis over REST or MCP when **all** of the
following land in the same change, superseding this ADR and ADR-0014 decision 1:

1. The shared-secret header, constant-time compare, and unset-secret disable
   SPEC.md §7 requires — built and tested, not merely argued to be unnecessary.
2. A per-request cost bound: a cap on context size and output tokens that a
   request cannot raise, resolved at serve time exactly as `base_url` is
   (ADR-0014 decision 6), so no request field can widen it.
3. An explicit decision about egress from an authenticated surface — whether a
   server configured against a cloud provider may serve synthesis at all, or
   whether that combination is refused at startup.

What does **not** reopen it: a client asking for it, or synthesis working well
locally. Both were true the day this was written.

## Alternatives considered

- **`grk search --synthesize`.** Rejected per decision 2: it makes one command
  emit two contracts, and the ambiguity would propagate into `SearchResponse`,
  which the service returns unchanged.
- **Both a flag and a command.** Rejected: two entry points to one feature, with
  the flag's output shape being the problem decision 2 describes. The
  documentation cost alone exceeds the convenience.
- **Name the verb `synthesize` or `ask`.** `answer` was chosen because it names
  the artifact rather than the mechanism, which is what the other verbs do
  (`search`, not `retrieve`; `ingest`, not `chunk`). It also survives the
  synthesizer being replaced.
- **A fifth MCP tool, `synthesize`, behind the shared-secret header built at the
  same time.** Rejected for this phase: it is decision 5's path, and taking it
  now means making SPEC.md §7's four unmade product decisions (deletion, file
  permissions, backup scope, retention) inside a phase about boundary features.
  ADR-0014 refused a half-built control for the same reason — a guard protecting
  one route establishes a place to attach the next one without re-deriving why it
  was needed.
- **Expose synthesis as a per-request option on `search`, the way `rerank` is.**
  Rejected: `rerank` is local computation on results the server already had, and
  its worst case is a slow response. Synthesis is outbound network egress with a
  price attached, so the analogy holds only in shape.
- **Leave the service question unanswered and simply not build it.** Rejected
  explicitly: that is the scope-by-omission ADR-0014 decision 1 exists to
  prevent. A later reader must find a decision, not a gap.

## Consequences

- **`grk answer` is CLI-only, so `agentic-evalkit` cannot grade groundkit's
  synthesis through its HTTP/MCP `ExecutionTarget` boundary.** The portfolio
  composition claim in SPEC.md §3 still holds for retrieval — the four read tools
  are unchanged — but the generative half is not reachable from that boundary,
  and that is a real, named cost of decision 4 rather than an oversight.
- The service surface stays exactly as ADR-0014 described it, so Phase 4's five
  structural checks keep meaning what they meant, and the extended import scan
  makes the Phase 5 refusal enforced rather than asserted.
- Adding a verb sets a precedent. The rule this ADR establishes for the next one:
  a new verb is justified when a command's *contract* differs — determinism,
  cost, failure vocabulary, output shape — and a flag is correct when only the
  computation differs. `--dense`, `--rerank` and `--mode` are all on the flag side
  of that line, which is why they are flags.
- SPEC.md §4's verb list must be updated to include `answer` when this ADR is
  accepted, or the spec and the CLI disagree, and this repo's convention is that
  the spec is the contract.
- An operator who wants synthesis behind an API has to write the wrapper
  themselves. That is the intended friction, and the reopening trigger in
  decision 5 is what it takes to remove it.

## References

- SPEC.md §4 (the verb list this deviates from), §7 (the shared-secret rule and
  the four unmade product decisions), §9 (Phase 5), §10 (zero cloud credentials
  end to end — preserved, since every existing command is unchanged).
- [ADR-0014](ADR-0014-read-only-service-surface-and-outbound-endpoint-safety.md)
  — decision 1 (no authentication, and why the guard was not built), decision 2
  check 5 (the import scan decision 4 extends), decision 6 (serve-time
  resolution), decision 7 (the loopback bind as the only access control),
  decision 8 (`SearchResponse` returned unchanged).
- [ADR-0012](ADR-0012-rerank-eval-stage-reorders-upstream-stage.md) — decision 2,
  the deferral pattern this ADR reuses for a named reopening trigger.
- [ADR-0017](ADR-0017-chat-seam-and-redaction-boundary.md),
  [ADR-0018](ADR-0018-llm-output-is-validated-never-trusted.md).
- `docs/specs/phase-5-boundary-features.md` §9.

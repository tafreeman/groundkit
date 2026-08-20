# Answer

The composition root behind `grk answer` (ADR-0019; Phase 5 of SPEC.md §9):
an optional query rewrite, a retrieval search, cited synthesis, and an
optional advisory faithfulness judge, wired into one call and one report.
Every collaborator *instance* is injected rather than constructed here —
resolving a chat config, constructing a concrete chat provider, opening a
`Retriever` over a persisted collection is the caller's job, so
`AnswerPipeline` itself never imports `groundkit.config` or
`groundkit.providers.llm`. The *types* are concrete rather than abstract, and
this page does not claim otherwise: only the search collaborator is a
structural Protocol, while the synthesizer, rewriter and judge are the named
classes, which the module therefore imports. All three are `ChatProtocol`
consumers living under `providers`, so the substitution a caller wants is
reached by injecting a different chat provider into them — and composing them
here costs the module no dependency on the eval harness.

`grk answer` deliberately does **not** reach the [service](service.md)
surface (ADR-0019) — the read-only REST API and MCP server never run
synthesis, so this module's contract has no counterpart there.

Abstention has one representation, and this module doesn't add a second:
`AnswerReport` carries no separate `abstained` field. An empty `citations`
tuple on a synthesized answer already *is* abstention, and a second boolean
here would only be a second, potentially disagreeing, way to say the same
thing.

When a faithfulness judge is injected, it runs after synthesis on
**everything retrieval returned**, not only the subset the answer ended up
citing, and its verdict is recorded unconditionally — including when the
answer abstained. If the judge itself raises, that error propagates out of
`AnswerPipeline.answer` uncaught: a requested component that breaks is a
typed failure, not a silently-skipped opinion (ADR-0018 decision 5).

::: groundkit.answer

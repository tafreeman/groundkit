# Answer

The composition root behind `grk answer` (ADR-0019; Phase 5 of SPEC.md §9):
an optional query rewrite, a retrieval search, cited synthesis, and an
optional advisory faithfulness judge, wired into one call and one report.
Every collaborator is injected rather than constructed here — resolving a
chat config, constructing a concrete chat provider, opening a `Retriever`
over a persisted collection is the caller's job, so `AnswerPipeline` itself
never imports `groundkit.config` or `groundkit.providers.llm`, and composing
`providers.synthesis` with `evals.judge` inside it creates no dependency edge
between those two packages.

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

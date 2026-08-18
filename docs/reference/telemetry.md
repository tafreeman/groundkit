# Telemetry

OpenTelemetry spans and structured JSON logging (ADR-0022). A deliberate
leaf module: it imports nothing from `groundkit`, because
`retrieval.search` imports *this* module to instrument its `search` span, so
an import in the other direction would be circular — the `RetrievalMode` and
stage literal types below are re-declared locally rather than imported from
`retrieval.search` or `evals.schema`, which define the same shapes for their
own reasons.

`opentelemetry-api` is a **base** dependency (ADR-0022 decision 1), so it is
imported unconditionally — there is no guarded import and no hand-rolled
no-op tracer shim to keep in signature parity by hand. With no
`opentelemetry-sdk` installed and nothing configured, `get_tracer()` still
returns a tracer, but every span it creates is non-recording: no export, no
collector, no error. That's documented `opentelemetry-api` behavior, not a
groundkit fallback, and it's why this module is excluded from the
core-subset coverage gate on the same test that keeps `runtime.py` inside
it (see `CONTRIBUTING.md`): whether a module's absence changes what a
response *contains*. A tracer's absence never does — it changes only what
gets shipped to a collector alongside the response, so this module is
covered by the whole-package 80% floor instead, the same treatment
`service/` gets for the same reason.

The `otel` extra (not installed by default) is what turns a span into
something a collector actually receives; see
[ADR-0022](../adr/ADR-0022-observability-dependency-shape-and-span-attribute-allowlist.md)
for the dependency shape and the span attribute allowlist that keeps query
text, source paths and document content out of every exported span.

::: groundkit.telemetry

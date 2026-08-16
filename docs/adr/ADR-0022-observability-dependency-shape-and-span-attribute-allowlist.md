# ADR-0022 — The OTel API is a base dependency, the SDK is an extra, and span attributes are an allowlist

- **Status:** Accepted (owner, 2026-08-15)
- **Date:** 2026-08-15
- **Deciders:** Andy Freeman (owner)

## Context

SPEC.md §3 requires "OpenTelemetry spans on ingest/retrieve/synthesize (ARP otel
collector-config conventions); structured JSON logs with request ID, latency, result
counts, typed failure codes. Never log document content or queries at info level."
SPEC.md §9's Phase 6 row adds "OTel verified end-to-end in compose."

That paragraph contains two requirements with very different dependency profiles and one
hazard, and conflating them produces the wrong answer to all three.

**Structured JSON logging needs no dependency.** `logging` already emits everything the
requirement names; what is missing is a formatter and a convention for carrying fields
alongside the message.

**Tracing needs a dependency, and OpenTelemetry ships it as two packages on purpose.**
`opentelemetry-api` is the instrumentation surface — pure Python, no transport, and its
documented behaviour with no SDK configured is to return non-recording spans that cost
almost nothing. `opentelemetry-sdk` plus an exporter is the part that batches, serialises
and ships. The split exists precisely so a library can be instrumented unconditionally
while an application decides whether anything is collected.

**The hazard is that a span is an off-process export of whatever is attached to it.**
SPEC.md §3 keeps query text and document content out of INFO logs, and
`service/api.py` implements that carefully: the access line carries id, method, route,
tool, status, latency and result count, and the request contents go to DEBUG only.
A span attribute is strictly worse than an INFO log for the same data — it leaves the
process by design, lands in a collector, and is then wherever the collector's exporters
put it. Instrumentation added without a rule about attributes is the shortest path to
re-introducing exactly the leak the logging rule was written to prevent, in a channel
where it is harder to notice.

This record is written before the instrumentation code, per SPEC.md §8. The code lands
in the second Phase 6 change; nothing here is a description of something already built.

## Decision

### 1. `opentelemetry-api` is a base dependency; `opentelemetry-sdk` and the OTLP exporter are an `otel` extra

```
dependencies       = [..., "opentelemetry-api>=1.27,<2"]
optional-dependencies.otel = [
  "opentelemetry-sdk>=1.27,<2",
  "opentelemetry-exporter-otlp-proto-http>=1.27,<2",
]
```

Instrumentation sites import the API unconditionally and call
`trace.get_tracer(__name__)`. With no SDK installed and nothing configured, every span is
a non-recording span: no export, no collector, no configuration, no error.

**Amendment (2026-08-16), added because the paragraph above is true and was still
read as saying something false.** "Installing the SDK and setting `OTEL_*`" is *not*
sufficient to make a span record. Those variables are consumed by
`opentelemetry.sdk._configuration`, which runs under the `opentelemetry-instrument`
launcher — **not on import**. A process with the `otel` extra installed and every
variable in decision 2 set correctly still gets a `ProxyTracerProvider` and exports
nothing unless something calls `trace.set_tracer_provider`. The instrumentation change
shipped exactly that state first: all three span sites correct, the allowlist enforced,
the unit suite green, and zero spans reaching the collector. It surfaced only on a real
compose run, which is the argument for SPEC.md §1.4's verification rule in miniature.

`groundkit.telemetry.configure_tracing()` is therefore part of this decision, not an
implementation detail below it: it is called from the CLI entry point, it no-ops unless
an OTLP endpoint is configured (so a default `grk search` still makes no network call,
as the consequences below promise), and it builds the provider from the SDK's own
env-reading components so decision 2's "standard `OTEL_*`, not groundkit config keys"
still holds exactly.

It also contains the **only guarded `opentelemetry` import in the package**, and that is
consistent with this decision rather than an exception to it. What is rejected above is a
guarded import *at every instrumentation site* plus a hand-rolled no-op shim to fall back
to. Neither exists: the sites still import only the API, unconditionally, and when the
extra is absent the fallback is `opentelemetry-api`'s own no-op path, not a fake of it. A
bootstrap whose entire job is to tolerate an optional extra being absent is the one place
a guard belongs.

The alternative — putting the API behind the extra too — means every instrumentation site
needs a guarded import and a hand-written no-op shim to fall back to. That shim is a
second implementation of the thing `opentelemetry-api` already is, it has to be kept in
signature parity with the real tracer by hand, and this repo has a written position on
exactly that failure mode: `tests/test_protocol_conformance.py` exists because
member-name-level compatibility is not compatibility. Taking a small pure-Python
dependency to avoid maintaining a fake of it is the right trade.

**This is not a weakening of SPEC.md §2's fail-closed rule, and the distinction is worth
stating so it is not cited as precedent.** Fail-closed governs things whose absence
changes an *answer*: an unconfigured embedding provider must raise rather than
substitute, because a substituted semantic space corrupts results silently. A tracer's
absence changes no answer. Refusing to serve because nobody is collecting traces would
make observability a hard runtime requirement of a local-first tool, which is the
opposite of what §3 is asking for. What *does* fail closed is the export path: a
configured endpoint that cannot be reached is an error the SDK reports, not a silent
downgrade to no tracing.

### 2. Enabling export is standard OTel environment configuration, not a groundkit config key

`OTEL_SERVICE_NAME`, `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_PROTOCOL`,
`OTEL_TRACES_EXPORTER`, `OTEL_RESOURCE_ATTRIBUTES` — read by the SDK, documented in
`.env.example` and in the compose stack, and absent from `config.py` entirely.

groundkit's own config models are frozen with real invariant validators and an unknown
key is a startup failure (SPEC.md §2). Adding `telemetry.endpoint` next to that would
create a second vocabulary for a setting that already has a cross-ecosystem standard one,
and the two would then need a precedence rule. Every collector, sidecar and operator
tutorial already speaks the `OTEL_*` names.

The consequence is accepted rather than hidden: these variables are *not* validated by
groundkit's config layer, so a typo in `OTEL_EXPORTER_OTLP_ENDPOINT` produces the SDK's
own error, not a groundkit `ConfigurationError`. `config.py`'s guarantees do not extend
here, and the docs say so.

**ERRATUM: `OTEL_EXPORTER_OTLP_PROTOCOL` was listed above as "read by the SDK" from the
first version of this decision, and for `configure_tracing`'s exporter it was not —
`telemetry.py` unconditionally imported `OTLPSpanExporter` from the `http` module, which
always speaks HTTP/protobuf regardless of that variable.** The confusion is real rather
than a typo: `opentelemetry.sdk._configuration`, which runs under the
`opentelemetry-instrument` launcher, *does* read this variable and select an exporter by
its own entry-point resolution — but `configure_tracing` does not go through that launcher
(decision 1's whole point is avoiding it), so its manually-constructed provider never saw
that resolution. The result: `OTEL_EXPORTER_OTLP_PROTOCOL=grpc` against a `grpc`-only
collector — the common case, since most collectors default to `grpc` on `4317` — applied
cleanly and then failed at every export call, the exact "documented variable, silently not
honoured" shape ADR-0020's own amendments record repeatedly for a different module.
`configure_tracing` now reads `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` then
`OTEL_EXPORTER_OTLP_PROTOCOL` itself and imports the matching exporter class — `grpc` or
`http`, both now shipped in the `otel` extra rather than only the `http` one — so the
variable means for this function what the sentence above already claimed it did. See
`tests/test_telemetry.py::TestConfigureTracing` for the regression coverage of both
transports.

### 3. Span attributes are an allowlist, and free text is not on it

**Permitted:** collection name, retrieval mode (`bm25`/`dense`/`hybrid`), stage,
`top_k`, result count, candidate count, chunk count, document count, duration, typed
failure code (the `kind` `service/errors.py` already renders), request id, HTTP status,
and the embedding identity triple ADR-0004 defines.

**Forbidden, on every span, at every level:** query text, chunk or document content,
citation spans, absolute source paths, and the values inside a `metadata_filter`. Not
"forbidden at INFO" — forbidden, because a span has no levels and every attribute on a
recording span is exported.

Two of these need their reasoning on the record, because both look permissible:

- **Absolute source paths are excluded even though `search` returns them to callers.**
  A caller who receives a path already had to reach the service, which ADR-0014
  decision 7 treats as the whole access-control boundary. A collector is a different
  audience with a different retention policy and, in the compose stack, a different
  container. Returning a path to an authorised caller and shipping it to a telemetry
  backend are not the same disclosure.
- **`metadata_filter` values are excluded even though keys would be useful.** The values
  are caller-supplied and are exactly where a tenant id, a user id or a customer name
  would appear. Attributing the key alone loses little and cannot leak a subject.

The rule is enforced by a test over the instrumentation helper rather than by review:
the helper takes typed keyword arguments and has no `**kwargs`, so an attribute outside
the allowlist is a type error rather than a code-review miss. That shape is chosen
deliberately for the reason ADR-0001 hazard 3 records — `**kwargs` absorbing something it
should have rejected is a defect class this repo has already been bitten by.

### 4. JSON logs carry the existing message *and* structured fields; the message text does not change

The formatter emits one JSON object per record: timestamp, level, logger, message, plus
whatever the call site passed in `extra`. Existing call sites keep their `key=value`
message strings and gain `extra={...}` alongside.

This is the low-risk shape and the reason is specific: ADR-0014 decision 9 makes
`service/api.py`'s access line something tests assert against, and rewriting those
messages into pure structured records would rewrite those tests in the same change that
introduces the formatter — leaving no test that was written against the old behaviour and
survived the new. Keeping the message and adding fields means the existing assertions
keep meaning what they meant.

Human-readable formatting stays the default for a terminal; JSON is opt-in
(`GROUNDKIT_LOG_FORMAT=json`), and the container images set it. A local `grk search` that
started printing JSON to a developer's terminal would be a regression in the tool's
primary use.

### 5. All three of SPEC.md §3's span sites are instrumented in Phase 6 — one of their names here was wrong

SPEC.md §3 names three: `ingest`, `retrieve`, `synthesize`. All three exist —
`Indexer.index_source` and `Indexer.index_directory`, `Retriever.search`, and
`Synthesizer.synthesize` — and Phase 6's instrumentation change covers all three.

**This decision originally deferred the third, and that deferral is withdrawn rather
than reworded.** It was written while Phase 5 was under construction concurrently, and
its reasoning was sound then: instrumenting a module that does not exist is impossible,
and guessing at a seam still being written is worse than waiting. Phase 5 then landed
(SPEC.md §9 records it done 2026-08-15) and this branch merged `main`, so
`src/groundkit/providers/synthesis.py` has been present in this very tree since that
merge. The premise expired without the text changing, which is the failure mode to
notice: a deferral justified by "it does not exist yet" has a shelf life, and nothing
about merging the thing into existence makes the record update itself.

Left as written, the consequence was concrete rather than cosmetic — the outstanding
instrumentation change would have read an accepted ADR telling it to skip the
`synthesize` span, and Phase 6 would have closed at two-thirds of a SPEC.md §3
requirement by following its own documentation.

**ERRATUM: this decision also named a method that has never existed — `Indexer.run` —
and that is corrected here rather than silently.** Unlike the deferral above, whose
premise expired between this record being written and the instrumenting change reading
it, this one was simply wrong the day it was written: the tree has never had a method
by that name. The real public ingest entry points, verified against
`src/groundkit/indexer.py`, are `Indexer.index_source(source: str) -> IndexReport`
(line 217) and `Indexer.index_directory(source_dir, max_concurrent=...) -> IndexReport`
(line 256), both `async`, reached from the CLI (`cli.py`) and from `evals/runner.py`.
An accepted ADR describing code that is not there is the same failure this decision
already corrects once above for a different reason, so it is fixed the same way: named
as an erratum, not quietly reworded. `docs/specs/phase-6-iac-observability.md` carried
the same name and is corrected alongside this one.

Two entry points rather than one settles a question the original text left implicit,
so it is decided here rather than left to the instrumenting change: **one span per
public entry point** — `index_source` and `index_directory` each get their own,
attributed with the document and chunk counts each returns in its `IndexReport`.
`index_directory` fans out concurrently over files internally (an inner `_one` onto
`_process`); a child span per file is the natural next step if per-file latency is ever
wanted, and it is deliberately not taken here. It keeps span cardinality bounded to the
call rather than to the file count, and it avoids deciding span-per-file semantics
under concurrent fan-out in the same change that introduces the tracer at all.
Recorded as a follow-up, not a gap this change forgot.

The seam for the third site is settled and public: `async Synthesizer.synthesize(query,
results) -> SynthesizedAnswer`. The span's attributes come from the same allowlist as
the other two (decision 3), which matters more here than anywhere else: a synthesis
span sits closest to prompt text, completion text and citation spans, and **none of
those may become an attribute.** Result count, model identity, latency and a typed
failure code may.

## Alternatives considered

- **Everything in the `otel` extra, with a hand-rolled no-op tracer.** Rejected per
  decision 1: it replaces a small maintained dependency with an unmaintained fake of it,
  and this repo already treats hand-maintained parity with an interface as a defect
  source rather than a saving.
- **Everything in base, SDK included.** Rejected: the SDK and an OTLP exporter pull
  protobuf and a transport stack into a default install for a feature a local
  single-user run never uses, and they grow the `pip-audit` surface ADR-0015 decision 4
  is careful about. The API alone is the part instrumentation genuinely needs.
- **`structlog` or `loguru` for the JSON logs.** Rejected. The package logs through
  stdlib `logging` in every module today, so adopting either means either a
  repo-wide rewrite — landing in the same change as an IaC phase, across files Phase 5 is
  editing concurrently — or two logging idioms coexisting. A `logging.Formatter`
  subclass produces the same JSON with no dependency and no migration.
- **`opentelemetry-instrumentation-fastapi` for automatic HTTP spans.** Rejected for now.
  It would produce spans without any of decision 3's discipline — its default attribute
  set includes the full request URL, which for a GET route carries the collection and
  chunk id in the path, and it would be one upstream default change away from carrying
  more. The four routes are generated from one registry (ADR-0014), so a single
  hand-written span in the dispatch path covers all of them with an allowlist that is
  ours. Revisit if the route set stops being generated from one place.
- **Prometheus metrics alongside traces.** Out of scope: SPEC.md §3 names spans and
  structured logs, and metrics would be a third telemetry surface with its own
  cardinality hazards — a `collection` label is unbounded — decided on its own.
- **Attribute *denylist* rather than allowlist.** Rejected on the standard argument, which
  holds unusually well here: a denylist is wrong the moment a new field appears, and the
  failure is a silent export of the new field rather than a missing attribute.

## Consequences

- A default `pip install groundkit` gains `opentelemetry-api` and its two small
  transitives; `uv.lock` and `requirements-audit.txt` regenerate with them, and an
  advisory against any of them will fail CI — the intended behaviour of that gate
  (ADR-0015 decision 4).
- With no `otel` extra and no `OTEL_*` configuration, groundkit behaves exactly as it
  does today: no spans, no exporter, no network calls, no measurable overhead.
- The compose stack's collector and Jaeger receive nothing until the instrumentation
  change lands. That ordering is deliberate — the topology is one change, the emission is
  the next — and both the compose README and the phase spec say so rather than implying
  a working trace pipeline.
- SPEC.md §3's three-site list is satisfied in full by Phase 6's instrumentation change,
  `synthesize` included. That was not the original plan — see decision 5 for why the
  deferral was withdrawn — and it means the change carries a span over a module whose
  inputs and outputs are the most sensitive in the codebase. The allowlist is what keeps
  that safe, so it is the part of this ADR to read before writing that span.
- The allowlist means a trace cannot answer "which query was slow" — only "a query
  against collection X in hybrid mode returning N results was slow." That is a real loss
  of debugging power and it is the intended trade; DEBUG logs, which stay in-process,
  remain where query text is available to an operator who has the machine.

## References

- SPEC.md **§2** (fail closed; real data only; no in-process state shadowing persisted
  state), §3 (OTel spans on ingest/retrieve/synthesize, structured JSON logs with request
  id, latency, result counts and typed failure codes, never document content or queries at
  info level), §8 (spec-driven: the spec section lands before the feature code), §9
  (Phase 6 row: OTel verified end-to-end in compose).
- [ADR-0014](ADR-0014-read-only-service-surface-and-outbound-endpoint-safety.md) —
  decision 9's typed failure `kind`, which is the failure attribute decision 3 permits,
  and the access-log line decision 4 preserves.
- [ADR-0015](ADR-0015-service-dependencies-are-base-not-an-extra.md) — the base-versus-extra
  reasoning and the `pip-audit` surface consequence this record follows.
- [ADR-0004](ADR-0004-embedding-identity-binding.md) — the embedding identity triple,
  the one provider-shaped value decision 3 permits on a span.
- [ADR-0001](ADR-0001-promote-vs-rewrite.md) — hazard 3 (`**kwargs` absorbing what it
  should reject), the reason decision 3's helper takes typed keywords.

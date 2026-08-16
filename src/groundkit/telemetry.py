"""OpenTelemetry spans and structured JSON logging (ADR-0022).

**This is a LEAF module: it imports nothing from ``groundkit``.**
``retrieval/search.py`` imports this module to instrument its ``search``
span, so an import in the other direction would be circular. The
``RetrievalMode`` and ``Stage`` literal types below are therefore
re-declared locally rather than imported from ``retrieval.search`` or
``evals.schema``, which define the same shapes for their own reasons.
``evals/metrics.py`` is this repo's existing precedent for a deliberate
leaf: no groundkit imports, so its tests construct plain values and this
module's tests construct plain values too.

``opentelemetry-api`` is a **base** dependency (ADR-0022 decision 1), so it
is imported unconditionally below. There is no guarded import, no
``try``/``except ImportError``, and no hand-rolled no-op tracer shim — the
ADR rejects that shape explicitly: a hand-maintained fake has to be kept in
signature parity with the real tracer by hand, which is exactly the defect
class ``tests/test_protocol_conformance.py`` exists to catch elsewhere in
this codebase. With no ``opentelemetry-sdk`` installed and nothing
configured, ``get_tracer()`` returns a tracer whose spans are
non-recording: no export, no collector, no configuration, no error. That is
documented ``opentelemetry-api`` behaviour, not a groundkit fallback, and
SPEC.md §2's fail-closed rule does **not** apply to it — fail-closed governs
things whose absence changes an *answer*, and a tracer's absence changes no
answer (ADR-0022 decision 1).

## The span attribute allowlist (ADR-0022 decision 3)

:func:`span_attributes` is the **only** way an instrumentation site may
attach data to a span. It is keyword-only and declares no ``**kwargs``, so
an attribute outside the allowlist is a type error caught by mypy and by
the call site's own test suite, not something that depends on a reviewer
noticing it. That shape mirrors ADR-0001 hazard 3: a ``**kwargs`` that
silently absorbs whatever it is handed is a defect class this repo has
already been bitten by, and the fix here is structural rather than a
convention to remember.

Permitted, and nothing else: collection name, retrieval mode, retrieval
stage, ``top_k``, result/candidate/chunk/document counts, duration, a typed
failure code, request id, HTTP status, the embedding identity triple
ADR-0004 defines (provider, model name, dimensions), and the chat
provider/model pair decision 5 permits on the ``synthesize`` span
specifically. Every one of those is either a bounded enum, a count, a
timing, or configuration the operator chose — never a value a user or a
document supplied, which is the line the allowlist is actually drawing.

**Forbidden on every span, at every level** — not just at ``INFO``, because
a span has no levels and every attribute on a recording span leaves the
process: query text, chunk or document content, citation spans, absolute
source paths, and the values inside a ``metadata_filter`` (its *keys* would
be fine; its values are exactly where a tenant id or a customer name would
appear). There is no parameter for any of these on :func:`span_attributes`
and there must never be one — see ``tests/test_telemetry.py`` for the tests
that hold this allowlist to that promise structurally rather than by
review.

## Attribute key naming scheme

Keys are dotted, OTel-style, under a ``groundkit.`` prefix. Concepts
specific to how a retrieval query was shaped nest under
``groundkit.retrieval.*``; the embedding identity triple nests under
``groundkit.embedding.*``; everything else — counts, duration, failure
kind, request id, HTTP status — is shared across the ingest/retrieve/
synthesize span sites and stays flat under ``groundkit.*``:

- ``collection``            -> ``groundkit.collection``
- ``retrieval_mode``        -> ``groundkit.retrieval.mode``
- ``stage``                 -> ``groundkit.retrieval.stage``
- ``top_k``                 -> ``groundkit.retrieval.top_k``
- ``result_count``          -> ``groundkit.result_count``
- ``candidate_count``       -> ``groundkit.candidate_count``
- ``chunk_count``           -> ``groundkit.chunk_count``
- ``document_count``        -> ``groundkit.document_count``
- ``duration_ms``           -> ``groundkit.duration_ms``
- ``failure_kind``          -> ``groundkit.failure_kind``
- ``request_id``            -> ``groundkit.request_id``
- ``http_status``           -> ``groundkit.http.status_code``
- ``embedding_provider``    -> ``groundkit.embedding.provider``
- ``embedding_model``       -> ``groundkit.embedding.model``
- ``embedding_dimensions``  -> ``groundkit.embedding.dimensions``
- ``chat_provider``         -> ``groundkit.chat.provider``
- ``chat_model``            -> ``groundkit.chat.model``

## Structured JSON logging

There is no central logging configuration in this repo today — the CLI
never calls ``logging.basicConfig``. :func:`configure_logging` is the new
entry point that makes ``GROUNDKIT_LOG_FORMAT`` mean anything, and it is
deliberately a thin ``logging.Formatter`` subclass plus a handler
attachment rather than a dependency on ``structlog`` or ``loguru``
(ADR-0022's alternatives section): every module already logs through
stdlib ``logging``, so a formatter subclass gets the same JSON with no new
dependency and no repo-wide rewrite.

:class:`JsonLogFormatter` does not change any existing log message string
(ADR-0022 decision 4) — call sites keep their ``key=value`` message text
and gain structured fields via ``extra={...}`` alongside it, so assertions
written against today's message strings keep meaning what they meant.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from typing import Any, Final, Literal

from opentelemetry import trace
from opentelemetry.trace import Tracer

# Local literal types — deliberately NOT imported from retrieval.search or
# evals.schema (this module is a leaf; see the module docstring).
RetrievalMode = Literal["bm25", "dense", "hybrid"]
Stage = Literal["bm25", "dense", "fusion", "rerank"]

#: Instrumentation scope name shared by every span this package emits, so
#: `Indexer`, `Retriever` and `Synthesizer` all publish under one scope
#: rather than one per module.
_TRACER_NAME: Final[str] = "groundkit"

logger = logging.getLogger(__name__)

#: Environment variable ADR-0022 decision 4 wires to the log formatter.
_LOG_FORMAT_ENV_VAR: Final[str] = "GROUNDKIT_LOG_FORMAT"

#: Standard OTel variables :func:`configure_tracing` reads to decide whether
#: the operator actually asked for export. Deliberately the spec's own names
#: (ADR-0022 decision 2), not groundkit config keys.
_OTLP_ENDPOINT_ENV_VARS: Final[tuple[str, ...]] = (
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
)
_TRACES_EXPORTER_ENV_VAR: Final[str] = "OTEL_TRACES_EXPORTER"

#: The standard value meaning "export nothing", honoured so setting it is a
#: reliable off switch even when an endpoint is also present in the
#: environment.
_EXPORTER_NONE: Final[str] = "none"

#: The one value that selects JSON. Anything else, including unset, keeps
#: human-readable formatting — see :func:`_select_formatter`.
_JSON_FORMAT_VALUE: Final[str] = "json"

#: Human-readable format string, the default for a developer's terminal.
_HUMAN_LOG_FORMAT: Final[str] = "%(asctime)s %(levelname)s %(name)s: %(message)s"

#: Name stamped on the handler :func:`configure_logging` attaches, so a
#: second call can detect its own handler and stay idempotent without a
#: module-level flag that would not survive, e.g., a test re-importing this
#: module in isolation.
_HANDLER_NAME: Final[str] = "groundkit.telemetry.stderr"

#: The attribute names present on a bare ``LogRecord`` with no ``extra``.
#: :func:`_extra_fields` diffs against this rather than hardcoding the
#: attribute list, so it tracks stdlib additions automatically (``taskName``
#: landed in Python 3.12) instead of silently going stale on an interpreter
#: upgrade. ``message`` and ``asctime`` are added on top: neither is present
#: on a fresh record, but ``logging.Formatter.format`` sets ``message`` as a
#: side effect, and a formatter that ran earlier on the same record could
#: have left either behind.
_RESERVED_LOG_RECORD_ATTRS: Final[frozenset[str]] = frozenset(
    logging.LogRecord(
        name="",
        level=0,
        pathname="",
        lineno=0,
        msg="",
        args=None,
        exc_info=None,
    ).__dict__
) | frozenset({"message", "asctime"})


def get_tracer() -> Tracer:
    """Return the package-wide OpenTelemetry tracer.

    Every instrumentation site (``Indexer.index_source``,
    ``Indexer.index_directory``, ``Retriever.search``,
    ``Synthesizer.synthesize``) calls this rather than
    ``opentelemetry.trace.get_tracer`` directly, so they all publish under
    the same ``"groundkit"`` instrumentation scope.

    With no ``opentelemetry-sdk`` installed and no exporter configured, the
    returned tracer's spans are non-recording: ``span.is_recording()`` is
    ``False``, attribute calls are accepted and discarded, and nothing is
    exported. That is the documented behaviour of ``opentelemetry-api``
    alone (ADR-0022 decision 1), not a groundkit fallback path, and it costs
    the caller nothing to instrument unconditionally.

    Returns:
        A ``Tracer`` bound to the ``"groundkit"`` instrumentation scope.
    """
    return trace.get_tracer(_TRACER_NAME)


def _export_requested() -> bool:
    """Return whether the environment asks for traces to be exported.

    Returns:
        ``True`` only if an OTLP endpoint is configured and
        ``OTEL_TRACES_EXPORTER`` is not the standard ``"none"``. Both halves
        matter: without the endpoint check, a default ``grk search`` on a
        developer's machine would start attempting network calls to
        ``localhost:4318`` merely because the ``otel`` extra happened to be
        installed, which ADR-0022's consequences rule out ("no ``OTEL_*``
        configuration ... no network calls"); without the ``none`` check,
        there would be no way to switch export off in an environment that
        sets an endpoint globally.
    """
    if os.environ.get(_TRACES_EXPORTER_ENV_VAR, "").strip().lower() == _EXPORTER_NONE:
        return False
    return any(os.environ.get(name, "").strip() for name in _OTLP_ENDPOINT_ENV_VARS)


def configure_tracing() -> None:
    """Install an SDK tracer provider when the environment asks for one.

    **This function is why any span in this package is ever recorded.**
    ``opentelemetry-api`` on its own returns a ``ProxyTracerProvider`` whose
    spans are non-recording, and — this is the part that is easy to get
    wrong and was gotten wrong here first — *installing*
    ``opentelemetry-sdk`` and setting the ``OTEL_*`` variables does not
    change that. Those variables are read by
    ``opentelemetry.sdk._configuration``, which runs under the
    ``opentelemetry-instrument`` launcher, not on import. Absent an explicit
    ``set_tracer_provider`` call, a fully-installed, fully-configured
    process still exports nothing, silently and with no error — a
    verification against the compose stack found exactly that.

    The SDK import here is guarded, and it is the **only** guarded
    opentelemetry import in this package. That does not contradict ADR-0022
    decision 1, which requires *instrumentation sites* to import
    unconditionally and rejects a hand-rolled no-op tracer shim: the sites
    still import only ``opentelemetry-api`` and still need no guard, and no
    shim exists — when the SDK is absent the real ``opentelemetry-api``
    no-op path is what runs. A bootstrap that must tolerate an optional
    extra being absent is the one place a guard genuinely belongs.

    Configuration is entirely the standard ``OTEL_*`` environment (ADR-0022
    decision 2) — ``Resource.create()`` reads ``OTEL_SERVICE_NAME`` and
    ``OTEL_RESOURCE_ATTRIBUTES``, and ``OTLPSpanExporter()`` reads the
    endpoint and protocol variables. Nothing is passed explicitly, so there
    is no second vocabulary and no precedence rule to invent, and a typo in
    an endpoint produces the SDK's own error rather than a groundkit
    ``ConfigurationError``.

    Calling this is optional and its absence changes no answer, so a missing
    SDK is a warning rather than a failure — SPEC.md §2's fail-closed rule
    governs things whose absence changes a *result*, and ADR-0022 decision 1
    says so explicitly to stop this being cited as precedent either way.
    What is *not* softened is the export path: a configured endpoint that
    cannot be reached surfaces as the SDK's own export error.

    Idempotent: a second call finds a real provider already registered and
    returns without replacing it (OTel itself only warns and ignores a
    second ``set_tracer_provider``, which would otherwise make the outcome
    depend on call order).
    """
    if not _export_requested():
        return
    if not isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider):
        return

    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning(
            "tracing requested via OTEL_* but the 'otel' extra is not installed; "
            "no spans will be exported (install groundkit[otel])",
        )
        return

    provider = TracerProvider(resource=Resource.create())
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)


def span_attributes(
    *,
    collection: str | None = None,
    retrieval_mode: RetrievalMode | None = None,
    stage: Stage | None = None,
    top_k: int | None = None,
    result_count: int | None = None,
    candidate_count: int | None = None,
    chunk_count: int | None = None,
    document_count: int | None = None,
    duration_ms: float | None = None,
    failure_kind: str | None = None,
    request_id: str | None = None,
    http_status: int | None = None,
    embedding_provider: str | None = None,
    embedding_model: str | None = None,
    embedding_dimensions: int | None = None,
    chat_provider: str | None = None,
    chat_model: str | None = None,
) -> dict[str, str | int | float]:
    """Build a span attribute dict from the ADR-0022 allowlist, and nothing else.

    Every permitted attribute is its own explicitly typed keyword parameter.
    There is no ``**kwargs``: an attribute outside this allowlist is a
    ``TypeError`` at the call site (and, before that, a mypy error), rather
    than something a reviewer has to notice is missing from a list. See the
    module docstring for the full allowlist, the forbidden set it
    deliberately excludes, and the key naming scheme applied below.

    Args:
        collection: The collection name, when reachable at the call site.
        retrieval_mode: ``"bm25"``, ``"dense"``, or ``"hybrid"``.
        stage: ``"bm25"``, ``"dense"``, ``"fusion"``, or ``"rerank"``.
        top_k: The requested result cutoff.
        result_count: Number of results returned.
        candidate_count: Number of candidates considered before cutoff.
        chunk_count: Number of chunks involved (e.g. produced by ingest).
        document_count: Number of documents involved.
        duration_ms: Elapsed wall-clock time in milliseconds.
        failure_kind: A typed, bounded failure label, when the span
            recorded a failure. ADR-0022 decision 3 names
            :mod:`groundkit.service.errors`'s rendered ``kind`` (e.g.
            ``"index_inconsistent"``) as the value it had in mind, and that
            is what a caller at the service boundary should pass. The
            ``retrieve`` and ``synthesize`` span sites sit *below* that
            boundary, though: ``retrieval/`` and ``providers/`` importing
            ``service/`` would invert the package layering, so they pass the
            raised exception's class name (e.g. ``"RetrievalError"``)
            instead. Both are closed, non-user-supplied vocabularies that
            cannot carry content, which is the property the allowlist is
            protecting; they are simply two different taxonomies, and a
            consumer reading this attribute across span sites should not
            assume one.
        request_id: The request id assigned at the service boundary.
        http_status: The HTTP status code returned for the request.
        embedding_provider: ADR-0004's embedding identity triple, part 1.
        embedding_model: ADR-0004's embedding identity triple, part 2.
        embedding_dimensions: ADR-0004's embedding identity triple, part 3.
        chat_provider: Chat provider backing a synthesis call.
        chat_model: Chat model identifier backing a synthesis call.
            ADR-0022 decision 5 permits "model identity" on the
            ``synthesize`` span specifically, and this pair is it — kept
            distinct from the ``embedding_*`` triple above because they name
            different things and a collector that conflated them would
            attribute a slow chat model to the embedding backend. Neither
            carries prompt or completion text; both are configuration the
            operator chose, not content a user supplied.

    Returns:
        A dict containing only the arguments that were not ``None``, keyed
        by the dotted ``groundkit.*`` name documented in the module
        docstring. OTel attribute values may not be ``None``, so an omitted
        argument is omitted from the result rather than included with a
        ``None`` value.
    """
    attributes: dict[str, str | int | float] = {}
    if collection is not None:
        attributes["groundkit.collection"] = collection
    if retrieval_mode is not None:
        attributes["groundkit.retrieval.mode"] = retrieval_mode
    if stage is not None:
        attributes["groundkit.retrieval.stage"] = stage
    if top_k is not None:
        attributes["groundkit.retrieval.top_k"] = top_k
    if result_count is not None:
        attributes["groundkit.result_count"] = result_count
    if candidate_count is not None:
        attributes["groundkit.candidate_count"] = candidate_count
    if chunk_count is not None:
        attributes["groundkit.chunk_count"] = chunk_count
    if document_count is not None:
        attributes["groundkit.document_count"] = document_count
    if duration_ms is not None:
        attributes["groundkit.duration_ms"] = duration_ms
    if failure_kind is not None:
        attributes["groundkit.failure_kind"] = failure_kind
    if request_id is not None:
        attributes["groundkit.request_id"] = request_id
    if http_status is not None:
        attributes["groundkit.http.status_code"] = http_status
    if embedding_provider is not None:
        attributes["groundkit.embedding.provider"] = embedding_provider
    if embedding_model is not None:
        attributes["groundkit.embedding.model"] = embedding_model
    if embedding_dimensions is not None:
        attributes["groundkit.embedding.dimensions"] = embedding_dimensions
    if chat_provider is not None:
        attributes["groundkit.chat.provider"] = chat_provider
    if chat_model is not None:
        attributes["groundkit.chat.model"] = chat_model
    return attributes


class JsonLogFormatter(logging.Formatter):
    """Render one JSON object per log record.

    The object always carries ``timestamp`` (ISO 8601, UTC — machine-
    parseable across a collector's timezone rather than the host's local
    one), ``level``, ``logger``, and ``message`` (the same string the
    record's ``msg %% args`` would have produced under the human-readable
    formatter — ADR-0022 decision 4 keeps existing message text unchanged).
    Every field the call site passed via ``extra={...}`` is merged in
    alongside those four; an ``exception`` field is added when the record
    carries exception info.

    A field from ``extra`` that collides with one of the four base keys is
    dropped rather than allowed to overwrite it — an accidental
    ``extra={"level": ...}`` must not be able to corrupt the record's actual
    severity. Non-JSON-serializable values are coerced with ``str()``
    (``json.dumps(..., default=str)``) rather than raising, so a bad value
    in ``extra`` can never crash the logging call site that produced it.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Render ``record`` as a single-line JSON string.

        Args:
            record: The log record being emitted.

        Returns:
            A JSON-encoded object; see the class docstring for its shape.
        """
        payload: dict[str, Any] = {
            **_extra_fields(record),
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _extra_fields(record: logging.LogRecord) -> dict[str, Any]:
    """Return the fields the call site passed via ``extra=``, if any.

    ``logging.Logger.makeRecord`` copies every key of an ``extra=`` mapping
    onto the record as an attribute, so the call site's structured fields
    and ``LogRecord``'s own bookkeeping attributes end up living in the same
    ``record.__dict__``. This tells them apart by diffing against
    :data:`_RESERVED_LOG_RECORD_ATTRS` instead of re-deriving the base
    attribute set by hand.

    Args:
        record: The record being formatted.

    Returns:
        A mapping of every attribute on ``record`` outside the base
        ``LogRecord`` shape.
    """
    return {
        key: value
        for key, value in record.__dict__.items()
        if key not in _RESERVED_LOG_RECORD_ATTRS
    }


def configure_logging(level: int = logging.INFO) -> None:
    """Attach one stderr log handler to the root logger, idempotently.

    There is no central logging configuration in this repo today, so this
    is the entry point that makes ``GROUNDKIT_LOG_FORMAT`` mean anything.
    ``GROUNDKIT_LOG_FORMAT=json`` selects :class:`JsonLogFormatter`; any
    other value, or unset, keeps human-readable formatting, which is the
    default (ADR-0022 decision 4) — a local ``grk search`` that started
    printing JSON to a developer's terminal would be a regression in the
    tool's primary use, not an improvement.

    The handler always writes to **stderr**, never stdout.
    ``service/mcp_server.py``'s ``_force_logging_to_stderr`` exists because
    ADR-0014 decision 5 makes stdout JSON-RPC-only on the stdio transport —
    one stray log record on stdout corrupts the rest of that session. This
    function does not fight that guard: it never attaches to stdout in the
    first place, so if ``run_stdio`` later calls
    ``_force_logging_to_stderr``, that sweep finds nothing of this
    function's to re-point.

    Calling this twice does not attach a second handler — a handler with
    :data:`_HANDLER_NAME` already present on the root logger is treated as
    "already configured," and the call becomes a level update only. That
    matters because nothing prevents ``configure_logging`` from being
    called from more than one entry point (a CLI command and a service
    startup path, say) without either one knowing the other already ran.

    Args:
        level: The root logger's level. Applied on every call — including a
            repeat call that finds the handler already attached — so a
            caller can still raise or lower verbosity without first
            checking whether an earlier call already configured logging.
    """
    root = logging.getLogger()
    if any(handler.name == _HANDLER_NAME for handler in root.handlers):
        root.setLevel(level)
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(_select_formatter())
    root.addHandler(handler)
    root.setLevel(level)


def _select_formatter() -> logging.Formatter:
    """Choose the log formatter honouring ``GROUNDKIT_LOG_FORMAT``.

    Returns:
        :class:`JsonLogFormatter` when the environment variable is exactly
        ``"json"``; a human-readable :class:`logging.Formatter` for any
        other value, including unset (ADR-0022 decision 4).
    """
    if os.environ.get(_LOG_FORMAT_ENV_VAR) == _JSON_FORMAT_VALUE:
        return JsonLogFormatter()
    return logging.Formatter(_HUMAN_LOG_FORMAT)

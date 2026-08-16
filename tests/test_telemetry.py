"""Tests for :mod:`groundkit.telemetry` (ADR-0022).

``telemetry.py`` is new in this branch, so most of these guards cannot be
demonstrated by SPEC.md §8's revert procedure — there is no prior behaviour
to revert to. Mirroring ``test_service_errors.py``'s own note for the same
situation: each guard here is instead demonstrated either by injecting the
exact violation it exists to catch, or — for the allowlist — by asserting a
structural property of :func:`~groundkit.telemetry.span_attributes`'s
signature that a future ``**kwargs`` or a future free-text parameter would
break. ADR-0022 decision 3 says this in so many words: "the rule is enforced
by a test over the instrumentation helper rather than by review," which
makes this file, not a reviewer's read of ``telemetry.py``, the thing that
actually holds the allowlist to its promise.
"""

from __future__ import annotations

import inspect
import io
import json
import logging
import os
import subprocess
import sys
import textwrap
from collections.abc import Iterator

import pytest
from opentelemetry import trace

from groundkit.telemetry import (
    _HANDLER_NAME,
    JsonLogFormatter,
    _export_requested,
    configure_logging,
    configure_tracing,
    get_tracer,
    span_attributes,
)

#: Substrings ADR-0022 decision 3 forbids from ever appearing on a span, at
#: any level: query text, chunk/document content, citation spans, absolute
#: source paths, and metadata_filter values. Checking substrings rather than
#: exact names means a future ``query_text``, ``chunk_content``, or
#: ``citation_spans`` parameter fails this test too, not just the four exact
#: names spelled out in the ADR.
_FORBIDDEN_SUBSTRINGS: tuple[str, ...] = ("query", "content", "citation", "path", "metadata_filter")

#: The standard OTel endpoint variables :func:`_export_requested` consults,
#: and the exporter switch that can veto them.
_ENDPOINT_VARS: tuple[str, ...] = (
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
)
_EXPORTER_VAR: str = "OTEL_TRACES_EXPORTER"

#: Every variable a tracing test must clear to start from a known-unset
#: environment. Named once so a future addition cannot be cleared in three
#: tests and forgotten in a fourth.
_ALL_OTEL_VARS: tuple[str, ...] = (*_ENDPOINT_VARS, _EXPORTER_VAR)


class TestSpanAttributesSignatureShape:
    """ADR-0022 decision 3: an attribute outside the allowlist must be a type error.

    That guarantee only holds if the helper has no ``**kwargs`` and takes
    every parameter keyword-only. These tests check the signature
    structurally with :mod:`inspect` rather than trusting a reviewer to
    notice a ``**kwargs`` creeping back in — the exact ADR-0001 hazard 3
    failure mode this helper exists to close off.
    """

    def test_no_var_keyword_parameter(self) -> None:
        """No ``**kwargs`` (or any other VAR_KEYWORD) parameter exists."""
        params = inspect.signature(span_attributes).parameters.values()
        assert not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params)

    def test_no_var_positional_parameter(self) -> None:
        """No ``*args`` either — the allowlist is entirely named keywords."""
        params = inspect.signature(span_attributes).parameters.values()
        assert not any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params)

    def test_every_parameter_is_keyword_only(self) -> None:
        """Every declared parameter is KEYWORD_ONLY, matching the frozen ``*,`` signature."""
        params = inspect.signature(span_attributes).parameters.values()
        assert params, "expected the allowlist to declare at least one parameter"
        assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params)

    def test_every_parameter_defaults_to_none(self) -> None:
        """Every parameter is optional, so a caller only sets what it has."""
        params = inspect.signature(span_attributes).parameters.values()
        assert all(p.default is None for p in params)


class TestSpanAttributesForbiddenSet:
    """The values ADR-0022 forbids must be structurally unreachable.

    Query text, chunk/document content, citation spans, absolute source
    paths, and metadata_filter values have no parameter today, and these
    tests are written so that adding one — under any of the obvious names —
    fails the suite rather than only a code review.
    """

    def test_no_exact_forbidden_parameter_name(self) -> None:
        """None of the specific names ADR-0022 calls out by category exist."""
        names = set(inspect.signature(span_attributes).parameters)
        forbidden_exact = {
            "query",
            "query_text",
            "content",
            "chunk_content",
            "document_content",
            "citation",
            "citation_span",
            "citation_spans",
            "source_path",
            "absolute_path",
            "path",
            "metadata_filter",
        }
        assert names.isdisjoint(forbidden_exact)

    def test_no_parameter_name_contains_a_forbidden_substring(self) -> None:
        """A drift-resistant version of the exact-name check.

        This is the test that would start failing if someone added, say,
        ``citation_span_start`` or ``metadata_filter_values`` — a name this
        test has never seen before but that still names a forbidden
        category.
        """
        names = inspect.signature(span_attributes).parameters
        for name in names:
            lowered = name.lower()
            for forbidden in _FORBIDDEN_SUBSTRINGS:
                assert forbidden not in lowered, (
                    f"span_attributes parameter {name!r} contains the forbidden "
                    f"substring {forbidden!r} (ADR-0022 decision 3)"
                )


class TestSpanAttributesAllowlistRoundTrip:
    """Every permitted attribute round-trips through its documented dotted key."""

    def test_no_arguments_returns_empty_dict(self) -> None:
        assert span_attributes() == {}

    def test_every_permitted_attribute_round_trips(self) -> None:
        result = span_attributes(
            collection="docs",
            retrieval_mode="hybrid",
            stage="rerank",
            top_k=10,
            result_count=8,
            candidate_count=40,
            chunk_count=120,
            document_count=6,
            duration_ms=12.5,
            failure_kind="index_inconsistent",
            request_id="req-123",
            http_status=200,
            embedding_provider="ollama",
            embedding_model="nomic-embed-text",
            embedding_dimensions=768,
            chat_provider="openai_compatible",
            chat_model="qwen2.5",
        )
        assert result == {
            "groundkit.collection": "docs",
            "groundkit.retrieval.mode": "hybrid",
            "groundkit.retrieval.stage": "rerank",
            "groundkit.retrieval.top_k": 10,
            "groundkit.result_count": 8,
            "groundkit.candidate_count": 40,
            "groundkit.chunk_count": 120,
            "groundkit.document_count": 6,
            "groundkit.duration_ms": 12.5,
            "groundkit.failure_kind": "index_inconsistent",
            "groundkit.request_id": "req-123",
            "groundkit.http.status_code": 200,
            "groundkit.embedding.provider": "ollama",
            "groundkit.embedding.model": "nomic-embed-text",
            "groundkit.embedding.dimensions": 768,
            "groundkit.chat.provider": "openai_compatible",
            "groundkit.chat.model": "qwen2.5",
        }

    def test_round_trip_covers_every_declared_parameter(self) -> None:
        """The round-trip above must not silently fall behind the signature.

        Without this, adding a parameter to ``span_attributes`` and
        forgetting to exercise it leaves an allowlisted attribute with no
        test that it maps to the key the module docstring advertises.
        """
        exercised = {
            "collection",
            "retrieval_mode",
            "stage",
            "top_k",
            "result_count",
            "candidate_count",
            "chunk_count",
            "document_count",
            "duration_ms",
            "failure_kind",
            "request_id",
            "http_status",
            "embedding_provider",
            "embedding_model",
            "embedding_dimensions",
            "chat_provider",
            "chat_model",
        }
        assert set(inspect.signature(span_attributes).parameters) == exercised

    def test_partial_arguments_include_only_those_provided(self) -> None:
        result = span_attributes(collection="docs", result_count=3)
        assert result == {"groundkit.collection": "docs", "groundkit.result_count": 3}

    def test_none_arguments_are_omitted_entirely(self) -> None:
        """A ``None`` argument must never appear as a ``None``-valued key.

        OTel span attributes may not be ``None``, so the contract is
        omission, not a null placeholder.
        """
        result = span_attributes(collection="docs", top_k=None, request_id=None)
        assert "groundkit.retrieval.top_k" not in result
        assert "groundkit.request_id" not in result
        assert None not in result.values()


def _make_capturing_logger(
    name: str, formatter: logging.Formatter
) -> tuple[logging.Logger, io.StringIO]:
    """Build an isolated logger that writes formatted records into an in-memory buffer.

    Args:
        name: A unique logger name, so tests do not share handlers.
        formatter: The formatter under test.

    Returns:
        The configured logger and the buffer its single handler writes to.
    """
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.handlers = [handler]
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger, stream


class TestJsonLogFormatter:
    """One parseable JSON object per record, carrying `extra` and exception info."""

    def test_produces_one_parseable_json_object_with_core_fields(self) -> None:
        logger, stream = _make_capturing_logger("telemetry-test.basic", JsonLogFormatter())
        logger.info("index built collection=docs chunk_count=12")
        record = json.loads(stream.getvalue().strip())
        assert record["level"] == "INFO"
        assert record["logger"] == "telemetry-test.basic"
        assert record["message"] == "index built collection=docs chunk_count=12"
        assert "timestamp" in record

    def test_message_text_is_unchanged_by_percent_formatting(self) -> None:
        """ADR-0022 decision 4: existing ``key=value`` message strings do not change."""
        logger, stream = _make_capturing_logger("telemetry-test.percent", JsonLogFormatter())
        logger.info("search completed route=%s status=%d", "/search", 200)
        record = json.loads(stream.getvalue().strip())
        assert record["message"] == "search completed route=/search status=200"

    def test_carries_extra_fields(self) -> None:
        logger, stream = _make_capturing_logger("telemetry-test.extra", JsonLogFormatter())
        logger.info(
            "search completed",
            extra={"groundkit.result_count": 5, "route": "/search"},
        )
        record = json.loads(stream.getvalue().strip())
        assert record["groundkit.result_count"] == 5
        assert record["route"] == "/search"

    def test_survives_non_serializable_extra_value(self) -> None:
        class _Unserializable:
            def __str__(self) -> str:
                return "<unserializable>"

        logger, stream = _make_capturing_logger(
            "telemetry-test.nonserializable", JsonLogFormatter()
        )
        logger.info("weird value", extra={"weird": _Unserializable()})
        # The point of this test is that formatting does not raise; if it
        # did, this call itself would blow up before json.loads ever runs.
        record = json.loads(stream.getvalue().strip())
        assert record["weird"] == "<unserializable>"

    def test_includes_exception_info_when_present(self) -> None:
        logger, stream = _make_capturing_logger("telemetry-test.exception", JsonLogFormatter())
        try:
            raise ValueError("boom")
        except ValueError:
            logger.exception("failed")
        record = json.loads(stream.getvalue().strip())
        assert "exception" in record
        assert "ValueError" in record["exception"]
        assert "boom" in record["exception"]

    def test_omits_exception_key_when_no_exception(self) -> None:
        logger, stream = _make_capturing_logger("telemetry-test.no-exception", JsonLogFormatter())
        logger.info("all fine")
        record = json.loads(stream.getvalue().strip())
        assert "exception" not in record

    def test_extra_field_cannot_overwrite_a_core_field(self) -> None:
        """A colliding ``extra`` key must not corrupt the record's real fields.

        ``"logger"`` is not a genuine ``LogRecord`` attribute name (the real
        one is ``name``), so stdlib ``logging`` itself does not block this
        collision — the formatter has to.
        """
        logger, stream = _make_capturing_logger("telemetry-test.collision", JsonLogFormatter())
        logger.info("careful", extra={"logger": "spoofed-logger-name"})
        record = json.loads(stream.getvalue().strip())
        assert record["logger"] == "telemetry-test.collision"


@pytest.fixture
def _isolated_root_logger() -> Iterator[logging.Logger]:
    """Save and restore the root logger's handlers/level around a test.

    ``configure_logging`` mutates process-global state (the root logger), so
    tests exercising it must not leak handlers into whatever runs next in
    the same pytest process.
    """
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    for handler in original_handlers:
        root.removeHandler(handler)
    try:
        yield root
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()
        for handler in original_handlers:
            root.addHandler(handler)
        root.setLevel(original_level)


class TestConfigureLogging:
    """Idempotent, stderr-only, and honours ``GROUNDKIT_LOG_FORMAT``."""

    def test_first_call_attaches_exactly_one_handler(
        self, _isolated_root_logger: logging.Logger
    ) -> None:
        configure_logging()
        matching = [h for h in _isolated_root_logger.handlers if h.name == _HANDLER_NAME]
        assert len(matching) == 1

    def test_second_call_does_not_attach_a_second_handler(
        self, _isolated_root_logger: logging.Logger
    ) -> None:
        configure_logging()
        configure_logging()
        matching = [h for h in _isolated_root_logger.handlers if h.name == _HANDLER_NAME]
        assert len(matching) == 1

    def test_repeat_call_still_updates_the_level(
        self, _isolated_root_logger: logging.Logger
    ) -> None:
        configure_logging(level=logging.WARNING)
        configure_logging(level=logging.DEBUG)
        assert _isolated_root_logger.level == logging.DEBUG

    def test_never_attaches_a_stdout_handler(self, _isolated_root_logger: logging.Logger) -> None:
        """ADR-0014 decision 5: stdout is JSON-RPC-only on the stdio MCP transport.

        Mirrors the exact check ``service/mcp_server.py:_force_logging_to_stderr``
        uses to find handlers it needs to re-point, applied here to prove
        this function never creates one of those in the first place.
        """
        configure_logging()
        assert not any(
            isinstance(handler, logging.StreamHandler) and handler.stream is sys.stdout
            for handler in _isolated_root_logger.handlers
        )

    def test_handler_stream_is_stderr(self, _isolated_root_logger: logging.Logger) -> None:
        configure_logging()
        handler = next(h for h in _isolated_root_logger.handlers if h.name == _HANDLER_NAME)
        assert isinstance(handler, logging.StreamHandler)
        assert handler.stream is sys.stderr

    def test_json_format_env_var_selects_json_formatter(
        self, _isolated_root_logger: logging.Logger, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GROUNDKIT_LOG_FORMAT", "json")
        configure_logging()
        handler = next(h for h in _isolated_root_logger.handlers if h.name == _HANDLER_NAME)
        assert isinstance(handler.formatter, JsonLogFormatter)

    @pytest.mark.parametrize("value", [None, "yaml", "JSON", ""])
    def test_unset_or_unrecognized_format_env_var_stays_human_readable(
        self,
        _isolated_root_logger: logging.Logger,
        monkeypatch: pytest.MonkeyPatch,
        value: str | None,
    ) -> None:
        """ADR-0022 decision 4: human-readable is the default, not an opt-out.

        A local ``grk search`` printing JSON to a developer's terminal would
        be a regression, so only the exact value ``"json"`` may switch it.
        """
        if value is None:
            monkeypatch.delenv("GROUNDKIT_LOG_FORMAT", raising=False)
        else:
            monkeypatch.setenv("GROUNDKIT_LOG_FORMAT", value)
        configure_logging()
        handler = next(h for h in _isolated_root_logger.handlers if h.name == _HANDLER_NAME)
        assert not isinstance(handler.formatter, JsonLogFormatter)


class TestGetTracer:
    """``opentelemetry-api`` is a base dependency; no SDK is installed in this suite."""

    def test_returns_a_tracer(self) -> None:
        tracer = get_tracer()
        assert hasattr(tracer, "start_as_current_span")

    def test_span_is_non_recording_and_accepts_attributes_without_raising(self) -> None:
        """ADR-0022 decision 1: with no SDK, spans are non-recording, not absent.

        The call must succeed silently rather than raising, guarding
        against the alternative design (a hand-rolled no-op shim) the ADR
        rejected — a real ``opentelemetry-api`` tracer already behaves this
        way with nothing configured.
        """
        tracer = get_tracer()
        with tracer.start_as_current_span("test-span") as span:
            assert span.is_recording() is False
            span.set_attribute("groundkit.collection", "docs")
            span.set_attributes(span_attributes(collection="docs", top_k=5, result_count=3))

    def test_get_tracer_is_stable_across_calls(self) -> None:
        """Every instrumentation site shares one instrumentation scope."""
        first = get_tracer()
        second = get_tracer()
        assert type(first) is type(second)


class TestExportRequested:
    """The gate deciding whether :func:`configure_tracing` does anything.

    Kept as its own pure function, and tested directly, because the
    alternative — inferring intent inside ``configure_tracing`` after the
    SDK import — would make "did a plain ``grk search`` just try to reach a
    collector?" untestable without a network namespace.
    """

    def test_no_environment_means_no_export(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ADR-0022's consequences: no ``OTEL_*`` config, no network calls."""
        for name in _ALL_OTEL_VARS:
            monkeypatch.delenv(name, raising=False)
        assert _export_requested() is False

    @pytest.mark.parametrize("endpoint_var", _ENDPOINT_VARS)
    def test_an_endpoint_requests_export(
        self, endpoint_var: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Either standard endpoint variable is enough to request export."""
        for name in _ALL_OTEL_VARS:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv(endpoint_var, "http://collector:4318")
        assert _export_requested() is True

    def test_exporter_none_overrides_a_configured_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``OTEL_TRACES_EXPORTER=none`` must switch export off.

        Without this, an environment that sets an endpoint globally (a
        cluster-wide default, say) would give an operator no way to opt a
        single workload out.
        """
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")
        monkeypatch.setenv(_EXPORTER_VAR, "none")
        assert _export_requested() is False

    def test_whitespace_only_endpoint_is_not_a_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty or blank value is "unset", not "export to nowhere"."""
        for name in _ALL_OTEL_VARS:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "   ")
        assert _export_requested() is False


class TestConfigureTracing:
    """The bootstrap that makes any span in this package actually record.

    **This class is the regression test for a defect that was found by
    running the compose stack, not by reading the code.** The first version
    of this change installed ``opentelemetry-sdk``, set every ``OTEL_*``
    variable in the container, opened spans correctly at all three SPEC.md
    §3 sites — and exported nothing at all, because nothing ever called
    ``set_tracer_provider``. The SDK reads those variables inside
    ``opentelemetry.sdk._configuration``, which runs under the
    ``opentelemetry-instrument`` launcher and *not* on import, so the API
    kept handing out a ``ProxyTracerProvider`` whose spans are
    non-recording. There was no error, no warning, and a green test suite.

    The positive case runs in a **subprocess**, deliberately.
    ``trace.set_tracer_provider`` mutates process-global state and cannot be
    undone, so installing a real provider in-process would leak into every
    other test in this suite — including
    ``TestGetTracer``'s assertion that a span is non-recording, which would
    then pass or fail depending on test execution order.
    """

    def test_does_nothing_when_export_is_not_requested(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A default ``grk search`` must not install an exporter."""
        for name in _ALL_OTEL_VARS:
            monkeypatch.delenv(name, raising=False)
        configure_tracing()
        assert isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider)

    def test_installs_a_recording_provider_when_configured(self) -> None:
        """The defect this whole class exists for: spans must actually record.

        Asserts in a subprocess that after ``configure_tracing()`` the
        global provider is no longer the API's proxy and a span opened
        through :func:`get_tracer` reports ``is_recording()``. No collector
        is reachable at the configured endpoint; that is deliberate and
        irrelevant — the export attempt is asynchronous and failing to
        deliver is not the same as failing to record.
        """
        script = textwrap.dedent(
            """
            from opentelemetry import trace
            from groundkit.telemetry import configure_tracing, get_tracer

            configure_tracing()
            provider_is_proxy = isinstance(
                trace.get_tracer_provider(), trace.ProxyTracerProvider
            )
            with get_tracer().start_as_current_span("probe") as span:
                recording = span.is_recording()
            print(f"proxy={provider_is_proxy} recording={recording}")
            """
        )
        env = {
            **os.environ,
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:4318",
            "OTEL_TRACES_EXPORTER": "otlp",
            "OTEL_SERVICE_NAME": "groundkit-test",
        }
        completed = subprocess.run(  # noqa: S603 - fixed argv, interpreter is sys.executable
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        assert completed.returncode == 0, completed.stderr
        assert "proxy=False recording=True" in completed.stdout, completed.stdout

    def test_is_idempotent_and_does_not_replace_an_installed_provider(self) -> None:
        """A second call must leave the first provider in place.

        OTel's own ``set_tracer_provider`` only logs a warning and ignores a
        second call, so without the explicit guard the observable outcome
        would depend on which caller ran first — a race between, say, the
        CLI entry point and a library embedding it.
        """
        script = textwrap.dedent(
            """
            from opentelemetry import trace
            from groundkit.telemetry import configure_tracing

            configure_tracing()
            first = trace.get_tracer_provider()
            configure_tracing()
            second = trace.get_tracer_provider()
            print(f"same={first is second}")
            """
        )
        env = {
            **os.environ,
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:4318",
            "OTEL_TRACES_EXPORTER": "otlp",
        }
        completed = subprocess.run(  # noqa: S603 - fixed argv, interpreter is sys.executable
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        assert completed.returncode == 0, completed.stderr
        assert "same=True" in completed.stdout, completed.stdout

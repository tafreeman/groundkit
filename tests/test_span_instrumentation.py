"""Regression tests for the three SPEC.md §3 span sites (ADR-0022).

Four call sites, three conceptual SPEC.md §3 sites: ``Indexer.index_source``
and ``Indexer.index_directory`` (ingest — one span per public entry point,
never one per file; ADR-0022 decision 5's ``Indexer.run`` no longer exists),
``Retriever.search`` (retrieve), and ``Synthesizer.synthesize`` (synthesize).

These tests do not depend on the optional ``otel`` extra. Constructing a real
``TracerProvider`` + ``InMemorySpanExporter`` pair would only prove the
OpenTelemetry SDK itself works, and — more to the point — a *non-recording*
span (what ``opentelemetry-api`` alone produces with no SDK configured)
silently discards every attribute set on it, so there would be nothing left
to assert against without the ``otel`` extra actually installed. Each test
below instead monkeypatches the module-level ``tracer`` object each call site
calls through — ``indexer.py``, ``retrieval/search.py`` and
``providers/synthesis.py`` each bind ``tracer = get_tracer()`` once at import
time, mirroring this repo's ``logger = logging.getLogger(__name__)``
convention — with a hand-rolled fake that records every span's name and every
attribute ever set on it. Deterministic regardless of whether
``opentelemetry-sdk`` happens to be installed in the dev environment
(ADR-0022 decision 1: the API is a base dependency, the SDK is not).

The single most important assertion in this file is the "no leak" family:
a distinctive sentinel is passed as the query (search, synthesize) or baked
into a path/content (ingest, synthesize's retrieved content) and asserted
absent from every attribute KEY and every attribute VALUE recorded on every
span opened during the call — the regression test for the exact leak
ADR-0022 exists to prevent (SPEC.md §3: never query text, chunk/document
content, citation spans, or absolute source paths on a span, at any level).

Async code under test is driven with ``asyncio.run()`` inside plain ``def``
test functions, matching this repo's house style (``tests/test_synthesis.py``,
``tests/test_retrieval.py``, ...); pytest-asyncio is not a dependency.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType

import pytest

from groundkit import indexer as indexer_module
from groundkit.contracts import Chunk, Document, RetrievalResult, SearchResponse
from groundkit.errors import ChatError, IngestionError, RetrievalError
from groundkit.index.metadata import SQLiteMetadataStore
from groundkit.indexer import Indexer
from groundkit.ingestion.loaders import FileLoader
from groundkit.providers import synthesis as synthesis_module
from groundkit.providers.synthesis import SynthesizedAnswer, Synthesizer
from groundkit.retrieval import search as search_module
from groundkit.retrieval.search import Retriever

#: Distinctive enough that its accidental presence anywhere is unambiguous —
#: a short, plausible-looking query would not prove a leak test means
#: anything, since a common word could coincidentally match an unrelated
#: label or count.
_SENTINEL_QUERY = "zzz-sentinel-9f3c2f61-must-never-leave-this-process-zzz"


# ── A hand-rolled Tracer/Span double (see the module docstring for why) ────


class _FakeSpan:
    """Records its own name and every attribute ever set on it."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.attributes: dict[str, str | int | float] = {}

    def set_attributes(self, attributes: Mapping[str, str | int | float]) -> None:
        self.attributes.update(attributes)

    def __enter__(self) -> _FakeSpan:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # Never suppress: a span must not swallow an exception, and this
        # fake exercising that property is part of what these tests pin down.
        return None


class _FakeTracer:
    """Records every span opened via ``start_as_current_span``, in call order."""

    def __init__(self) -> None:
        self.spans: list[_FakeSpan] = []

    def start_as_current_span(self, name: str) -> _FakeSpan:
        span = _FakeSpan(name)
        self.spans.append(span)
        return span


@pytest.fixture
def fake_tracer(monkeypatch: pytest.MonkeyPatch) -> _FakeTracer:
    """Patch the module-level ``tracer`` each of the three call sites uses.

    Patching the module attribute directly — rather than patching
    ``groundkit.telemetry.get_tracer`` — reaches every call site regardless
    of import order, and matches exactly what production code does: each
    module calls ``tracer.start_as_current_span(...)`` against its own
    already-bound module-level name, never ``get_tracer()`` again per call.
    """
    tracer = _FakeTracer()
    monkeypatch.setattr(indexer_module, "tracer", tracer)
    monkeypatch.setattr(search_module, "tracer", tracer)
    monkeypatch.setattr(synthesis_module, "tracer", tracer)
    return tracer


def _values(span: _FakeSpan) -> list[str | int | float]:
    return list(span.attributes.values())


def _assert_no_leak(tracer: _FakeTracer, secret: str) -> None:
    """Assert ``secret`` appears in no attribute key or value on any recorded span."""
    for span in tracer.spans:
        for key, value in span.attributes.items():
            assert secret not in key, f"leaked into attribute key {key!r} on {span.name!r}"
            assert secret not in str(value), (
                f"leaked into attribute {key!r}={value!r} on {span.name!r}"
            )


# ── Shared fixtures: a store, a chat double, a retrieval result ────────────


async def _open_store(tmp_path: Path, name: str = "default") -> SQLiteMetadataStore:
    return await SQLiteMetadataStore.open(tmp_path / "idx", name)


async def _populated_store(tmp_path: Path) -> SQLiteMetadataStore:
    """A store with one document already indexed, for search-only tests."""
    store = await _open_store(tmp_path)
    content = "Retrieval systems rank documents by relevance to a query."
    doc = Document(source=str(tmp_path / "alpha.md"), content=content)
    await store.upsert_document(source=doc.source, document_id=doc.document_id, content_hash="h")
    chunk = Chunk(
        document_id=doc.document_id,
        chunk_index=0,
        content=content,
        start_offset=0,
        end_offset=len(content),
        metadata={},
    )
    await store.add_chunks([chunk], source=doc.source)
    return store


class _ScriptedChat:
    """Minimal ``ChatProtocol`` double: a fixed completion, or a scripted error."""

    def __init__(self, completion: str = "", *, error: Exception | None = None) -> None:
        self._completion = completion
        self._error = error

    @property
    def provider(self) -> str:
        return "scripted"

    @property
    def model_name(self) -> str:
        return "scripted-v1"

    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        if self._error is not None:
            raise self._error
        return self._completion


def _result(content: str = "the sky is blue") -> RetrievalResult:
    return RetrievalResult(
        content=content,
        score=1.0,
        document_id="doc-1",
        chunk_id="chunk-1",
        source="doc.txt",
        start_offset=0,
        end_offset=len(content),
    )


# ── Ingest: groundkit.ingest.index_source / groundkit.ingest.index_directory ──


class TestIngestSpans:
    def test_index_source_span_carries_counts_and_collection(
        self, tmp_path: Path, fake_tracer: _FakeTracer
    ) -> None:
        async def run() -> None:
            docs_dir = tmp_path / "docs"
            docs_dir.mkdir()
            source = docs_dir / "alpha.md"
            source.write_text("Groundkit indexes one document for this test.", encoding="utf-8")
            store = await _open_store(tmp_path)
            try:
                idx = Indexer(store, FileLoader(allowed_base_dir=docs_dir), collection="default")
                report = await idx.index_source(str(source))
            finally:
                await store.close()

            assert report.documents_indexed == 1
            [span] = fake_tracer.spans
            assert span.name == "groundkit.ingest.index_source"
            values = _values(span)
            assert report.documents_indexed in values
            assert report.chunks_written in values
            assert "default" in values

        asyncio.run(run())

    def test_index_source_failure_records_typed_failure_code(
        self, tmp_path: Path, fake_tracer: _FakeTracer
    ) -> None:
        async def run() -> None:
            docs_dir = tmp_path / "docs"
            docs_dir.mkdir()
            store = await _open_store(tmp_path)
            try:
                idx = Indexer(store, FileLoader(allowed_base_dir=docs_dir))
                with pytest.raises(IngestionError):
                    # Never written: loading it fails closed as IngestionError.
                    await idx.index_source(str(docs_dir / "missing.md"))
            finally:
                await store.close()

            [span] = fake_tracer.spans
            assert span.name == "groundkit.ingest.index_source"
            assert "IngestionError" in _values(span)

        asyncio.run(run())

    def test_index_directory_span_carries_document_and_chunk_counts(
        self, tmp_path: Path, fake_tracer: _FakeTracer
    ) -> None:
        async def run() -> None:
            docs_dir = tmp_path / "docs"
            docs_dir.mkdir()
            (docs_dir / "alpha.md").write_text("First document body text.", encoding="utf-8")
            (docs_dir / "beta.md").write_text("Second document body text.", encoding="utf-8")
            store = await _open_store(tmp_path)
            try:
                idx = Indexer(store, FileLoader(allowed_base_dir=docs_dir), collection="default")
                report = await idx.index_directory(str(docs_dir))
            finally:
                await store.close()

            assert report.documents_indexed == 2
            [span] = fake_tracer.spans
            assert span.name == "groundkit.ingest.index_directory"
            values = _values(span)
            assert report.documents_indexed in values
            assert report.chunks_written in values
            assert "default" in values

        asyncio.run(run())

    def test_index_directory_failure_records_typed_failure_code(
        self, tmp_path: Path, fake_tracer: _FakeTracer
    ) -> None:
        async def run() -> None:
            store = await _open_store(tmp_path)
            try:
                idx = Indexer(store, FileLoader(allowed_base_dir=tmp_path))
                with pytest.raises(IngestionError, match="not found"):
                    await idx.index_directory(str(tmp_path / "does-not-exist"))
            finally:
                await store.close()

            [span] = fake_tracer.spans
            assert span.name == "groundkit.ingest.index_directory"
            assert "IngestionError" in _values(span)

        asyncio.run(run())

    def test_ingest_spans_never_carry_the_source_path(
        self, tmp_path: Path, fake_tracer: _FakeTracer
    ) -> None:
        """ADR-0022 decision 3: an absolute source path is exactly as
        forbidden on a span as query text — ``index_source`` takes one
        directly, and it must never reach either span, at any point."""

        async def run() -> None:
            docs_dir = tmp_path / "leaky-path-marker-83a1"
            docs_dir.mkdir()
            source = docs_dir / "alpha.md"
            source.write_text("Body text unrelated to the path.", encoding="utf-8")
            store = await _open_store(tmp_path)
            try:
                idx = Indexer(store, FileLoader(allowed_base_dir=docs_dir))
                await idx.index_source(str(source))
                await idx.index_directory(str(docs_dir))
            finally:
                await store.close()

        asyncio.run(run())
        assert len(fake_tracer.spans) == 2
        _assert_no_leak(fake_tracer, "leaky-path-marker-83a1")


# ── Retrieve: groundkit.retrieve.search ─────────────────────────────────────


class TestRetrieveSpan:
    def test_search_span_name_attributes_and_no_query_leak(
        self, tmp_path: Path, fake_tracer: _FakeTracer
    ) -> None:
        async def run() -> SearchResponse:
            store = await _populated_store(tmp_path)
            try:
                retriever = await Retriever.open(store, collection="default")
                return await retriever.search(_SENTINEL_QUERY)
            finally:
                await store.close()

        response = asyncio.run(run())
        [span] = fake_tracer.spans
        assert span.name == "groundkit.retrieve.search"
        values = _values(span)
        assert "bm25" in values  # retrieval_mode and stage, both "bm25" here
        assert response.total_results in values  # result_count
        assert "default" in values  # collection
        _assert_no_leak(fake_tracer, _SENTINEL_QUERY)

    def test_search_failure_records_typed_failure_code_and_no_query_leak(
        self, tmp_path: Path, fake_tracer: _FakeTracer
    ) -> None:
        async def run() -> None:
            store = await _populated_store(tmp_path)
            try:
                retriever = await Retriever.open(store)
                with pytest.raises(RetrievalError, match="top_k"):
                    await retriever.search(_SENTINEL_QUERY, top_k=0)
            finally:
                await store.close()

        asyncio.run(run())
        [span] = fake_tracer.spans
        assert span.name == "groundkit.retrieve.search"
        assert "RetrievalError" in _values(span)
        _assert_no_leak(fake_tracer, _SENTINEL_QUERY)


# ── Synthesize: groundkit.synthesize.synthesize ─────────────────────────────


class TestSynthesizeSpan:
    def test_synthesize_span_name_result_count_and_no_query_leak(
        self, fake_tracer: _FakeTracer
    ) -> None:
        async def run() -> SynthesizedAnswer:
            chat = _ScriptedChat("[1] a cited answer.")
            synthesizer = Synthesizer(chat)
            return await synthesizer.synthesize(_SENTINEL_QUERY, [_result()])

        answer = asyncio.run(run())
        assert answer.citations  # sanity: the scripted completion actually cited [1]
        [span] = fake_tracer.spans
        assert span.name == "groundkit.synthesize.synthesize"
        assert 1 in _values(span)  # result_count == len(results)
        _assert_no_leak(fake_tracer, _SENTINEL_QUERY)

    def test_synthesize_failure_records_typed_failure_code_and_no_query_leak(
        self, fake_tracer: _FakeTracer
    ) -> None:
        async def run() -> None:
            chat = _ScriptedChat(error=ChatError("simulated provider failure"))
            synthesizer = Synthesizer(chat)
            with pytest.raises(ChatError):
                await synthesizer.synthesize(_SENTINEL_QUERY, [_result()])

        asyncio.run(run())
        [span] = fake_tracer.spans
        assert span.name == "groundkit.synthesize.synthesize"
        assert "ChatError" in _values(span)
        _assert_no_leak(fake_tracer, _SENTINEL_QUERY)

    def test_synthesize_never_attaches_retrieved_content_or_completion_text(
        self, fake_tracer: _FakeTracer
    ) -> None:
        """The sharpest allowlist boundary (ADR-0022 decision 5): retrieved
        content and the completion text sit right next to this span and must
        never reach it, even though both are in scope throughout the call."""

        async def run() -> None:
            chat = _ScriptedChat("[1] a distinctively-worded-completion-9f2a marker.")
            synthesizer = Synthesizer(chat)
            await synthesizer.synthesize(
                "an unrelated ordinary query",
                [_result(content="distinctively-worded-source-content-7c1b marker")],
            )

        asyncio.run(run())
        _assert_no_leak(fake_tracer, "distinctively-worded-source-content-7c1b")
        _assert_no_leak(fake_tracer, "distinctively-worded-completion-9f2a")


# ── All four call sites together: span names, in order, no leak ────────────


class TestAllSpanSitesTogether:
    def test_all_four_call_sites_open_the_expected_span_names_with_no_leak(
        self, tmp_path: Path, fake_tracer: _FakeTracer
    ) -> None:
        async def run() -> None:
            docs_dir = tmp_path / "docs"
            docs_dir.mkdir()
            (docs_dir / "alpha.md").write_text("A short document body.", encoding="utf-8")
            store = await _open_store(tmp_path)
            try:
                idx = Indexer(store, FileLoader(allowed_base_dir=docs_dir))
                await idx.index_source(str(docs_dir / "alpha.md"))
                await idx.index_directory(str(docs_dir))
                retriever = await Retriever.open(store)
                await retriever.search(_SENTINEL_QUERY)
            finally:
                await store.close()
            synthesizer = Synthesizer(_ScriptedChat("[1] an answer."))
            await synthesizer.synthesize(_SENTINEL_QUERY, [_result()])

        asyncio.run(run())
        assert [span.name for span in fake_tracer.spans] == [
            "groundkit.ingest.index_source",
            "groundkit.ingest.index_directory",
            "groundkit.retrieve.search",
            "groundkit.synthesize.synthesize",
        ]
        _assert_no_leak(fake_tracer, _SENTINEL_QUERY)

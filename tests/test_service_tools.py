"""Tests for the transport-agnostic tool registry (ADR-0014).

The enforcement tests here are the load-bearing ones. None of them can be shown
to fail by reverting source — ``service/`` did not exist before this phase — so
each is demonstrated instead by *injecting the violation it exists to catch*,
which is the meaningful direction for a guard and is explicitly NOT the SPEC.md
§8 revert procedure. Each says so in its own docstring rather than leaving a
reader to assume a revert was performed.
"""

from __future__ import annotations

import ast
import asyncio
import re
import sqlite3
import threading
import typing
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from groundkit.contracts import Chunk, RetrievalResult, SearchResponse
from groundkit.errors import ConfigurationError, RerankerNotConfiguredError
from groundkit.index.metadata import SQLiteMetadataStore
from groundkit.runtime import CollectionRegistry, CollectionRuntime
from groundkit.service import tools as tools_module

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from groundkit.service.schemas import (
    FetchChunkRequest,
    IndexStatusRequest,
    ListCollectionsRequest,
    SearchRequest,
)
from groundkit.service.tools import (
    MAX_CONCURRENT_CORPUS_SCANS,
    TOOL_NAMES,
    TOOLS,
    ServiceContext,
    SideEffect,
)

#: Store members that commit durable state. A service surface must be unable to
#: reach any of them.
_MUTATING_STORE_MEMBERS = frozenset(
    {
        "upsert_document",
        "add_chunks",
        "replace_document",
        "delete_document",
        "write_manifest",
    }
)

_SERVICE_DIR = Path(__file__).resolve().parents[1] / "src" / "groundkit" / "service"

#: Modules that write. A service module importing any of these is three steps
#: upstream of the route that would have exposed a write.
_WRITE_PATH_MODULES = frozenset(
    {
        "groundkit.indexer",
        "groundkit.ingestion.loaders",
        "groundkit.ingestion.pipeline",
    }
)

#: Modules on the synthesis boundary. Barred from the service package by
#: ADR-0019 decision 4: synthesis is a read, but serving it would add cost
#: amplification (an unauthenticated request triggering a billable model
#: call) and egress amplification (a caller making the operator publish
#: their corpus to a cloud provider) — neither bounded by the loopback bind.
_SYNTHESIS_PATH_MODULES = frozenset(
    {
        "groundkit.providers.llm",
        "groundkit.providers.query_rewrite",
        "groundkit.providers.synthesis",
        "groundkit.evals.judge",
        "groundkit.answer",
    }
)

#: The full barred set the AST scan enforces.
_BARRED_SERVICE_IMPORTS = _WRITE_PATH_MODULES | _SYNTHESIS_PATH_MODULES


async def _seed(tmp_path: Path) -> tuple[Path, Path, str]:
    """Write a real source file and index it so citations genuinely resolve."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    text = "Turbine maintenance intervals depend on load factor and ambient temperature."
    (corpus / "a.md").write_text(text, encoding="utf-8")

    index_dir = tmp_path / "index"
    index_dir.mkdir()
    store = await SQLiteMetadataStore.open(index_dir, "default")
    try:
        chunk = Chunk(
            chunk_id="c1",
            document_id="doc-1",
            chunk_index=0,
            content=text,
            start_offset=0,
            end_offset=len(text),
        )
        await store.replace_document(str(corpus / "a.md"), "doc-1", "h1", [chunk])
    finally:
        await store.close()
    return index_dir, corpus, "c1"


def _context(index_dir: Path, corpus: Path, reranker: object | None = None) -> ServiceContext:
    return ServiceContext(
        registry=CollectionRegistry(index_dir),
        index_dir=index_dir,
        base_dir=corpus,
        reranker=reranker,  # type: ignore[arg-type]
    )


# -- Enforcement -----------------------------------------------------------


def test_every_registered_tool_is_read_only() -> None:
    """Guard, demonstrated by injection: flip one spec's side_effect and this fails."""
    assert TOOLS, "the registry is empty"
    assert all(spec.side_effect == "read_only" for spec in TOOLS)


def test_side_effect_literal_admits_exactly_one_value() -> None:
    """The type system, not a convention, is what keeps Phase 4 read-only.

    Adding a mutating operation requires widening this Literal, which
    ``mypy --strict`` rejects at every existing construction site until someone
    does it deliberately and visibly. Per SPEC.md §7 that widening must arrive
    in the same change as the shared-secret header, the constant-time compare,
    and the unset-secret disable.
    """
    assert typing.get_args(SideEffect) == ("read_only",)


def test_registry_declares_exactly_the_four_tools_spec_names() -> None:
    """SPEC.md §1.2 names four tools; ingest is not among them.

    Pinned so a later reader finds a decision rather than an omission, and so
    that adding a fifth operation is a deliberate edit here.
    """
    assert {"search", "fetch_chunk", "list_collections", "index_status"} == TOOL_NAMES


def test_no_request_model_accepts_provider_or_filesystem_configuration() -> None:
    """ADR-0014 decision 6, made executable.

    The structural argument — the service receives a constructed
    EmbeddingConfig, so there is no resolution path inside ``service/`` — is
    what makes this true today. This test is what keeps it true as the surface
    grows, because it walks every registered request model rather than a list
    someone has to remember to update.

    Guard, demonstrated by injection: add ``base_url: str | None = None`` to
    SearchRequest and this fails. It is what keeps SECURITY.md's sentence "no
    network-facing caller can set base_url" honest.
    """
    forbidden_exact = {"base_url", "index_dir", "base_dir", "api_key_env"}
    for spec in TOOLS:
        for field in spec.request_model.model_fields:
            assert field not in forbidden_exact, f"{spec.name}.{field} exposes server config"
            assert not field.startswith("embed"), f"{spec.name}.{field} exposes embedder config"


def test_request_models_reject_unknown_fields() -> None:
    """extra="forbid" everywhere: a caller learns their field was not understood.

    Silently ignoring it is the worse failure — the caller believes a setting
    took effect.
    """
    with pytest.raises(ValueError, match=r"base_url|extra"):
        SearchRequest(query="x", base_url="http://evil.example")  # type: ignore[call-arg]


def test_service_package_imports_no_write_path() -> None:
    """AST scan: no service module reaches the ingest path or the synthesis boundary.

    Guard, demonstrated by injection: add ``from groundkit.indexer import
    Indexer`` to any service module and this fails. It fires three steps
    upstream of the route that would have exposed a write, which is why it is
    worth having alongside the registry check. ADR-0019 decision 4 extends
    the barred set to the synthesis boundary for the same three-steps-upstream
    reason — see ``_SYNTHESIS_PATH_MODULES``.
    """
    offenders: list[str] = []
    for path in _SERVICE_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in _BARRED_SERVICE_IMPORTS:
                offenders.append(f"{path.name} -> {node.module}")
            elif isinstance(node, ast.Import):
                offenders.extend(
                    f"{path.name} -> {alias.name}"
                    for alias in node.names
                    if alias.name in _BARRED_SERVICE_IMPORTS
                )
    assert not offenders, f"service package reaches a barred module: {offenders}"


def test_collection_runtime_exposes_no_mutating_member() -> None:
    """The runtime IS the read-only facade handlers reach collections through.

    Asserted as an exact disjointness against a named mutation set rather than
    as "does not contain delete_document", so a rename does not defeat it.
    """
    public = {name for name in dir(CollectionRuntime) if not name.startswith("_")}
    assert public.isdisjoint(_MUTATING_STORE_MEMBERS), (
        f"CollectionRuntime exposes a mutating member: {public & _MUTATING_STORE_MEMBERS}"
    )


# -- Behaviour -------------------------------------------------------------


def test_search_returns_the_existing_contract_unchanged(tmp_path: Path) -> None:
    """search reuses SearchResponse, so REST and `grk search --json` agree in shape."""

    async def run() -> None:
        index_dir, corpus, _ = await _seed(tmp_path)
        ctx = _context(index_dir, corpus)
        try:
            response = await tools_module.handle_search(ctx, SearchRequest(query="turbine"))
            assert isinstance(response, SearchResponse)
            assert response.results
            assert response.metadata["reranked"] is False
        finally:
            await ctx.registry.aclose()

    asyncio.run(run())


def test_rerank_without_a_configured_reranker_is_refused(tmp_path: Path) -> None:
    """Never a 200 carrying an unreranked list.

    Runs in the default suite because it needs no model, so the "never a silent
    passthrough" guarantee is proved offline without torch.

    Guard, demonstrated by injection: replace the raise with a passthrough
    returning the unreranked results and this fails.
    """

    async def run() -> None:
        index_dir, corpus, _ = await _seed(tmp_path)
        ctx = _context(index_dir, corpus)
        try:
            with pytest.raises(RerankerNotConfiguredError, match="--rerank"):
                await tools_module.handle_search(ctx, SearchRequest(query="turbine", rerank=True))
        finally:
            await ctx.registry.aclose()

    asyncio.run(run())


def test_a_configured_reranker_is_not_applied_unless_requested(tmp_path: Path) -> None:
    """rerank defaults to false, and a server that has one does not force it.

    The narrowing in handle_search binds the reranker only when the request
    asked; an earlier draft bound it from the context first, which would have
    reranked every request on a rerank-enabled server.
    """

    class _Stub:
        model_name = "stub-cross-encoder"
        called = False

        async def rerank(
            self, query: str, results: list[RetrievalResult], *, top_k: int = 5
        ) -> list[RetrievalResult]:
            type(self).called = True
            return list(reversed(results))[:top_k]

    async def run() -> None:
        index_dir, corpus, _ = await _seed(tmp_path)
        ctx = _context(index_dir, corpus, reranker=_Stub())
        try:
            response = await tools_module.handle_search(ctx, SearchRequest(query="turbine"))
            assert response.metadata["reranked"] is False
            assert _Stub.called is False

            reranked = await tools_module.handle_search(
                ctx, SearchRequest(query="turbine", rerank=True)
            )
            assert reranked.metadata["reranked"] is True
            assert reranked.metadata["rerank_model"] == "stub-cross-encoder"
            # Recorded because a reranked hybrid result is not hybrid@top_k
            # reordered -- RRF is not depth-invariant (ADR-0012 decision 1).
            assert reranked.metadata["rerank_input_depth"] > 0
            assert _Stub.called is True
        finally:
            await ctx.registry.aclose()

    asyncio.run(run())


def test_fetch_chunk_verifies_against_the_source(tmp_path: Path) -> None:
    """content comes from the source file, not the indexed copy."""

    async def run() -> None:
        index_dir, corpus, chunk_id = await _seed(tmp_path)
        ctx = _context(index_dir, corpus)
        try:
            response = await tools_module.handle_fetch_chunk(
                ctx, FetchChunkRequest(chunk_id=chunk_id)
            )
            assert response.verification == "verified"
            assert response.content is not None
            assert "Turbine" in response.content
            assert response.citation.chunk_id == chunk_id
        finally:
            await ctx.registry.aclose()

    asyncio.run(run())


def test_fetch_chunk_reports_drift_and_withholds_content(tmp_path: Path) -> None:
    """A same-length in-place edit is caught here, and no text is returned.

    This byte comparison is strictly stronger than resolve_citation's
    length-only drift check -- which KNOWN_LIMITATIONS records as unable to see
    this exact edit -- but only at this one boundary. Withholding content on a
    non-verified verdict is the fail-closed half: there is no text for a client
    to mistakenly attribute to the source.
    """

    async def run() -> None:
        index_dir, corpus, chunk_id = await _seed(tmp_path)
        original = (corpus / "a.md").read_text(encoding="utf-8")
        # Same length, different bytes: invisible to a length check.
        (corpus / "a.md").write_text(original.replace("Turbine", "Tvrbine"), encoding="utf-8")

        ctx = _context(index_dir, corpus)
        try:
            response = await tools_module.handle_fetch_chunk(
                ctx, FetchChunkRequest(chunk_id=chunk_id)
            )
            assert response.verification == "drifted"
            assert response.content is None
            assert response.detail
        finally:
            await ctx.registry.aclose()

    asyncio.run(run())


def test_fetch_chunk_on_an_unknown_chunk_is_refused(tmp_path: Path) -> None:
    async def run() -> None:
        index_dir, corpus, _ = await _seed(tmp_path)
        ctx = _context(index_dir, corpus)
        try:
            with pytest.raises(ConfigurationError, match="no chunk"):
                await tools_module.handle_fetch_chunk(ctx, FetchChunkRequest(chunk_id="nope"))
        finally:
            await ctx.registry.aclose()

    asyncio.run(run())


def test_list_collections_opens_nothing(tmp_path: Path) -> None:
    """Enumeration must not create or stamp a store -- that would make listing a write."""

    async def run() -> None:
        index_dir, corpus, _ = await _seed(tmp_path)
        ctx = _context(index_dir, corpus)
        try:
            before = sorted(p.name for p in index_dir.iterdir())
            names = await tools_module.handle_list_collections(ctx, ListCollectionsRequest())
            assert names == ["default"]
            assert sorted(p.name for p in index_dir.iterdir()) == before
        finally:
            await ctx.registry.aclose()

    asyncio.run(run())


def test_list_collections_does_not_block_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The probe must run off the loop thread, not on it.

    ``is_groundkit_store`` opens a real ``sqlite3`` connection and reads a
    pragma for every candidate file. Called inline from an ``async def``, that
    is blocking work on the single event loop, in the discovery tool a client
    typically calls first -- so it stalls every other in-flight coroutine for N
    sequential sqlite opens.

    Correctness is unchanged either way, so this asserts the concurrency
    property directly rather than an observable output: the identity of the
    thread the blocking probe runs on. That is deterministic, not timing-based
    -- ``asyncio.to_thread`` always dispatches to an executor worker, and
    running inline always reports the loop's own thread -- so there is no
    sleep, no timeout and nothing to make it flaky.
    """
    probe_threads: list[int] = []

    def _recording_probe(_db_path: Path) -> bool:
        probe_threads.append(threading.get_ident())
        return True

    monkeypatch.setattr(tools_module, "is_groundkit_store", _recording_probe)

    async def run() -> None:
        index_dir, corpus, _ = await _seed(tmp_path)
        (index_dir / "second.sqlite3").write_bytes(b"")
        ctx = _context(index_dir, corpus)
        try:
            loop_thread = threading.get_ident()
            names = await tools_module.handle_list_collections(ctx, ListCollectionsRequest())

            assert len(probe_threads) >= 2, "the probe never ran over both candidates"
            assert sorted(names) == ["default", "second"]
            offenders = [t for t in probe_threads if t == loop_thread]
            assert not offenders, (
                f"is_groundkit_store ran on the event loop thread ({loop_thread}) "
                f"for {len(offenders)} of {len(probe_threads)} candidates"
            )
        finally:
            await ctx.registry.aclose()

    asyncio.run(run())


def test_list_collections_skips_a_foreign_sqlite_file(tmp_path: Path) -> None:
    """An unrelated database in the index directory is not a collection.

    Advertising it was the first half of a read-only surface mutating a
    stranger's database: a caller who saw the name then asked ``index_status``
    for it, and ``SQLiteMetadataStore.open`` wrote groundkit's four tables
    into it. The filter here and the refusal in ``open`` close the two halves.
    """

    async def run() -> None:
        index_dir, corpus, _ = await _seed(tmp_path)
        foreign = index_dir / "someone-elses.sqlite3"
        conn = sqlite3.connect(str(foreign))
        try:
            conn.execute("CREATE TABLE journal (id INTEGER PRIMARY KEY, body TEXT)")
            conn.commit()
        finally:
            conn.close()

        ctx = _context(index_dir, corpus)
        try:
            names = await tools_module.handle_list_collections(ctx, ListCollectionsRequest())
            assert names == ["default"]
            assert "someone-elses" not in names
        finally:
            await ctx.registry.aclose()

    asyncio.run(run())


def test_index_status_reports_counts_without_leaking_content(tmp_path: Path) -> None:
    """Counts and identity only -- never sources, chunk text, or paths.

    SPEC.md §7 records that SQLite here is content-bearing data, so a status
    surface that enumerated sources would disclose corpus layout to an
    unauthenticated reader.
    """

    async def run() -> None:
        index_dir, corpus, _ = await _seed(tmp_path)
        ctx = _context(index_dir, corpus)
        try:
            status = await tools_module.handle_index_status(ctx, IndexStatusRequest())
            assert status.document_count == 1
            assert status.chunk_count == 1
            assert status.embedding is None
            assert status.dense_search_available is False
            assert status.cache_enabled is True
            assert status.generation is not None

            serialized = status.model_dump_json()
            assert "Turbine" not in serialized
            assert str(corpus) not in serialized
            assert str(index_dir) not in serialized
        finally:
            await ctx.registry.aclose()

    asyncio.run(run())


def test_index_status_on_an_unknown_collection_creates_no_file(tmp_path: Path) -> None:
    """The strongest of the guards, because a naive implementation ships the bug.

    ``SQLiteMetadataStore.open`` creates and stamps a store it does not find, so
    without the registry's existence pre-check every request for an arbitrary
    name would leave an empty collection behind -- an unauthenticated read
    surface turned into a disk-fill primitive.
    """

    async def run() -> None:
        index_dir, corpus, _ = await _seed(tmp_path)
        ctx = _context(index_dir, corpus)
        try:
            before = sorted(p.name for p in index_dir.iterdir())
            with pytest.raises(ConfigurationError, match="does not exist"):
                await tools_module.handle_index_status(
                    ctx, IndexStatusRequest(collection="nonexistent")
                )
            assert sorted(p.name for p in index_dir.iterdir()) == before
        finally:
            await ctx.registry.aclose()

    asyncio.run(run())


# -- The concurrency bound on corpus-scale work ----------------------------
#
# These two drive the bound rather than reading back a configured value, which
# is the whole point: the first version of this control set uvicorn's
# `limit_concurrency` and asserted the number it had been given, never issuing
# a request. A review found that the setting counts *open connections*
# (idle keep-alive included) and answers 503 before routing -- so it could
# never have expressed "bound the expensive work", and no test that reads a
# config value could have noticed.
#
# The store is faked out here deliberately. Everything below the handler --
# opening a collection, rebuilding BM25 -- dispatches to worker threads, and no
# amount of `await asyncio.sleep(0)` waits for a thread. Faking it keeps every
# await on one loop, which is what makes `_settle` a deterministic "let
# everything runnable run" rather than a sleep to tune. The real store path is
# covered by `test_index_status_counts_without_materializing_either_table`
# below, which uses a genuine collection.

#: Loop turns allowed for every runnable task to reach the bound. Not a
#: timeout: ``asyncio.sleep(0)`` advances no clock, so this is "let the loop run
#: until nothing more can proceed", with a ceiling so a bug cannot spin forever.
_SETTLE_TURNS = 200

_SCAN_STATE: dict[str, Any] = {}


async def _settle() -> None:
    """Yield until every runnable task has had ample opportunity to proceed."""
    for _ in range(_SETTLE_TURNS):
        await asyncio.sleep(0)


def _reset_scan_state() -> None:
    _SCAN_STATE.clear()
    _SCAN_STATE.update(in_flight=0, peak=0, entered=asyncio.Event(), release=asyncio.Event())


class _ParkedRetriever:
    """Retriever stand-in that parks in ``search`` until the test releases it.

    The signature mirrors :meth:`Retriever.search` exactly, positional ``top_k``
    and keyword-only ``mode`` included: a stand-in that renames or reorders what
    it replaces is not standing in. ``tests/**`` ignores ARG002 for this reason.
    """

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        *,
        mode: str = "bm25",
    ) -> SearchResponse:
        _SCAN_STATE["in_flight"] += 1
        _SCAN_STATE["peak"] = max(_SCAN_STATE["peak"], _SCAN_STATE["in_flight"])
        _SCAN_STATE["entered"].set()
        try:
            await _SCAN_STATE["release"].wait()
        finally:
            _SCAN_STATE["in_flight"] -= 1
        return SearchResponse(query=query, results=[], total_results=0, metadata={"stage": "bm25"})


class _FakeAcquired:
    def __init__(self) -> None:
        self.retriever = _ParkedRetriever()
        self.generation = 0


class _FakeRuntime:
    """Runtime stand-in whose every operation completes on the loop."""

    async def acquire(self) -> _FakeAcquired:
        return _FakeAcquired()

    async def document_count(self) -> int:
        return 1

    async def chunk_count(self) -> int:
        return 1

    async def get_manifest(self) -> None:
        return None

    async def get_generation(self) -> int:
        return 0


class _FakeRegistry:
    """Registry stand-in: no SQLite, no worker threads, no real collection."""

    @asynccontextmanager
    async def acquire(self, collection: str) -> AsyncIterator[_FakeRuntime]:
        yield _FakeRuntime()

    async def aclose(self) -> None:
        return None


def _parked_context(tmp_path: Path) -> ServiceContext:
    return ServiceContext(
        registry=_FakeRegistry(),  # type: ignore[arg-type]
        index_dir=tmp_path,
        base_dir=tmp_path,
    )


def test_concurrent_corpus_scans_are_bounded(tmp_path: Path) -> None:
    """More search requests than the bound cannot all be scanning at once.

    ``search`` is O(corpus) regardless of ``top_k`` -- BM25 scores every indexed
    chunk -- so each one in flight holds its own corpus-scale working set and,
    unbounded, arrival rate alone decides peak memory. On the single replica
    ``infra/k8s/deployment.yaml`` describes, that is an OOMKill rather than a
    slowdown.

    Deterministic, not timing-based: everything runs on one loop, the parked
    scans are released by an ``asyncio.Event`` the test owns, and ``_settle``
    advances no clock -- there is no sleep to tune and nothing to make flaky.
    """

    async def run() -> None:
        _reset_scan_state()
        ctx = _parked_context(tmp_path)
        overshoot = 3
        try:
            tasks = [
                asyncio.ensure_future(
                    tools_module.handle_search(ctx, SearchRequest(query="turbine"))
                )
                for _ in range(MAX_CONCURRENT_CORPUS_SCANS + overshoot)
            ]
            await _settle()

            assert _SCAN_STATE["in_flight"] == MAX_CONCURRENT_CORPUS_SCANS, (
                f"{_SCAN_STATE['in_flight']} scans admitted, expected exactly "
                f"{MAX_CONCURRENT_CORPUS_SCANS}; the bound is not holding"
            )

            # The excess is queued, not refused: releasing the first wave lets
            # every remaining request through rather than failing it.
            _SCAN_STATE["release"].set()
            results = await asyncio.gather(*tasks)
            assert len(results) == MAX_CONCURRENT_CORPUS_SCANS + overshoot
            assert _SCAN_STATE["peak"] == MAX_CONCURRENT_CORPUS_SCANS, (
                f"peak concurrency reached {_SCAN_STATE['peak']} once the first "
                f"wave drained; the bound is not holding"
            )
        finally:
            _SCAN_STATE["release"].set()

    asyncio.run(run())


def test_index_status_shares_the_scan_budget_with_search(tmp_path: Path) -> None:
    """``index_status`` draws on the same budget, not a second one.

    It aggregates over every row of two tables, on the cheapest-*looking* call
    on an unauthenticated surface. Two transports over one runtime must contend
    for one budget rather than two, so this asserts the budget is genuinely
    shared: a saturated ``search`` load blocks ``index_status`` too.
    """

    async def run() -> None:
        _reset_scan_state()
        ctx = _parked_context(tmp_path)
        try:
            searches = [
                asyncio.ensure_future(
                    tools_module.handle_search(ctx, SearchRequest(query="turbine"))
                )
                for _ in range(MAX_CONCURRENT_CORPUS_SCANS)
            ]
            await _settle()
            assert _SCAN_STATE["entered"].is_set(), "the searches never reached the retriever"
            assert _SCAN_STATE["in_flight"] == MAX_CONCURRENT_CORPUS_SCANS

            status = asyncio.ensure_future(
                tools_module.handle_index_status(ctx, IndexStatusRequest())
            )
            await _settle()
            assert not status.done(), (
                "index_status completed while the scan budget was fully held -- "
                "it is not sharing the bound"
            )

            _SCAN_STATE["release"].set()
            await asyncio.gather(*searches)
            response = await status
            assert response.document_count == 1
            assert response.chunk_count == 1
        finally:
            _SCAN_STATE["release"].set()

    asyncio.run(run())


def test_index_status_counts_without_materializing_either_table(tmp_path: Path) -> None:
    """Both counts are aggregates -- the document half as well as the chunk half.

    The chunk half was fixed first and the document half missed: one line above
    it, ``len(await runtime.get_document_sources())`` built the whole
    ``{document_id: source}`` mapping to take its length. Asserted on the SQL
    the store executes rather than on which method was called, so a later
    refactor that materializes either table by another route is still caught.

    Runs against a genuine collection, unlike the two bound tests above.
    """

    async def run() -> None:
        index_dir, corpus, _ = await _seed(tmp_path)
        ctx = _context(index_dir, corpus)
        statements: list[str] = []
        try:
            async with ctx.registry.acquire("default") as runtime:
                runtime._store._conn.set_trace_callback(statements.append)
                try:
                    response = await tools_module.handle_index_status(ctx, IndexStatusRequest())
                finally:
                    runtime._store._conn.set_trace_callback(None)

            assert response.document_count == 1
            assert response.chunk_count == 1

            executed = " ".join(statements).lower()
            assert executed.count("count(") >= 2, (
                f"expected one aggregate per table; statements were {statements!r}"
            )
            # Word-bounded so a future `SELECT content_hash` is not a false
            # alarm, while any read of the `content` column still is.
            assert not re.search(r"\bcontent\b", executed), (
                f"index_status read chunk content: {statements!r}"
            )
            assert "source" not in executed, f"index_status read document sources: {statements!r}"
        finally:
            await ctx.registry.aclose()

    asyncio.run(run())

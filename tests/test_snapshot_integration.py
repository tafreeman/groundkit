"""Wave 4 integration: URL ingestion -> local snapshot -> citation resolution
(ADR-0016 decision 4; docs/specs/loaders-extracted-and-remote-sources.md §10).

Unit-level tests exercise each piece in isolation: ``tests/test_url_loader.py``
covers ``UrlLoader`` alone, ``tests/test_source_class.py`` covers
``resolve_citation``'s dispatch alone. Nothing before this file wires all
three together as one path: a real ``UrlLoader`` (constructed against
``httpx.MockTransport``, never real network) feeding a real ``Indexer`` /
``SQLiteMetadataStore``, and ``resolve_citation``'s ``snapshot`` branch
reading what that ingest wrote back off disk — the exact round trip
``handle_fetch_chunk`` performs in production.

Async helpers are driven with ``asyncio.run()`` inside plain ``def`` test
functions, matching this repo's established pattern (pytest-asyncio is not
part of this repo's dependency set — see ``tests/test_url_loader.py``'s
module docstring).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

import httpx
import pytest

from groundkit import snapshots
from groundkit.contracts import Citation
from groundkit.errors import RetrievalError
from groundkit.index.metadata import SQLiteMetadataStore
from groundkit.indexer import Indexer
from groundkit.ingestion.loaders import FileLoader
from groundkit.ingestion.url_loader import UrlLoader
from groundkit.retrieval import citations as citations_module
from groundkit.retrieval.citations import resolve_citation
from groundkit.runtime import CollectionRegistry
from groundkit.service.schemas import FetchChunkRequest
from groundkit.service.tools import ServiceContext, handle_fetch_chunk
from groundkit.utils import url_safety
from groundkit.utils.path_safety import ensure_within_base

Handler = Callable[[httpx.Request], httpx.Response]

#: See ``tests/test_url_loader.py``'s ``_PUBLIC_ADDRESS``: a stand-in DNS
#: answer, never contacted.
_PUBLIC_ADDRESS: Final[str] = "93.184.216.34"


@pytest.fixture(autouse=True)
def _no_real_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """No real DNS either -- ``MockTransport`` stops the HTTP request, not the
    ``ensure_safe_endpoint`` lookup that precedes it.

    Every ingest here targets ``https://example.com/...``, a non-literal host,
    so without this the module docstring's "never real network" claim would be
    true of the fetch and false of the resolution. Same seam and same reasoning
    as ``tests/test_url_loader.py``'s fixture of this name.
    """

    async def _resolve_to_public_address(host: str) -> Sequence[str]:
        del host
        return [_PUBLIC_ADDRESS]

    monkeypatch.setattr(url_safety, "_default_resolver", _resolve_to_public_address)


def _client(handler: Handler) -> httpx.AsyncClient:
    """Build an httpx.AsyncClient wired to a MockTransport -- no network."""
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _citation(source: str, **kwargs: object) -> Citation:
    base: dict[str, object] = {
        "document_id": "doc-1",
        "chunk_id": "c1",
        "source": source,
        "start_offset": 0,
        "end_offset": 5,
    }
    base.update(kwargs)
    return Citation(**base)  # type: ignore[arg-type]


async def _ingest_url(index_dir: Path, collection: str, url: str, handler: Handler) -> None:
    """Ingest one URL into ``collection``, exactly as ``grk ingest <url>`` does."""
    store = await SQLiteMetadataStore.open(index_dir, collection)
    try:
        snapshot_dir = snapshots.snapshot_dir_for(index_dir, collection)
        loader = UrlLoader(snapshot_dir, client=_client(handler))
        indexer = Indexer(store, loader, collection=collection)
        await indexer.index_source(url)
    finally:
        await store.close()


async def _first_chunk_citation(index_dir: Path, collection: str) -> Citation:
    """Read back the one document/chunk an ingest just wrote, as a Citation.

    Mirrors ``handle_fetch_chunk``'s own construction: the citation's
    ``source``/``source_class``/``extractor`` come from the joined
    ``DocumentRecord``, never from a bare source string (ADR-0016).
    """
    store = await SQLiteMetadataStore.open(index_dir, collection)
    try:
        records = await store.get_document_records()
        chunks = await store.get_chunks()
    finally:
        await store.close()
    chunk = chunks[0]
    record = records[chunk.document_id]
    return Citation(
        document_id=chunk.document_id,
        chunk_id=chunk.chunk_id,
        source=record.source,
        source_class=record.source_class,
        extractor=record.extractor,
        start_offset=chunk.start_offset,
        end_offset=chunk.end_offset,
    )


# -- A URL ingests as `snapshot`, and its citation verifies against the local
# -- copy, never a re-fetch. -------------------------------------------------


def test_url_ingest_produces_a_snapshot_class_document(tmp_path: Path) -> None:
    index_dir = tmp_path / ".groundkit"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"Snapshot integration content.")

    asyncio.run(_ingest_url(index_dir, "default", "https://example.com/doc", handler))

    async def check() -> None:
        store = await SQLiteMetadataStore.open(index_dir, "default")
        try:
            records = await store.get_document_records()
        finally:
            await store.close()
        assert len(records) == 1
        record = next(iter(records.values()))
        assert record.source_class == "snapshot"
        assert record.extractor is None
        assert record.source == "https://example.com/doc"

    asyncio.run(check())


def test_snapshot_citation_verifies_with_no_network_call_at_resolution_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ADR-0016 decision 4 property: resolution reads the local copy, never
    re-fetches. Pinned by making every httpx network entry point raise, so a
    future change that re-fetches at verification time fails this test loudly
    rather than merely being slow or flaky.
    """
    index_dir = tmp_path / ".groundkit"
    content = "Snapshot integration content, exact and verifiable."

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=content.encode())

    asyncio.run(_ingest_url(index_dir, "default", "https://example.com/doc", handler))
    citation = asyncio.run(_first_chunk_citation(index_dir, "default"))
    assert citation.source_class == "snapshot"
    snapshot_dir = snapshots.snapshot_dir_for(index_dir, "default")

    def _network_forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError(
            "resolve_citation must not touch the network to resolve a snapshot citation"
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", _network_forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "request", _network_forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "stream", _network_forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "send", _network_forbidden)

    resolved = asyncio.run(
        resolve_citation(citation, tmp_path / "unrelated-base-dir", snapshot_dir=snapshot_dir)
    )
    assert resolved == content[citation.start_offset : citation.end_offset]


def test_fetch_chunk_verifies_a_url_ingested_chunk_end_to_end(tmp_path: Path) -> None:
    """The real service path: ``handle_fetch_chunk`` threads ``snapshot_dir``
    through on its own, from ``ctx.index_dir`` + the request's collection --
    nothing about this test computes it manually."""
    index_dir = tmp_path / ".groundkit"
    content = "End-to-end snapshot verification content."

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=content.encode())

    asyncio.run(_ingest_url(index_dir, "default", "https://example.com/e2e", handler))

    async def run() -> None:
        registry = CollectionRegistry(index_dir)
        ctx = ServiceContext(registry=registry, index_dir=index_dir, base_dir=tmp_path)
        try:
            store = await SQLiteMetadataStore.open(index_dir, "default")
            try:
                chunk = (await store.get_chunks())[0]
            finally:
                await store.close()
            response = await handle_fetch_chunk(
                ctx, FetchChunkRequest(chunk_id=chunk.chunk_id, collection="default")
            )
        finally:
            await registry.aclose()

        assert response.verification == "verified"
        assert response.content == chunk.content
        assert response.citation.source_class == "snapshot"
        assert response.detail is None

    asyncio.run(run())


# -- A missing/deleted snapshot is `unresolvable`, never `drifted`. ----------


def test_deleted_snapshot_file_is_unresolvable_not_drifted(tmp_path: Path) -> None:
    index_dir = tmp_path / ".groundkit"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"content that will be deleted from disk")

    asyncio.run(_ingest_url(index_dir, "default", "https://example.com/gone", handler))
    citation = asyncio.run(_first_chunk_citation(index_dir, "default"))
    snapshot_dir = snapshots.snapshot_dir_for(index_dir, "default")
    snapshots.snapshot_path_for(snapshot_dir, citation.document_id).unlink()

    async def run() -> None:
        with pytest.raises(RetrievalError) as excinfo:
            await resolve_citation(citation, tmp_path, snapshot_dir=snapshot_dir)
        assert excinfo.value.verdict == "unresolvable"

    asyncio.run(run())


def test_no_snapshot_dir_supplied_is_unresolvable(tmp_path: Path) -> None:
    """A caller that forgets to compute and pass ``snapshot_dir`` gets a typed,
    actionable refusal -- never a crash, and never treated as drift (nothing
    was read to disagree with anything)."""
    index_dir = tmp_path / ".groundkit"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"content")

    asyncio.run(_ingest_url(index_dir, "default", "https://example.com/x", handler))
    citation = asyncio.run(_first_chunk_citation(index_dir, "default"))

    async def run() -> None:
        with pytest.raises(RetrievalError, match="snapshot_dir") as excinfo:
            await resolve_citation(citation, tmp_path)  # snapshot_dir defaults to None
        assert excinfo.value.verdict == "unresolvable"

    asyncio.run(run())


def test_drifted_when_the_snapshot_no_longer_covers_the_cited_offsets(tmp_path: Path) -> None:
    """Distinguishes `drifted` from `unresolvable`: the snapshot file exists
    and reads fine, but is now too short for what was indexed. Complements
    the deleted-file case above, which is `unresolvable` because nothing
    could be read at all."""
    snapshot_dir = tmp_path / "default.snapshots"
    snapshot_dir.mkdir()
    citation = _citation(
        "https://example.com/shrunk",
        document_id="doc-2",
        source_class="snapshot",
        start_offset=0,
        end_offset=50,
    )
    (snapshot_dir / citation.document_id).write_text("too short now", encoding="utf-8")

    async def run() -> None:
        with pytest.raises(RetrievalError) as excinfo:
            await resolve_citation(citation, tmp_path, snapshot_dir=snapshot_dir)
        assert excinfo.value.verdict == "drifted"

    asyncio.run(run())


# -- A URL source never reaches ensure_within_base. --------------------------


def test_url_source_never_reaches_ensure_within_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ADR-0016 decision 4's ordering constraint, pinned directly: whatever
    ``resolve_citation`` hands to ``ensure_within_base`` for a ``snapshot``
    citation must be the derived local snapshot path, never ``citation.source``
    (a URL). ``os.path.realpath`` resolves a URL string as a relative
    filesystem path with no error at all (``test_source_class.py``'s
    ``test_a_url_passes_path_containment_which_is_why_class_is_checked_first``
    pins that hazard directly), so a regression here would not raise -- it
    would silently "succeed" at containment-checking a string that was never
    a path.
    """
    snapshot_dir = tmp_path / "default.snapshots"
    snapshot_dir.mkdir()
    citation = _citation(
        "https://example.com/never-a-path", document_id="doc-3", source_class="snapshot"
    )
    (snapshot_dir / citation.document_id).write_text("hello", encoding="utf-8")

    seen: list[str] = []
    # Bound from its real home rather than off the citations namespace: the
    # patch below targets that namespace by name (correct — patch where it
    # is used), but reading it back through the module is an implicit
    # re-export mypy --strict rejects. Same function object either way.
    real_ensure_within_base = ensure_within_base

    def tracking(path: str | Path, base_dir: str | Path) -> Path:
        seen.append(str(path))
        return real_ensure_within_base(path, base_dir)

    monkeypatch.setattr(citations_module, "ensure_within_base", tracking)

    resolved = asyncio.run(resolve_citation(citation, tmp_path, snapshot_dir=snapshot_dir))

    assert resolved == "hello"
    assert seen == [str(snapshot_dir / citation.document_id)]
    assert citation.source not in seen


# -- Regression guard: `.md` ingestion and citation resolution are untouched.


def test_markdown_ingestion_and_citation_are_byte_identical_to_before(tmp_path: Path) -> None:
    """`text`-class ingestion and resolution behave exactly as before Wave 4:
    ``resolve_citation``'s new ``snapshot_dir`` keyword is defaulted and never
    consulted outside the ``snapshot`` branch (spec §10.1's "no existing
    caller breaks" guarantee), and ``FileLoader``/``Indexer`` are untouched by
    this wave's own file list."""
    source_dir = tmp_path / "corpus"
    source_dir.mkdir()
    doc_path = source_dir / "a.md"
    doc_path.write_text("Hello, groundkit world.", encoding="utf-8")

    index_dir = tmp_path / ".groundkit"

    async def run() -> None:
        store = await SQLiteMetadataStore.open(index_dir, "default")
        try:
            indexer = Indexer(store, FileLoader(allowed_base_dir=source_dir), collection="default")
            await indexer.index_source(str(doc_path))
            records = await store.get_document_records()
            chunks = await store.get_chunks()
        finally:
            await store.close()

        assert len(records) == 1
        record = next(iter(records.values()))
        assert record.source_class == "text"
        assert record.extractor is None

        chunk = chunks[0]
        citation = Citation(
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            source=record.source,
            source_class=record.source_class,
            extractor=record.extractor,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
        )
        expected = doc_path.read_text(encoding="utf-8")[citation.start_offset : citation.end_offset]

        # Called both without snapshot_dir (the exact pre-Wave-4 call shape)
        # and with one supplied (as every real caller now does
        # unconditionally, e.g. handle_fetch_chunk) -- proving the new
        # keyword-only parameter changes nothing for a `text` citation either
        # way.
        resolved_without = await resolve_citation(citation, source_dir)
        resolved_with = await resolve_citation(
            citation, source_dir, snapshot_dir=index_dir / "default.snapshots"
        )
        assert resolved_without == resolved_with == expected

    asyncio.run(run())

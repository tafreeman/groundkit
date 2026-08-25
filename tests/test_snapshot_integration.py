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
import errno
import os
import sys
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
        # ``snapshot_dir`` goes to the Indexer as well as the loader, exactly
        # as ``_cmd_ingest_url`` wires it (ADR-0023 decision 2): the loader
        # writes snapshots, the Indexer removes the ones no document
        # references any more. Omitting it here would leave every lifecycle
        # test below silently exercising a cleanup-disabled indexer.
        indexer = Indexer(store, loader, collection=collection, snapshot_dir=snapshot_dir)
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


# -- Snapshot lifecycle: a snapshot exists iff a documents row references it
# -- (ADR-0023). ------------------------------------------------------------
#
# ADR-0016 decision 4 deferred retention and cleanup for these files. In the
# absence of a decision the implementation defaulted to the worst answer:
# `UrlLoader.load()` wrote a snapshot on every load, named after a
# `document_id` that is a fresh uuid4 each time, and no code path anywhere in
# `src/` ever removed one. These pin both halves of the fix.


def _snapshot_files(index_dir: Path, collection: str = "default") -> list[Path]:
    """Every snapshot file this collection currently has on disk."""
    snapshot_dir = snapshots.snapshot_dir_for(index_dir, collection)
    if not snapshot_dir.exists():
        return []
    return sorted(p for p in snapshot_dir.iterdir() if p.is_file())


def test_reingesting_an_unchanged_url_does_not_accumulate_snapshots(tmp_path: Path) -> None:
    """ADR-0023 defect 1, and the reason it was invisible.

    Re-running `grk ingest <url>` is the documented incremental workflow, and
    the one an operator is most likely to automate. The second run hits
    `Indexer._process`'s fingerprint gate and skips -- but the loader has
    already written a *second* snapshot under a fresh uuid4 by then, which
    nothing references and nothing reports. The ingest report counts
    `documents_skipped`, not bytes written, so this grew silently forever.
    """
    index_dir = tmp_path / ".groundkit"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"Stable content that never changes.")

    asyncio.run(_ingest_url(index_dir, "default", "https://example.com/doc", handler))
    after_first = _snapshot_files(index_dir)
    assert len(after_first) == 1

    asyncio.run(_ingest_url(index_dir, "default", "https://example.com/doc", handler))
    after_second = _snapshot_files(index_dir)

    assert len(after_second) == 1
    # The surviving file is the one the stored document actually points at --
    # the original -- not the throwaway the second load wrote.
    assert after_second[0].name == after_first[0].name
    assert after_second[0].read_text(encoding="utf-8") == "Stable content that never changes."


def test_reingesting_changed_content_leaves_only_the_new_snapshot(tmp_path: Path) -> None:
    """The replace path. The prior document's snapshot stops being referenced
    the moment `replace_document` commits, so it must not survive -- otherwise
    every edit to a remote document leaves its previous text on disk."""
    index_dir = tmp_path / ".groundkit"
    bodies = iter([b"First version of the document.", b"Second version of the document."])

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=next(bodies))

    asyncio.run(_ingest_url(index_dir, "default", "https://example.com/doc", handler))
    asyncio.run(_ingest_url(index_dir, "default", "https://example.com/doc", handler))

    remaining = _snapshot_files(index_dir)
    assert len(remaining) == 1
    assert remaining[0].read_text(encoding="utf-8") == "Second version of the document."

    async def check_row_points_at_it() -> None:
        store = await SQLiteMetadataStore.open(index_dir, "default")
        try:
            records = await store.get_document_records()
        finally:
            await store.close()
        assert len(records) == 1
        assert next(iter(records)) == remaining[0].name

    asyncio.run(check_row_points_at_it())


def test_pruning_a_document_removes_its_snapshot(tmp_path: Path) -> None:
    """ADR-0023 defect 2, the sharper half.

    A user who deletes a document has stated an intent about *content*.
    Honouring it in SQLite while the fetched text stays in a sibling
    directory makes the retention story a false statement. Here the second
    fetch returns an empty body, which makes `UrlLoader` yield no documents
    and `Indexer._prune_emptied_source` delete the stored one.
    """
    index_dir = tmp_path / ".groundkit"
    bodies = iter([b"Content that will later be pruned.", b"   "])

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=next(bodies))

    asyncio.run(_ingest_url(index_dir, "default", "https://example.com/doc", handler))
    assert len(_snapshot_files(index_dir)) == 1

    asyncio.run(_ingest_url(index_dir, "default", "https://example.com/doc", handler))

    async def check_document_is_gone() -> None:
        store = await SQLiteMetadataStore.open(index_dir, "default")
        try:
            records = await store.get_document_records()
        finally:
            await store.close()
        assert records == {}

    asyncio.run(check_document_is_gone())
    assert _snapshot_files(index_dir) == []


def test_a_document_id_escaping_the_snapshot_dir_is_never_unlinked(tmp_path: Path) -> None:
    """ADR-0023 decision 4.

    `snapshots.snapshot_path_for` performs no containment check and says so:
    `document_id` is a plain string field with no character class. The read
    side already guards this; unlinking is strictly more dangerous than
    reading, so it takes the same guard. Without it, a crafted `document_id`
    turns ordinary cleanup into an arbitrary file delete.
    """
    index_dir = tmp_path / ".groundkit"
    snapshot_dir = snapshots.snapshot_dir_for(index_dir, "default")
    snapshot_dir.mkdir(parents=True)
    # The sentinel sits exactly where ``../outside.txt`` lands from inside
    # snapshot_dir -- one level up, in index_dir. Getting this wrong is not a
    # cosmetic slip: an escape aimed at a path that happens not to exist is
    # absorbed by ``unlink(missing_ok=True)``, so the test passes whether or
    # not the containment check is there. An earlier draft of this test put
    # the sentinel in tmp_path and was decorative for exactly that reason.
    outsider = index_dir / "outside.txt"
    outsider.write_text("must survive", encoding="utf-8")
    assert (snapshot_dir / "../outside.txt").resolve() == outsider.resolve()

    async def attempt() -> None:
        store = await SQLiteMetadataStore.open(index_dir, "default")
        try:
            indexer = Indexer(
                store,
                UrlLoader(snapshot_dir, client=_client(lambda _r: httpx.Response(200))),
                collection="default",
                snapshot_dir=snapshot_dir,
            )
            await indexer._remove_snapshot("../outside.txt")
        finally:
            await store.close()

    asyncio.run(attempt())

    assert outsider.exists()
    assert outsider.read_text(encoding="utf-8") == "must survive"


def test_snapshot_cleanup_is_inert_when_no_snapshot_dir_is_configured(tmp_path: Path) -> None:
    """Every `FileLoader` caller passes no `snapshot_dir`, so the cleanup path
    must be a no-op rather than an error -- a collection that never ingested a
    URL has no snapshot directory and pays nothing (ADR-0023 decision 2)."""
    index_dir = tmp_path / ".groundkit"

    async def attempt() -> None:
        store = await SQLiteMetadataStore.open(index_dir, "default")
        try:
            indexer = Indexer(store, FileLoader(allowed_base_dir=tmp_path), collection="default")
            await indexer._remove_snapshot("anything-at-all")
        finally:
            await store.close()

    asyncio.run(attempt())


# -- The snapshot READ does not follow a symlink planted after the containment
# -- check, and does not translate line endings on the way back in (GK-030). --


class TestSnapshotReadDoesNotFollowASymlink:
    """The read side of the gap ``GK-028`` closed on the write side.

    ``ensure_within_base`` resolves symlinks, so a snapshot path that is
    *already* a link out of the root is refused by the check itself. What the
    check cannot see is a link created after it returned and before the file
    is opened. This is the more exploitable half of the pair: the write side
    could corrupt a file, the read side returns whatever the link points at
    to a service caller through ``fetch_chunk``.

    The race is driven from the code path itself rather than from a second
    thread, exactly as ``tests/test_url_loader.py``'s write-side sibling does:
    the containment check is wrapped so that planting the symlink is the last
    thing it does before returning. That makes the interleaving deterministic
    instead of timing-dependent, and it is the only way to reach the window at
    all -- a symlink planted before the call is caught by the check, and one
    planted after the read is too late to matter.
    """

    def test_a_symlink_planted_after_the_containment_check_is_not_followed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if not hasattr(os, "O_NOFOLLOW"):
            pytest.skip("O_NOFOLLOW does not exist on this platform (Windows)")

        secret = tmp_path / "secret.txt"
        secret.write_text("SECRET-EXFILTRATED", encoding="utf-8")
        probe = tmp_path / "symlink-probe"
        try:
            probe.symlink_to(secret)
        except OSError:
            pytest.skip("this platform or user cannot create symlinks")
        probe.unlink()

        snapshot_dir = tmp_path / "default.snapshots"
        snapshot_dir.mkdir()
        citation = _citation(
            "https://example.com/raced",
            document_id="doc-raced",
            source_class="snapshot",
            start_offset=0,
            end_offset=18,
        )
        # Bind the real function from its defining module, not from
        # ``citations``' re-export: the monkeypatch below replaces the latter.
        real_ensure_within_base = ensure_within_base

        def _plant_symlink_after_checking(path: str | Path, base_dir: str | Path) -> Path:
            resolved = real_ensure_within_base(path, base_dir)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.symlink_to(secret)
            return resolved

        monkeypatch.setattr(citations_module, "ensure_within_base", _plant_symlink_after_checking)

        async def run() -> None:
            with pytest.raises(RetrievalError) as excinfo:
                await resolve_citation(citation, tmp_path, snapshot_dir=snapshot_dir)
            assert excinfo.value.verdict == "unresolvable"
            assert "symbolic link" in str(excinfo.value)
            # The whole point: the linked-to bytes never became the answer.
            assert "SECRET-EXFILTRATED" not in str(excinfo.value)

        asyncio.run(run())

    @pytest.mark.skipif(
        sys.platform.startswith(("freebsd", "netbsd", "openbsd", "dragonfly")),
        reason="EMLINK legitimately means 'symlinked final component' on these BSDs",
    )
    def test_emlink_is_not_reported_as_a_symlink_attack(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``EMLINK`` is "Too many links" off the BSDs, not a planted symlink.

        The read side classifies with the same shared ``SYMLINK_ERRNOS`` the
        write side uses, so this pins that the sharing did not widen it: an
        unrelated I/O failure must still read as a plain unresolvable, never
        as "something replaced the file under us".
        """
        snapshot_dir = tmp_path / "default.snapshots"
        snapshot_dir.mkdir()
        citation = _citation(
            "https://example.com/emlink", document_id="doc-emlink", source_class="snapshot"
        )
        (snapshot_dir / citation.document_id).write_text("content", encoding="utf-8")

        def _raise_emlink(*_args: object, **_kwargs: object) -> int:
            raise OSError(errno.EMLINK, "Too many links")

        monkeypatch.setattr("groundkit.retrieval.citations.os.open", _raise_emlink)

        async def run() -> None:
            with pytest.raises(RetrievalError) as excinfo:
                await resolve_citation(citation, tmp_path, snapshot_dir=snapshot_dir)
            assert excinfo.value.verdict == "unresolvable"
            assert "symbolic link" not in str(excinfo.value)

        asyncio.run(run())


def test_a_crlf_snapshot_resolves_to_the_span_that_was_indexed(tmp_path: Path) -> None:
    """Offsets are measured against the decoded ``content``; the read must not
    translate line endings back out from under them.

    ``UrlLoader`` decodes the response body and writes it byte-for-byte, so a
    source served with CRLF line endings has real CRLFs in both ``content``
    and the snapshot. ``Path.read_text`` defaults to universal-newline mode, which
    turns each of those into a bare LF -- one character shorter -- shifting
    every offset after the first line break. The failure is silent: the span
    comes back wrong rather than refused, or the citation is reported
    ``drifted`` when nothing drifted (GK-030).
    """
    index_dir = tmp_path / ".groundkit"
    body = b"first line\r\nsecond line\r\nthird line"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=body, headers={"content-type": "text/plain; charset=utf-8"}
        )

    asyncio.run(_ingest_url(index_dir, "default", "https://example.com/crlf", handler))
    citation = asyncio.run(_first_chunk_citation(index_dir, "default"))
    snapshot_dir = snapshots.snapshot_dir_for(index_dir, "default")

    # What the chunk actually holds, straight off the decoded body the
    # indexer saw -- derived, never a hand-copied literal.
    expected = body.decode("utf-8")[citation.start_offset : citation.end_offset]
    assert "\r\n" in expected, "the fixture stopped exercising CRLF"

    async def run() -> None:
        resolved = await resolve_citation(citation, tmp_path, snapshot_dir=snapshot_dir)
        assert resolved == expected

    asyncio.run(run())

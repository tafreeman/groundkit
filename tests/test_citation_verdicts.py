"""Typed citation verdicts and the extractor-identity check (ADR-0016 Wave 2).

Covers two things Wave 2 adds on top of the dispatch ``tests/test_source_class.py``
already pins:

1. ``resolve_citation``'s ``extracted`` branch now does a real (if permanently
   empty pre-Wave-3) membership check against the active extractor set, and
   every raise site inside it sets ``RetrievalError.verdict`` explicitly
   instead of leaving callers to guess from the message.
2. ``service.tools.handle_fetch_chunk`` classifies a resolution failure by
   reading that typed attribute, not by searching the message for the retired
   phrase ``"changed since indexing"``. The regression test that matters is
   the one that would *fail* under the old string-sniff implementation --
   see :func:`test_fetch_chunk_classifies_by_verdict_not_by_message_text`.

Async helpers are driven with ``asyncio.run()`` inside sync test functions,
matching the rest of this repo's async test style (pytest-asyncio is not
configured here).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from groundkit import extraction
from groundkit.contracts import Chunk, Citation
from groundkit.errors import RetrievalError
from groundkit.index.metadata import SQLiteMetadataStore
from groundkit.retrieval.citations import resolve_citation
from groundkit.runtime import CollectionRegistry
from groundkit.service import tools as tools_module
from groundkit.service.schemas import FetchChunkRequest
from groundkit.service.tools import ServiceContext

#: Stands in for real document text. Never expected to appear in any
#: exception message or ``ChunkFetchResponse.detail`` produced in this file --
#: those describe *offsets and identities*, never the source's own content.
_CONTENT_SENTINEL = "xqzv7-body-text-must-never-appear-in-a-verdict-message"


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


async def _seed(tmp_path: Path, *, text: str) -> tuple[Path, Path, str]:
    """Write a real source file and index one chunk covering all of it."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
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


def _context(index_dir: Path, corpus: Path) -> ServiceContext:
    return ServiceContext(
        registry=CollectionRegistry(index_dir), index_dir=index_dir, base_dir=corpus
    )


# -- resolve_citation: extractor-identity check -----------------------------


def test_extractor_identity_mismatch_is_unresolvable_and_names_both_identities(
    tmp_path: Path,
) -> None:
    """The regression ADR-0016 decision 2 exists for: a mismatch fails closed,
    as ``unresolvable`` (it could not be checked), never ``drifted`` (checked
    and found to differ) -- and names what it recorded and what is active.
    """

    async def run() -> None:
        with pytest.raises(RetrievalError) as exc_info:
            await resolve_citation(
                _citation(str(tmp_path / "a.pdf"), source_class="extracted", extractor="pdf-x/9"),
                tmp_path,
            )
        exc = exc_info.value
        assert exc.verdict == "unresolvable"
        message = str(exc)
        assert "pdf-x/9" in message  # the recorded identity
        # The active set, derived from the registry rather than hardcoded:
        # Wave 3 populates it per extra installed, so a literal string here
        # would assert the test environment's extras rather than the message
        # contract ("name what is active"). Pre-Wave-3 this read
        # "none registered", which the empty-registry branch still produces.
        active = sorted(extraction.active_extractors())
        assert (str(active) if active else "none registered") in message
        # This particular message never contained the retired phrase either --
        # it is not what distinguishes the old sniff from the new mechanism.
        # See test_fetch_chunk_classifies_by_verdict_not_by_message_text for
        # the case that actually would have been misclassified by the sniff.
        assert "changed since indexing" not in message

    asyncio.run(run())


def test_snapshot_citation_is_unresolvable_with_a_reason_distinct_from_extracted(
    tmp_path: Path,
) -> None:
    """Both non-text classes refuse, but no longer by sharing one generic message."""

    async def run() -> None:
        with pytest.raises(RetrievalError) as exc_info:
            await resolve_citation(
                _citation("https://example.com/doc", source_class="snapshot"), tmp_path
            )
        exc = exc_info.value
        assert exc.verdict == "unresolvable"
        message = str(exc)
        assert "local snapshot" in message
        # The extracted branch's specific wording must not leak into the
        # snapshot refusal -- proof the two no longer share a single message.
        assert "extractor identity recorded at ingest" not in message

    asyncio.run(run())


def test_resolve_still_verifies_a_text_citation(tmp_path: Path) -> None:
    """The class that already worked keeps its exact behaviour under the new branching."""
    source = tmp_path / "a.md"
    source.write_text("hello world", encoding="utf-8")

    async def run() -> None:
        span = await resolve_citation(_citation(str(source), end_offset=5), tmp_path)
        assert span == "hello"

    asyncio.run(run())


def test_genuine_drift_still_yields_drifted_verdict(tmp_path: Path) -> None:
    """A real offset-overflow (source shorter than the cited span) is ``drifted``,
    not ``unresolvable`` -- it was read and found to disagree, which is a
    different fact from "could not be checked".
    """
    source = tmp_path / "a.md"
    source.write_text(_CONTENT_SENTINEL, encoding="utf-8")

    async def run() -> None:
        with pytest.raises(RetrievalError) as exc_info:
            await resolve_citation(
                _citation(str(source), start_offset=0, end_offset=9999), tmp_path
            )
        exc = exc_info.value
        assert exc.verdict == "drifted"
        message = str(exc)
        assert "changed since indexing" in message
        # Offsets and a length are reported; the source's own text is not.
        assert _CONTENT_SENTINEL not in message

    asyncio.run(run())


def test_extractor_mismatch_message_does_not_leak_source_content(tmp_path: Path) -> None:
    """The extracted-mismatch branch never reads the source at all, but assert
    it directly rather than trust that: the message must still be clean.

    The sentinel goes in the file's **content only**, never its name. An
    earlier version of this test named the file after the sentinel too, and
    failed — not because content leaked, but because the message quotes
    ``citation.source``, which it is supposed to do: an operator needs to know
    *which* source was refused. A test that cannot tell "the message names the
    path" from "the message leaked the body" is asserting the wrong property,
    and would have to be satisfied by removing information the message owes
    its reader.
    """
    source = tmp_path / "extracted-source.pdf"
    source.write_text(_CONTENT_SENTINEL, encoding="utf-8")

    async def run() -> None:
        with pytest.raises(RetrievalError) as exc_info:
            await resolve_citation(
                _citation(str(source), source_class="extracted", extractor="pdf-x/1"), tmp_path
            )
        assert _CONTENT_SENTINEL not in str(exc_info.value)

    asyncio.run(run())


# -- handle_fetch_chunk: the typed verdict replaces the string sniff --------


def test_fetch_chunk_classifies_by_verdict_not_by_message_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression that matters.

    Constructs a ``RetrievalError`` whose message deliberately omits the
    literal phrase ``"changed since indexing"`` but carries
    ``verdict="drifted"``. The retired implementation
    (``"drifted" if "changed since indexing" in str(exc) else "unresolvable"``)
    would misclassify this as ``"unresolvable"``, because it had nothing to
    read but the message. A test that would still pass under that old
    implementation is testing nothing (per the work order) -- this one does
    not, because it fails under it and passes only under the fix.
    """

    async def run() -> None:
        index_dir, corpus, chunk_id = await _seed(tmp_path, text="turbine maintenance")
        ctx = _context(index_dir, corpus)

        async def _fake_resolve_citation(citation: Citation, allowed_base_dir: Path) -> str:
            del citation, allowed_base_dir
            raise RetrievalError(
                "the recorded extractor identity diverged from the active build",
                verdict="drifted",
            )

        monkeypatch.setattr(tools_module, "resolve_citation", _fake_resolve_citation)
        try:
            response = await tools_module.handle_fetch_chunk(
                ctx, FetchChunkRequest(chunk_id=chunk_id)
            )
            assert response.verification == "drifted"
            assert response.content is None
            assert response.detail == (
                "the recorded extractor identity diverged from the active build"
            )
        finally:
            await ctx.registry.aclose()

    asyncio.run(run())


def test_fetch_chunk_does_not_key_off_the_retired_phrase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The flip side: a message that happens to *contain* the retired phrase
    but carries ``verdict="unresolvable"`` must still classify as
    unresolvable. If ``fetch_chunk`` still searched the message text, this
    would misclassify as ``"drifted"``.
    """

    async def run() -> None:
        index_dir, corpus, chunk_id = await _seed(tmp_path, text="turbine maintenance")
        ctx = _context(index_dir, corpus)

        async def _fake_resolve_citation(citation: Citation, allowed_base_dir: Path) -> str:
            del citation, allowed_base_dir
            raise RetrievalError(
                "unrelated failure that happens to mention: source changed since indexing",
                verdict="unresolvable",
            )

        monkeypatch.setattr(tools_module, "resolve_citation", _fake_resolve_citation)
        try:
            response = await tools_module.handle_fetch_chunk(
                ctx, FetchChunkRequest(chunk_id=chunk_id)
            )
            assert response.verification == "unresolvable"
        finally:
            await ctx.registry.aclose()

    asyncio.run(run())


def test_fetch_chunk_detail_never_leaks_document_content_on_drift(tmp_path: Path) -> None:
    """End-to-end through the real (non-mocked) drift path: the source shrinks,
    the cited span overflows, and the ``detail`` a client sees must not
    contain the source's own text -- only that it changed.
    """

    async def run() -> None:
        text = f"Turbine maintenance intervals. {_CONTENT_SENTINEL} more text after."
        index_dir, corpus, chunk_id = await _seed(tmp_path, text=text)
        # Shrink the source so the cited span overflows -- a genuine drift.
        (corpus / "a.md").write_text("short", encoding="utf-8")

        ctx = _context(index_dir, corpus)
        try:
            response = await tools_module.handle_fetch_chunk(
                ctx, FetchChunkRequest(chunk_id=chunk_id)
            )
            assert response.verification == "drifted"
            assert response.content is None
            assert response.detail is not None
            assert _CONTENT_SENTINEL not in response.detail
        finally:
            await ctx.registry.aclose()

    asyncio.run(run())

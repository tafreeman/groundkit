"""Source classification and citation verifiability (ADR-0016).

Async helpers are driven with ``asyncio.run()`` inside sync test functions
(pytest-asyncio is not configured in this repo).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError

from groundkit.contracts import Citation, Document, RetrievalResult
from groundkit.errors import RetrievalError
from groundkit.retrieval.citations import resolve_citation
from groundkit.utils.path_safety import ensure_within_base


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


# -- Document invariants ---------------------------------------------------


def test_an_extracted_document_must_record_its_extractor() -> None:
    """Without it, a citation's offsets cannot be re-derived at verification time."""
    with pytest.raises(ValidationError, match="extractor identity"):
        Document(source="/a.pdf", content="text", source_class="extracted")


def test_a_non_extracted_document_may_not_record_an_extractor() -> None:
    """Both directions are enforced: a stray extractor implies a step that never runs."""
    with pytest.raises(ValidationError, match="only meaningful"):
        Document(source="/a.md", content="text", extractor="pdf-x/1")


def test_source_class_defaults_to_text() -> None:
    """The class every loader produced before ADR-0016, and the one the plain
    re-read-and-compare check assumes."""
    doc = Document(source="/a.md", content="text")
    assert doc.source_class == "text"
    assert doc.extractor is None


def test_retrieval_result_propagates_the_class_onto_its_citation() -> None:
    """A resolver must not have to look the class up separately.

    Looking it up at resolution time would let a document ingested under one
    class be verified under another's assumptions — the drift ADR-0004 closed
    for embeddings, one layer out.
    """
    result = RetrievalResult(
        content="hello",
        score=1.0,
        document_id="doc-1",
        chunk_id="c1",
        source="/a.pdf",
        source_class="extracted",
        extractor="pdf-x/1",
        start_offset=0,
        end_offset=5,
    )
    assert result.citation.source_class == "extracted"
    assert result.citation.extractor == "pdf-x/1"


# -- Resolution dispatch ---------------------------------------------------


def test_resolve_refuses_an_extracted_citation(tmp_path: Path) -> None:
    """Reading a PDF as UTF-8 is not a weaker check, it is the wrong one.

    Offsets index deterministic extractor output; comparing them against the
    file's raw bytes compares against text they were never measured against.
    """
    (tmp_path / "a.pdf").write_text("not really a pdf", encoding="utf-8")

    async def run() -> None:
        with pytest.raises(RetrievalError, match="extracted"):
            await resolve_citation(
                _citation(str(tmp_path / "a.pdf"), source_class="extracted", extractor="pdf-x/1"),
                tmp_path,
            )

    asyncio.run(run())


def test_resolve_refuses_a_snapshot_citation(tmp_path: Path) -> None:
    """A URL is refused by CLASS, before it can reach the path helper.

    That ordering is the fix, not an optimization — see the containment test
    below for what happens if a URL reaches ``ensure_within_base``.
    """

    async def run() -> None:
        with pytest.raises(RetrievalError, match="snapshot"):
            await resolve_citation(
                _citation("https://example.com/doc", source_class="snapshot"), tmp_path
            )

    asyncio.run(run())


def test_a_url_passes_path_containment_which_is_why_class_is_checked_first() -> None:
    """The latent hazard ADR-0016 decision 4 closes, pinned so it cannot be forgotten.

    ``ensure_within_base`` validates only that its input is non-empty and
    null-byte-free, then hands it to ``os.path.realpath`` — which resolves a
    URL as a *relative path under the current directory*. So containment
    **passes**, and the failure surfaces later from ``read_text`` as a
    confusing file-not-found rather than as "this is not a path".

    This test asserts the hazard still exists rather than that it was fixed:
    hardening the path helper to sniff URLs was explicitly rejected (a path
    helper that knows about URLs is a blurred boundary). If this ever starts
    failing, ``path_safety`` gained URL awareness and ADR-0016 decision 4's
    reasoning needs revisiting rather than the test being deleted.
    """
    base = Path.cwd()
    resolved = ensure_within_base("https://example.com/evil", base)
    assert base in resolved.parents
    assert not resolved.exists()


def test_resolve_still_works_for_text(tmp_path: Path) -> None:
    """The class that already held keeps its exact behaviour."""
    source = tmp_path / "a.md"
    source.write_text("hello world", encoding="utf-8")

    async def run() -> None:
        span = await resolve_citation(_citation(str(source)), tmp_path)
        assert span == "hello"

    asyncio.run(run())

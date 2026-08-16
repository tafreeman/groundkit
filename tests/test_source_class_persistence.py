"""Persistence and the join for ``source_class``/``extractor`` (ADR-0016 Wave 1).

Before this module's fix, ``Indexer`` accepted a ``Document`` carrying
``source_class="extracted"`` and an extractor identity, persisted it, and then
every read path (``get_document_records``, ``Retriever.search``) reported it
back as ``source_class="text"`` regardless — a silent downgrade, not an absent
feature, since the value existed on the ingested ``Document`` the whole time
and was simply dropped on write. The headline test below
(``test_ingested_extracted_document_is_not_silently_reported_as_text``)
demonstrates exactly that regression: it must fail against the pre-fix
``replace_document``/``Retriever._resolve`` (verified with ``git stash``, per
SPEC.md §8) and pass against the fix.

Async helpers are driven with ``asyncio.run()`` inside sync test functions,
matching every other async test module in this suite (pytest-asyncio is not
configured here).
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from groundkit.contracts import Document, SearchResponse
from groundkit.errors import StorageError
from groundkit.index.metadata import SCHEMA_VERSION, SQLiteMetadataStore
from groundkit.index.protocols import DocumentRecord, DocumentRecordStoreProtocol
from groundkit.indexer import Indexer
from groundkit.retrieval.search import Retriever
from test_protocol_conformance import assert_signature_parity


class _StubLoader:
    """Loader stand-in that returns one pre-built ``Document`` regardless of source.

    Satisfies ``LoaderProtocol`` structurally (``supported_extensions``,
    ``load``). No PDF/HTML/URL loader exists yet (ADR-0016 Waves 3-4 are out
    of scope for this change) — this is exactly what a keyword-only,
    defaulted ``source_class``/``extractor`` on ``Document`` is for:
    exercising the persistence path with a hand-built document a real loader
    will produce later.
    """

    def __init__(self, document: Document) -> None:
        self._document = document

    @property
    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    async def load(self, source: str) -> list[Document]:
        del source  # always returns the one fixed document; source is unused
        return [self._document]


# -- Schema -------------------------------------------------------------------


def test_schema_version_is_3() -> None:
    assert SCHEMA_VERSION == 3


# -- The new capability seam (index/protocols.py) ------------------------------


def test_sqlite_metadata_store_conforms_to_document_record_store_protocol(
    tmp_path: Path,
) -> None:
    """SQLiteMetadataStore implements the narrower, optional capability protocol.

    Deliberately a separate protocol from ``MetadataStoreProtocol`` (see
    ``DocumentRecordStoreProtocol``'s docstring) — this pins that
    ``SQLiteMetadataStore`` is nonetheless a real implementer, so
    ``Retriever._document_records``'s ``isinstance`` check takes the
    enriched branch for every real store, never the ``text``/``None``
    fallback meant only for a store double that predates ADR-0016.
    """

    async def run() -> SQLiteMetadataStore:
        return await SQLiteMetadataStore.open(tmp_path, "col")

    store = asyncio.run(run())
    try:
        assert isinstance(store, DocumentRecordStoreProtocol)
    finally:
        asyncio.run(store.close())


def test_sqlite_metadata_store_matches_document_record_store_protocol_signature() -> None:
    assert_signature_parity(DocumentRecordStoreProtocol, SQLiteMetadataStore)


# -- Store round trip (metadata.py, direct) ------------------------------------


def test_text_document_round_trips_as_text_none(tmp_path: Path) -> None:
    """A plain document, untouched by ADR-0016, keeps its exact prior defaults."""

    async def run() -> dict[str, DocumentRecord]:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            await store.replace_document(
                source="a.md", document_id="doc-1", content_hash="h1", chunks=[]
            )
            return await store.get_document_records()
        finally:
            await store.close()

    records = asyncio.run(run())
    record = records["doc-1"]
    assert record.source == "a.md"
    assert record.source_class == "text"
    assert record.extractor is None


def test_extracted_document_round_trips_through_replace_document(tmp_path: Path) -> None:
    async def run() -> dict[str, DocumentRecord]:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            await store.replace_document(
                source="a.pdf",
                document_id="doc-1",
                content_hash="h1",
                chunks=[],
                source_class="extracted",
                extractor="pdf-x/1",
            )
            return await store.get_document_records()
        finally:
            await store.close()

    records = asyncio.run(run())
    record = records["doc-1"]
    assert record.source == "a.pdf"
    assert record.source_class == "extracted"
    assert record.extractor == "pdf-x/1"


def test_snapshot_document_round_trips_through_upsert_document(tmp_path: Path) -> None:
    async def run() -> dict[str, DocumentRecord]:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            await store.upsert_document(
                source="https://example.com/doc",
                document_id="doc-1",
                content_hash="h1",
                source_class="snapshot",
            )
            return await store.get_document_records()
        finally:
            await store.close()

    records = asyncio.run(run())
    record = records["doc-1"]
    assert record.source == "https://example.com/doc"
    assert record.source_class == "snapshot"
    assert record.extractor is None


def test_get_document_records_refuses_a_pre_v3_store(tmp_path: Path) -> None:
    """A v1/v2 store genuinely lacks the two new columns; refuse cleanly rather
    than surface a raw ``sqlite3.OperationalError``."""
    db_path = tmp_path / "col.sqlite3"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(
            """
            CREATE TABLE documents (
                document_id TEXT PRIMARY KEY,
                source TEXT UNIQUE NOT NULL,
                content_hash TEXT NOT NULL,
                ingested_at TEXT NOT NULL
            );
            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                start_offset INTEGER NOT NULL,
                end_offset INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                metadata TEXT NOT NULL
            );
            """
        )
        conn.commit()
    finally:
        conn.close()

    async def run() -> None:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            with pytest.raises(StorageError, match="source_class"):
                await store.get_document_records()
        finally:
            await store.close()

    asyncio.run(run())


# -- The join: Indexer -> store -> Retriever (the headline regression) --------


def test_ingested_extracted_document_is_not_silently_reported_as_text(tmp_path: Path) -> None:
    """THE fail-open regression (SPEC.md §8).

    Ingests a hand-built ``Document(source_class="extracted", extractor=...)``
    through the real ``Indexer`` and asserts that both ``get_document_records``
    and a ``Retriever.search`` result — and, through it, the result's
    ``.citation`` — report ``"extracted"``/``"pdf-x/1"``, never the
    ``("text", None)`` default. Must fail against the pre-fix
    ``Indexer._persist_document`` (which dropped both fields) and
    ``Retriever._resolve`` (which never read them back); verified both
    directions with ``git stash`` per SPEC.md §8.
    """
    doc = Document(
        source=str(tmp_path / "a.pdf"),
        content="Extracted PDF text about turbines and hydrofoils.",
        source_class="extracted",
        extractor="pdf-x/1",
    )

    async def run() -> tuple[dict[str, DocumentRecord], SearchResponse]:
        store = await SQLiteMetadataStore.open(tmp_path / "idx", "default")
        try:
            indexer = Indexer(store, _StubLoader(doc))
            report = await indexer.index_source(doc.source)
            assert report.documents_indexed == 1

            records = await store.get_document_records()

            retriever = await Retriever.open(store)
            response = await retriever.search("extracted pdf turbines")
            return records, response
        finally:
            await store.close()

    records, response = asyncio.run(run())

    record = records[doc.document_id]
    assert record.source_class == "extracted", "silently downgraded to the ('text', None) default"
    assert record.extractor == "pdf-x/1"

    assert response.total_results == 1
    result = response.results[0]
    assert result.source_class == "extracted"
    assert result.extractor == "pdf-x/1"
    assert result.citation.source_class == "extracted"
    assert result.citation.extractor == "pdf-x/1"


def test_ingested_text_document_still_reports_text_none_via_search(tmp_path: Path) -> None:
    """The class every loader produced before ADR-0016 keeps its exact behavior
    end-to-end through ``Indexer`` and ``Retriever``, not just at the contract
    level (``tests/test_source_class.py`` already covers the contract)."""
    doc = Document(
        source=str(tmp_path / "a.md"), content="Plain markdown content about hydrofoils."
    )

    async def run() -> SearchResponse:
        store = await SQLiteMetadataStore.open(tmp_path / "idx", "default")
        try:
            indexer = Indexer(store, _StubLoader(doc))
            await indexer.index_source(doc.source)
            retriever = await Retriever.open(store)
            return await retriever.search("plain markdown hydrofoils")
        finally:
            await store.close()

    response = asyncio.run(run())
    assert response.total_results == 1
    result = response.results[0]
    assert result.source_class == "text"
    assert result.extractor is None
    assert result.citation.source_class == "text"
    assert result.citation.extractor is None

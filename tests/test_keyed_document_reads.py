"""GK-019: the read paths ask a keyed question with a keyed query.

Three read paths answered a question about at most ``top_k`` document IDs by
materializing the whole ``documents`` table — one validated
:class:`~groundkit.index.protocols.DocumentRecord` per stored row — and then
indexing into the result. ``Retriever.search`` did it on **every** query;
``handle_fetch_chunk`` did it for a single ID; ``chunk_count`` did the chunk
half (closed earlier, by GK-009). Cost scaled with the corpus while the
question scaled with ``top_k``.

The assertions here are on **the SQL the store executes**, not on which method
was called, following ``test_runtime.py::test_chunk_count_does_not_read_chunk_content``
and its ``index_status`` sibling. That matters because the defect is a cost
defect: a later refactor that reintroduces a full scan by another route — a
different method, a join, a cached-then-invalidated map — is still a full scan,
and asserting on the query is the only formulation that catches it.

What is deliberately *not* fixed by caching: the document read must stay
**live** per search. A retriever is an ``open()``-time snapshot on the lexical
side (ADR-0002), but the document join is what makes a hit against deleted
content fail closed, so hoisting it to ``open()`` would trade a cost defect for
a correctness one. :func:`test_the_document_row_is_read_live_not_snapshotted_at_open`
pins that direction explicitly and is labelled below for what it is.
"""

from __future__ import annotations

import asyncio
import sqlite3
from typing import TYPE_CHECKING

import pytest

from groundkit.config import RetrievalConfig
from groundkit.contracts import Chunk, SearchResponse
from groundkit.errors import ConfigurationError, RetrievalError, StorageError
from groundkit.index.metadata import SQLiteMetadataStore
from groundkit.index.protocols import (
    DocumentRecord,
    DocumentRecordStoreProtocol,
    MetadataStoreProtocol,
)
from groundkit.retrieval.search import Retriever
from groundkit.runtime import CollectionRegistry
from groundkit.service.schemas import FetchChunkRequest
from groundkit.service.tools import ServiceContext, handle_fetch_chunk
from metadata_store_doubles import (
    DelegatingMetadataStore,
    RefusingDocumentRecordStore,
    RefusingMetadataStore,
)
from test_protocol_conformance import assert_signature_parity

if TYPE_CHECKING:
    from pathlib import Path

#: Body of every seeded document. Long enough that a full-table read of
#: ``documents`` is visibly different from a keyed one in the traced SQL, and
#: distinct enough per document that BM25 ranks them apart.
_TEXTS: dict[str, str] = {
    "alpha.md": "Turbine maintenance intervals depend on load factor and ambient temperature.",
    "beta.md": "Hydrofoil lift scales with the square of forward speed through the water.",
    "gamma.md": "Pasta water should be salted generously before the pasta goes in.",
}


def _chunk(
    chunk_id: str,
    document_id: str,
    content: str,
    *,
    chunk_index: int = 0,
    start_offset: int = 0,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        content=content,
        start_offset=start_offset,
        end_offset=start_offset + len(content),
    )


async def _seed(tmp_path: Path) -> tuple[Path, Path]:
    """Write three real source files and index them. Returns ``(index_dir, corpus)``."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    index_dir = tmp_path / "index"
    index_dir.mkdir()

    store = await SQLiteMetadataStore.open(index_dir, "default")
    try:
        for position, (name, text) in enumerate(_TEXTS.items()):
            (corpus / name).write_text(text, encoding="utf-8")
            await store.replace_document(
                str(corpus / name),
                f"doc-{position}",
                f"hash-{position}",
                [_chunk(f"chunk-{position}", f"doc-{position}", text)],
            )
    finally:
        await store.close()
    return index_dir, corpus


def _documents_reads(statements: list[str]) -> list[str]:
    """Every traced statement that reads the ``documents`` table."""
    return [s for s in statements if "from documents" in s.lower()]


def _assert_every_documents_read_is_keyed(statements: list[str]) -> None:
    """Assert the ``documents`` table was read, and only ever by primary key.

    Both halves matter. "Only ever keyed" alone would pass vacuously against
    an implementation that stopped reading ``documents`` at all — which is
    precisely the wrong fix here, since the read is what makes a deleted
    document fail closed.
    """
    reads = _documents_reads(statements)
    assert reads, f"no read of `documents` happened at all; statements were {statements!r}"
    unkeyed = [s for s in reads if "where document_id" not in s.lower()]
    assert not unkeyed, (
        "the whole `documents` table was scanned to answer a question about a "
        f"bounded set of document ids: {unkeyed!r}"
    )


# -- Retriever.search ---------------------------------------------------------


def test_search_reads_documents_only_by_primary_key(tmp_path: Path) -> None:
    """The hot path: one query must not scan ``documents``.

    Fail-first: this is the defect itself. Against the pre-GK-019
    ``Retriever``, ``search`` opens with
    ``SELECT document_id, source, source_class, extractor FROM documents`` —
    no ``WHERE`` — before the mode branch even runs, and
    :func:`_assert_every_documents_read_is_keyed` names that statement in its
    failure message.

    Tracing starts *after* ``Retriever.open`` so the open()-time BM25 rebuild
    (``get_chunks``, and on the dense path ``get_document_sources``) is not
    what is measured. Those are ADR-0002's accepted O(corpus) startup cost;
    this test is about the per-query cost that had no such licence.
    """

    async def run() -> None:
        index_dir, _ = await _seed(tmp_path)
        store = await SQLiteMetadataStore.open(index_dir, "default")
        try:
            retriever = await Retriever.open(store)
            statements: list[str] = []
            store._conn.set_trace_callback(statements.append)
            try:
                response = await retriever.search("hydrofoil lift")
            finally:
                store._conn.set_trace_callback(None)
        finally:
            await store.close()

        assert response.total_results >= 1, "the query matched nothing, so nothing was joined"
        _assert_every_documents_read_is_keyed(statements)

    asyncio.run(run())


def test_search_that_returns_nothing_reads_no_document_row(tmp_path: Path) -> None:
    """A search below ``score_threshold`` resolves nothing, so it must read nothing.

    Fail-first: the pre-GK-019 ``search`` read every ``documents`` row before
    it knew whether any hit would survive the threshold, so a query that
    returned zero results still paid corpus-proportional cost. Here the join
    happens per surviving hit, after the filter — there are no survivors, so
    there is no document read.
    """

    async def run() -> None:
        index_dir, _ = await _seed(tmp_path)
        store = await SQLiteMetadataStore.open(index_dir, "default")
        try:
            retriever = await Retriever.open(store, RetrievalConfig(score_threshold=1_000_000.0))
            statements: list[str] = []
            store._conn.set_trace_callback(statements.append)
            try:
                response = await retriever.search("hydrofoil lift")
            finally:
                store._conn.set_trace_callback(None)
        finally:
            await store.close()

        reads = _documents_reads(statements)
        assert response.total_results == 0
        assert not reads, f"a search that resolved nothing still read documents: {reads!r}"

    asyncio.run(run())


def test_repeated_hits_on_one_document_are_read_once(tmp_path: Path) -> None:
    """Per-hit lookup, memoized per search — not one round trip per chunk.

    Several chunks of one document is the normal shape of a result list, and a
    naive per-hit keyed read would pay a lock acquisition and a thread hop for
    each of them. Bounded at one read per *distinct* document, which is still
    no less live than the single whole-table snapshot it replaces: every answer
    here is read at or after the moment the old one was.

    Not a regression test for the original defect and must not be reported as
    one — it guards the fix's own cost, and the pre-GK-019 code also reads
    ``documents`` once. It fails against a *plausible* wrong fix (drop the
    memo), which is what makes it worth keeping.
    """

    async def run() -> None:
        index_dir = tmp_path / "index"
        index_dir.mkdir()
        store = await SQLiteMetadataStore.open(index_dir, "default")
        try:
            parts = [
                "Turbine maintenance intervals depend on load factor.",
                "Turbine blades are inspected on the same maintenance schedule.",
                "Maintenance records for each turbine are kept for ten years.",
            ]
            chunks: list[Chunk] = []
            offset = 0
            for position, part in enumerate(parts):
                chunks.append(
                    _chunk(f"c{position}", "doc-0", part, chunk_index=position, start_offset=offset)
                )
                offset += len(part) + 1
            await store.replace_document("one.md", "doc-0", "h0", chunks)
            retriever = await Retriever.open(store)
            statements: list[str] = []
            store._conn.set_trace_callback(statements.append)
            try:
                response = await retriever.search("turbine maintenance")
            finally:
                store._conn.set_trace_callback(None)
        finally:
            await store.close()

        reads = _documents_reads(statements)
        assert response.total_results == 3, "all three chunks should have matched"
        assert len(reads) == 1, f"three hits on one document cost {len(reads)} reads: {reads!r}"

    asyncio.run(run())


def test_the_document_row_is_read_live_not_snapshotted_at_open(tmp_path: Path) -> None:
    """A document deleted after ``open()`` fails closed on the next search.

    NOT a regression test for GK-019 and must never be reported as one: the
    pre-fix code passes it too. It is the guard on the property the fix had to
    preserve, and the reason the obvious cheaper fix — read the table once at
    ``open()`` and keep it — is wrong. A retriever holding a stale document map
    would resolve this chunk to a source that no longer has a row, emitting a
    citation nothing can verify.
    """

    async def run() -> None:
        index_dir, _ = await _seed(tmp_path)
        store = await SQLiteMetadataStore.open(index_dir, "default")
        try:
            retriever = await Retriever.open(store)
            # Deleted through a *second* handle: this is what an out-of-process
            # `grk ingest --prune` looks like to a retriever already open.
            other = await SQLiteMetadataStore.open(index_dir, "default")
            try:
                assert await other.delete_document("doc-1") == 1
            finally:
                await other.close()

            with pytest.raises(RetrievalError, match="inconsistency"):
                await retriever.search("hydrofoil lift")
        finally:
            await store.close()

    asyncio.run(run())


# -- Retriever.search, over the two capability branches -----------------------


def test_the_keyed_branch_fails_closed_on_a_missing_document() -> None:
    """A store with the keyed capability refuses a dangling chunk, as the fallback does.

    Fail-first: the double implements ``get_document_record`` (returning
    ``None``) but leaves the inherited ``get_document_records`` refusing, so
    the pre-GK-019 ``Retriever`` — which reaches for the whole-table read —
    raises ``NotImplementedError`` here instead of the typed
    ``RetrievalError`` this asserts. That is exactly the distinction being
    pinned: which read the retrieval path performs, and that the keyed branch
    fails closed rather than fabricating a record for a row that is gone.
    """

    class _KeyedButEmpty(RefusingDocumentRecordStore):
        """Holds a chunk whose document row does not exist."""

        async def get_chunks(self) -> list[Chunk]:
            return [_chunk("chunk-x", "doc-gone", "orphaned chunk content")]

        async def get_document_record(self, document_id: str) -> DocumentRecord | None:
            return None

    store = _KeyedButEmpty()
    assert isinstance(store, MetadataStoreProtocol)
    assert isinstance(store, DocumentRecordStoreProtocol)

    async def run() -> None:
        retriever = await Retriever.open(store)
        with pytest.raises(RetrievalError, match="inconsistency"):
            await retriever.search("orphaned chunk")

    asyncio.run(run())


def test_the_fallback_branch_still_resolves_at_the_text_default() -> None:
    """A store without the optional capability keeps working, honestly degraded.

    NOT a regression test: it guards the fallback branch the fix had to keep.
    The branch is now *lazy* — the whole-table read happens on the first
    unresolved id rather than before the mode branch — so a fix that dropped
    the fallback entirely, or wired it to a member that no longer exists,
    would surface here rather than only for a downstream consumer with a
    hand-built store.
    """

    class _SourcesOnly(RefusingMetadataStore):
        async def get_chunks(self) -> list[Chunk]:
            return [_chunk("chunk-x", "doc-0", "orphaned chunk content")]

        async def get_document_sources(self) -> dict[str, str]:
            return {"doc-0": "ghost.md"}

    store = _SourcesOnly()
    assert not isinstance(store, DocumentRecordStoreProtocol)

    async def run() -> SearchResponse:
        retriever = await Retriever.open(store)
        return await retriever.search("orphaned chunk")

    response = asyncio.run(run())
    assert response.total_results == 1
    result = response.results[0]
    assert result.source == "ghost.md"
    # The honest degradation: a store that cannot report provenance reads back
    # as exactly what a plain `text` ingest would have produced, never as a
    # fabricated richer answer.
    assert result.source_class == "text"
    assert result.extractor is None


# -- handle_fetch_chunk -------------------------------------------------------


def test_fetch_chunk_reads_documents_only_by_primary_key(tmp_path: Path) -> None:
    """The per-result tool a client calls: one chunk, one document, one keyed read.

    Fail-first: the pre-GK-019 handler called ``runtime.get_document_records()``
    — an unfiltered ``SELECT ... FROM documents`` — to look up the single
    document its already-keyed chunk read had just named.
    """

    async def run() -> None:
        index_dir, corpus = await _seed(tmp_path)
        ctx = ServiceContext(
            registry=CollectionRegistry(index_dir), index_dir=index_dir, base_dir=corpus
        )
        statements: list[str] = []
        try:
            async with ctx.registry.acquire("default") as runtime:
                runtime._store._conn.set_trace_callback(statements.append)
                request = FetchChunkRequest(chunk_id="chunk-1")
                try:
                    response = await handle_fetch_chunk(ctx, request)
                finally:
                    runtime._store._conn.set_trace_callback(None)
        finally:
            await ctx.registry.aclose()

        assert response.verification == "verified"
        _assert_every_documents_read_is_keyed(statements)

    asyncio.run(run())


def test_fetch_chunk_of_an_unknown_chunk_reads_no_document_row(tmp_path: Path) -> None:
    """An unknown chunk id is refused without touching ``documents`` at all.

    Fail-first: the pre-GK-019 handler read every ``documents`` row *before*
    checking whether the chunk existed, so the cheapest possible rejection on
    an unauthenticated read-only surface — a request naming a chunk that is
    not there — was also corpus-proportional. That ordering is what an
    unauthenticated caller would have used to make a refusal expensive.
    """

    async def run() -> None:
        index_dir, corpus = await _seed(tmp_path)
        ctx = ServiceContext(
            registry=CollectionRegistry(index_dir), index_dir=index_dir, base_dir=corpus
        )
        statements: list[str] = []
        try:
            async with ctx.registry.acquire("default") as runtime:
                runtime._store._conn.set_trace_callback(statements.append)
                request = FetchChunkRequest(chunk_id="chunk-does-not-exist")
                try:
                    with pytest.raises(ConfigurationError, match="no chunk"):
                        await handle_fetch_chunk(ctx, request)
                finally:
                    runtime._store._conn.set_trace_callback(None)
        finally:
            await ctx.registry.aclose()

        reads = _documents_reads(statements)
        assert not reads, f"an unknown chunk id still cost a documents read: {reads!r}"

    asyncio.run(run())


# -- The store's two accessors must agree -------------------------------------


def test_the_keyed_and_whole_table_reads_report_the_same_record(tmp_path: Path) -> None:
    """One stored fact, two accessors, one answer — across all three source classes.

    Two read-only accessors disagreeing about one stored fact is exactly the
    ADR-0016 defect class (``search`` reporting ``extracted`` while
    ``fetch_chunk`` reported ``text`` for the same chunk). Adding a second
    accessor re-opens that possibility, so this pins the two together at the
    store's own boundary rather than trusting that one SQL statement was
    copied correctly from the other.
    """

    async def run() -> dict[str, tuple[DocumentRecord | None, DocumentRecord]]:
        index_dir = tmp_path / "index"
        index_dir.mkdir()
        store = await SQLiteMetadataStore.open(index_dir, "default")
        try:
            await store.replace_document("plain.md", "doc-text", "h1", [])
            await store.replace_document(
                "paper.pdf",
                "doc-extracted",
                "h2",
                [],
                source_class="extracted",
                extractor="pdf-x/1",
            )
            await store.upsert_document(
                "https://example.com/page",
                "doc-snapshot",
                "h3",
                source_class="snapshot",
            )
            whole_table = await store.get_document_records()
            return {
                document_id: (await store.get_document_record(document_id), record)
                for document_id, record in whole_table.items()
            }
        finally:
            await store.close()

    pairs = asyncio.run(run())
    assert set(pairs) == {"doc-text", "doc-extracted", "doc-snapshot"}
    for document_id, (keyed, from_table) in pairs.items():
        assert keyed == from_table, f"the two accessors disagree about {document_id}"


def test_get_document_record_of_an_unknown_id_is_none(tmp_path: Path) -> None:
    """``None`` means *no such document* — the value both callers fail closed on.

    A store that raised instead, or that returned a
    ``DocumentRecord(source="")`` placeholder, would either break the callers'
    typed error vocabulary or hand them a fabricated row to build an
    unverifiable citation from.
    """

    async def run() -> DocumentRecord | None:
        store = await SQLiteMetadataStore.open(tmp_path, "default")
        try:
            return await store.get_document_record("no-such-document")
        finally:
            await store.close()

    assert asyncio.run(run()) is None


def test_get_document_record_refuses_a_pre_v3_store(tmp_path: Path) -> None:
    """A v1/v2 store lacks the two columns; refuse cleanly, not with raw driver noise.

    Fail-first: drop the ``_require_source_class_capable`` call from
    ``get_document_record`` and the ``SELECT source, source_class, extractor``
    below raises ``sqlite3.OperationalError: no such column: source_class``,
    which is neither a ``GroundkitError`` nor a message naming the
    delete-and-re-ingest remedy. The whole-table sibling has this guard; a
    keyed sibling added without it would reopen the hole for one method.
    """
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
            with pytest.raises(StorageError, match="re-ingest"):
                await store.get_document_record("doc-0")
        finally:
            await store.close()

    asyncio.run(run())


# -- CollectionRuntime --------------------------------------------------------


def test_runtime_exposes_both_reads_and_they_agree(tmp_path: Path) -> None:
    """``fetch_chunk``'s keyed accessor and the whole-table one report the same row.

    The runtime is a thin facade over the store, and both accessors are part
    of its surface: the keyed one because every request path wants it, the
    whole-table one for a caller that genuinely wants every row. A facade whose
    two methods disagreed would be worse than either alone.
    """

    async def run() -> tuple[DocumentRecord | None, dict[str, DocumentRecord]]:
        index_dir, _ = await _seed(tmp_path)
        registry = CollectionRegistry(index_dir)
        try:
            async with registry.acquire("default") as runtime:
                keyed = await runtime.get_document_record("doc-1")
                whole_table = await runtime.get_document_records()
                return keyed, whole_table
        finally:
            await registry.aclose()

    keyed, whole_table = asyncio.run(run())
    assert keyed is not None
    assert keyed == whole_table["doc-1"]


# -- The shared test-double base is itself conformance-checked ----------------


class TestSharedStoreDoubleConformance:
    """The doubles base satisfies both store protocols, by signature not by hope.

    This is what actually removes the constraint GK-019 names. The recorded
    reason for keeping a capability off ``MetadataStoreProtocol`` was that
    widening it would break several hand-built doubles; the doubles now derive
    from one base, and these checks are what make that base a real
    stand-in — a protocol member added without a corresponding member here
    fails in one place, loudly, instead of surfacing as an ``AttributeError``
    from inside whichever test happens to drive it first.

    ``assert_signature_parity``, never ``isinstance``: ``isinstance`` against a
    ``runtime_checkable`` Protocol only checks that members of the same *name*
    exist, which is the vacuous pass this suite has had to close twice.
    """

    def test_refusing_base_matches_metadata_store_protocol(self) -> None:
        assert_signature_parity(MetadataStoreProtocol, RefusingMetadataStore)

    def test_refusing_record_base_matches_both_protocols(self) -> None:
        assert_signature_parity(MetadataStoreProtocol, RefusingDocumentRecordStore)
        assert_signature_parity(DocumentRecordStoreProtocol, RefusingDocumentRecordStore)

    def test_delegating_base_matches_both_protocols(self) -> None:
        assert_signature_parity(MetadataStoreProtocol, DelegatingMetadataStore)
        assert_signature_parity(DocumentRecordStoreProtocol, DelegatingMetadataStore)

    def test_the_plain_refusing_base_is_not_record_capable(self) -> None:
        """It must NOT satisfy the optional protocol, or the fallback branch is untestable.

        A double built on :class:`RefusingMetadataStore` exists precisely to
        drive the whole-table fallback. If the base ever acquired the optional
        members, every such double would silently switch to the keyed branch
        and the fallback would go untested while every test still passed.
        """
        assert not isinstance(RefusingMetadataStore(), DocumentRecordStoreProtocol)

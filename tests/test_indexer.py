"""Indexer tests: incremental persisted ingestion and the restart guarantee.

The restart test is the flagship: it closes every handle, reopens the store
from disk, and searches — the exact capability ARP defined but never wired
(ADR-0001 gap #1).

The Wave B block (Phase 3, ADR-0004) pins the dense write path onto the same
lifecycle: one vector per chunk, replace/prune/rename parity between SQLite
and the vector store, the embedding-identity manifest failing closed, and the
ordering invariant that the dense store is updated *before* SQLite commits —
SQLite's ``content_hash`` is the incremental skip key, so a SQLite-first
ordering that then failed on the vector write would leave a document
permanently skipped and silently absent from dense results.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path
from typing import Any, Final

import pytest

from groundkit.contracts import Chunk, Document
from groundkit.errors import (
    ConfigurationError,
    IndexIdentityError,
    IngestionError,
    StorageError,
)
from groundkit.index.dense import InMemoryVectorStore
from groundkit.index.metadata import SQLiteMetadataStore
from groundkit.index.protocols import VectorStoreProtocol
from groundkit.indexer import Indexer
from groundkit.ingestion.chunking import RecursiveChunker
from groundkit.ingestion.loaders import FileLoader
from groundkit.providers.embeddings import InMemoryEmbedder
from groundkit.retrieval.search import Retriever


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    (docs / "sub").mkdir(parents=True)
    (docs / ".hidden").mkdir()
    (docs / "alpha.md").write_text(
        "Groundkit persists its index in SQLite. Restarts do not lose data.",
        encoding="utf-8",
    )
    (docs / "beta.txt").write_text(
        "Citations carry character offsets back into the source file.",
        encoding="utf-8",
    )
    (docs / "sub" / "gamma.md").write_text(
        "Hybrid retrieval fuses BM25 with dense embeddings.", encoding="utf-8"
    )
    (docs / ".hidden" / "skipme.md").write_text("hidden", encoding="utf-8")
    (docs / "delta.pdf").write_text("unsupported", encoding="utf-8")
    return docs


async def _open(tmp_path: Path) -> SQLiteMetadataStore:
    return await SQLiteMetadataStore.open(tmp_path / "idx", "default")


class _CrashOnceChunker:
    """Wraps :class:`RecursiveChunker` but sabotages its first ``chunk()`` call.

    Stands in for "the process dies between the document row write and the
    chunk write" without needing an actual crash: the first call returns a
    chunk list whose first chunk carries non-JSON-serializable metadata, so
    ``SQLiteMetadataStore.replace_document`` fails partway through — after
    the document row would already have been written under the old,
    non-atomic ``upsert_document`` + ``add_chunks`` sequence. Every call
    after the first delegates to a real :class:`RecursiveChunker` untouched.
    """

    def __init__(self) -> None:
        self._inner = RecursiveChunker()
        self._armed = True

    def chunk(self, document: Document, **kwargs: Any) -> list[Chunk]:
        chunks = self._inner.chunk(document, **kwargs)
        if self._armed:
            self._armed = False
            sabotaged = chunks[0].model_copy(update={"metadata": {"bad": {1, 2}}})
            return [sabotaged, *chunks[1:]]
        return chunks


#: Vector width for every dense-path test embedder — small and readable. The
#: identity checks under test care about the (provider, model_name,
#: dimensions) triple, never the arithmetic, so 8 exercises everything 768
#: would.
_DIMS: Final[int] = 8

#: A top_k large enough to return every row a test ever stores, used as a
#: black-box "what does the dense store hold" probe via search() rather than
#: reaching into a store's private state.
_PROBE_TOP_K: Final[int] = 10_000


class _CountingEmbedder:
    """Wraps :class:`InMemoryEmbedder`, counting calls and letting identity vary.

    Two jobs. The counters (``embed_calls``, ``texts_embedded``) make "an
    unchanged document is never re-embedded" directly observable: a re-embed
    that produced byte-identical vectors would be invisible to every
    state-based assertion, but not to the call count. The per-instance
    ``provider``/``model_name`` let two embedders present *different*
    identities at the *same* width — the 768-vs-768 substitution ADR-0004
    exists to catch, scaled down to test size.

    Satisfies :class:`~groundkit.providers.protocols.EmbeddingProtocol`
    structurally (properties ``provider``/``model_name``/``dimensions``,
    ``async embed``).
    """

    def __init__(
        self,
        *,
        provider: str = "inmemory",
        model_name: str = "inmemory-hash-v1",
        dimensions: int = _DIMS,
    ) -> None:
        self._inner: InMemoryEmbedder = InMemoryEmbedder(dimensions=dimensions)
        self._provider: str = provider
        self._model_name: str = model_name
        self.embed_calls: int = 0
        self.texts_embedded: int = 0

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls += 1
        self.texts_embedded += len(texts)
        return await self._inner.embed(texts)


class _RecordingVectorStore:
    """Wraps :class:`InMemoryVectorStore`, recording call order; ``delete`` can fail once.

    The G1 regression seam. The replace path's ``add``/``delete`` order is
    not observable from final state — both orders converge on the same
    stored rows when nothing fails — so the order itself has to be recorded
    to be asserted. ``fail_delete_once`` then stops the run in the window
    between them, which *is* state-observable.

    Satisfies :class:`~groundkit.index.protocols.VectorStoreProtocol`
    structurally.
    """

    def __init__(self, *, fail_delete_once: bool = False) -> None:
        self._inner: InMemoryVectorStore = InMemoryVectorStore()
        self.calls: list[str] = []
        self._armed: bool = fail_delete_once

    async def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self.calls.append("add")
        await self._inner.add(chunks, embeddings)

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[tuple[Chunk, float]]:
        return await self._inner.search(query_embedding, top_k, metadata_filter)

    async def delete(self, document_id: str) -> int:
        self.calls.append("delete")
        if self._armed:
            self._armed = False
            raise StorageError("simulated dense-store outage on delete()")
        return await self._inner.delete(document_id)


class _SlowFailingChunker:
    """Raises for one target source; sleeps on every other, recording completions.

    The G2 regression seam. "One file's failure must not abandon its
    siblings" is only observable while the siblings are still in flight, so
    the target has to fail *fast* and the siblings have to be *slow*: the
    target is the first entry in path-sorted order, and every other source
    blocks long enough to still be mid-``_process`` when it does. Chunking
    runs through ``asyncio.to_thread``, so a blocking sleep here suspends
    that source without blocking the loop.
    """

    def __init__(self, target_name: str) -> None:
        self._inner: RecursiveChunker = RecursiveChunker()
        self._target_name: str = target_name
        self.completed: list[str] = []

    def chunk(self, document: Document, **kwargs: Any) -> list[Chunk]:
        if Path(document.source).name == self._target_name:
            raise RuntimeError(f"simulated chunker failure for {self._target_name}")
        time.sleep(0.25)
        chunks = self._inner.chunk(document, **kwargs)
        self.completed.append(Path(document.source).name)
        return chunks


class _FailOnceVectorStore:
    """Wraps :class:`InMemoryVectorStore`; the first ``add()`` raises ``StorageError``.

    Stands in for "the dense backend went down mid-ingest" — the dense
    analogue of :class:`_CrashOnceChunker`. Every call after the first
    ``add()``, and every ``search``/``delete`` throughout, delegates to the
    real in-memory store untouched, so a retry exercises the genuine write
    path rather than a second fake.

    Satisfies :class:`~groundkit.index.protocols.VectorStoreProtocol`
    structurally.
    """

    def __init__(self) -> None:
        self._inner: InMemoryVectorStore = InMemoryVectorStore()
        self._armed: bool = True

    async def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if self._armed:
            self._armed = False
            raise StorageError("simulated dense-store outage on first add()")
        await self._inner.add(chunks, embeddings)

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[tuple[Chunk, float]]:
        return await self._inner.search(query_embedding, top_k, metadata_filter)

    async def delete(self, document_id: str) -> int:
        return await self._inner.delete(document_id)


async def _dense_rows(vector_store: VectorStoreProtocol) -> list[tuple[Chunk, float]]:
    """Black-box dump of every stored dense row, via ``search()``.

    The ``tests/test_dense.py`` row-count idiom, reproduced locally: a dense
    search never drops zero-scored results (unlike BM25, which excludes
    score == 0), so an unfiltered search with a large ``top_k`` returns
    exactly the stored rows — no reaching into private attributes.
    """
    probe = [1.0] + [0.0] * (_DIMS - 1)
    return await vector_store.search(probe, top_k=_PROBE_TOP_K)


async def _assert_dense_matches_sqlite(
    store: SQLiteMetadataStore, vector_store: VectorStoreProtocol
) -> None:
    """Assert total dense/SQLite parity: exactly one vector per persisted chunk.

    The invariant every dense-enabled mutation must restore (Wave B
    "replace_document parity"): no orphaned vectors for chunks SQLite no
    longer holds, no chunk SQLite holds that the dense store is missing, and
    no duplicate vector rows.
    """
    sqlite_ids = {chunk.chunk_id for chunk in await store.get_chunks()}
    dense_ids = [chunk.chunk_id for chunk, _ in await _dense_rows(vector_store)]
    assert len(dense_ids) == len(set(dense_ids))  # no duplicate vector rows
    assert set(dense_ids) == sqlite_ids


def _write_pre_manifest_store(index_dir: Path, collection: str) -> None:
    """Create a pre-ADR-0004 store file: documents/chunks tables, no manifest stamp.

    Locally reproduces the legacy-store setup ``tests/test_metadata_store.py``
    uses (not imported from it — that module is being edited concurrently):
    the same documents/chunks schema ``SQLiteMetadataStore`` would create,
    but built over a raw ``sqlite3`` connection with no
    ``collection_manifest`` table and, crucially, neither
    ``PRAGMA application_id`` nor ``user_version`` set — both stay at
    SQLite's default of 0. The file exists *before*
    ``SQLiteMetadataStore.open`` ever sees it, matching how a real
    pre-ADR-0004 store is only ever encountered on reopen, never on
    creation.
    """
    index_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(index_dir / f"{collection}.sqlite3"))
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
            CREATE INDEX idx_chunks_document_id ON chunks (document_id);
            """
        )
        conn.commit()
    finally:
        conn.close()


def test_directory_indexing_counts(corpus: Path, tmp_path: Path) -> None:
    async def run() -> None:
        store = await _open(tmp_path)
        try:
            indexer = Indexer(store, FileLoader(allowed_base_dir=corpus))
            report = await indexer.index_directory(str(corpus))
            assert report.files_seen == 3
            assert report.documents_indexed == 3
            assert report.documents_skipped == 0
            assert report.chunks_written > 0
        finally:
            await store.close()

    asyncio.run(run())


def test_reindex_skips_unchanged_and_replaces_changed(corpus: Path, tmp_path: Path) -> None:
    async def run() -> None:
        store = await _open(tmp_path)
        try:
            indexer = Indexer(store, FileLoader(allowed_base_dir=corpus))
            await indexer.index_directory(str(corpus))
            baseline = len(await store.get_chunks())

            second = await indexer.index_directory(str(corpus))
            assert second.documents_indexed == 0
            assert second.documents_skipped == 3
            assert second.chunks_written == 0
            assert len(await store.get_chunks()) == baseline

            (corpus / "alpha.md").write_text(
                "Completely new alpha content about persistence.", encoding="utf-8"
            )
            third = await indexer.index_directory(str(corpus))
            assert third.documents_indexed == 1
            assert third.documents_skipped == 2

            contents = " ".join(c.content for c in await store.get_chunks())
            assert "Completely new alpha content" in contents
            assert "Restarts do not lose data" not in contents
        finally:
            await store.close()

    asyncio.run(run())


def test_search_survives_restart(corpus: Path, tmp_path: Path) -> None:
    async def ingest() -> None:
        store = await _open(tmp_path)
        try:
            indexer = Indexer(store, FileLoader(allowed_base_dir=corpus))
            await indexer.index_directory(str(corpus))
        finally:
            await store.close()

    async def search_fresh() -> None:
        store = await _open(tmp_path)
        try:
            retriever = await Retriever.open(store)
            response = await retriever.search("dense embeddings hybrid")
        finally:
            await store.close()
        assert response.total_results >= 1
        assert response.results[0].source.endswith("gamma.md")

    asyncio.run(ingest())
    # Separate event loop + fresh store handle == process-restart equivalent.
    asyncio.run(search_fresh())


def test_single_file_and_error_paths(corpus: Path, tmp_path: Path) -> None:
    async def run() -> None:
        store = await _open(tmp_path)
        try:
            indexer = Indexer(store, FileLoader(allowed_base_dir=corpus))
            report = await indexer.index_source(str(corpus / "alpha.md"))
            assert report.documents_indexed == 1

            with pytest.raises(IngestionError, match="Directory not found"):
                await indexer.index_directory(str(corpus / "nope"))
            with pytest.raises(ValueError, match="max_concurrent"):
                await indexer.index_directory(str(corpus), max_concurrent=0)
        finally:
            await store.close()

    asyncio.run(run())


def test_crash_between_document_and_chunk_write_does_not_orphan_document(
    corpus: Path, tmp_path: Path
) -> None:
    """A failure between the document row write and the chunk write must not
    permanently orphan the document.

    Regression test for the CRITICAL defect where ``Indexer._process`` called
    ``upsert_document`` and ``add_chunks`` as two separate committed store
    calls: a crash between them left the document row durably committed with
    the new content hash and zero chunks. Because ``_process`` skips a
    source whenever ``get_document_hash(source) == doc_hash``, that document
    was then skipped forever — permanently unsearchable, with no error ever
    surfaced. The fix routes both writes through
    ``SQLiteMetadataStore.replace_document``, one atomic commit; this
    simulates the crash by making the chunk-write step fail on the first
    attempt via ``_CrashOnceChunker``.
    """

    async def run() -> None:
        store = await _open(tmp_path)
        try:
            crash_once = _CrashOnceChunker()
            indexer = Indexer(store, FileLoader(allowed_base_dir=corpus), chunker=crash_once)

            with pytest.raises(StorageError):
                await indexer.index_source(str(corpus / "alpha.md"))

            # The "crash" must not have left a document row behind — under
            # the old defect this would already report the new hash despite
            # zero chunks existing for it.
            assert await store.get_document_sources() == {}
            assert await store.get_chunks() == []

            # A subsequent run (the chunker is no longer sabotaged) must
            # actually re-index the file rather than silently skipping it
            # forever.
            report = await indexer.index_source(str(corpus / "alpha.md"))
            assert report.documents_indexed == 1
            assert report.documents_skipped == 0
            assert report.chunks_written > 0
        finally:
            await store.close()

    asyncio.run(run())


def test_rename_prunes_old_source_and_leaves_no_duplicates(corpus: Path, tmp_path: Path) -> None:
    """Renaming a file between two directory-index runs must drop the old
    document row rather than leaving both old and new rows (and chunks)
    behind.
    """

    async def run() -> None:
        store = await _open(tmp_path)
        try:
            indexer = Indexer(store, FileLoader(allowed_base_dir=corpus))
            first = await indexer.index_directory(str(corpus))
            assert first.documents_pruned == 0

            (corpus / "alpha.md").rename(corpus / "renamed-alpha.md")

            second = await indexer.index_directory(str(corpus))
            assert second.documents_pruned == 1
            assert second.documents_indexed == 1  # renamed-alpha.md is a new source

            sources = list((await store.get_document_sources()).values())
            assert len(sources) == 3
            assert len(sources) == len(set(sources))  # no duplicate document rows
            assert not any(s.endswith("alpha.md") and "renamed" not in s for s in sources)
            assert any(s.endswith("renamed-alpha.md") for s in sources)

            chunk_ids = [c.chunk_id for c in await store.get_chunks()]
            assert len(chunk_ids) == len(set(chunk_ids))  # no duplicate chunks
        finally:
            await store.close()

    asyncio.run(run())


def test_delete_prunes_document_and_chunks(corpus: Path, tmp_path: Path) -> None:
    """Deleting a source file must remove its stored document and chunks on
    the next directory-index run.
    """

    async def run() -> None:
        store = await _open(tmp_path)
        try:
            indexer = Indexer(store, FileLoader(allowed_base_dir=corpus))
            await indexer.index_directory(str(corpus))

            (corpus / "beta.txt").unlink()

            report = await indexer.index_directory(str(corpus))
            assert report.documents_pruned == 1

            sources = list((await store.get_document_sources()).values())
            assert len(sources) == 2
            assert not any(s.endswith("beta.txt") for s in sources)

            remaining = await store.get_chunks()
            assert all("character offsets" not in c.content for c in remaining)
        finally:
            await store.close()

    asyncio.run(run())


def test_indexing_subdirectory_does_not_prune_sibling_documents(
    corpus: Path, tmp_path: Path
) -> None:
    """Indexing a subdirectory must never prune documents ingested from
    outside that subdirectory — the scoping constraint on the prune pass.
    """

    async def run() -> None:
        store = await _open(tmp_path)
        try:
            indexer = Indexer(store, FileLoader(allowed_base_dir=corpus))
            await indexer.index_directory(str(corpus))
            before = set((await store.get_document_sources()).values())
            assert len(before) == 3

            report = await indexer.index_directory(str(corpus / "sub"))
            assert report.documents_pruned == 0

            after = set((await store.get_document_sources()).values())
            assert after == before
        finally:
            await store.close()

    asyncio.run(run())


def test_index_source_never_prunes(corpus: Path, tmp_path: Path) -> None:
    """The single-file entry point must never prune a *vanished* source."""

    async def run() -> None:
        store = await _open(tmp_path)
        try:
            indexer = Indexer(store, FileLoader(allowed_base_dir=corpus))
            await indexer.index_directory(str(corpus))

            (corpus / "beta.txt").unlink()

            report = await indexer.index_source(str(corpus / "alpha.md"))
            assert report.documents_pruned == 0

            sources = list((await store.get_document_sources()).values())
            assert any(s.endswith("beta.txt") for s in sources)  # untouched
        finally:
            await store.close()

    asyncio.run(run())


def test_index_directory_prunes_emptied_file(corpus: Path, tmp_path: Path) -> None:
    """Blanking a previously-indexed file's content must prune its stored
    document and chunks on the next directory-index run.

    Regression test for the CRITICAL defect where ``FileLoader.load``
    returns ``[]`` for empty content, so ``Indexer._process``'s
    ``for doc in documents`` loop never ran and never deleted the stale
    document — and the file still exists on disk, so ``_prune_missing``
    never saw it as "missing" either. The chunks from the old content
    stayed searchable forever with no source text left to back them.
    """

    async def run() -> None:
        store = await _open(tmp_path)
        try:
            indexer = Indexer(store, FileLoader(allowed_base_dir=corpus))
            await indexer.index_directory(str(corpus))
            assert any("Restarts do not lose data" in c.content for c in await store.get_chunks())

            (corpus / "alpha.md").write_text("", encoding="utf-8")

            report = await indexer.index_directory(str(corpus))
            assert report.documents_pruned == 1
            assert report.documents_indexed == 0

            sources = list((await store.get_document_sources()).values())
            assert len(sources) == 2
            assert not any(s.endswith("alpha.md") for s in sources)

            remaining = await store.get_chunks()
            assert not any("Restarts do not lose data" in c.content for c in remaining)
        finally:
            await store.close()

    asyncio.run(run())


def test_index_directory_prunes_whitespace_only_file(corpus: Path, tmp_path: Path) -> None:
    """Whitespace-only content, not just zero bytes, must also trigger the prune."""

    async def run() -> None:
        store = await _open(tmp_path)
        try:
            indexer = Indexer(store, FileLoader(allowed_base_dir=corpus))
            await indexer.index_directory(str(corpus))

            (corpus / "beta.txt").write_text("   \n\t  \n", encoding="utf-8")

            report = await indexer.index_directory(str(corpus))
            assert report.documents_pruned == 1

            sources = list((await store.get_document_sources()).values())
            assert not any(s.endswith("beta.txt") for s in sources)

            remaining = await store.get_chunks()
            assert not any("character offsets" in c.content for c in remaining)
        finally:
            await store.close()

    asyncio.run(run())


def test_index_source_prunes_emptied_file(corpus: Path, tmp_path: Path) -> None:
    """The single-file entry point must prune the exact source it was given
    when that source's content is emptied, via the same lifecycle as
    ``index_directory`` (index, blank the file, re-index).
    """

    async def run() -> None:
        store = await _open(tmp_path)
        try:
            indexer = Indexer(store, FileLoader(allowed_base_dir=corpus))
            first = await indexer.index_source(str(corpus / "alpha.md"))
            assert first.documents_indexed == 1

            (corpus / "alpha.md").write_text("   ", encoding="utf-8")

            report = await indexer.index_source(str(corpus / "alpha.md"))
            assert report.documents_indexed == 0
            assert report.documents_skipped == 0
            assert report.chunks_written == 0
            assert report.documents_pruned == 1

            assert await store.get_document_sources() == {}
            assert await store.get_chunks() == []
        finally:
            await store.close()

    asyncio.run(run())


def test_index_source_empty_file_never_indexed_is_noop(corpus: Path, tmp_path: Path) -> None:
    """An empty file that was never indexed must not error and must not
    report a spurious prune — there is nothing stored to delete.
    """

    async def run() -> None:
        store = await _open(tmp_path)
        try:
            indexer = Indexer(store, FileLoader(allowed_base_dir=corpus))
            never_indexed = corpus / "never-indexed.md"
            never_indexed.write_text("", encoding="utf-8")

            report = await indexer.index_source(str(never_indexed))
            assert report.documents_indexed == 0
            assert report.documents_skipped == 0
            assert report.chunks_written == 0
            assert report.documents_pruned == 0

            assert await store.get_document_sources() == {}
        finally:
            await store.close()

    asyncio.run(run())


def test_emptying_one_file_does_not_prune_a_different_document(
    corpus: Path, tmp_path: Path
) -> None:
    """Emptying one source must only ever remove *its own* stored document —
    never another file's, even though both were discovered in the same
    directory-index run.
    """

    async def run() -> None:
        store = await _open(tmp_path)
        try:
            indexer = Indexer(store, FileLoader(allowed_base_dir=corpus))
            await indexer.index_directory(str(corpus))

            (corpus / "alpha.md").write_text("", encoding="utf-8")
            report = await indexer.index_directory(str(corpus))
            assert report.documents_pruned == 1

            sources = list((await store.get_document_sources()).values())
            assert any(s.endswith("beta.txt") for s in sources)
            assert any(s.endswith("gamma.md") for s in sources)

            remaining_contents = " ".join(c.content for c in await store.get_chunks())
            assert "character offsets" in remaining_contents  # beta.txt survives
            assert "Hybrid retrieval" in remaining_contents  # sub/gamma.md survives
        finally:
            await store.close()

    asyncio.run(run())


# ── Wave B: the dense write path (Phase 3, ADR-0004) ───────────────────────


def test_bm25_only_run_reports_zero_vector_counts(corpus: Path, tmp_path: Path) -> None:
    """With no embedder and no vector store, Phase 1 behaviour is unchanged.

    Both-``None`` is Wave B's compatibility contract: every Phase 1 count is
    exactly what it was, the two new report fields exist and read zero, and
    no embedding-identity manifest gets bound. The manifest assertion has
    teeth — a BM25-only collection must stay bindable later under *any*
    embedding identity, and a spuriously written manifest would permanently
    lock it to one that never produced a single vector.
    """

    async def run() -> None:
        store = await _open(tmp_path)
        try:
            indexer = Indexer(store, FileLoader(allowed_base_dir=corpus))
            report = await indexer.index_directory(str(corpus))
            assert report.files_seen == 3
            assert report.documents_indexed == 3
            assert report.documents_skipped == 0
            assert report.chunks_written > 0
            assert report.documents_pruned == 0
            assert report.vectors_written == 0
            assert report.vectors_deleted == 0
            assert await store.get_manifest() is None
        finally:
            await store.close()

    asyncio.run(run())


def test_embedder_without_vector_store_is_a_construction_error(
    corpus: Path, tmp_path: Path
) -> None:
    """An embedder with no vector store must fail at construction, not mid-run.

    Half a dense path is a misconfiguration, and discovering it lazily — on
    the first changed document, after other documents were already
    processed — would strand a partially updated index behind an error that
    only some corpora ever trigger. ``ConfigurationError`` at ``Indexer(...)``
    time means no file has been read, no hash advanced, nothing to clean up
    (errors.py: invalid configuration is a startup failure, SPEC.md §2).
    """

    async def run() -> None:
        store = await _open(tmp_path)
        try:
            with pytest.raises(ConfigurationError):
                Indexer(
                    store,
                    FileLoader(allowed_base_dir=corpus),
                    embedder=_CountingEmbedder(),
                )
        finally:
            await store.close()

    asyncio.run(run())


def test_vector_store_without_embedder_is_a_construction_error(
    corpus: Path, tmp_path: Path
) -> None:
    """A vector store with no embedder is the same construction-time refusal.

    The mirror half of the previous test: a vector store with nothing to
    fill cannot mean "dense, but skip the embedding" — a run that silently
    wrote no vectors while the caller believed dense indexing was on would
    surface only as quietly empty dense results much later.
    """

    async def run() -> None:
        store = await _open(tmp_path)
        try:
            with pytest.raises(ConfigurationError):
                Indexer(
                    store,
                    FileLoader(allowed_base_dir=corpus),
                    vector_store=InMemoryVectorStore(),
                )
        finally:
            await store.close()

    asyncio.run(run())


def test_dense_ingest_writes_one_vector_per_chunk(corpus: Path, tmp_path: Path) -> None:
    """A dense-enabled ingest stores exactly one vector per chunk it writes.

    ``vectors_written`` must equal ``chunks_written`` *and* the store must
    actually hold that many rows: a report that counts intents rather than
    completed writes, or a write path that drops or duplicates chunks
    between the chunker and the vector store, both break here. A fresh
    ingest also deletes nothing — there is no old document whose vectors
    could need removing — so ``vectors_deleted`` must read zero.
    """

    async def run() -> None:
        store = await _open(tmp_path)
        try:
            vector_store = InMemoryVectorStore()
            indexer = Indexer(
                store,
                FileLoader(allowed_base_dir=corpus),
                embedder=_CountingEmbedder(),
                vector_store=vector_store,
            )
            report = await indexer.index_directory(str(corpus))
            assert report.documents_indexed == 3
            assert report.chunks_written > 0
            assert report.vectors_written == report.chunks_written
            assert report.vectors_deleted == 0
            assert len(await _dense_rows(vector_store)) == report.vectors_written
            await _assert_dense_matches_sqlite(store, vector_store)
        finally:
            await store.close()

    asyncio.run(run())


def test_reindex_of_unchanged_corpus_never_reembeds(corpus: Path, tmp_path: Path) -> None:
    """An unchanged corpus must not cost a single embedding call on re-index.

    The content-hash gate is extended to the dense path, not duplicated: a
    skipped document is skipped everywhere. The embedder counters are the
    teeth — a "delete and re-add identical vectors" implementation would
    pass every state-based assertion (same rows, same contents) while
    quietly re-paying the full embedding cost (real deployments: network +
    GPU, per document, per run) forever. The counters must be completely
    unmoved, not merely "not much higher".
    """

    async def run() -> None:
        store = await _open(tmp_path)
        try:
            embedder = _CountingEmbedder()
            vector_store = InMemoryVectorStore()
            indexer = Indexer(
                store,
                FileLoader(allowed_base_dir=corpus),
                embedder=embedder,
                vector_store=vector_store,
            )
            await indexer.index_directory(str(corpus))
            calls_before = embedder.embed_calls
            texts_before = embedder.texts_embedded
            rows_before = len(await _dense_rows(vector_store))

            second = await indexer.index_directory(str(corpus))
            assert second.documents_skipped == 3
            assert second.vectors_written == 0
            assert second.vectors_deleted == 0
            assert embedder.embed_calls == calls_before
            assert embedder.texts_embedded == texts_before
            assert len(await _dense_rows(vector_store)) == rows_before
        finally:
            await store.close()

    asyncio.run(run())


def test_replaced_document_swaps_its_vectors_without_orphans(corpus: Path, tmp_path: Path) -> None:
    """Rewriting a document replaces its vectors in the same logical operation.

    Wave B replace parity: a document row without its vectors — or the
    inverse, stale vectors surviving a replace — is the Phase 1
    ``replace_document`` hazard repeated one store over. The stale-vector
    direction is the poisonous one: dense search keeps returning content no
    source file contains, and citation resolution can only fail closed on
    every such hit, one query at a time, forever. Afterwards the vector
    store must hold exactly SQLite's chunk set — old gone, new present, no
    orphans in either direction.
    """

    async def run() -> None:
        store = await _open(tmp_path)
        try:
            vector_store = InMemoryVectorStore()
            indexer = Indexer(
                store,
                FileLoader(allowed_base_dir=corpus),
                embedder=_CountingEmbedder(),
                vector_store=vector_store,
            )
            await indexer.index_directory(str(corpus))

            (corpus / "alpha.md").write_text(
                "Completely new alpha content about persistence.", encoding="utf-8"
            )
            report = await indexer.index_directory(str(corpus))
            assert report.documents_indexed == 1
            assert report.vectors_deleted > 0
            assert report.vectors_written == report.chunks_written
            assert report.vectors_written > 0

            contents = " ".join(c.content for c, _ in await _dense_rows(vector_store))
            assert "Completely new alpha content" in contents
            assert "Restarts do not lose data" not in contents
            await _assert_dense_matches_sqlite(store, vector_store)
        finally:
            await store.close()

    asyncio.run(run())


def test_pruned_document_loses_its_vectors(corpus: Path, tmp_path: Path) -> None:
    """Deleting a source file prunes its vectors along with its SQLite rows.

    A prune that removed only the SQLite side would leave orphaned vectors
    that dense search happily returns — hits for a document whose source no
    longer exists on disk and whose row no longer exists in SQLite, which
    the retriever cannot even join to a source path. Prune order is vectors
    first, then the SQLite row (the ordering invariant: SQLite may never be
    ahead of the dense store), and the observable result is total parity.
    """

    async def run() -> None:
        store = await _open(tmp_path)
        try:
            vector_store = InMemoryVectorStore()
            indexer = Indexer(
                store,
                FileLoader(allowed_base_dir=corpus),
                embedder=_CountingEmbedder(),
                vector_store=vector_store,
            )
            await indexer.index_directory(str(corpus))
            seeded = await _dense_rows(vector_store)
            assert any("character offsets" in c.content for c, _ in seeded)

            (corpus / "beta.txt").unlink()

            report = await indexer.index_directory(str(corpus))
            assert report.documents_pruned == 1
            assert report.vectors_deleted > 0
            assert report.vectors_written == 0

            rows = await _dense_rows(vector_store)
            assert not any("character offsets" in c.content for c, _ in rows)
            await _assert_dense_matches_sqlite(store, vector_store)
        finally:
            await store.close()

    asyncio.run(run())


def test_renamed_source_moves_vectors_without_duplicates(corpus: Path, tmp_path: Path) -> None:
    """Renaming a file re-homes its vectors: old document out, new one in.

    A rename is a prune of the old source plus a fresh ingest of the new
    one, and the content is byte-identical — so content-based assertions
    cannot tell old vectors from new, and a lazy implementation that kept
    the old document's vectors alongside the new one's would double every
    dense hit for this document. The discriminator is ``document_id``: the
    pruned document's vectors must be gone, the new document's present, and
    the vector store must hold exactly one row per SQLite chunk.
    """

    async def run() -> None:
        store = await _open(tmp_path)
        try:
            vector_store = InMemoryVectorStore()
            indexer = Indexer(
                store,
                FileLoader(allowed_base_dir=corpus),
                embedder=_CountingEmbedder(),
                vector_store=vector_store,
            )
            await indexer.index_directory(str(corpus))
            sources = await store.get_document_sources()
            old_doc_id = next(d for d, s in sources.items() if s.endswith("alpha.md"))

            (corpus / "alpha.md").rename(corpus / "renamed-alpha.md")

            report = await indexer.index_directory(str(corpus))
            assert report.documents_pruned == 1
            assert report.documents_indexed == 1
            assert report.vectors_deleted > 0
            assert report.vectors_written > 0

            after = await store.get_document_sources()
            new_doc_id = next(d for d, s in after.items() if s.endswith("renamed-alpha.md"))
            dense_doc_ids = {c.document_id for c, _ in await _dense_rows(vector_store)}
            assert old_doc_id not in dense_doc_ids
            assert new_doc_id in dense_doc_ids
            await _assert_dense_matches_sqlite(store, vector_store)
        finally:
            await store.close()

    asyncio.run(run())


def test_index_source_emptied_file_prunes_its_vectors(corpus: Path, tmp_path: Path) -> None:
    """Blanking a dense-indexed file and ``index_source``-ing it prunes its vectors.

    The single-file entry point's emptied-source prune (the Phase 1
    regression above, ``test_index_source_prunes_emptied_file``) must reach
    the dense store too: the loader yields no documents for whitespace-only
    content, so a dense path wired only through the per-document loop would
    never see this document again and its vectors would survive as
    permanent orphans — retrievable content whose source text no longer
    exists anywhere.
    """

    async def run() -> None:
        store = await _open(tmp_path)
        try:
            vector_store = InMemoryVectorStore()
            indexer = Indexer(
                store,
                FileLoader(allowed_base_dir=corpus),
                embedder=_CountingEmbedder(),
                vector_store=vector_store,
            )
            first = await indexer.index_source(str(corpus / "alpha.md"))
            assert first.documents_indexed == 1
            assert first.vectors_written == first.chunks_written
            assert first.vectors_written > 0

            (corpus / "alpha.md").write_text("   \n\t  \n", encoding="utf-8")

            report = await indexer.index_source(str(corpus / "alpha.md"))
            assert report.documents_pruned == 1
            assert report.vectors_deleted > 0
            assert report.vectors_written == 0
            assert await _dense_rows(vector_store) == []
            assert await store.get_document_sources() == {}
            assert await store.get_chunks() == []
        finally:
            await store.close()

    asyncio.run(run())


def test_first_dense_write_binds_manifest_to_embedder_identity(
    corpus: Path, tmp_path: Path
) -> None:
    """The first dense write binds the collection to the *embedder's* identity.

    The manifest triple must be read off the embedder that actually produced
    the vectors — provider, model name, dimensions, exactly — not off some
    config object that merely should agree with it
    (``providers/protocols.py`` sources all three from the embedder for
    precisely this reason). The non-default values are the proof of
    provenance: a manifest populated from a default ``EmbeddingConfig``
    would read nomic-embed-text/768, not model-a/8.
    """

    async def run() -> None:
        store = await _open(tmp_path)
        try:
            embedder = _CountingEmbedder(model_name="model-a", dimensions=8)
            indexer = Indexer(
                store,
                FileLoader(allowed_base_dir=corpus),
                embedder=embedder,
                vector_store=InMemoryVectorStore(),
            )
            await indexer.index_directory(str(corpus))

            manifest = await store.get_manifest()
            assert manifest is not None
            assert manifest.provider == "inmemory"
            assert manifest.model_name == "model-a"
            assert manifest.dimensions == 8
        finally:
            await store.close()

    asyncio.run(run())


def test_identity_mismatch_fails_closed_with_nothing_mutated(corpus: Path, tmp_path: Path) -> None:
    """ADR-0004's flagship: same width, different model — refused before any write.

    The collection is bound to ``("inmemory", "model-a", 8)``; a second
    indexer arrives over the same store and vector store with
    ``("inmemory", "model-b", 8)`` — identical arithmetic, mutually
    incomprehensible semantic spaces, the 768-vs-768 substitution a
    width-only check would wave through. ``alpha.md`` is modified first so
    the mismatched run has real dense work queued (delete old vectors,
    embed, replace the SQLite row): without that pending work, "nothing was
    mutated" would hold trivially even under a too-late check. The run must
    raise ``IndexIdentityError`` and leave everything untouched — vector
    rows, SQLite sources and chunk contents, and the new embedder's call
    counter — which is what proves verification happens per-run *before*
    any load/chunk/embed/delete, not somewhere in the middle.
    """

    async def run() -> None:
        store = await _open(tmp_path)
        try:
            vector_store = InMemoryVectorStore()
            indexer = Indexer(
                store,
                FileLoader(allowed_base_dir=corpus),
                embedder=_CountingEmbedder(model_name="model-a", dimensions=8),
                vector_store=vector_store,
            )
            await indexer.index_directory(str(corpus))

            rows_before = {c.chunk_id for c, _ in await _dense_rows(vector_store)}
            sources_before = await store.get_document_sources()
            chunks_before = {c.chunk_id for c in await store.get_chunks()}

            (corpus / "alpha.md").write_text("Entirely different alpha content.", encoding="utf-8")

            swapped_embedder = _CountingEmbedder(model_name="model-b", dimensions=8)
            with pytest.raises(IndexIdentityError):
                swapped = Indexer(
                    store,
                    FileLoader(allowed_base_dir=corpus),
                    embedder=swapped_embedder,
                    vector_store=vector_store,
                )
                await swapped.index_directory(str(corpus))

            rows_after = await _dense_rows(vector_store)
            assert {c.chunk_id for c, _ in rows_after} == rows_before
            assert len(rows_after) == len(rows_before)
            assert await store.get_document_sources() == sources_before
            assert {c.chunk_id for c in await store.get_chunks()} == chunks_before
            # The *old* alpha content is still what both stores hold …
            assert any("Restarts do not lose data" in c.content for c, _ in rows_after)
            assert not any("Entirely different alpha" in c.content for c, _ in rows_after)
            # … and the mismatched embedder was never even invoked.
            assert swapped_embedder.embed_calls == 0
        finally:
            await store.close()

    asyncio.run(run())


def test_noop_dense_ingest_does_not_bind_manifest(tmp_path: Path) -> None:
    """A dense-enabled run that writes nothing must not bind the collection.

    The manifest is written on the first real dense write, not on intent: a
    run over a directory with no supported files must leave
    ``get_manifest()`` as ``None``. An indexer that bound at construction
    or at run start would lock an *empty* collection to whatever embedder
    happened to be configured — and every later, genuine first ingest under
    a different (perhaps corrected) embedder would fail its identity check
    against vectors that never existed.
    """
    barren = tmp_path / "barren"
    barren.mkdir()
    (barren / "unsupported.pdf").write_text("not a supported format", encoding="utf-8")

    async def run() -> None:
        store = await _open(tmp_path)
        try:
            embedder = _CountingEmbedder()
            indexer = Indexer(
                store,
                FileLoader(allowed_base_dir=barren),
                embedder=embedder,
                vector_store=InMemoryVectorStore(),
            )
            report = await indexer.index_directory(str(barren))
            assert report.files_seen == 0
            assert report.vectors_written == 0
            assert embedder.embed_calls == 0
            assert await store.get_manifest() is None
        finally:
            await store.close()

    asyncio.run(run())


def test_failed_vector_write_leaves_sqlite_behind_never_ahead(corpus: Path, tmp_path: Path) -> None:
    """A dense-store failure mid-ingest must leave SQLite behind, never ahead.

    The ordering invariant at its failure point, and the dense analogue of
    ``test_crash_between_document_and_chunk_write_does_not_orphan_document``:
    SQLite's ``content_hash`` is the incremental skip key, so a SQLite-first
    ordering that then failed on the vector write would commit the new hash
    with zero vectors — the document is skipped on every later run and
    silently absent from dense results, permanently, with no error ever
    surfaced. Dense-first ordering makes the same failure self-healing: no
    hash was committed, so the next run (the store recovered) re-indexes
    the document completely, vectors included. The reverse residue —
    vectors present for an uncommitted document — is the tolerable
    direction, because ``Retriever.search`` already fails closed on a hit
    whose document has no stored source.
    """

    async def run() -> None:
        store = await _open(tmp_path)
        try:
            fail_once = _FailOnceVectorStore()
            indexer = Indexer(
                store,
                FileLoader(allowed_base_dir=corpus),
                embedder=_CountingEmbedder(),
                vector_store=fail_once,
            )

            with pytest.raises(StorageError):
                await indexer.index_source(str(corpus / "alpha.md"))

            # SQLite recorded nothing: the content hash never advanced.
            assert await store.get_document_sources() == {}
            assert await store.get_chunks() == []

            # The store no longer fails; the same source must now index
            # fully rather than being skipped forever.
            report = await indexer.index_source(str(corpus / "alpha.md"))
            assert report.documents_indexed == 1
            assert report.documents_skipped == 0
            assert report.chunks_written > 0
            assert report.vectors_written == report.chunks_written

            rows = await _dense_rows(fail_once)
            assert len(rows) == report.vectors_written
            assert any("Restarts do not lose data" in c.content for c, _ in rows)
            await _assert_dense_matches_sqlite(store, fail_once)
        finally:
            await store.close()

    asyncio.run(run())


def test_legacy_unstamped_store_is_refused_for_dense_indexing(corpus: Path, tmp_path: Path) -> None:
    """A pre-ADR-0004 store is refused for dense work before anything runs.

    A store with no ``PRAGMA application_id``/``user_version`` stamp
    predates the manifest and recorded no embedding identity; guessing one
    is exactly the silent-corruption path ADR-0004 closes. A dense-enabled
    indexer over such a store must raise ``IndexIdentityError`` up front —
    before any load, chunk, embed, or delete has touched either store, so
    both remain exactly as found. The refusal is scoped to dense work only:
    the same legacy store must keep serving BM25-only indexing unchanged
    (pre-1.0 the documented remedy is delete-and-reingest, not a migration
    and not a lockout).
    """
    _write_pre_manifest_store(tmp_path / "idx", "default")

    async def run() -> None:
        store = await _open(tmp_path)  # opens the legacy file just written
        try:
            vector_store = InMemoryVectorStore()
            embedder = _CountingEmbedder()
            with pytest.raises(IndexIdentityError):
                dense = Indexer(
                    store,
                    FileLoader(allowed_base_dir=corpus),
                    embedder=embedder,
                    vector_store=vector_store,
                )
                await dense.index_directory(str(corpus))

            # Refused before anything happened, on either store.
            assert await store.get_document_sources() == {}
            assert await store.get_chunks() == []
            assert embedder.embed_calls == 0
            assert await _dense_rows(vector_store) == []

            # …but only for dense work: BM25-only indexing still works.
            bm25_only = Indexer(store, FileLoader(allowed_base_dir=corpus))
            report = await bm25_only.index_directory(str(corpus))
            assert report.documents_indexed == 3
        finally:
            await store.close()

    asyncio.run(run())


def test_dense_replace_adds_new_vectors_before_deleting_old(corpus: Path, tmp_path: Path) -> None:
    """The replace path must add the new vectors before deleting the old ones.

    G1. Deleting first opens a window in which the document has no vectors
    at all, and that window is not self-healing: a crash inside it leaves
    SQLite's ``content_hash`` still matching the *previous* content, so
    reverting the file (``git checkout``) makes every later run hash-skip
    the document — permanently, silently absent from dense results, which
    is exactly what SPEC.md §2's fail-closed rule forbids. Ordering is
    invisible in final state (both orders converge when nothing fails), so
    it is pinned by call order here and by consequence in
    ``test_crash_between_dense_add_and_delete_preserves_old_vectors``.
    """

    async def run() -> None:
        store = await _open(tmp_path)
        try:
            recorder = _RecordingVectorStore()
            indexer = Indexer(
                store,
                FileLoader(allowed_base_dir=corpus),
                embedder=_CountingEmbedder(),
                vector_store=recorder,
            )
            alpha = corpus / "alpha.md"

            await indexer.index_source(str(alpha))
            # First ingest: nothing stored for this source, so no delete.
            assert recorder.calls == ["add"]

            alpha.write_text("Entirely different content for the replace.", encoding="utf-8")
            report = await indexer.index_source(str(alpha))
            assert report.documents_indexed == 1

            # The replace: add strictly precedes delete.
            assert recorder.calls == ["add", "add", "delete"]
            await _assert_dense_matches_sqlite(store, recorder)
        finally:
            await store.close()

    asyncio.run(run())


def test_crash_between_dense_add_and_delete_preserves_old_vectors(
    corpus: Path, tmp_path: Path
) -> None:
    """A crash mid-replace must leave the previous content still dense-searchable.

    G1's defect, reproduced end to end: fail between the two dense writes,
    then revert the file. Under delete-first the old vectors were already
    gone and the un-advanced ``content_hash`` now matches the reverted
    bytes, so the document is skipped forever with no error — a silent
    permanent hole. Under add-first the old vectors are still there, so the
    reverted content resolves correctly.

    What this does *not* claim: the interrupted run's vectors remain in the
    store under a ``document_id`` SQLite never committed, and no prune sweep
    can reach them (sweeps iterate SQLite). That residue is loud rather than
    silent — ``Retriever.search`` fails closed on it — and is recorded in
    ``KNOWN_LIMITATIONS.md``.
    """

    async def run() -> None:
        store = await _open(tmp_path)
        try:
            recorder = _RecordingVectorStore(fail_delete_once=True)
            indexer = Indexer(
                store,
                FileLoader(allowed_base_dir=corpus),
                embedder=_CountingEmbedder(),
                vector_store=recorder,
            )
            alpha = corpus / "alpha.md"
            original = alpha.read_text(encoding="utf-8")

            await indexer.index_source(str(alpha))

            alpha.write_text("Replacement content that will fail to land.", encoding="utf-8")
            with pytest.raises(StorageError):
                await indexer.index_source(str(alpha))

            # The add landed, the delete did not, and SQLite never advanced.
            assert recorder.calls == ["add", "add", "delete"]
            stored = await store.get_chunks()
            assert any(original[:20] in chunk.content for chunk in stored)

            # Revert: the hash now matches what SQLite still holds, so the
            # document is skipped -- and must still be dense-searchable.
            alpha.write_text(original, encoding="utf-8")
            report = await indexer.index_source(str(alpha))
            assert report.documents_skipped == 1
            assert report.documents_indexed == 0

            rows = await _dense_rows(recorder)
            assert any("Restarts do not lose data" in chunk.content for chunk, _ in rows)
        finally:
            await store.close()

    asyncio.run(run())


def test_one_failing_file_does_not_abandon_its_siblings(corpus: Path, tmp_path: Path) -> None:
    """A single file's failure must not cancel siblings mid-write.

    G2. ``asyncio.gather`` without ``return_exceptions=True`` propagates the
    first failure immediately and cancels every in-flight sibling, landing a
    ``CancelledError`` inside whatever store call each was awaiting -- for a
    dense-enabled indexer, potentially between the vector write and the
    SQLite commit, the one torn state the ordering invariant cannot make
    self-healing. Every sibling must therefore run to completion before the
    failure propagates.

    ``alpha.md`` sorts first and fails instantly; the rest sleep, so under a
    bare gather they are still in flight when it does.
    """

    async def run() -> None:
        store = await _open(tmp_path)
        try:
            chunker = _SlowFailingChunker("alpha.md")
            indexer = Indexer(store, FileLoader(allowed_base_dir=corpus), chunker)

            with pytest.raises(IngestionError):
                await indexer.index_directory(str(corpus))

            # Both siblings finished rather than being cancelled mid-flight.
            assert sorted(chunker.completed) == ["beta.txt", "gamma.md"]
            sources = await store.get_document_sources()
            assert sorted(Path(s).name for s in sources.values()) == ["beta.txt", "gamma.md"]
        finally:
            await store.close()

    asyncio.run(run())


def test_lost_dense_side_is_refused_rather_than_silently_reindexed(
    corpus: Path, tmp_path: Path
) -> None:
    """A manifest-bound collection whose vectors vanished must fail closed.

    G10. SQLite's ``content_hash`` records whether the *content* changed,
    never whether the vectors derived from it still exist -- so a collection
    that kept its SQLite file while losing its dense store reports every
    document unchanged, re-embeds nothing, and answers every dense query
    from an empty index, permanently and silently.

    Reproduced the way a library caller reaches it: an ephemeral
    ``InMemoryVectorStore`` paired with a persisted store, then a fresh
    vector store standing in for the restart. The likelier operational
    route -- a deleted ``.lance`` directory -- lands in the identical state,
    which is why the check keys on the ADR-0004 manifest (proof that vectors
    once existed) rather than on the store's type.
    """

    async def run() -> None:
        store = await _open(tmp_path)
        try:
            embedder = _CountingEmbedder()
            first = Indexer(
                store,
                FileLoader(allowed_base_dir=corpus),
                embedder=embedder,
                vector_store=InMemoryVectorStore(),
            )
            report = await first.index_directory(str(corpus))
            assert report.vectors_written > 0

            # The restart: SQLite survived, the vectors did not.
            restarted = InMemoryVectorStore()
            after = Indexer(
                store,
                FileLoader(allowed_base_dir=corpus),
                embedder=_CountingEmbedder(),
                vector_store=restarted,
            )
            with pytest.raises(StorageError, match="Dense side is empty"):
                await after.index_directory(str(corpus))

            # The read path refuses it too, rather than answering from empty.
            with pytest.raises(StorageError, match="Dense side is empty"):
                await Retriever.open(store, embedder=_CountingEmbedder(), vector_store=restarted)
        finally:
            await store.close()

    asyncio.run(run())


def test_bm25_only_collection_is_not_mistaken_for_a_lost_dense_side(
    corpus: Path, tmp_path: Path
) -> None:
    """No manifest means dense was never used -- not that vectors were lost.

    The counterpart to the check above, and the case that decides it is
    keyed on the manifest rather than on emptiness alone. Enabling the dense
    path over a collection previously ingested BM25-only leaves every
    unchanged document without vectors by design (the documented
    no-backfill limitation), and that legitimate upgrade must not be
    refused as corruption.
    """

    async def run() -> None:
        store = await _open(tmp_path)
        try:
            bm25_only = Indexer(store, FileLoader(allowed_base_dir=corpus))
            assert (await bm25_only.index_directory(str(corpus))).documents_indexed == 3

            # Dense enabled afterwards: no manifest was ever bound, so the
            # empty vector store is expected, not evidence of loss.
            upgraded = Indexer(
                store,
                FileLoader(allowed_base_dir=corpus),
                embedder=_CountingEmbedder(),
                vector_store=InMemoryVectorStore(),
            )
            report = await upgraded.index_directory(str(corpus))
            assert report.documents_skipped == 3
        finally:
            await store.close()

    asyncio.run(run())

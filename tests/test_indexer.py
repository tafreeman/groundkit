"""Indexer tests: incremental persisted ingestion and the restart guarantee.

The restart test is the flagship: it closes every handle, reopens the store
from disk, and searches — the exact capability ARP defined but never wired
(ADR-0001 gap #1).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from groundkit.contracts import Chunk, Document
from groundkit.errors import IngestionError, StorageError
from groundkit.index.metadata import SQLiteMetadataStore
from groundkit.indexer import Indexer
from groundkit.ingestion.chunking import RecursiveChunker
from groundkit.ingestion.loaders import FileLoader
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

"""Indexer tests: incremental persisted ingestion and the restart guarantee.

The restart test is the flagship: it closes every handle, reopens the store
from disk, and searches — the exact capability ARP defined but never wired
(ADR-0001 gap #1).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from groundkit.errors import IngestionError
from groundkit.index.metadata import SQLiteMetadataStore
from groundkit.indexer import Indexer
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

"""Retriever and citation-resolution tests (Phase 1, BM25-only path)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from groundkit.config import RetrievalConfig
from groundkit.contracts import (
    Chunk,
    Citation,
    CollectionManifest,
    Document,
    EmbeddingIdentity,
)
from groundkit.errors import RetrievalError
from groundkit.index.metadata import SQLiteMetadataStore
from groundkit.index.protocols import MetadataStoreProtocol
from groundkit.indexer import Indexer
from groundkit.ingestion.loaders import FileLoader
from groundkit.retrieval.citations import resolve_citation, verify_citation
from groundkit.retrieval.search import MAX_TOP_K, Retriever

DOC_TEXTS = {
    "alpha.md": "Retrieval systems rank documents. BM25 is a lexical ranking function.",
    "beta.md": "Cooking pasta requires boiling water and a generous pinch of salt.",
}


def _chunks_for(doc: Document) -> list[Chunk]:
    return [
        Chunk(
            document_id=doc.document_id,
            chunk_index=0,
            content=doc.content,
            start_offset=0,
            end_offset=len(doc.content),
            metadata={"source": doc.source},
        )
    ]


async def _populated_store(tmp_path: Path) -> SQLiteMetadataStore:
    store = await SQLiteMetadataStore.open(tmp_path / "idx", "default")
    for name, text in DOC_TEXTS.items():
        doc = Document(source=str(tmp_path / name), content=text)
        await store.upsert_document(
            source=doc.source, document_id=doc.document_id, content_hash="h"
        )
        await store.add_chunks(_chunks_for(doc), source=doc.source)
    return store


class TestRetriever:
    def test_search_returns_ranked_citation_bearing_results(self, tmp_path: Path) -> None:
        async def run() -> None:
            store = await _populated_store(tmp_path)
            try:
                retriever = await Retriever.open(store)
                response = await retriever.search("BM25 lexical ranking")
            finally:
                await store.close()
            assert response.total_results == 1
            top = response.results[0]
            assert top.source.endswith("alpha.md")
            assert top.start_offset == 0
            assert top.end_offset == len(DOC_TEXTS["alpha.md"])
            assert top.score > 0
            assert top.citation.chunk_id == top.chunk_id
            assert response.metadata["stage"] == "bm25"

        asyncio.run(run())

    def test_empty_query_raises(self, tmp_path: Path) -> None:
        async def run() -> None:
            store = await _populated_store(tmp_path)
            try:
                retriever = await Retriever.open(store)
                with pytest.raises(RetrievalError, match="empty"):
                    await retriever.search("   ")
            finally:
                await store.close()

        asyncio.run(run())

    @pytest.mark.parametrize("bad_k", [0, -1, MAX_TOP_K + 1])
    def test_out_of_range_top_k_raises(self, tmp_path: Path, bad_k: int) -> None:
        async def run() -> None:
            store = await _populated_store(tmp_path)
            try:
                retriever = await Retriever.open(store)
                with pytest.raises(RetrievalError, match="top_k"):
                    await retriever.search("pasta", top_k=bad_k)
            finally:
                await store.close()

        asyncio.run(run())

    def test_score_threshold_filters(self, tmp_path: Path) -> None:
        async def run() -> None:
            store = await _populated_store(tmp_path)
            try:
                strict = await Retriever.open(store, RetrievalConfig(score_threshold=1_000_000.0))
                response = await strict.search("pasta")
            finally:
                await store.close()
            assert response.total_results == 0

        asyncio.run(run())

    def test_missing_source_mapping_fails_closed(self) -> None:
        doc = Document(source="ghost.md", content="orphaned chunk content")

        class OrphanStore:
            async def upsert_document(
                self, source: str, document_id: str, content_hash: str
            ) -> None:
                raise NotImplementedError

            async def get_document_hash(self, source: str) -> str | None:
                return None

            async def get_document_id(self, source: str) -> str | None:
                return None

            async def get_document_sources(self) -> dict[str, str]:
                return {}

            async def add_chunks(self, chunks: list[Chunk], source: str) -> None:
                raise NotImplementedError

            async def replace_document(
                self, source: str, document_id: str, content_hash: str, chunks: list[Chunk]
            ) -> None:
                raise NotImplementedError

            async def get_chunks(self) -> list[Chunk]:
                return _chunks_for(doc)

            async def get_chunk(self, chunk_id: str) -> Chunk | None:
                return None

            async def delete_document(self, document_id: str) -> int:
                return 0

            async def write_manifest(self, identity: EmbeddingIdentity) -> None:
                raise NotImplementedError

            async def verify_manifest(self, identity: EmbeddingIdentity) -> None:
                raise NotImplementedError

            async def get_manifest(self) -> CollectionManifest | None:
                return None

        store = OrphanStore()
        assert isinstance(store, MetadataStoreProtocol)

        async def run() -> None:
            retriever = await Retriever.open(store)
            with pytest.raises(RetrievalError, match="inconsistency"):
                await retriever.search("orphaned chunk")

        asyncio.run(run())


class TestStaleRetriever:
    """ADR-0002: ``Retriever.open`` snapshots BM25 once and never refreshes.

    Pins both halves of that deliberate, currently-undocumented-in-code
    behavior with a real end-to-end re-ingest (``Indexer`` + a persisted
    ``SQLiteMetadataStore``, not a hand-built stub store):

    - a retriever holding a chunk whose document was re-ingested (so its
      old ``document_id`` no longer resolves to a source) fails closed —
      ``RetrievalError`` — exactly like the pre-existing orphaned-chunk case
      above, just reached via a real re-ingest instead of a stub.
    - a retriever queried for content ingested *after* it was opened finds
      nothing and raises nothing: BM25 was never rebuilt, so the new
      chunks were never tokenized into it. There is no signal that the
      index is stale — this is the surprising half.

    ``src/groundkit/retrieval/search.py`` is not owned by this change; its
    class docstring still needs a staleness note (see the accompanying
    report — that file is out of scope here).
    """

    def test_stale_retriever_raises_on_document_modified_after_open(self, tmp_path: Path) -> None:
        """A retriever opened before a re-ingest fails closed on a doc changed underneath it."""

        async def run() -> None:
            docs_dir = tmp_path / "docs"
            docs_dir.mkdir()
            target = docs_dir / "doc.md"
            target.write_text("Retrieval systems rank documents by relevance.", encoding="utf-8")

            store = await SQLiteMetadataStore.open(tmp_path / "idx", "default")
            try:
                indexer = Indexer(store, FileLoader(allowed_base_dir=docs_dir))
                await indexer.index_directory(str(docs_dir))

                retriever = await Retriever.open(store)

                # Re-ingest with different content: replace_document deletes
                # the old document row (and its document_id) and inserts a
                # fresh one — the stale retriever's BM25 snapshot still
                # holds a chunk pointing at the now-gone document_id.
                target.write_text("Something completely unrelated now.", encoding="utf-8")
                await indexer.index_directory(str(docs_dir))

                with pytest.raises(RetrievalError, match="inconsistency"):
                    await retriever.search("relevance")
            finally:
                await store.close()

        asyncio.run(run())

    def test_stale_retriever_returns_zero_results_for_content_ingested_after_open(
        self, tmp_path: Path
    ) -> None:
        """A retriever opened before a new document is ingested silently misses it: zero
        results, no error — no signal at all that the index is stale."""

        async def run() -> None:
            docs_dir = tmp_path / "docs"
            docs_dir.mkdir()
            (docs_dir / "alpha.md").write_text(
                "Retrieval systems rank documents by relevance.", encoding="utf-8"
            )

            store = await SQLiteMetadataStore.open(tmp_path / "idx", "default")
            try:
                indexer = Indexer(store, FileLoader(allowed_base_dir=docs_dir))
                await indexer.index_directory(str(docs_dir))

                retriever = await Retriever.open(store)

                (docs_dir / "beta.md").write_text(
                    "Zebras migrate across the savanna every dry season.", encoding="utf-8"
                )
                await indexer.index_directory(str(docs_dir))

                response = await retriever.search("zebras savanna")

                assert response.total_results == 0
                assert response.results == []
            finally:
                await store.close()

        asyncio.run(run())


class TestCitations:
    def _write_source(self, tmp_path: Path) -> tuple[Citation, str]:
        text = "Alpha beta gamma delta epsilon."
        path = tmp_path / "doc.md"
        path.write_text(text, encoding="utf-8")
        span = (6, 16)
        citation = Citation(
            document_id="d",
            chunk_id="c",
            source=str(path),
            start_offset=span[0],
            end_offset=span[1],
        )
        return citation, text[span[0] : span[1]]

    def test_resolve_returns_exact_span(self, tmp_path: Path) -> None:
        citation, expected = self._write_source(tmp_path)
        assert asyncio.run(resolve_citation(citation, tmp_path)) == expected

    def test_verify_roundtrip(self, tmp_path: Path) -> None:
        citation, expected = self._write_source(tmp_path)
        assert asyncio.run(verify_citation(citation, expected, tmp_path))
        assert not asyncio.run(verify_citation(citation, "tampered", tmp_path))

    def test_source_escape_rejected(self, tmp_path: Path) -> None:
        self._write_source(tmp_path)
        outside = Citation(
            document_id="d",
            chunk_id="c",
            source=str(tmp_path / ".." / "outside.md"),
            start_offset=0,
            end_offset=4,
        )
        with pytest.raises(RetrievalError, match="escapes"):
            asyncio.run(resolve_citation(outside, tmp_path))

    def test_changed_source_detected(self, tmp_path: Path) -> None:
        citation, _ = self._write_source(tmp_path)
        Path(citation.source).write_text("tiny", encoding="utf-8")
        with pytest.raises(RetrievalError, match="source changed"):
            asyncio.run(resolve_citation(citation, tmp_path))

    def test_missing_source_raises(self, tmp_path: Path) -> None:
        citation = Citation(
            document_id="d",
            chunk_id="c",
            source=str(tmp_path / "never.md"),
            start_offset=0,
            end_offset=4,
        )
        with pytest.raises(RetrievalError, match="Cannot read"):
            asyncio.run(resolve_citation(citation, tmp_path))


class TestCitationsInvalidUtf8:
    """A source that becomes invalid UTF-8 after indexing must surface the
    typed ``RetrievalError`` (SPEC.md §2), not a raw ``UnicodeDecodeError`` —
    ``UnicodeDecodeError`` is a ``ValueError`` subclass, not an ``OSError``,
    so a naive ``except OSError`` around the read lets it escape uncaught."""

    def _write_invalid_utf8_source(self, tmp_path: Path) -> Citation:
        path = tmp_path / "invalid.md"
        path.write_bytes(b"\xff\xfe\x00invalid")
        return Citation(
            document_id="d",
            chunk_id="c",
            source=str(path),
            start_offset=0,
            end_offset=4,
        )

    def test_resolve_raises_retrieval_error_not_unicode_decode_error(self, tmp_path: Path) -> None:
        citation = self._write_invalid_utf8_source(tmp_path)
        with pytest.raises(RetrievalError, match="not valid UTF-8") as exc_info:
            asyncio.run(resolve_citation(citation, tmp_path))
        assert isinstance(exc_info.value.__cause__, UnicodeDecodeError)

    def test_verify_raises_retrieval_error_not_unicode_decode_error(self, tmp_path: Path) -> None:
        citation = self._write_invalid_utf8_source(tmp_path)
        with pytest.raises(RetrievalError, match="not valid UTF-8") as exc_info:
            asyncio.run(verify_citation(citation, "anything", tmp_path))
        assert isinstance(exc_info.value.__cause__, UnicodeDecodeError)

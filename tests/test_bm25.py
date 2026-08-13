"""Tests for BM25Index — ported behaviors from ARP's test_rag_retrieval.py
(ADR-0001) plus the persistence-rebuild guarantee ADR-0002 adds.

Async methods are driven with ``asyncio.run()`` inside sync test functions
(pytest-asyncio is not configured in this repo).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from groundkit.contracts import Chunk
from groundkit.index.bm25 import BM25Index
from groundkit.index.metadata import SQLiteMetadataStore


def _make_chunk(
    content: str,
    *,
    chunk_id: str,
    document_id: str = "doc-1",
    chunk_index: int = 0,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        content=content,
        start_offset=0,
        end_offset=len(content),
    )


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Three chunks with distinct vocabularies, mirroring ARP's fixture."""
    return [
        _make_chunk(
            "Python is a popular programming language for data science",
            chunk_id="c1",
            chunk_index=0,
        ),
        _make_chunk(
            "Rust is a systems programming language focused on safety",
            chunk_id="c2",
            chunk_index=1,
        ),
        _make_chunk(
            "Data science uses Python and R for statistical analysis",
            chunk_id="c3",
            chunk_index=2,
        ),
    ]


class TestBM25Index:
    """Behaviors ported from ARP's ``TestBM25Index``."""

    def test_single_chunk_search(self) -> None:
        chunk = _make_chunk("hello world", chunk_id="c1")
        index = BM25Index()
        index.index_chunks([chunk])

        results = index.search("hello", top_k=5)

        assert len(results) == 1
        result_chunk, score = results[0]
        assert result_chunk.chunk_id == "c1"
        assert score > 0.0

    def test_multiple_chunks_ranking(self, sample_chunks: list[Chunk]) -> None:
        index = BM25Index()
        index.index_chunks(sample_chunks)

        results = index.search("Python data science", top_k=3)

        result_ids = [chunk.chunk_id for chunk, _ in results]
        assert "c1" in result_ids
        assert "c3" in result_ids
        if "c2" in result_ids:
            scores = dict(zip(result_ids, (score for _, score in results), strict=True))
            assert scores["c1"] > scores["c2"]

    def test_term_frequency_scoring_order(self) -> None:
        """A chunk repeating a query term more often scores higher."""
        single = _make_chunk("python is great", chunk_id="single")
        triple = _make_chunk("python python python", chunk_id="triple")
        index = BM25Index()
        index.index_chunks([single, triple])

        results = index.search("python", top_k=2)

        assert len(results) == 2
        (top_chunk, top_score), (_, second_score) = results
        assert top_chunk.chunk_id == "triple"
        assert top_score > second_score

    def test_empty_query_returns_empty_list(self) -> None:
        chunk = _make_chunk("some content", chunk_id="c1")
        index = BM25Index()
        index.index_chunks([chunk])

        assert index.search("", top_k=5) == []

    def test_case_insensitive_tokenization(self) -> None:
        chunk = _make_chunk("Python PYTHON python", chunk_id="c1")
        index = BM25Index()
        index.index_chunks([chunk])

        results = index.search("PYTHON", top_k=5)

        assert len(results) == 1
        result_chunk, score = results[0]
        assert result_chunk.chunk_id == "c1"
        assert score > 0.0

    def test_no_matching_terms_returns_empty_list(self) -> None:
        chunk = _make_chunk("hello world", chunk_id="c1")
        index = BM25Index()
        index.index_chunks([chunk])

        assert index.search("xyzzy foobar", top_k=5) == []

    def test_empty_index_returns_empty_list(self) -> None:
        index = BM25Index()

        assert index.search("hello", top_k=5) == []

    def test_top_k_respected(self) -> None:
        chunks = [
            _make_chunk(f"word word chunk {i}", chunk_id=f"c{i}", chunk_index=i) for i in range(10)
        ]
        index = BM25Index()
        index.index_chunks(chunks)

        results = index.search("word", top_k=3)

        assert len(results) == 3

    def test_all_scores_non_negative(self, sample_chunks: list[Chunk]) -> None:
        index = BM25Index()
        index.index_chunks(sample_chunks)

        results = index.search("Python data science language", top_k=10)

        assert results
        assert all(score >= 0.0 for _, score in results)

    def test_custom_k1_b_change_scores_but_preserve_ranking(self) -> None:
        """Different k1/b values yield different scores for the same ranking."""
        chunks = [
            _make_chunk("python is great for scripting", chunk_id="low"),
            _make_chunk("python python python is the best python", chunk_id="high"),
        ]
        default_index = BM25Index()
        default_index.index_chunks(chunks)
        custom_index = BM25Index(k1=3.0, b=0.25)
        custom_index.index_chunks(chunks)

        default_results = default_index.search("python", top_k=2)
        custom_results = custom_index.search("python", top_k=2)

        assert [chunk.chunk_id for chunk, _ in default_results] == ["high", "low"]
        assert [chunk.chunk_id for chunk, _ in custom_results] == ["high", "low"]
        assert default_results[0][1] != custom_results[0][1]

    def test_incremental_indexing_preserves_prior_chunks(self) -> None:
        """index_chunks() accumulates rather than replacing."""
        index = BM25Index()
        index.index_chunks([_make_chunk("alpha beta", chunk_id="c1")])
        index.index_chunks([_make_chunk("gamma delta", chunk_id="c2")])

        assert index.size == 2
        results = index.search("alpha", top_k=5)
        assert [chunk.chunk_id for chunk, _ in results] == ["c1"]

    def test_tied_scores_sort_by_content_hash_not_insertion_order(self) -> None:
        """Two chunks with identical tf/idf/length inputs score exactly
        equal for "term"; the tie must resolve by ascending content_hash,
        not by the order they were appended in."""
        alpha = _make_chunk("alpha term term", chunk_id="alpha")
        beta = _make_chunk("beta term term", chunk_id="beta")
        assert alpha.content_hash < beta.content_hash  # pins the expected tie-break direction

        index = BM25Index()
        index.index_chunks([beta, alpha])  # insertion order is the opposite of hash order

        results = index.search("term", top_k=2)

        assert len(results) == 2
        (first_chunk, first_score), (second_chunk, second_score) = results
        assert first_score == pytest.approx(second_score)  # confirms a genuine tie, not luck
        assert first_chunk.chunk_id == "alpha"
        assert second_chunk.chunk_id == "beta"

    def test_tie_break_order_independent_of_insertion_order(self) -> None:
        """Regression pin: indexing the same tied chunks in either order
        must yield identical search-result ordering. A position-derived
        tie-break (e.g. the raw doc_idx) would fail this."""
        alpha = _make_chunk("alpha term term", chunk_id="alpha")
        beta = _make_chunk("beta term term", chunk_id="beta")

        forward_index = BM25Index()
        forward_index.index_chunks([alpha, beta])
        backward_index = BM25Index()
        backward_index.index_chunks([beta, alpha])

        forward_ids = [chunk.chunk_id for chunk, _ in forward_index.search("term", top_k=2)]
        backward_ids = [chunk.chunk_id for chunk, _ in backward_index.search("term", top_k=2)]

        assert forward_ids == backward_ids == ["alpha", "beta"]


class TestBM25FromStore:
    """The ADR-0002 persistence-rebuild guarantee: rebuilding from SQLite
    reproduces the same ranking and scores as the original in-memory index."""

    def test_from_store_rebuild_matches_pre_restart_index(
        self, tmp_path: Path, sample_chunks: list[Chunk]
    ) -> None:
        original = BM25Index()
        original.index_chunks(sample_chunks)
        original_results = original.search("python data science", top_k=5)

        async def _persist_and_rebuild() -> list[tuple[Chunk, float]]:
            store = await SQLiteMetadataStore.open(tmp_path, "col")
            try:
                await store.upsert_document(source="doc.md", document_id="doc-1", content_hash="h1")
                await store.add_chunks(sample_chunks, source="doc.md")
                rebuilt = await BM25Index.from_store(store)
                return rebuilt.search("python data science", top_k=5)
            finally:
                await store.close()

        rebuilt_results = asyncio.run(_persist_and_rebuild())

        assert [chunk.chunk_id for chunk, _ in rebuilt_results] == [
            chunk.chunk_id for chunk, _ in original_results
        ]
        for (_, original_score), (_, rebuilt_score) in zip(
            original_results, rebuilt_results, strict=True
        ):
            assert rebuilt_score == pytest.approx(original_score)

    def test_from_store_empty_store_returns_empty_index(self, tmp_path: Path) -> None:
        async def _rebuild() -> BM25Index:
            store = await SQLiteMetadataStore.open(tmp_path, "col")
            try:
                return await BM25Index.from_store(store)
            finally:
                await store.close()

        rebuilt = asyncio.run(_rebuild())

        assert rebuilt.size == 0
        assert rebuilt.search("anything", top_k=5) == []

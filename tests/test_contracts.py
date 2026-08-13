"""Contract tests — adapted from ARP's test_rag_contracts.py per ADR-0001,
extended with the offset invariants groundkit adds."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from groundkit.contracts import (
    HIGH_CONFIDENCE_SCORE,
    Chunk,
    Citation,
    Document,
    EmbeddingIdentity,
    RetrievalResult,
    SearchResponse,
)


def make_chunk(content: str = "hello world", start: int = 0, **kwargs: object) -> Chunk:
    return Chunk(
        document_id="doc1",
        chunk_index=0,
        content=content,
        start_offset=start,
        end_offset=start + len(content),
        **kwargs,  # type: ignore[arg-type]
    )


class TestDocument:
    def test_auto_generates_id(self) -> None:
        d1 = Document(source="a.md", content="x")
        d2 = Document(source="a.md", content="x")
        assert d1.document_id != d2.document_id

    def test_empty_content_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Document(source="a.md", content="")

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Document(source="a.md", content="x", surprise=True)  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        d = Document(source="a.md", content="x")
        with pytest.raises(ValidationError):
            d.content = "y"


class TestChunk:
    def test_offsets_must_match_content_length(self) -> None:
        with pytest.raises(ValidationError, match="offset span"):
            Chunk(
                document_id="d",
                chunk_index=0,
                content="abc",
                start_offset=0,
                end_offset=5,
            )

    def test_end_must_exceed_start(self) -> None:
        with pytest.raises(ValidationError, match="must be greater"):
            Chunk(
                document_id="d",
                chunk_index=0,
                content="abc",
                start_offset=7,
                end_offset=7,
            )

    def test_valid_offsets_accepted(self) -> None:
        c = make_chunk("abc", start=10)
        assert (c.start_offset, c.end_offset) == (10, 13)

    def test_content_hash_is_deterministic_dedup_key(self) -> None:
        a, b = make_chunk("same text"), make_chunk("same text", start=50)
        assert a.content_hash == b.content_hash
        assert a.content_hash != make_chunk("other text").content_hash

    def test_substring_property_holds_against_parent(self) -> None:
        doc = Document(source="a.md", content="0123456789")
        c = Chunk(
            document_id=doc.document_id,
            chunk_index=0,
            content=doc.content[2:6],
            start_offset=2,
            end_offset=6,
        )
        assert doc.content[c.start_offset : c.end_offset] == c.content


class TestRetrievalResult:
    def make(self, score: float) -> RetrievalResult:
        return RetrievalResult(
            content="text",
            score=score,
            document_id="d",
            chunk_id="c",
            source="a.md",
            start_offset=0,
            end_offset=4,
        )

    def test_negative_score_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self.make(-0.1)

    def test_high_confidence_uses_named_constant(self) -> None:
        assert self.make(HIGH_CONFIDENCE_SCORE).is_high_confidence
        assert not self.make(HIGH_CONFIDENCE_SCORE - 1e-9).is_high_confidence

    def test_citation_resolves_offsets(self) -> None:
        cit = self.make(1.0).citation
        assert isinstance(cit, Citation)
        assert (cit.source, cit.start_offset, cit.end_offset) == ("a.md", 0, 4)


class TestEmbeddingIdentity:
    def make(self, dimensions: int = 768) -> EmbeddingIdentity:
        return EmbeddingIdentity(
            provider="ollama", model_name="nomic-embed-text", dimensions=dimensions
        )

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EmbeddingIdentity(
                provider="ollama",
                model_name="nomic-embed-text",
                dimensions=768,
                surprise=True,  # type: ignore[call-arg]
            )

    def test_frozen(self) -> None:
        identity = self.make()
        with pytest.raises(ValidationError):
            identity.dimensions = 1024

    def test_zero_dimensions_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self.make(dimensions=0)

    def test_negative_dimensions_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self.make(dimensions=-1)


class TestSearchResponse:
    def test_defaults(self) -> None:
        r = SearchResponse(query="q", total_results=0)
        assert r.results == []

    def test_negative_total_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SearchResponse(query="q", total_results=-1)

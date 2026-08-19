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

    def test_inverted_offsets_rejected(self) -> None:
        """An inverted span used to construct cleanly, and the damage was
        silent downstream: ``"hello world"[100:5]`` is ``""``, not an error, so
        a caller slicing source text with these offsets reads an empty
        verification as a successful one."""
        with pytest.raises(ValidationError, match="must be greater"):
            RetrievalResult(
                content="text",
                score=1.0,
                document_id="d",
                chunk_id="c",
                source="a.md",
                start_offset=100,
                end_offset=5,
            )

    def test_equal_offsets_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must be greater"):
            RetrievalResult(
                content="text",
                score=1.0,
                document_id="d",
                chunk_id="c",
                source="a.md",
                start_offset=4,
                end_offset=4,
            )

    def test_content_longer_than_span_is_allowed(self) -> None:
        """The deliberate asymmetry with ``Chunk``: a result's offsets address
        the source-document span while its content may be a rendering of that
        span. ``ContextAssembler`` wraps content in a provenance envelope and
        keeps the original offsets, so enforcing the length arithmetic here
        would reject a shipped path. Pinned so a future "make it match
        ``Chunk``" tightening has to confront the reason."""
        result = RetrievalResult(
            content="<retrieved_context>text</retrieved_context>",
            score=1.0,
            document_id="d",
            chunk_id="c",
            source="a.md",
            start_offset=0,
            end_offset=4,
        )
        assert len(result.content) != result.end_offset - result.start_offset


class TestCitationOffsets:
    def make(self, start: int, end: int) -> Citation:
        return Citation(
            document_id="d",
            chunk_id="c",
            source="a.md",
            start_offset=start,
            end_offset=end,
        )

    def test_inverted_offsets_rejected(self) -> None:
        """``Citation`` is the model a resolver slices source text with, so an
        inverted span here is the one that reaches disk."""
        with pytest.raises(ValidationError, match="must be greater"):
            self.make(100, 5)

    def test_equal_offsets_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must be greater"):
            self.make(4, 4)

    def test_valid_span_accepted(self) -> None:
        cit = self.make(10, 20)
        assert (cit.start_offset, cit.end_offset) == (10, 20)


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


class TestMetadataIsolation:
    """GK-017: ``frozen=True`` blocks rebinding ``model.metadata`` but never
    touched nested mutable values -- a caller mutating a list or dict they
    passed in after construction used to reach inside an
    already-constructed, "frozen" model, because a plain ``dict[str, Any]``
    field let pydantic pass the caller's original object straight through.

    One test per model carrying a ``metadata`` field: each attaches its own
    ``field_validator`` independently (``contracts.py``'s
    ``_isolated_json_safe_metadata`` is shared, but the decorator wiring is
    not), so a mistake on any one of the four would not be caught by testing
    only one.
    """

    def test_document_metadata_is_isolated_from_caller_mutation(self) -> None:
        caller_tags = ["a", "b"]
        doc = Document(source="a.md", content="x", metadata={"tags": caller_tags})

        caller_tags.append("MUTATED")

        assert doc.metadata["tags"] == ["a", "b"]

    def test_chunk_metadata_is_isolated_from_caller_mutation(self) -> None:
        caller_nested = {"inner": "original"}
        chunk = make_chunk(metadata={"nested": caller_nested})

        caller_nested["inner"] = "MUTATED"

        assert chunk.metadata["nested"] == {"inner": "original"}

    def test_retrieval_result_metadata_is_isolated_from_caller_mutation(self) -> None:
        caller_tags = ["a", "b"]
        result = RetrievalResult(
            content="text",
            score=1.0,
            document_id="d",
            chunk_id="c",
            source="a.md",
            start_offset=0,
            end_offset=4,
            metadata={"tags": caller_tags},
        )

        caller_tags.append("MUTATED")

        assert result.metadata["tags"] == ["a", "b"]

    def test_search_response_metadata_is_isolated_from_caller_mutation(self) -> None:
        caller_tags = ["a", "b"]
        response = SearchResponse(query="q", total_results=0, metadata={"tags": caller_tags})

        caller_tags.append("MUTATED")

        assert response.metadata["tags"] == ["a", "b"]

    def test_non_json_serializable_metadata_is_rejected_at_construction(self) -> None:
        """Fails closed at construction rather than at the far-away
        persistence boundary (``index/metadata.py`` and ``index/dense.py``
        both store metadata as JSON) -- a set is a real, easy-to-reach
        example of a value that constructs cleanly against ``dict[str,
        Any]`` but has never been persistable."""
        with pytest.raises(ValidationError, match="JSON-serializable"):
            make_chunk(metadata={"oops": {1, 2, 3}})

    @pytest.mark.parametrize("bad_float", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_metadata_float_is_rejected_at_construction(self, bad_float: float) -> None:
        """``json.dumps``'s stdlib default accepts NaN/Infinity and renders
        them as bare (non-JSON) identifiers -- a plain default-``json.dumps``
        check would let this through construction as "JSON-serializable"
        while it is still unpersistable at the real boundary, and
        ``service/mcp_server.py``'s ``model_dump(mode="json")`` silently
        turns it into ``null`` rather than raising, so a caller-visible
        number would become caller-visible ``None`` with nothing failing
        anywhere. Mirrors ``retrieval/rerank.py``'s existing
        ``math.isfinite`` guard on ``score``, extended to metadata values.
        """
        with pytest.raises(ValidationError, match="JSON-serializable"):
            make_chunk(metadata={"score": bad_float})

    def test_deeply_nested_metadata_is_rejected_as_validation_error_not_recursion_error(
        self,
    ) -> None:
        """A ``RecursionError`` from ``json.dumps`` on an extremely deep
        value must surface as the same ``ValidationError`` every other
        malformed ``metadata`` value gets, not escape past pydantic's
        error-wrapping (which only auto-wraps ``ValueError``, not
        ``RecursionError``) and past every typed-error boundary a caller of
        this constructor has (``ingestion/chunking.py`` catches only
        ``pydantic.ValidationError`` around its own ``Chunk(...)`` call).
        """
        deep: dict[str, object] = {}
        cursor = deep
        for _ in range(3000):
            cursor["n"] = {}
            cursor = cursor["n"]  # type: ignore[assignment]

        with pytest.raises(ValidationError, match="JSON-serializable"):
            make_chunk(metadata=deep)

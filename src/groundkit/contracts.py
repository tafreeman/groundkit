"""Core data model: Document, Chunk, RetrievalResult, Citation, SearchResponse.

Ported from ARP's ``agentic_v2/rag/contracts.py`` per ADR-0001, with two
deliberate changes: chunks carry **character offsets** into their source
document (the substrate citation resolution depends on — SPEC.md §5.2), and
the high-confidence threshold is a named constant instead of an inline
literal.

All models are frozen and reject unknown fields. Functions that transform
these models return new objects; nothing here is mutated.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

#: Score at or above which a retrieval result is considered high-confidence.
#: Producers must normalize scores to be >= 0 (see RetrievalResult.score).
HIGH_CONFIDENCE_SCORE: float = 0.7


class Document(BaseModel):
    """A source document before chunking.

    Attributes:
        document_id: Unique identifier (auto-generated UUID if omitted).
        source: File path, URL, or other source identifier.
        content: Raw text content of the document.
        metadata: Arbitrary key-value metadata (author, date, etc.).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    source: str
    content: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A chunk of text extracted from a document, addressable by offsets.

    ``content`` must be the exact substring
    ``document.content[start_offset:end_offset]`` of its parent document —
    chunker tests enforce the substring property; this model enforces the
    length arithmetic so a drifting chunker cannot construct a valid Chunk.

    Attributes:
        chunk_id: Unique identifier for this chunk.
        document_id: ID of the parent document.
        chunk_index: Position of this chunk within the document.
        content: The chunk text.
        start_offset: Character offset of ``content`` in the parent document.
        end_offset: One past the last character of ``content`` in the parent.
        metadata: Inherited + chunk-specific metadata.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    document_id: str
    chunk_index: int = Field(ge=0)
    content: str = Field(min_length=1)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_offsets(self) -> Chunk:
        if self.end_offset <= self.start_offset:
            raise ValueError(
                f"end_offset ({self.end_offset}) must be greater than "
                f"start_offset ({self.start_offset})"
            )
        span = self.end_offset - self.start_offset
        if span != len(self.content):
            raise ValueError(
                f"offset span ({span}) does not match content length ({len(self.content)})"
            )
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def content_hash(self) -> str:
        """SHA-256 hash of the chunk content for deduplication."""
        return hashlib.sha256(self.content.encode()).hexdigest()


class Citation(BaseModel):
    """A verifiable pointer from a retrieved passage back to its source.

    Attributes:
        document_id: ID of the source document.
        chunk_id: ID of the cited chunk.
        source: The document's source identifier (path/URL).
        start_offset: Character offset where the cited span starts.
        end_offset: One past the last character of the cited span.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    chunk_id: str
    source: str
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)


class RetrievalResult(BaseModel):
    """A single retrieval result, always citation-bearing.

    Producers must normalize scores into ``>= 0.0`` before construction —
    feeding raw model logits into this contract is the ARP defect ADR-0001
    hazard 2 records; it is a producer bug, not a contract relaxation.

    Attributes:
        content: The matched chunk text.
        score: Relevance score (>= 0.0, higher is better).
        document_id: ID of the source document.
        chunk_id: ID of the matched chunk.
        source: The document's source identifier (path/URL).
        start_offset: Character offset of the match in its document.
        end_offset: One past the last character of the match.
        metadata: Chunk metadata.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    content: str
    score: float = Field(ge=0.0)
    document_id: str
    chunk_id: str
    source: str
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_high_confidence(self) -> bool:
        """True if score meets :data:`HIGH_CONFIDENCE_SCORE`."""
        return self.score >= HIGH_CONFIDENCE_SCORE

    @computed_field  # type: ignore[prop-decorator]
    @property
    def citation(self) -> Citation:
        """The verifiable source pointer for this result."""
        return Citation(
            document_id=self.document_id,
            chunk_id=self.chunk_id,
            source=self.source,
            start_offset=self.start_offset,
            end_offset=self.end_offset,
        )


class SearchResponse(BaseModel):
    """Response from a search query.

    Attributes:
        query: The original query text.
        results: Ordered list of retrieval results.
        total_results: Total number of results found.
        metadata: Pipeline metadata (latency, stages, etc.).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str
    results: list[RetrievalResult] = Field(default_factory=list)
    total_results: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

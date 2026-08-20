"""Core data model: Document, Chunk, RetrievalResult, Citation, SearchResponse,
EmbeddingIdentity, CollectionManifest.

Ported from ARP's ``agentic_v2/rag/contracts.py`` per ADR-0001, with two
deliberate changes: chunks carry **character offsets** into their source
document (the substrate citation resolution depends on — SPEC.md §5.2), and
the high-confidence threshold is a named constant instead of an inline
literal.

All models are frozen and reject unknown fields. Functions that transform
these models return new objects; nothing here is mutated.
"""

from __future__ import annotations

import copy
import hashlib
import json
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

#: Score at or above which a retrieval result is considered high-confidence.
#: Producers must normalize scores to be >= 0 (see RetrievalResult.score).
HIGH_CONFIDENCE_SCORE: float = 0.7


def _isolated_json_safe_metadata(value: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy ``metadata`` and reject anything that would not survive ``json.dumps``.

    Shared by every model below carrying a ``metadata`` field (GK-017).
    ``frozen=True`` blocks *rebinding* ``model.metadata`` but never touches
    nested mutable values: a plain ``dict[str, Any]`` field lets pydantic
    pass the caller's original list/dict straight through unvalidated, so a
    caller mutating a value they still hold reaches inside an
    already-constructed, "frozen" model. The type is deliberately left
    ``Any`` here rather than narrowed to a recursive JSON-value alias: every
    real caller already passes JSON-safe scalars/containers (verified
    against every ``metadata=`` construction site in this codebase), and
    narrowing the field's public type ripples ``Any`` into a real union
    everywhere a caller *reads* ``metadata`` back out -- a much larger,
    higher-risk change than this bug calls for. ``json.dumps`` is the
    correctness check that matters: ``metadata`` is persisted as JSON
    (``index/metadata.py``, ``index/dense.py``), so failing here means
    failing at construction rather than at that far-away persistence
    boundary, and raising outright (never silently coercing a non-JSON
    value into something that happens to serialize) matches this project's
    fail-closed rule for malformed input.

    ``allow_nan=False``, not the stdlib default: ``json.dumps`` accepts
    ``NaN``/``Infinity`` by default and renders them as bare (non-JSON)
    identifiers, so the plain default-``json.dumps`` check this validator
    used to run would let a ``NaN`` score through construction as
    "JSON-serializable" and it would still be unpersistable at the real
    boundary -- and worse, `service/mcp_server.py`'s
    ``model_dump(mode="json")`` silently turns such a value into ``null``
    rather than raising, so a caller-visible number would become
    caller-visible ``None`` with nothing failing anywhere. This mirrors
    ``retrieval/rerank.py``'s existing ``math.isfinite`` guard on ``score``
    for the same reason, extended to metadata values.

    ``RecursionError`` is caught alongside ``TypeError``/``ValueError``:
    both operations below recurse per nesting level, so a sufficiently deep
    ``metadata`` value overflows the interpreter's call stack rather than
    raising either of those. Pydantic auto-wraps a ``ValueError`` raised
    inside a field validator into a clean ``ValidationError``; it does not
    auto-wrap a bare ``RecursionError``, which would otherwise propagate
    past every typed-error boundary a caller of this constructor has
    (``ingestion/chunking.py`` catches only ``pydantic.ValidationError``
    around its own ``Chunk(...)`` call).

    **Both** operations are guarded, not just ``json.dumps``, because which
    one overflows first is an interpreter-version detail rather than a
    property of the value. ``json.dumps``'s C encoder is bounded by the
    separate C recursion limit CPython 3.12 introduced (~10k frames, not
    ``sys.getrecursionlimit()``), while ``copy.deepcopy`` is pure Python
    costing ~3 Python frames per nesting level. So the same 3000-deep dict
    is rejected by ``json.dumps`` on 3.11 and sails through it on 3.13,
    where ``deepcopy`` is what overflows — which is exactly how an
    unguarded ``deepcopy`` reached CI green on 3.11 and red on 3.13. Only
    ``RecursionError`` is caught on the copy: everything that survives the
    ``json.dumps`` check above is a JSON scalar or container, all of which
    are deep-copyable, so any other exception there is a bug worth seeing
    rather than a malformed input worth reporting.

    Passing the ``json.dumps`` check is necessary, not sufficient, for
    round-trip fidelity: ``copy.deepcopy`` preserves the exact Python type
    of whatever was handed in, so a tuple value survives as a ``tuple``
    here but would come back as a ``list`` after a real trip through
    ``index/metadata.py``'s ``json.dumps``/``json.loads``. No shipped
    caller passes a tuple as of this writing, so this is not fixed
    pre-emptively -- worth revisiting if one starts to.
    """
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(f"metadata must be JSON-serializable: {exc}") from exc
    try:
        return copy.deepcopy(value)
    except RecursionError as exc:
        raise ValueError(
            f"metadata must be JSON-serializable and shallow enough to copy: {exc}"
        ) from exc


#: How a document's ``content`` relates to its source, and therefore how a
#: citation into it is verified (ADR-0016).
#:
#: ``text`` — ``content`` **is** the source file's decoded bytes, so offsets into
#: content are offsets into the file and re-reading reproduces them exactly. This
#: is the only class that holds today, and the property the original
#: ``resolve_citation`` quietly depends on.
#:
#: ``extracted`` — ``content`` is deterministic extractor output (PDF text, HTML
#: with markup stripped). Re-reading the file does not reproduce the offsets;
#: re-running the *same* extractor does, which is why an ``extracted`` document
#: also records the extractor identity it was produced with.
#:
#: ``snapshot`` — ``content`` came from a remote resource whose bytes were stored
#: locally at ingest. Verification resolves against that snapshot, never against
#: a re-fetch: a re-fetch is a different observation at a different time, so a
#: mismatch could not distinguish a stale index from a changed server.
SourceClass = Literal["text", "extracted", "snapshot"]


class Document(BaseModel):
    """A source document before chunking.

    Attributes:
        document_id: Unique identifier (auto-generated UUID if omitted).
        source: File path, URL, or other source identifier.
        content: Raw text content of the document.
        source_class: How ``content`` relates to ``source``, and therefore how a
            citation into this document is verified (ADR-0016). Defaults to
            ``"text"``, which is the only class a loader produced before
            PDF/HTML/URL support and the one the plain re-read-and-compare
            check assumes.
        extractor: Identity of the extractor that produced ``content``, for
            ``extracted`` documents only. Recorded because two extractor
            versions produce two incompatible *offset* spaces under one name —
            the same argument ADR-0004 makes about two embedding models and one
            semantic space. ``None`` for every other class.
        metadata: Arbitrary key-value metadata (author, date, etc.).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    source: str
    content: str = Field(min_length=1)
    source_class: SourceClass = "text"
    extractor: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata", mode="after")
    @classmethod
    def _isolate_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _isolated_json_safe_metadata(value)

    @model_validator(mode="after")
    def _validate_extractor(self) -> Document:
        """An extractor identity is required for ``extracted`` and meaningless otherwise.

        Both directions are enforced, because both are wrong in a way that
        surfaces late: an ``extracted`` document with no extractor cannot have
        its offsets re-derived at citation time, and a ``text`` document
        carrying one implies a re-extraction step that will never run.
        """
        if self.source_class == "extracted" and not self.extractor:
            raise ValueError(
                "an 'extracted' document must record the extractor identity that "
                "produced its content — without it a citation's offsets cannot be "
                "re-derived and verification would silently compare against the "
                "wrong text (ADR-0016)"
            )
        if self.source_class != "extracted" and self.extractor is not None:
            raise ValueError(
                f"extractor is only meaningful for an 'extracted' document, "
                f"got source_class={self.source_class!r}"
            )
        return self


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

    @field_validator("metadata", mode="after")
    @classmethod
    def _isolate_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _isolated_json_safe_metadata(value)

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
        source_class: How to resolve ``source`` back to text (ADR-0016).
            Carried on the citation rather than looked up at resolution time so
            a resolver cannot verify a document under a different class's
            assumptions than the one it was ingested under.
        extractor: Extractor identity for an ``extracted`` source; a resolver
            must refuse rather than slice when its own differs.
        start_offset: Character offset where the cited span starts.
        end_offset: One past the last character of the cited span.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    chunk_id: str
    source: str
    source_class: SourceClass = "text"
    extractor: str | None = None
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)

    @model_validator(mode="after")
    def _validate_offsets(self) -> Citation:
        """Reject an inverted or empty span.

        ``Chunk`` has enforced this since Phase 1; the two models that carry
        the same pair of fields did not, so ``Citation(start_offset=100,
        end_offset=5)`` constructed cleanly and a resolver's
        ``source_text[100:5]`` yielded ``""`` — an empty verification that is
        indistinguishable from a successful one at the call site.

        Only the ordering half of :meth:`Chunk._validate_offsets` applies here:
        a citation is a *pointer*, carrying no content whose length the span
        could be checked against.
        """
        if self.end_offset <= self.start_offset:
            raise ValueError(
                f"end_offset ({self.end_offset}) must be greater than "
                f"start_offset ({self.start_offset})"
            )
        return self


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
    source_class: SourceClass = "text"
    extractor: str | None = None
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata", mode="after")
    @classmethod
    def _isolate_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _isolated_json_safe_metadata(value)

    @model_validator(mode="after")
    def _validate_offsets(self) -> RetrievalResult:
        """Reject an inverted or empty span.

        ``retrieval/rerank.py`` rebuilds every reranked result through this
        constructor specifically so the contract is re-checked; before this
        validator existed, what it re-checked did not include the one
        invariant that makes the offsets usable.

        **The length half of :meth:`Chunk._validate_offsets` is deliberately
        not enforced here.** On a ``Chunk``, ``content`` *is* the span, so
        ``end_offset - start_offset == len(content)`` is definitional. On a
        result, the offsets address the span in the *source document* while
        ``content`` is free to be a rendering of it:
        :class:`~groundkit.providers.context_assembly.ContextAssembler`
        constructs results whose content is the chunk wrapped in a provenance
        envelope, necessarily longer than the span, and asserting equality
        here would reject that shipped, correct path. The arithmetic belongs
        where content and span are the same object, and that is ``Chunk``.
        """
        if self.end_offset <= self.start_offset:
            raise ValueError(
                f"end_offset ({self.end_offset}) must be greater than "
                f"start_offset ({self.start_offset})"
            )
        return self

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
            source_class=self.source_class,
            extractor=self.extractor,
            start_offset=self.start_offset,
            end_offset=self.end_offset,
        )


class EmbeddingIdentity(BaseModel):
    """The embedding identity triple a collection is bound to (ADR-0004 decision 2).

    The *input* to a manifest write or verify, and the exact set of fields
    ADR-0004 defines identity as — no more. Distinct from
    :class:`CollectionManifest`, which is what comes back *out* of the store
    and additionally carries the ``created_at`` stamp the store assigns.

    Deliberately not an ``EmbeddingConfig``: that model also carries
    operational settings (``base_url``, ``batch_size``, ``timeout_seconds``)
    that have nothing to do with which semantic space a collection lives in,
    and its ``provider`` is a closed ``Literal`` over the providers this repo
    ships, which a third-party
    :class:`~groundkit.providers.protocols.EmbeddingProtocol` implementation
    could not satisfy. Taking the narrow triple lets the identity be read
    straight off the embedder that actually produces the vectors.

    Attributes:
        provider: Embedding provider identity (e.g. ``"ollama"``).
        model_name: Model identifier used to embed.
        dimensions: Embedding vector width.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str
    model_name: str
    dimensions: int = Field(gt=0)


class CollectionManifest(BaseModel):
    """The persisted embedding identity a collection was built with (ADR-0004).

    Identity is the triple ``(provider, model_name, dimensions)``, checked
    as a triple — never dimensions alone. Distinct embedding models can
    share a vector width (``nomic-embed-text`` and ``all-mpnet-base-v2`` are
    both 768-dimensional), so a width-only check would admit exactly the
    model swap this manifest exists to reject. Written once, by
    ``SQLiteMetadataStore.write_manifest`` on a collection's first dense
    write, and immutable for the collection's lifetime thereafter.

    Attributes:
        provider: Embedding provider the collection was built with.
        model_name: Model identifier the collection was built with.
        dimensions: Embedding vector width the collection was built with.
        created_at: ISO-8601 UTC timestamp of the manifest's creation.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str
    model_name: str
    dimensions: int = Field(gt=0)
    created_at: str


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

    @field_validator("metadata", mode="after")
    @classmethod
    def _isolate_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _isolated_json_safe_metadata(value)

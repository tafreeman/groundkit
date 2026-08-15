"""Request and response models for the read-only service surface (ADR-0014).

Two rules govern this module, and both are load-bearing rather than stylistic.

**Reuse ``contracts.py`` wherever it already models the domain.** ``search``
returns the existing :class:`~groundkit.contracts.SearchResponse` unchanged, so
a client parsing ``grk search --json`` and one parsing the REST route parse the
same shape. Exactly two response wrappers are new, each because no contract
covers what it carries — see their docstrings, which record *why* rather than
asserting that it was necessary.

**No request model may carry provider or filesystem configuration.** There is
no ``base_url``, ``index_dir``, ``base_dir``, or ``embed_*`` field anywhere in
this module, and there must never be one: the service resolves all of that at
serve time, so a caller cannot reach it (ADR-0014 decision 6). Every model here
is frozen with ``extra="forbid"``, so an unknown field is a rejection rather
than a silently ignored one — the difference between a caller learning their
request was misunderstood and a caller believing a setting took effect.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from groundkit.contracts import Citation, CollectionManifest
from groundkit.retrieval.search import MAX_QUERY_LEN, MAX_TOP_K, SearchMode

#: Outcome of re-reading a cited span from its source file.
#:
#: ``verified`` — the span read back from disk equals the indexed chunk.
#: ``drifted`` — the source resolved but no longer matches, or is now shorter
#: than the cited offsets.
#: ``unresolvable`` — the source escapes the containment root, cannot be read,
#: or is not valid UTF-8.
VerificationVerdict = Literal["verified", "drifted", "unresolvable"]


class SearchRequest(BaseModel):
    """A search over one collection.

    ``query`` and ``top_k`` carry real bounds rather than being validated in
    the handler, and that placement is what makes ADR-0014's error mapping
    correct. ``Retriever.search`` raises ``RetrievalError`` both for a caller
    error (empty query, out-of-range ``top_k``) and for a server-side index
    inconsistency, and the type does not distinguish them. Bounding here means
    the caller-error cases are rejected by schema validation before reaching
    the retriever, so the only ``RetrievalError`` that can surface from a
    handler is the inconsistency one — which is why it maps to a conflict
    rather than a bad request. Relax these bounds and that mapping silently
    becomes wrong.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1, max_length=MAX_QUERY_LEN)
    collection: str = "default"
    top_k: int | None = Field(default=None, ge=1, le=MAX_TOP_K)
    mode: SearchMode = "bm25"
    rerank: bool = False


class FetchChunkRequest(BaseModel):
    """Fetch one chunk and re-verify its citation against the source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(min_length=1)
    collection: str = "default"


class IndexStatusRequest(BaseModel):
    """Report one collection's size and embedding identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    collection: str = "default"


class ListCollectionsRequest(BaseModel):
    """No parameters.

    A model rather than ``None`` so every entry in the tool registry has a
    ``request_model`` to generate an MCP input schema from and to parse a REST
    body with. A uniform registry is what lets the parity tests compare sets
    instead of special-casing one operation, and ``extra="forbid"`` still
    means a caller who sends a field learns it was not understood.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class ChunkFetchResponse(BaseModel):
    """A chunk, its citation, and whether that citation still holds.

    New rather than reused, for a reason specific to each candidate.
    :class:`~groundkit.contracts.RetrievalResult` requires ``score`` with
    ``ge=0.0``; a fetch has no score, and supplying ``0.0`` would invent a
    number SPEC.md §2 forbids *and* drive ``is_high_confidence`` off a
    fabricated value. :class:`~groundkit.contracts.Chunk` carries no ``source``
    — only ``Document`` does (ADR-0006) — so it cannot express a citation. The
    real :class:`~groundkit.contracts.Citation` is therefore nested rather than
    restated field by field.

    ``content`` is the text read **from the source**, not the indexed copy, and
    is ``None`` unless the verdict is ``verified``. That is the point of the
    operation: ``search`` returns indexed text at speed, and this is the step a
    client performs on a hit it intends to quote. A non-``verified`` response
    is still HTTP 200 — a drifted source is an ordinary state of a live local
    corpus, and the verdict is the useful part of the answer — but it carries
    no text a caller could mistakenly attribute to the source.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str
    document_id: str
    chunk_index: int = Field(ge=0)
    citation: Citation
    verification: VerificationVerdict
    content: str | None = None
    detail: str | None = None


class IndexStatusResponse(BaseModel):
    """Collection size and embedding identity — never content or layout.

    Nests the real :class:`~groundkit.contracts.CollectionManifest` rather than
    projecting it: ``contracts.py`` already documents that model as identity
    minus operational settings, which is exactly the leak-safe subset a status
    surface needs, so reusing it means the safe subset is a contract instead of
    a hand-picked list that could drift.

    Deliberately absent: document sources, chunk content, queries, the index
    directory, the containment root, ``base_url``, and ``api_key_env``. SPEC.md
    §7 records that SQLite here is content-bearing data; a status endpoint that
    enumerated sources would be disclosing corpus layout to an unauthenticated
    reader.

    ``generation`` is ``None`` when the collection predates ADR-0013's marker,
    which is also when ``cache_enabled`` is ``False`` — surfaced so a degraded
    collection is visible directly rather than inferred from latency.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    collection: str
    document_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    embedding: CollectionManifest | None = None
    dense_search_available: bool = False
    generation: int | None = None
    cache_enabled: bool = True
    schema_version: int = Field(ge=0)

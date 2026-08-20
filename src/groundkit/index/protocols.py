"""Structural seams for the persisted index (adapted from ARP per ADR-0001).

``VectorStoreProtocol`` keeps ARP's shape minus the ``**kwargs`` catch-all
that silently absorbed misspelled filter arguments (ADR-0001 hazard 3): the
signature is exact, so a wrong keyword is a ``TypeError`` at the call site,
and ``metadata_filter`` must actually filter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from groundkit.contracts import SourceClass

if TYPE_CHECKING:
    from groundkit.contracts import Chunk, CollectionManifest, EmbeddingIdentity


class DocumentRecord(BaseModel):
    """A stored document's provenance, projected for the citation join (ADR-0016).

    Read-only: a view over a ``documents`` row, never a second place that fact
    is asserted from. ``source_class``/``extractor`` default exactly as
    :class:`~groundkit.contracts.Document`'s own fields do, so a document a
    store can only describe at the ``text`` default (see
    :class:`DocumentRecordStoreProtocol`, below, for when that happens) reads
    back as exactly what a plain ``text`` ingest would have produced — never
    a fabricated richer answer.

    Defined here, beside :class:`DocumentRecordStoreProtocol`, rather than in
    ``contracts.py`` next to :data:`~groundkit.contracts.SourceClass` — its
    most natural home, alongside the analogous
    :class:`~groundkit.contracts.CollectionManifest` and
    :class:`~groundkit.contracts.EmbeddingIdentity` view types — because this
    change's scope is ``index/*.py`` and ``retrieval/search.py``. A later
    change with ``contracts.py`` in scope is free to relocate it.

    Attributes:
        source: The document's source identifier (path/URL).
        source_class: How ``source`` maps to stored content (ADR-0016).
        extractor: Extractor identity for an ``extracted`` source; ``None``
            for every other class.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str
    source_class: SourceClass = "text"
    extractor: str | None = None


@runtime_checkable
class MetadataStoreProtocol(Protocol):
    """Durable store for documents, chunks, and ingest state."""

    async def upsert_document(
        self,
        source: str,
        document_id: str,
        content_hash: str,
        *,
        source_class: SourceClass = "text",
        extractor: str | None = None,
    ) -> None:
        """Record (or replace) a document's ingest state.

        ``source_class``/``extractor`` record how ``source`` maps to stored
        content (ADR-0016) — keyword-only and defaulted so every caller that
        predates this pair keeps compiling, and keeps meaning exactly what it
        meant before, unchanged.
        """
        ...

    async def get_document_hash(self, source: str) -> str | None:
        """Return the stored content hash for ``source``, or None if unseen."""
        ...

    async def get_document_id(self, source: str) -> str | None:
        """Return the stored document ID for ``source``, or None if unseen."""
        ...

    async def get_document_sources(self) -> dict[str, str]:
        """Return a ``document_id -> source`` map for every stored document."""
        ...

    async def add_chunks(self, chunks: list[Chunk], source: str) -> None:
        """Persist chunks for a document."""
        ...

    async def replace_document(
        self,
        source: str,
        document_id: str,
        content_hash: str,
        chunks: list[Chunk],
        *,
        source_class: SourceClass = "text",
        extractor: str | None = None,
    ) -> None:
        """Atomically replace a document's row and its chunks in one transaction.

        The durable write path. An implementation must commit the document
        row and its chunks together or not at all: the two-call
        ``upsert_document`` + ``add_chunks`` sequence leaves a document row
        carrying a fresh content hash with zero chunks if it is interrupted
        between the calls, and incremental re-index then skips that document
        forever on hash match.

        ``source_class``/``extractor`` record how ``source`` maps to stored
        content (ADR-0016) — keyword-only and defaulted so every caller that
        predates this pair keeps compiling unchanged.
        """
        ...

    async def get_chunks(self) -> list[Chunk]:
        """Return all persisted chunks in the collection."""
        ...

    async def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Return one chunk by ID, or None."""
        ...

    async def delete_document(self, document_id: str) -> int:
        """Delete a document and its chunks. Returns deleted-chunk count."""
        ...

    async def write_manifest(self, identity: EmbeddingIdentity) -> None:
        """Write the collection's embedding-identity manifest, once (ADR-0004).

        Called on a collection's first dense write. Immutable thereafter: a
        later call with the same ``(provider, model_name, dimensions)``
        triple as the stored manifest is a no-op (re-ingesting into an
        already-bound collection must keep working); a call with a
        different triple is refused, never silently overwritten.

        Raises:
            IndexIdentityError: The store predates the embedding-identity
                manifest and cannot be used for dense work, or a manifest
                already exists with a different identity triple.
        """
        ...

    async def verify_manifest(self, identity: EmbeddingIdentity) -> CollectionManifest | None:
        """Verify ``identity`` matches the collection's stored identity manifest.

        A collection with no manifest yet (no dense write has ever
        happened) has nothing to conflict with, so verification passes
        trivially. Never a re-embed, never a fallback, never a
        warn-and-continue: a real mismatch always raises.

        Returns:
            The manifest that was verified against, or ``None`` when the
            collection has none. Returned rather than left to a follow-up
            :meth:`get_manifest` call because a caller that needs *both*
            "does this identity match" and "is this collection dense-bound
            at all" must get both answers from one read: two reads admit a
            state change in between, and a caller deciding "dense-bound"
            from a later read than the one it checked identity against
            would accept a collection bound to a different embedding space
            (see ``Retriever.open``).

        Raises:
            IndexIdentityError: The store predates the embedding-identity
                manifest and cannot be used for dense work, or the stored
                manifest's identity triple does not match ``identity``.
        """
        ...

    async def get_manifest(self) -> CollectionManifest | None:
        """Return the collection's embedding-identity manifest, or None if unset."""
        ...

    async def get_generation(self) -> int | None:
        """Return the collection's staleness marker, or ``None`` if unanswerable.

        The marker advances on every commit that changes durable state
        (ADR-0013). The only operation defined on it is **equality against a
        previously observed value** — a caller may ask "is this still the
        state I built against", never how far apart two values are.

        ``None`` means *freshness cannot be asserted*, never *unchanged*. A
        caller that receives it must assume the collection changed and
        rebuild whatever it derived from the store. Treating ``None`` as
        "unchanged" is the silent-staleness failure this member exists to
        prevent, which is why the two are different values rather than a
        default.
        """
        ...


@runtime_checkable
class DocumentRecordStoreProtocol(Protocol):
    """Optional capability: a store that answers keyed and aggregate questions directly.

    Two things at once, and deliberately one protocol rather than two. It
    reports each document's full ADR-0016 provenance (``get_document_record``,
    ``get_document_records``), and it answers "how many" without reading rows
    (``count_documents``, ``count_chunks``). What binds them is the property a
    caller actually selects on: **a store that can answer a keyed or aggregate
    question without materializing a table**. Splitting them would make a
    caller test two capabilities to establish one fact, and no store in this
    repo — or plausibly outside it — implements one half without the other,
    since both are the same single SQL statement against the same two tables.

    Deliberately **not** a member of :class:`MetadataStoreProtocol`. That
    protocol is held to two structural guards from outside this module —
    ``tests/test_protocol_conformance.py``'s exact signature-parity check and
    ``tests/test_metadata_store.py``'s mutating/read-only completeness
    classification — and hand-built protocol-conforming test doubles
    elsewhere in the suite implement only the pre-existing member set.
    Folding this in as a required member of ``MetadataStoreProtocol`` would
    fail every one of them: the ``isinstance`` check some are asserted
    against (structural Protocol conformance requires every declared member
    to be present), and a real call from
    :class:`~groundkit.retrieval.search.Retriever` would raise a bare
    ``AttributeError`` on the rest rather than the typed refusal SPEC.md §2
    asks for.

    That reasoning stands, but it is no longer load-bearing on the test side:
    ``tests/metadata_store_doubles.py`` now supplies the one shared base every
    hand-built double derives from (GK-019), so widening a protocol costs one
    edit rather than one per double. Keeping this capability separate is now a
    statement about stores, not about test maintenance — a store either can
    answer these cheaply or it cannot, and a caller is entitled to know which
    before choosing a query plan.

    A store either implements this narrower, separate capability
    (:class:`~groundkit.index.metadata.SQLiteMetadataStore` does) or it does
    not, and ``Retriever`` degrades to ``text``/``None`` defaults over a
    single full-table read for one that does not (see
    ``Retriever``'s ``_DocumentRecordLookup``) — which is honest rather than a
    silent downgrade: a store with no way to report richer provenance never
    had it to report in the first place. That is different from the actual
    ADR-0016 defect this capability closes, where a real
    ``SQLiteMetadataStore`` *did* have the value (it was on the ingested
    ``Document``) and dropped it on write.
    """

    async def get_document_record(self, document_id: str) -> DocumentRecord | None:
        """Return one document's :class:`DocumentRecord`, or ``None`` if unknown.

        The keyed form of :meth:`get_document_records`, and the one every
        read path on a query's critical path should use (GK-019):
        :meth:`~groundkit.retrieval.search.Retriever.search` and
        ``service.tools.handle_fetch_chunk`` each need at most ``top_k``
        document IDs, and answered them by materializing the whole
        ``documents`` table — one validated model per stored row — to look
        up a handful of keys.

        ``None`` means *no such document*, and the callers that fail closed
        on a dangling chunk depend on that being distinguishable from a
        stored document whose provenance happens to be at its defaults. An
        implementation must never substitute a fabricated record for a
        missing row.

        This read is **live**, never cached at open: the whole reason a
        retriever re-reads it per search rather than snapshotting it is that
        a document deleted after ``open()`` must fail closed rather than
        resolve against a stale row.
        """
        ...

    async def get_document_records(self) -> dict[str, DocumentRecord]:
        """Return ``{document_id: DocumentRecord}`` for every stored document.

        Unlike :meth:`MetadataStoreProtocol.get_document_sources`, this
        reports each document's full ADR-0016 provenance, not just its
        source string — the read half of the join
        :meth:`MetadataStoreProtocol.replace_document` writes.

        Whole-table by definition, so it belongs to callers that genuinely
        want every row (a diagnostic, an export, a fallback over a store
        without the keyed form). A caller holding a bounded set of document
        IDs wants :meth:`get_document_record` instead.
        """
        ...

    async def count_documents(self) -> int:
        """Return how many documents are stored, without reading their rows."""
        ...

    async def count_chunks(self) -> int:
        """Return how many chunks are stored, without reading any of them.

        The one ``index_status`` reports. Answering it as
        ``len(await get_chunks())`` pulls the entire corpus text into memory
        as re-validated :class:`~groundkit.contracts.Chunk` models to produce
        one integer, on the cheapest-*looking* call of a read-only service
        surface.
        """
        ...


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Dense vector store.

    Implemented by :class:`~groundkit.index.dense.InMemoryVectorStore` (the
    dev/test double; no ``lancedb`` import, safe for a BM25-only install) and
    :class:`~groundkit.index.dense.LanceDBVectorStore` (the persisted backend,
    constructed via its ``open`` classmethod).
    """

    async def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Store chunks with their embeddings.

        Raises:
            StorageError: On length/dimension mismatch or backend failure.
        """
        ...

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Return the most similar chunks as ``(chunk, score)`` pairs.

        ``metadata_filter`` keeps only chunks whose metadata contains all
        specified key/value pairs — it is never accepted-and-ignored.

        Returns ``(Chunk, score)`` rather than ``RetrievalResult`` for the
        same reason ``BM25Index.search`` does: only ``Document`` carries a
        source path, so a store that holds chunks cannot construct a
        verifiable citation on its own. Joining chunks to their document
        source is ``retrieval.Retriever``'s job, against the metadata store
        that owns that mapping (ADR-0002). A vector store that built
        ``RetrievalResult`` itself would have to be handed the source
        alongside each chunk, duplicating document-level truth into every
        chunk row — a second copy that drifts the moment a document is
        re-ingested from a new path. Scores are ``>= 0.0``; ordering is
        descending with a deterministic tie-break.
        """
        ...

    async def delete(self, document_id: str) -> int:
        """Delete all vectors for a document. Returns deleted count."""
        ...


@runtime_checkable
class LexicalIndexProtocol(Protocol):
    """In-memory lexical (keyword) index -- ``VectorStoreProtocol``'s
    lexical-side peer, both index-layer seams
    :class:`~groundkit.retrieval.search.Retriever` composes against.

    Implemented by :class:`~groundkit.index.bm25.BM25Index` (GK-016): the
    seams for loaders, chunkers, metadata stores, vector stores, embedders,
    chat and rerankers all existed before this one, which made GK-018 (a
    postings-list index, ADR-0002's named revisit trigger) and GK-020 more
    expensive than they needed to be -- both would otherwise mean editing
    ``Retriever.open`` directly to swap the concrete class.

    Unlike ``VectorStoreProtocol``, whose construction (``open``) is never
    called through the protocol type and is therefore not part of it,
    :meth:`from_store` IS part of this seam on purpose: it is the whole
    reason the seam exists, so a future implementation can be built and
    conformance-checked (``assert_signature_parity``) without this Protocol
    needing to change again. A classmethod factory's return annotation is
    exempt from that check's comparison (see
    ``tests/test_protocol_conformance.py``'s ``_assert_member_parity``): the
    whole point of a factory is that each implementation returns its own
    concrete type, not this Protocol's name.
    """

    @property
    def size(self) -> int:
        """Number of chunks currently indexed."""
        ...

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Incrementally add chunks to the index. Existing chunks are preserved."""
        ...

    def search(self, query: str, *, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Return ``(chunk, score)`` pairs ranked by descending relevance.

        Scores are ``>= 0.0``. An empty query or an empty index returns
        ``[]``.
        """
        ...

    @classmethod
    async def from_store(
        cls, store: MetadataStoreProtocol, *, k1: float = 1.5, b: float = 0.75
    ) -> LexicalIndexProtocol:
        """Rebuild a fresh index from every chunk currently persisted in ``store``."""
        ...

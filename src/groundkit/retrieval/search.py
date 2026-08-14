"""BM25 + dense + hybrid search returning citation-bearing results.

Deterministic — no LLM in this path (SPEC.md §2). Phase 1 shipped the BM25
path; Phase 3 Wave C adds the dense read path and RRF hybrid fusion behind
the same surface, with the document-source join happening exactly once, on
the surviving results (ADR-0006). Rerank arrives in Wave D.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Final, Literal

from groundkit.config import RetrievalConfig
from groundkit.contracts import EmbeddingIdentity, RetrievalResult, SearchResponse
from groundkit.errors import ConfigurationError, RetrievalError
from groundkit.index.bm25 import BM25Index
from groundkit.index.dense import verify_dense_side_present
from groundkit.retrieval.fusion import reciprocal_rank_fusion

if TYPE_CHECKING:
    from groundkit.contracts import Chunk
    from groundkit.index.protocols import MetadataStoreProtocol, VectorStoreProtocol
    from groundkit.providers.protocols import EmbeddingProtocol

logger = logging.getLogger(__name__)

#: Upper bound on a caller-supplied top_k (ported from ARP's tool validation).
MAX_TOP_K: int = 50

#: Search modes accepted by :meth:`Retriever.search`.
SearchMode = Literal["bm25", "dense", "hybrid"]

#: How many times the dense snapshot filter may double its fetch before
#: giving up and returning fewer than top_k. Each attempt doubles from k, so
#: the last attempt asks for k * 2**(n-1); the loop also stops early the
#: moment the store returns fewer rows than asked, which means exhausted.
_MAX_SNAPSHOT_FETCH_ATTEMPTS: Final[int] = 5

_NS_PER_MS = 1_000_000


class Retriever:
    """Search a persisted collection, resolving every hit to a citation.

    Construct via :meth:`open`, which rebuilds the in-memory BM25 index from
    the metadata store (ADR-0002) and — when a dense pair is supplied —
    verifies the collection's embedding-identity manifest (ADR-0004
    decision 3) and captures the open()-time document snapshot the dense
    staleness filter is defined over.

    A ``Retriever`` reflects the store's state at ``open()`` time only and is
    never refreshed — it must be reopened after any write to the store
    (re-ingest, edit, deletion) to observe that write. On the BM25 side that
    is structural: the in-memory index is a rebuild, so content ingested
    after ``open()`` has no representation in it at all and silently returns
    zero results, while a hit against modified or deleted content fails
    closed (:class:`~groundkit.errors.RetrievalError`, per the index
    inconsistency check in :meth:`search`). The dense side reads a *live*
    vector-store handle, so the same snapshot semantics are enforced by
    filtering instead of by rebuild: dense hits are restricted to documents
    that existed at ``open()`` time, making content ingested afterwards
    silently invisible on both paths in exactly the same way. Both paths
    fail closed identically on a hit whose document has no stored source —
    on the dense side that includes orphaned vectors, i.e. rows whose
    document a BM25-only ``Indexer`` deleted from SQLite without a vector
    store to delete from, or the residue of an interrupted ingest
    (``KNOWN_LIMITATIONS.md``). Stale *content* cannot be resolved through
    the dense path at all: a dense-enabled indexer deletes a replaced
    document's vectors in lockstep (ADR-0004 decision 6), and the fail-closed
    join covers the case where it did not. See ADR-0002's Consequences
    section for the full staleness discussion.
    """

    def __init__(
        self,
        store: MetadataStoreProtocol,
        bm25: BM25Index,
        config: RetrievalConfig | None = None,
        *,
        embedder: EmbeddingProtocol | None = None,
        vector_store: VectorStoreProtocol | None = None,
        documents_at_open: frozenset[str] | None = None,
    ) -> None:
        _validate_dense_pair(embedder, vector_store)
        if embedder is not None and documents_at_open is None:
            raise ConfigurationError(
                "A dense-paired Retriever requires documents_at_open — the open()-time "
                "document snapshot its staleness filter is defined over. Construct via "
                "Retriever.open(), which captures it."
            )
        self._store = store
        self._bm25 = bm25
        self._config = config or RetrievalConfig()
        self._embedder = embedder
        self._vector_store = vector_store
        self._documents_at_open: frozenset[str] = (
            documents_at_open if documents_at_open is not None else frozenset()
        )

    @classmethod
    async def open(
        cls,
        store: MetadataStoreProtocol,
        config: RetrievalConfig | None = None,
        *,
        embedder: EmbeddingProtocol | None = None,
        vector_store: VectorStoreProtocol | None = None,
    ) -> Retriever:
        """Build a retriever over ``store``, rebuilding BM25 from persisted chunks.

        With a dense pair supplied, the collection's embedding-identity
        manifest is verified *first* — before the O(corpus) BM25 rebuild does
        any work — closing the second of ADR-0004 decision 3's two
        boundaries (``Indexer`` closed the ingest boundary in Wave B). A
        mismatch is :class:`~groundkit.errors.IndexIdentityError`, never a
        re-embed and never a fallback. A collection with no manifest (never
        dense-ingested) verifies trivially and simply has nothing for the
        dense path to read.

        Args:
            store: The collection's metadata store.
            config: Retrieval settings (defaults if omitted).
            embedder: Optional embedding provider for the dense read path
                (keyword-only; both or neither with ``vector_store``). Used
                to embed queries; its ``(provider, model_name, dimensions)``
                triple is verified against the collection manifest.
            vector_store: Optional dense vector store (keyword-only).

        Returns:
            A ready-to-search :class:`Retriever`.

        Raises:
            ConfigurationError: Exactly one of ``embedder`` /
                ``vector_store`` was supplied.
            IndexIdentityError: The collection is bound to a different
                embedding identity (ADR-0004).
            StorageError: The collection is manifest-bound and holds
                documents, but its vector store is empty — a lost dense
                side, refused here rather than silently answering every
                dense and hybrid query from an empty index. See
                :func:`~groundkit.index.dense.verify_dense_side_present`.
        """
        cfg = config or RetrievalConfig()
        _validate_dense_pair(embedder, vector_store)
        if embedder is not None:
            await store.verify_manifest(_identity_of(embedder))
        if embedder is not None and vector_store is not None:
            await verify_dense_side_present(store, vector_store, embedder.dimensions)
        bm25 = await BM25Index.from_store(store, k1=cfg.bm25_k1, b=cfg.bm25_b)
        documents_at_open: frozenset[str] | None = None
        if embedder is not None:
            documents_at_open = frozenset(await store.get_document_sources())
        logger.info("Retriever opened over %d chunks", bm25.size)
        return cls(
            store,
            bm25,
            cfg,
            embedder=embedder,
            vector_store=vector_store,
            documents_at_open=documents_at_open,
        )

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        *,
        mode: SearchMode = "bm25",
    ) -> SearchResponse:
        """Search the collection.

        The default mode is ``"bm25"`` deliberately, and stays so under
        ADR-0007 even though Wave E's measured eval delta favoured hybrid on
        every retrieval-quality metric. The reason is abstention: ``"hybrid"``
        applies no ``score_threshold`` (ADR-0005 decision 6 — see below), so
        no configuration makes it return nothing for a question the corpus
        cannot answer, and a default that always answers is the wrong default
        for citation-verifiable retrieval. ``"dense"`` *does* honour
        ``score_threshold``; what it lacks is a measured, defensible value.

        ``score_threshold`` applies to the producer-scored modes (``bm25``,
        ``dense``) and never to hybrid: fused scores are rank-derived
        quantities an absolute threshold would silently reinterpret, so
        ADR-0005 decision 6 excludes them — thresholding the pre-fusion
        candidate lists would reintroduce the same threshold by a side door,
        so those are not thresholded either.

        Args:
            query: Non-empty query text.
            top_k: Result cap for this call; defaults to config. Must be in
                ``1..MAX_TOP_K``.
            mode: ``"bm25"`` (lexical only, the default), ``"dense"``
                (vector only), or ``"hybrid"`` (RRF fusion of both,
                ADR-0005; the response's ``metadata["stage"]`` reports it
                as ``"fusion"``).

        Returns:
            A :class:`SearchResponse` whose results each carry a resolvable
            citation (source + character offsets), with
            ``metadata["stage"]`` naming the stage that produced them
            honestly: ``"bm25"``, ``"dense"``, or ``"fusion"``.

        Raises:
            RetrievalError: Empty/whitespace query, out-of-range ``top_k``,
                or an index inconsistency (a hit whose document has no
                stored source — including orphaned dense vectors — fail
                closed rather than emit an unverifiable citation).
            ConfigurationError: ``mode`` is ``"dense"`` or ``"hybrid"`` but
                this retriever was opened without a dense pair.
        """
        if not query.strip():
            raise RetrievalError("Query must not be empty")
        k = top_k if top_k is not None else self._config.top_k
        if not 1 <= k <= MAX_TOP_K:
            raise RetrievalError(f"top_k must be between 1 and {MAX_TOP_K}, got {k}")

        started = time.perf_counter_ns()
        sources = await self._store.get_document_sources()

        metadata: dict[str, object]
        if mode == "bm25":
            pairs = self._bm25.search(query, top_k=k)
            results = self._resolve(pairs, sources, apply_threshold=True)
            metadata = {"stage": "bm25", "top_k": k}
        elif mode == "dense":
            embedder, vector_store = self._require_dense(mode)
            pairs = await self._dense_candidates(query, k, sources, embedder, vector_store)
            results = self._resolve(pairs, sources, apply_threshold=True)
            metadata = {"stage": "dense", "top_k": k}
        elif mode == "hybrid":
            embedder, vector_store = self._require_dense(mode)
            bm25_pairs = self._bm25.search(query, top_k=k)
            dense_pairs = await self._dense_candidates(query, k, sources, embedder, vector_store)
            fused = reciprocal_rank_fusion(
                [bm25_pairs, dense_pairs], rrf_k=self._config.rrf_k, top_k=k
            )
            results = self._resolve(fused, sources, apply_threshold=False)
            metadata = {"stage": "fusion", "top_k": k, "rrf_k": self._config.rrf_k}
        else:
            # SearchMode is a type hint, not a runtime guard. Falling through
            # to hybrid would answer a typo'd mode with fused results and
            # stamp metadata["stage"] = "fusion" on them — a wrong answer
            # reported as a valid one, which SPEC.md §2's fail-closed rule
            # forbids. An unknown mode is a caller bug; name it.
            raise RetrievalError(
                f"Unknown search mode {mode!r}; expected one of 'bm25', 'dense', 'hybrid'"
            )

        latency_ms = (time.perf_counter_ns() - started) / _NS_PER_MS
        logger.info("Search returned %d results in %.2f ms", len(results), latency_ms)
        return SearchResponse(
            query=query,
            results=results,
            total_results=len(results),
            metadata={**metadata, "latency_ms": latency_ms},
        )

    def _require_dense(self, mode: str) -> tuple[EmbeddingProtocol, VectorStoreProtocol]:
        """Return the dense pair, or fail closed if this retriever has none."""
        if self._embedder is None or self._vector_store is None:
            raise ConfigurationError(
                f"search mode {mode!r} requires a dense path, but this Retriever was "
                "opened without one. Reopen it via Retriever.open(store, config, "
                "embedder=..., vector_store=...) to search dense or hybrid."
            )
        return self._embedder, self._vector_store

    async def _dense_candidates(
        self,
        query: str,
        k: int,
        sources: dict[str, str],
        embedder: EmbeddingProtocol,
        vector_store: VectorStoreProtocol,
    ) -> list[tuple[Chunk, float]]:
        """Embed ``query`` and return snapshot-filtered dense ``(chunk, score)`` pairs.

        The vector-store handle is live, so this filter is what gives the
        dense path the same ``open()``-time snapshot semantics the BM25
        rebuild has structurally (see the class docstring). Three cases per
        hit, decided against the open()-time snapshot and the live
        ``sources`` map:

        - document known at ``open()`` → kept (the join may still fail
          closed later if the document has since been deleted).
        - unknown at ``open()`` but present in ``sources`` → ingested after
          ``open()``; dropped silently, exactly as invisible as it is to the
          stale in-memory BM25 index.
        - in neither → orphaned vectors; fail closed loudly.

        **Filter-then-truncate, by over-fetching.** Asking the store for
        exactly ``k`` and then dropping post-open hits would let content
        ingested after ``open()`` *displace* eligible results: enough new
        chunks ranking above them and a dense or hybrid search returns fewer
        than ``k``, or nothing at all, while perfectly good pre-open chunks
        sat just below the cut. That is not the snapshot semantics claimed
        above — a search over the old corpus would have returned them — and
        it is the same principle ``index/dense.py`` already applies to
        metadata filters. So the fetch widens (doubling, bounded) until
        either ``k`` results survive the filter or the store is exhausted,
        and only then truncates.

        Widening also widens the orphan check's window, which can surface an
        orphan that a ``k``-sized fetch would not have reached. That is the
        intended direction: an orphan is real corruption, and finding it is
        the fail-closed outcome, not a regression.

        Raises:
            RetrievalError: A hit references a document with no stored
                source — orphaned vectors (a document deleted from SQLite
                whose dense rows were left behind by a BM25-only indexer,
                or interrupted-ingest residue; ``KNOWN_LIMITATIONS.md``).
        """
        embedding = (await embedder.embed([query]))[0]
        kept: list[tuple[Chunk, float]] = []
        fetch = k
        for attempt in range(_MAX_SNAPSHOT_FETCH_ATTEMPTS):
            pairs = await vector_store.search(embedding, top_k=fetch)
            kept = self._apply_snapshot_filter(pairs, sources)
            if len(kept) >= k or len(pairs) < fetch:
                # Enough survivors, or the store returned less than asked and
                # therefore holds nothing more to widen into.
                break
            if attempt == _MAX_SNAPSHOT_FETCH_ATTEMPTS - 1:
                # Never silently: a caller seeing < k results is entitled to
                # know the cap truncated the search rather than the corpus.
                logger.warning(
                    "Dense snapshot filter still short of top_k after %d widening "
                    "attempts (fetched %d, kept %d, wanted %d); returning what survived",
                    _MAX_SNAPSHOT_FETCH_ATTEMPTS,
                    fetch,
                    len(kept),
                    k,
                )
                break
            fetch *= 2
        return kept[:k]

    def _apply_snapshot_filter(
        self, pairs: list[tuple[Chunk, float]], sources: dict[str, str]
    ) -> list[tuple[Chunk, float]]:
        """Keep only hits whose document existed at ``open()``; fail closed on orphans.

        Split out of :meth:`_dense_candidates` because the over-fetch loop
        applies it once per widening attempt.
        """
        kept: list[tuple[Chunk, float]] = []
        for chunk, score in pairs:
            if chunk.document_id in self._documents_at_open:
                kept.append((chunk, score))
                continue
            if chunk.document_id in sources:
                continue
            raise RetrievalError(
                f"Index inconsistency: dense hit for chunk {chunk.chunk_id} references "
                f"document {chunk.document_id} which has no stored source — orphaned "
                "vectors; failing closed rather than emitting an unverifiable citation"
            )
        return kept

    def _resolve(
        self,
        pairs: list[tuple[Chunk, float]],
        sources: dict[str, str],
        *,
        apply_threshold: bool,
    ) -> list[RetrievalResult]:
        """Join ``(chunk, score)`` pairs to their documents' sources — the ONE join.

        Every mode funnels through here exactly once per search, after
        fusion in hybrid mode (ADR-0006: the join happens on the surviving
        results, and this is the one place the fail-closed rule lives).
        ``apply_threshold=False`` is the hybrid path: ADR-0005 decision 6
        keeps ``score_threshold`` away from rank-derived fused scores.

        Raises:
            RetrievalError: A chunk's document has no stored source — fail
                closed rather than emit an unverifiable citation.
        """
        threshold = self._config.score_threshold if apply_threshold else None
        results: list[RetrievalResult] = []
        for chunk, score in pairs:
            if threshold is not None and score < threshold:
                continue
            source = sources.get(chunk.document_id)
            if source is None:
                raise RetrievalError(
                    f"Index inconsistency: chunk {chunk.chunk_id} references "
                    f"document {chunk.document_id} which has no stored source"
                )
            results.append(
                RetrievalResult(
                    content=chunk.content,
                    score=score,
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    source=source,
                    start_offset=chunk.start_offset,
                    end_offset=chunk.end_offset,
                    metadata=chunk.metadata,
                )
            )
        return results


def _validate_dense_pair(
    embedder: EmbeddingProtocol | None, vector_store: VectorStoreProtocol | None
) -> None:
    """Reject half a dense pair at construction (mirrors ``Indexer``'s check).

    Raises:
        ConfigurationError: Exactly one of ``embedder`` / ``vector_store``
            was supplied.
    """
    if embedder is not None and vector_store is None:
        raise ConfigurationError(
            "Retriever was given an embedder but no vector_store. The pair is "
            "inseparable: an embedder alone can embed the query, but there is no "
            "dense index to search with it. Pass both or neither."
        )
    if vector_store is not None and embedder is None:
        raise ConfigurationError(
            "Retriever was given a vector_store but no embedder. The pair is "
            "inseparable: without an embedder the query can never be embedded, so "
            "the store could never be searched. Pass both or neither."
        )


def _identity_of(embedder: EmbeddingProtocol) -> EmbeddingIdentity:
    """Read the ADR-0004 identity triple off the embedder itself.

    Mirrors ``indexer.py``'s helper of the same name (deliberately not
    imported from it — retrieval does not depend on the ingest pipeline):
    sourcing all three fields from the object that actually embeds makes
    "the manifest is checked against a different model than the one
    embedding the queries" unrepresentable.
    """
    return EmbeddingIdentity(
        provider=embedder.provider,
        model_name=embedder.model_name,
        dimensions=embedder.dimensions,
    )

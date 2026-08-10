"""BM25-backed search returning citation-bearing results (Phase 1).

Deterministic — no LLM in this path (SPEC.md §2). Hybrid fusion and rerank
arrive in Phase 3 behind the same surface.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from groundkit.config import RetrievalConfig
from groundkit.contracts import RetrievalResult, SearchResponse
from groundkit.errors import RetrievalError
from groundkit.index.bm25 import BM25Index

if TYPE_CHECKING:
    from groundkit.index.protocols import MetadataStoreProtocol

logger = logging.getLogger(__name__)

#: Upper bound on a caller-supplied top_k (ported from ARP's tool validation).
MAX_TOP_K: int = 50

_NS_PER_MS = 1_000_000


class Retriever:
    """Search a persisted collection with BM25, resolving every hit to a citation.

    Construct via :meth:`open`, which rebuilds the in-memory BM25 index from
    the metadata store (ADR-0002).
    """

    def __init__(
        self,
        store: MetadataStoreProtocol,
        bm25: BM25Index,
        config: RetrievalConfig | None = None,
    ) -> None:
        self._store = store
        self._bm25 = bm25
        self._config = config or RetrievalConfig()

    @classmethod
    async def open(
        cls, store: MetadataStoreProtocol, config: RetrievalConfig | None = None
    ) -> Retriever:
        """Build a retriever over ``store``, rebuilding BM25 from persisted chunks.

        Args:
            store: The collection's metadata store.
            config: Retrieval settings (defaults if omitted).

        Returns:
            A ready-to-search :class:`Retriever`.
        """
        cfg = config or RetrievalConfig()
        bm25 = await BM25Index.from_store(store, k1=cfg.bm25_k1, b=cfg.bm25_b)
        logger.info("Retriever opened over %d chunks", bm25.size)
        return cls(store, bm25, cfg)

    async def search(self, query: str, top_k: int | None = None) -> SearchResponse:
        """Search the collection.

        Args:
            query: Non-empty query text.
            top_k: Result cap for this call; defaults to config. Must be in
                ``1..MAX_TOP_K``.

        Returns:
            A :class:`SearchResponse` whose results each carry a resolvable
            citation (source + character offsets).

        Raises:
            RetrievalError: Empty/whitespace query, out-of-range ``top_k``,
                or an index inconsistency (a hit whose document has no
                stored source — fail closed rather than emit an
                unverifiable citation).
        """
        if not query.strip():
            raise RetrievalError("Query must not be empty")
        k = top_k if top_k is not None else self._config.top_k
        if not 1 <= k <= MAX_TOP_K:
            raise RetrievalError(f"top_k must be between 1 and {MAX_TOP_K}, got {k}")

        started = time.perf_counter_ns()
        pairs = self._bm25.search(query, top_k=k)
        sources = await self._store.get_document_sources()

        threshold = self._config.score_threshold
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

        latency_ms = (time.perf_counter_ns() - started) / _NS_PER_MS
        logger.info("Search returned %d results in %.2f ms", len(results), latency_ms)
        return SearchResponse(
            query=query,
            results=results,
            total_results=len(results),
            metadata={"stage": "bm25", "top_k": k, "latency_ms": latency_ms},
        )

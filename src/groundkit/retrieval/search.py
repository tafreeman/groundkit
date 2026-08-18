"""BM25 + dense + hybrid search returning citation-bearing results.

Deterministic — no LLM in this path (SPEC.md §2). Phase 1 shipped the BM25
path; Phase 3 Wave C adds the dense read path and RRF hybrid fusion behind
the same surface, with the document-source join happening exactly once, on
the surviving results (ADR-0006). Rerank arrives in Wave D.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Final, Literal

from groundkit.config import RetrievalConfig
from groundkit.contracts import (
    CollectionManifest,
    RetrievalResult,
    SearchResponse,
)
from groundkit.errors import ConfigurationError, RetrievalError
from groundkit.identity import identity_of, validate_dense_pair
from groundkit.index.bm25 import BM25Index
from groundkit.index.dense import verify_dense_side_present
from groundkit.index.protocols import DocumentRecord, DocumentRecordStoreProtocol
from groundkit.retrieval.fusion import reciprocal_rank_fusion
from groundkit.telemetry import get_tracer, span_attributes

if TYPE_CHECKING:
    from groundkit.contracts import Chunk
    from groundkit.index.protocols import MetadataStoreProtocol, VectorStoreProtocol
    from groundkit.providers.protocols import EmbeddingProtocol
    from groundkit.telemetry import Stage

logger = logging.getLogger(__name__)

#: Tracer for this module's one retrieval span, ``groundkit.retrieve.search``
#: (ADR-0022 decision 5).
tracer = get_tracer()

#: Upper bound on a caller-supplied top_k (ported from ARP's tool validation).
MAX_TOP_K: int = 50

#: Upper bound on a caller-supplied query, in characters. Unlike
#: :data:`MAX_TOP_K` this has no ARP ancestor: the CLI is bounded by whatever a
#: shell will pass, so nothing needed it until Phase 4 put a network surface in
#: front of retrieval (ADR-0014). It lives here, beside the bound it sits next
#: to, because a threshold belongs in one named place rather than inline at each
#: boundary that enforces it (SPEC.md §5.2) — and because the service is not the
#: only future caller that will need it.
#:
#: Not enforced by :meth:`Retriever.search` itself: the retriever's contract is
#: that an empty query is a :class:`~groundkit.errors.RetrievalError`, and
#: widening it to police length would change an existing typed behaviour that
#: ADR-0014's error mapping depends on. Callers at a trust boundary bound the
#: input before it reaches retrieval.
MAX_QUERY_LEN: int = 4096

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
        collection_is_dense_bound: bool = False,
        collection: str | None = None,
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
        self._collection_is_dense_bound = collection_is_dense_bound
        # Telemetry label only (ADR-0022 decision 5), never read back by this
        # class. There is no MetadataStoreProtocol accessor for a collection's
        # own name (see the class docstring's snapshot discussion for the
        # analogous reasoning on documents_at_open) — widening that protocol
        # for an observability nicety is out of proportion, so a caller that
        # knows the name (CollectionRuntime, which opened the store from it)
        # passes it through here instead.
        self._collection = collection

    @classmethod
    async def open(
        cls,
        store: MetadataStoreProtocol,
        config: RetrievalConfig | None = None,
        *,
        embedder: EmbeddingProtocol | None = None,
        vector_store: VectorStoreProtocol | None = None,
        collection: str | None = None,
    ) -> Retriever:
        """Build a retriever over ``store``, rebuilding BM25 from persisted chunks.

        With a dense pair supplied, the collection's embedding-identity
        manifest is verified *first* — before the O(corpus) BM25 rebuild does
        any work — closing the second of ADR-0004 decision 3's two
        boundaries (``Indexer`` closed the ingest boundary in Wave B). A
        mismatch is :class:`~groundkit.errors.IndexIdentityError`, never a
        re-embed and never a fallback. A collection with no manifest (never
        dense-ingested) verifies trivially here — opening is legitimate, and
        a caller may then search ``"bm25"`` — but a ``dense`` or ``hybrid``
        search against it is refused in :meth:`search` (ADR-0008), because
        there are no vectors for either mode to read.

        **That verification is the single source of the dense-bound
        verdict**, which is why ``verify_manifest`` returns the manifest it
        checked instead of this method reading it again afterwards. Deciding
        "is this collection dense-bound" from a *later* read than the one
        the identity was checked against is a TOCTOU hole with a silent
        wrong answer at the end of it: an unbound collection passes the
        identity check trivially, and if a concurrent dense ingest (another
        task, or another process — the store's lock spans neither) binds it
        to a different provider before the later read, that read reports
        "bound" and this retriever proceeds to answer ``dense`` and
        ``hybrid`` queries by matching *this* embedder's query vectors
        against *that* provider's index. Mixed embedding spaces, the exact
        corruption ADR-0004 exists to make unrepresentable, with no error
        anywhere. One read decides both, so the two answers cannot disagree.

        The residual window is deliberately biased closed: a collection
        bound *after* this read is treated as unbound, so ``dense`` and
        ``hybrid`` are refused rather than answered. That is also the
        already-documented snapshot semantics — a retriever never observes
        writes that land after ``open()``, and a dense side that did not
        exist at ``open()`` is no exception.

        Args:
            store: The collection's metadata store.
            config: Retrieval settings (defaults if omitted).
            embedder: Optional embedding provider for the dense read path
                (keyword-only; both or neither with ``vector_store``). Used
                to embed queries; its ``(provider, model_name, dimensions)``
                triple is verified against the collection manifest.
            vector_store: Optional dense vector store (keyword-only).
            collection: Optional collection name (keyword-only), attached
                to every span this retriever's :meth:`search` opens as the
                ``groundkit.collection`` attribute (ADR-0022 decision 5).
                Stored as-is on ``self._collection``; never validated
                against ``store`` here, since the store carries no
                collection-name accessor to validate it against (see
                :meth:`__init__`).

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
        manifest: CollectionManifest | None = None
        if embedder is not None:
            # Verdict and manifest from one read: see the docstring for the
            # mixed-embedding-space race a second read would open.
            manifest = await store.verify_manifest(identity_of(embedder))
        if embedder is not None and vector_store is not None:
            await verify_dense_side_present(
                store, vector_store, embedder.dimensions, manifest=manifest
            )
        bm25 = await BM25Index.from_store(store, k1=cfg.bm25_k1, b=cfg.bm25_b)
        documents_at_open: frozenset[str] | None = None
        if embedder is not None:
            documents_at_open = frozenset(await store.get_document_sources())
        # Recorded, not enforced, here. A collection with no manifest was
        # never dense-ingested, which is a legitimate state to open with a
        # dense pair — the caller may only ever search "bm25". It is
        # searching *dense* or *hybrid* against it that cannot be answered,
        # so the refusal lives in search(), per mode.
        dense_bound = manifest is not None
        logger.info("Retriever opened over %d chunks", bm25.size)
        return cls(
            store,
            bm25,
            cfg,
            embedder=embedder,
            vector_store=vector_store,
            documents_at_open=documents_at_open,
            collection_is_dense_bound=dense_bound,
            collection=collection,
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
                this retriever was opened without a dense pair, or the
                collection has no embedding-identity manifest and therefore
                no vectors either mode could read (ADR-0008). The second
                case is refused rather than answered from an empty dense
                side, which would return lexical results labelled as dense
                or fused.

        Wrapped in one OTel span, ``groundkit.retrieve.search`` (SPEC.md §3,
        ADR-0022 decision 5), covering the whole call — validation, the
        store read, and whichever mode branch runs. Only allowlisted
        attributes ever reach it: collection name, retrieval mode, stage,
        ``top_k``, result count, latency, and — on any failure — the raised
        exception's type name as a typed failure code. **``query`` itself
        is never attached, at any point in this method or anything it
        calls** — SPEC.md §3 and ADR-0022 decision 3 forbid query text on a
        span explicitly, and this is the one property
        ``tests/test_span_instrumentation.py`` exists to pin down with a
        sentinel value.
        """
        with tracer.start_as_current_span("groundkit.retrieve.search") as span:
            try:
                if not query.strip():
                    raise RetrievalError("Query must not be empty")
                k = top_k if top_k is not None else self._config.top_k
                if not 1 <= k <= MAX_TOP_K:
                    raise RetrievalError(f"top_k must be between 1 and {MAX_TOP_K}, got {k}")

                started = time.perf_counter_ns()
                records = await self._document_records()

                metadata: dict[str, object]
                stage: Stage
                if mode == "bm25":
                    pairs = await self._bm25_search(query, k)
                    results = self._resolve(pairs, records, apply_threshold=True)
                    stage = "bm25"
                    metadata = {"stage": stage, "top_k": k}
                elif mode == "dense":
                    embedder, vector_store = self._require_dense(mode)
                    pairs = await self._dense_candidates(query, k, records, embedder, vector_store)
                    results = self._resolve(pairs, records, apply_threshold=True)
                    stage = "dense"
                    metadata = {"stage": stage, "top_k": k}
                elif mode == "hybrid":
                    embedder, vector_store = self._require_dense(mode)
                    bm25_pairs = await self._bm25_search(query, k)
                    dense_pairs = await self._dense_candidates(
                        query, k, records, embedder, vector_store
                    )
                    fused = reciprocal_rank_fusion(
                        [bm25_pairs, dense_pairs], rrf_k=self._config.rrf_k, top_k=k
                    )
                    results = self._resolve(fused, records, apply_threshold=False)
                    stage = "fusion"
                    metadata = {"stage": stage, "top_k": k, "rrf_k": self._config.rrf_k}
                else:
                    # SearchMode is a type hint, not a runtime guard. Falling
                    # through to hybrid would answer a typo'd mode with fused
                    # results and stamp metadata["stage"] = "fusion" on them —
                    # a wrong answer reported as a valid one, which SPEC.md
                    # §2's fail-closed rule forbids. An unknown mode is a
                    # caller bug; name it.
                    raise RetrievalError(
                        f"Unknown search mode {mode!r}; expected one of 'bm25', 'dense', 'hybrid'"
                    )

                latency_ms = (time.perf_counter_ns() - started) / _NS_PER_MS
                logger.info("Search returned %d results in %.2f ms", len(results), latency_ms)
                response = SearchResponse(
                    query=query,
                    results=results,
                    total_results=len(results),
                    metadata={**metadata, "latency_ms": latency_ms},
                )
            except Exception as exc:
                # Telemetry only: this except exists to label the span, never
                # to handle the failure. The bare `raise` below re-raises the
                # exact exception, unwrapped.
                span.set_attributes(
                    span_attributes(collection=self._collection, failure_kind=type(exc).__name__)
                )
                raise
            span.set_attributes(
                span_attributes(
                    collection=self._collection,
                    retrieval_mode=mode,
                    stage=stage,
                    top_k=k,
                    result_count=len(results),
                    duration_ms=latency_ms,
                )
            )
            return response

    def _require_dense(self, mode: str) -> tuple[EmbeddingProtocol, VectorStoreProtocol]:
        """Return the dense pair, or fail closed if this mode cannot be served.

        Two separate ways a dense or hybrid search can be unanswerable, and
        both are caller errors rather than corruption:

        1. **This retriever has no dense pair.** Nothing to embed with.
        2. **This collection was never dense-ingested** — no ADR-0004
           manifest, so no vectors exist for any query to match. Answering
           anyway is the trap this check closes: the dense side contributes
           nothing, RRF over one non-empty list returns the BM25 ordering
           unchanged, and ``metadata["stage"]`` stamps it ``"fusion"``. The
           caller gets lexical results labelled hybrid, with no error — a
           wrong answer presented as a valid one, which SPEC.md §2 forbids
           and which this repo already ruled on once (the unknown-``mode``
           fallthrough closed in review on PR #3).

        Deliberately *not* checked in :meth:`open`: an unbound manifest is a
        legitimate collection state (a BM25-only collection, whose upgrade
        path is the documented no-backfill limitation, not a defect), and a
        caller may open with a dense pair and only ever search ``"bm25"``.
        The mode is what makes it unanswerable, so the mode is where it
        fails.
        """
        if self._embedder is None or self._vector_store is None:
            raise ConfigurationError(
                f"search mode {mode!r} requires a dense path, but this Retriever was "
                "opened without one. Reopen it via Retriever.open(store, config, "
                "embedder=..., vector_store=...) to search dense or hybrid."
            )
        if not self._collection_is_dense_bound:
            # ASCII only: this reaches a live console, and the repo's primary
            # dev platform defaults to a cp1252 code page that mangles
            # non-ASCII punctuation mid-message.
            raise ConfigurationError(
                f"search mode {mode!r} requires a collection that was ingested with a "
                "dense path, but this one has no embedding-identity manifest - no "
                "vectors have ever been written to it. Refusing rather than returning "
                "lexical results labelled as dense or fused. Note that enabling the "
                "dense path over an existing collection does not backfill it (the "
                "incremental skip gate runs before embedding): re-ingest the corpus into a "
                "fresh collection with `grk ingest --dense`."
            )
        return self._embedder, self._vector_store

    async def _bm25_search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        """Score the whole corpus off the event loop.

        :meth:`~groundkit.index.bm25.BM25Index.search` is synchronous and
        CPU-bound: it scores *every* indexed chunk before truncating to
        ``top_k`` (ADR-0002's accepted O(corpus) trade), so its cost tracks
        corpus size rather than ``k``. Called inline from this ``async def``,
        that arithmetic runs on the one event loop ``grk serve`` has — a
        single uvicorn worker — and stalls every other in-flight request for
        its duration, ``index_status`` and ``fetch_chunk`` included.

        Every other CPU- or IO-bound step on this path is already dispatched
        to a worker thread for exactly this reason
        (:mod:`groundkit.index.dense`, ``retrieval/rerank.py``'s
        ``model.predict`` call, and every ``SQLiteMetadataStore`` operation).
        BM25 was the omission, and it is the *default* retrieval mode — the
        one always available with no optional extra installed.

        This moves the stall, not the work: total CPU is unchanged and a
        concurrent caller still contends for the GIL, which pure-Python
        scoring holds for most of its run. What changes is that the loop
        keeps turning, so unrelated requests are served rather than queued
        behind a whole-corpus scan. Recorded as a residual in
        ``KNOWN_LIMITATIONS.md``; the standing fix is a postings list that
        scores only candidate chunks.

        Args:
            query: The raw query string, tokenized by the index itself.
            k: Already-validated ``top_k`` for this search.

        Returns:
            ``(chunk, score)`` pairs exactly as
            :meth:`~groundkit.index.bm25.BM25Index.search` returns them.
        """
        return await asyncio.to_thread(self._bm25.search, query, top_k=k)

    async def _document_records(self) -> dict[str, DocumentRecord]:
        """Read every stored document's provenance for this search's one join.

        Prefers ``get_document_records`` (:class:`DocumentRecordStoreProtocol`)
        when ``self._store`` implements that capability — every
        :class:`~groundkit.index.metadata.SQLiteMetadataStore` does, and this
        is the ADR-0016 read half of the join
        :meth:`~groundkit.index.protocols.MetadataStoreProtocol.replace_document`
        writes — and degrades to plain ``get_document_sources`` otherwise,
        wrapping each bare source string in a :class:`DocumentRecord` at the
        ``text``/``None`` default.

        That fallback is deliberately narrow rather than folded into
        :class:`~groundkit.index.protocols.MetadataStoreProtocol` itself —
        see :class:`DocumentRecordStoreProtocol`'s docstring for why
        widening the required protocol would break every hand-built
        protocol-conforming store double that predates ADR-0016. It is
        honest rather than a silent downgrade: a store with no way to report
        ``source_class``/``extractor`` never had richer data to report in
        the first place, unlike the actual fail-open defect this method
        closes (a real store dropping a value it *did* have).
        """
        if isinstance(self._store, DocumentRecordStoreProtocol):
            return await self._store.get_document_records()
        sources = await self._store.get_document_sources()
        return {
            document_id: DocumentRecord(source=source) for document_id, source in sources.items()
        }

    async def _dense_candidates(
        self,
        query: str,
        k: int,
        records: dict[str, DocumentRecord],
        embedder: EmbeddingProtocol,
        vector_store: VectorStoreProtocol,
    ) -> list[tuple[Chunk, float]]:
        """Embed ``query`` and return snapshot-filtered dense ``(chunk, score)`` pairs.

        The vector-store handle is live, so this filter is what gives the
        dense path the same ``open()``-time snapshot semantics the BM25
        rebuild has structurally (see the class docstring). Three cases per
        hit, decided against the open()-time snapshot and the live
        ``records`` map:

        - document known at ``open()`` → kept (the join may still fail
          closed later if the document has since been deleted).
        - unknown at ``open()`` but present in ``records`` → ingested after
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
            kept = self._apply_snapshot_filter(pairs, records)
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
        self, pairs: list[tuple[Chunk, float]], records: dict[str, DocumentRecord]
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
            if chunk.document_id in records:
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
        records: dict[str, DocumentRecord],
        *,
        apply_threshold: bool,
    ) -> list[RetrievalResult]:
        """Join ``(chunk, score)`` pairs to their documents' records — the ONE join.

        Every mode funnels through here exactly once per search, after
        fusion in hybrid mode (ADR-0006: the join happens on the surviving
        results, and this is the one place the fail-closed rule lives).
        ``apply_threshold=False`` is the hybrid path: ADR-0005 decision 6
        keeps ``score_threshold`` away from rank-derived fused scores.

        Every constructed :class:`RetrievalResult` carries the document's
        real ``source_class``/``extractor`` (ADR-0016) rather than the
        ``("text", None)`` default — the fail-open defect this closes: before
        this, every result silently reported ``"text"`` regardless of what
        was actually ingested, routing an ``extracted`` or ``snapshot``
        citation into a resolver path that would compare its offsets against
        text they were never measured against.

        Raises:
            RetrievalError: A chunk's document has no stored record — fail
                closed rather than emit an unverifiable citation.
        """
        threshold = self._config.score_threshold if apply_threshold else None
        results: list[RetrievalResult] = []
        for chunk, score in pairs:
            if threshold is not None and score < threshold:
                continue
            record = records.get(chunk.document_id)
            if record is None:
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
                    source=record.source,
                    source_class=record.source_class,
                    extractor=record.extractor,
                    start_offset=chunk.start_offset,
                    end_offset=chunk.end_offset,
                    metadata=chunk.metadata,
                )
            )
        return results


def _validate_dense_pair(
    embedder: EmbeddingProtocol | None, vector_store: VectorStoreProtocol | None
) -> None:
    """Reject half a dense pair at construction, in ``Retriever``'s vocabulary.

    The branch structure is shared with ``Indexer`` and ``run_eval``
    (:func:`groundkit.identity.validate_dense_pair`); only the consequence of
    each missing half is stated per caller, because for a read path it differs
    from a write path's.

    Raises:
        ConfigurationError: Exactly one of ``embedder`` / ``vector_store``
            was supplied.
    """
    validate_dense_pair(
        embedder,
        vector_store,
        subject="Retriever",
        without_store=(
            "an embedder alone can embed the query, but there is no dense index to search with it."
        ),
        without_embedder=(
            "without an embedder the query can never be embedded, so the store "
            "could never be searched."
        ),
    )

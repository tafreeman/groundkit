"""Collection lifecycle: a cached retriever whose validity comes from disk (ADR-0013).

A :class:`~groundkit.retrieval.search.Retriever` is an ``open()``-time snapshot
that never refreshes (ADR-0002). That is correct for a one-shot CLI invocation
and wrong for a long-lived process: content ingested after ``open()`` has no
representation in the rebuilt BM25 index and no way past the dense side's
snapshot filter, so it returns **zero results, silently** — the one staleness
direction this repo's fail-closed rule cannot detect. A server that opens one
retriever at startup is wrong the moment anyone runs ``grk ingest``.

Caching a retriever is nonetheless in-process state shadowing persisted state,
which SPEC.md §2 forbids — *unless* its validity is derived from something
persisted. That is the whole design: :class:`CollectionRuntime` holds an open
retriever and a generation number read from SQLite, and hands it out only while
the store still reports that same generation. An invalidate-on-my-own-writes
cache would be cheaper and would be correct exactly until someone ingested from
another terminal, which is the workflow SPEC.md §9 documents.

Placement follows :mod:`groundkit.identity`: a module outside whichever caller
happened to need it first, so sharing it creates no dependency between ingest
and retrieval, and nothing imports it back. One property does *not* carry over,
and saying so is clearer than implying it: ``identity`` is an import leaf, while
this is a composition root — it must import the retriever, the store and the
vector store, because composing them is its job.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

from groundkit.errors import ConfigurationError, StorageError
from groundkit.identity import validate_dense_pair
from groundkit.index.metadata import SQLiteMetadataStore
from groundkit.retrieval.search import Retriever

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable
    from pathlib import Path
    from typing import TypeAlias

    from groundkit.config import RetrievalConfig
    from groundkit.contracts import Chunk, CollectionManifest
    from groundkit.index.protocols import VectorStoreProtocol
    from groundkit.providers.protocols import EmbeddingProtocol

    #: Produces a fresh vector-store handle for each rebuild. A factory rather
    #: than an instance because the handle is collection-derived state:
    #: ``LanceDBVectorStore.open()`` caches its table and returns no results
    #: when that table did not exist at open time, never re-checking for one
    #: created later. Held across rebuilds, a runtime opened before a
    #: collection's first dense ingest would refuse dense reads permanently
    #: afterwards, on a collection that had become perfectly healthy.
    VectorStoreFactory: TypeAlias = Callable[[], Awaitable[VectorStoreProtocol]]

    #: The registry's factory, which takes the collection name. A
    #: :class:`CollectionRuntime` is *one* collection, so its factory rightly
    #: takes no argument; a :class:`CollectionRegistry` holds many, so its
    #: factory must be told which one it is opening.
    #:
    #: The distinction is load-bearing rather than cosmetic. The dense store is
    #: laid out per collection — ``<index_dir>/<collection>.lance``, sibling of
    #: ``<collection>.sqlite3`` — so a registry that handed every runtime the
    #: same handle would search one collection's vectors and then join the
    #: resulting chunk ids against a *different* collection's SQLite. Those ids
    #: do not match, so the failure is not an error: it is a silently empty or
    #: under-populated result on a collection that is perfectly healthy. That is
    #: the same silent-zero-results class ADR-0013 exists to close, reintroduced
    #: one level up.
    CollectionVectorStoreFactory: TypeAlias = Callable[[str], Awaitable[VectorStoreProtocol]]

logger = logging.getLogger(__name__)

#: Default ceiling on collections held open by a :class:`CollectionRegistry`.
#: The binding resource is memory, not file handles: ``BM25Index`` retains the
#: collection's chunk list, so a cached retriever pins that collection's entire
#: chunk text resident. N open collections means N corpora in RAM.
DEFAULT_MAX_OPEN_COLLECTIONS: int = 8


@dataclass(frozen=True, slots=True)
class AcquiredRetriever:
    """A retriever together with the store generation it was built against.

    ``generation`` is ``None`` when the store cannot answer the freshness
    question at all (a collection predating ADR-0013's schema). That is not a
    defect in this object — it is the honest label on a retriever that was
    rebuilt precisely because nothing could vouch for a cached one.

    Attributes:
        generation: Store generation at the moment the rebuild began, or
            ``None`` when freshness is unanswerable.
        retriever: The retriever, valid as of ``generation``.
    """

    generation: int | None
    retriever: Retriever


class CollectionRuntime:
    """Hands out a retriever for one collection, rebuilding it when the store changes.

    Not thread-safe and not intended to be: like everything else on this path
    it is single-event-loop, serialized through the store's own lock.
    """

    def __init__(
        self,
        store: SQLiteMetadataStore,
        config: RetrievalConfig | None = None,
        *,
        embedder: EmbeddingProtocol | None = None,
        vector_store_factory: Callable[[], Awaitable[VectorStoreProtocol]] | None = None,
    ) -> None:
        validate_dense_pair(
            embedder,
            vector_store_factory,
            subject="CollectionRuntime",
            without_store=(
                "a retriever built from it could embed a query and then have nothing to search."
            ),
            without_embedder=(
                "a vector store with no embedder cannot turn a query into the vector it "
                "would be searched with."
            ),
        )
        self._store = store
        self._config = config
        self._embedder = embedder
        self._vector_store_factory = vector_store_factory
        self._cached: AcquiredRetriever | None = None
        self._vector_store: VectorStoreProtocol | None = None
        self._rebuild_lock = asyncio.Lock()
        self._closed = False
        self._warned_unanswerable = False

    @classmethod
    async def open(
        cls,
        index_dir: Path,
        collection: str,
        config: RetrievalConfig | None = None,
        *,
        embedder: EmbeddingProtocol | None = None,
        vector_store_factory: Callable[[], Awaitable[VectorStoreProtocol]] | None = None,
    ) -> CollectionRuntime:
        """Open the collection's store and build a runtime over it.

        Opens the store eagerly but builds no retriever: the first
        :meth:`acquire` does that, so a runtime for a collection nobody queries
        never pays the O(corpus) rebuild.

        Raises:
            ConfigurationError: Invalid collection name, or exactly one half of
                the dense pair was supplied.
            StorageError: The store could not be opened.
        """
        store = await SQLiteMetadataStore.open(index_dir, collection)
        try:
            return cls(
                store,
                config,
                embedder=embedder,
                vector_store_factory=vector_store_factory,
            )
        except BaseException:
            # The pair validation above can reject; do not leak the handle.
            await store.close()
            raise

    async def acquire(self) -> AcquiredRetriever:
        """Return a retriever current as of the store's present generation.

        The generation is read **first**, before any part of the rebuild. That
        ordering is the correctness property, not an implementation detail:
        every read the retriever is built from then happens at or after that
        sample, so the stamp is a *lower bound* on the artifact's freshness. A
        write landing mid-rebuild leaves ``current > stamp``, the equality
        predicate fails, and the next acquire rebuilds — a redundant rebuild,
        never a stale answer.

        Sampling it *after* the rebuild would invert that: a write landing
        during the rebuild would raise the counter without being reflected in
        reads that already completed, the predicate would then hold
        indefinitely, and this runtime would serve an index missing that write
        until some later write happened to bump again. That is ADR-0002's
        silent-zero-results outcome made permanent by caching.

        A request arriving mid-rebuild waits rather than being handed the stale
        cached retriever. That costs real tail latency and is chosen anyway:
        serving stale "only for the duration of a rebuild" reintroduces silent
        zero results at exactly the moment someone is most likely to be
        actively ingesting.

        Raises:
            StorageError: The runtime is closed, or the store failed.
            ConfigurationError: A dense read was configured against a
                collection that cannot serve one.
            IndexIdentityError: The collection is bound to a different
                embedding identity (ADR-0004).
        """
        self._require_open()
        generation = await self._store.get_generation()
        if generation is None and not self._warned_unanswerable:
            self._warned_unanswerable = True
            logger.warning(
                "Collection at %s predates the ADR-0013 staleness marker, so freshness "
                "cannot be asserted and every request will rebuild its retriever. "
                "Re-ingest into a fresh collection to restore caching.",
                self._store.db_path,
            )

        cached = self._cached
        if cached is not None and generation is not None and cached.generation == generation:
            return cached

        await self._rebuild_lock.acquire()
        released = False
        try:
            self._require_open()
            # Re-check against the generation observed BEFORE queueing, not a
            # fresh read. Re-reading here would make every waiter chase a
            # moving target under sustained writes: each would see a newer
            # generation than the leader just satisfied and rebuild again,
            # turning write pressure into an unbounded rebuild storm. Reusing
            # the pre-queue sample caps rebuilds at one per observed
            # generation and still gives each waiter a defined freshness bound.
            cached = self._cached
            if cached is not None and generation is not None and cached.generation == generation:
                return cached

            worker: asyncio.Task[AcquiredRetriever] = asyncio.ensure_future(
                self._rebuild(generation)
            )
            worker.add_done_callback(self._release_after_rebuild)
            released = True
            # Shielded for throughput, not correctness: a Retriever owns
            # nothing closable, so an abandoned half-built one leaks nothing.
            # What the shield buys is that a disconnecting client does not
            # abandon a rebuild other waiters are already blocked on.
            return await asyncio.shield(worker)
        finally:
            if not released:
                self._rebuild_lock.release()

    async def _rebuild(self, generation: int | None) -> AcquiredRetriever:
        """Build a fresh retriever and publish it, stamped with ``generation``.

        The vector-store handle is rebuilt alongside the retriever rather than
        held for the runtime's lifetime — see :data:`VectorStoreFactory`.

        Publishes only on success: ``self._cached`` is assigned after
        ``Retriever.open`` returns, so a failed rebuild leaves the previous
        entry in place but stamped with a generation that no longer matches,
        which makes it unreachable through the equality predicate rather than
        merely unused. That is a structural consequence of the predicate, not
        defensive code.
        """
        vector_store: VectorStoreProtocol | None = None
        if self._vector_store_factory is not None:
            vector_store = await self._vector_store_factory()

        retriever = await Retriever.open(
            self._store,
            self._config,
            embedder=self._embedder,
            vector_store=vector_store,
        )
        acquired = AcquiredRetriever(generation=generation, retriever=retriever)

        # The superseded handle is dropped rather than closed: no vector store
        # in this repo exposes a close(). If one ever does, this is where the
        # old handle must be closed, and the same applies to Retriever itself.
        self._vector_store = vector_store
        if generation is not None:
            self._cached = acquired
        return acquired

    def _release_after_rebuild(self, worker: asyncio.Future[Any]) -> None:
        """Release the rebuild lock once the rebuild is done with the store.

        A done-callback rather than a ``finally``, for the reason
        :meth:`~groundkit.index.metadata.SQLiteMetadataStore._run` uses one: on
        cancellation the awaiting coroutine unwinds while the rebuild is still
        reading the store, and releasing then would let the next waiter start a
        second concurrent rebuild over the same handle. Also retrieves the
        exception so an abandoned failure does not surface as an "exception was
        never retrieved" warning.
        """
        self._rebuild_lock.release()
        if not worker.cancelled():
            worker.exception()

    # -- Read-only passthroughs ------------------------------------------
    #
    # These exist so a service surface never needs the store handle itself.
    # The runtime *is* the read-only facade ADR-0014 decision 2 check 4 calls
    # for: its public API is search-by-acquire plus the reads below, and it
    # exposes no way to reach upsert_document, add_chunks, replace_document,
    # delete_document or write_manifest. A wrapper class asserting the same
    # thing would add a layer whose only job is to re-state a property this
    # class can hold directly.

    @property
    def db_path(self) -> Path:
        """Path to the collection's SQLite file, for diagnostics."""
        return self._store.db_path

    async def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Return one chunk by id, or ``None``."""
        self._require_open()
        return await self._store.get_chunk(chunk_id)

    async def get_document_sources(self) -> dict[str, str]:
        """Return ``{document_id: source}`` for every document in the collection."""
        self._require_open()
        return await self._store.get_document_sources()

    async def get_manifest(self) -> CollectionManifest | None:
        """Return the collection's embedding-identity manifest, or ``None``."""
        self._require_open()
        return await self._store.get_manifest()

    async def get_generation(self) -> int | None:
        """Return the collection's staleness marker, or ``None`` if unanswerable."""
        self._require_open()
        return await self._store.get_generation()

    async def chunk_count(self) -> int:
        """Return the number of persisted chunks.

        O(corpus): the store exposes no count, so this materializes the chunk
        list and measures it. Adding ``COUNT(*)`` to
        :class:`~groundkit.index.protocols.MetadataStoreProtocol` would be
        cheaper and is deliberately not done — that protocol is held to exact
        signature parity by conformance tests, and widening it for a reporting
        convenience is the trade ADR-0012 decision 3 already refused for
        ``model_name``. Recorded in ``KNOWN_LIMITATIONS.md`` rather than
        hidden: this is a status call, not a hot path.
        """
        self._require_open()
        return len(await self._store.get_chunks())

    def _require_open(self) -> None:
        if self._closed:
            raise StorageError(
                f"CollectionRuntime for {self._store.db_path} is closed; its store "
                "connection is gone and no retriever can be built from it."
            )

    async def aclose(self) -> None:
        """Close the runtime and its store. Idempotent.

        Ordered deliberately: the rebuild lock is drained first so no rebuild
        is mid-flight over the connection, and the store is closed **last**,
        because the store's own lock is what every in-flight operation
        serializes on — closing it first would race them.

        The embedder is *not* closed here. It is process-lifetime state that
        may be shared across collections, and a runtime does not own what it
        was handed.
        """
        if self._closed:
            return
        async with self._rebuild_lock:
            self._closed = True
            self._cached = None
            self._vector_store = None
        await self._store.close()


class CollectionRegistry:
    """Holds one :class:`CollectionRuntime` per collection, bounded and refcounted.

    Exists because a service may serve more than one collection and each open
    one costs a resident corpus. Hands runtimes out through an async context
    manager, which is the shape both a FastAPI dependency and an MCP tool
    handler want.

    ``vector_store_factory`` takes the **collection name**, unlike
    :class:`CollectionRuntime`'s, which takes nothing — see
    :data:`CollectionVectorStoreFactory` for why that difference is load-bearing
    rather than an inconsistency to tidy away.
    """

    def __init__(
        self,
        index_dir: Path,
        config: RetrievalConfig | None = None,
        *,
        embedder: EmbeddingProtocol | None = None,
        vector_store_factory: Callable[[str], Awaitable[VectorStoreProtocol]] | None = None,
        max_open_collections: int = DEFAULT_MAX_OPEN_COLLECTIONS,
    ) -> None:
        if max_open_collections < 1:
            raise ConfigurationError(
                f"max_open_collections must be at least 1, got {max_open_collections}"
            )
        self._index_dir = index_dir
        self._config = config
        self._embedder = embedder
        self._vector_store_factory = vector_store_factory
        self._max_open = max_open_collections
        self._runtimes: dict[str, CollectionRuntime] = {}
        self._refcounts: dict[str, int] = {}
        self._lock = asyncio.Lock()
        self._closed = False

    @asynccontextmanager
    async def acquire(self, collection: str) -> AsyncIterator[CollectionRuntime]:
        """Yield the runtime for ``collection``, held open for the block's duration.

        Yields the runtime rather than an :class:`AcquiredRetriever` because a
        caller usually needs more than a retriever — ``index_status`` wants
        counts and the manifest, ``fetch_chunk`` wants a chunk and its
        document's source — and returning only the retriever would push those
        callers back to the store handle, which is the surface this whole
        layer exists to keep them away from. Call
        :meth:`CollectionRuntime.acquire` inside the block for a retriever.

        The refcount is held across the whole block, so eviction cannot close
        a runtime out from under an in-flight request.

        Raises:
            ConfigurationError: The collection name is invalid, or no such
                collection exists.
            StorageError: The registry is closed, or the store failed.
        """
        runtime = await self._checkout(collection)
        try:
            yield runtime
        finally:
            await self._checkin(collection)

    async def _checkout(self, collection: str) -> CollectionRuntime:
        async with self._lock:
            if self._closed:
                raise StorageError("CollectionRegistry is closed")
            runtime = self._runtimes.get(collection)
            if runtime is None:
                self._require_existing(collection)
                runtime = await CollectionRuntime.open(
                    self._index_dir,
                    collection,
                    self._config,
                    embedder=self._embedder,
                    vector_store_factory=self._bind_factory(collection),
                )
                self._runtimes[collection] = runtime
                self._refcounts[collection] = 0
            self._refcounts[collection] += 1
            # Re-insert so dict order is least-recently-checked-out first.
            self._runtimes[collection] = self._runtimes.pop(collection)
            return runtime

    def _bind_factory(self, collection: str) -> Callable[[], Awaitable[VectorStoreProtocol]] | None:
        """Bind the registry's collection-aware factory to one collection.

        ``functools.partial`` rather than a ``lambda`` closing over
        ``collection``: the loop-variable capture a closure would introduce is
        the classic late-binding bug, and here it would not raise — it would
        hand a runtime the *wrong* collection's vector store, which is exactly
        the silent mis-join :data:`CollectionVectorStoreFactory` describes.
        ``partial`` binds the value now, so the hazard cannot exist.
        """
        if self._vector_store_factory is None:
            return None
        return partial(self._vector_store_factory, collection)

    async def _checkin(self, collection: str) -> None:
        async with self._lock:
            if collection in self._refcounts:
                self._refcounts[collection] -= 1
            await self._evict_if_over_bound()

    def _require_existing(self, collection: str) -> None:
        """Refuse a collection that does not already exist, without creating it.

        ``SQLiteMetadataStore.open`` creates the file when it is absent, which
        is right for ``grk ingest`` and wrong at a request boundary: without
        this check, asking for the status of an arbitrary name would *create*
        an empty collection, so an unauthenticated read surface would become a
        disk-fill primitive.

        The name is validated by ``SQLiteMetadataStore.open`` itself, which
        also rejects traversal — this check only answers "does it exist", and
        deliberately runs before any handle is opened.
        """
        db_path = self._index_dir / f"{collection}.sqlite3"
        if not db_path.is_file():
            raise ConfigurationError(
                f"collection {collection!r} does not exist in {self._index_dir}. "
                "Collections are created by `grk ingest`, never by reading one."
            )

    async def _evict_if_over_bound(self) -> None:
        """Close idle runtimes until the bound is met, oldest checkout first.

        A **soft** bound: if every runtime is in use the registry exceeds it
        and logs rather than closing a store out from under a live request.
        Overshooting a memory target is recoverable; closing a connection
        mid-request is a failure handed to a caller who did nothing wrong.
        """
        while len(self._runtimes) > self._max_open:
            idle = next(
                (name for name in self._runtimes if self._refcounts.get(name, 0) <= 0),
                None,
            )
            if idle is None:
                logger.warning(
                    "All %d open collections are in use, exceeding max_open_collections=%d. "
                    "Holding them open rather than closing one mid-request.",
                    len(self._runtimes),
                    self._max_open,
                )
                return
            runtime = self._runtimes.pop(idle)
            self._refcounts.pop(idle, None)
            await runtime.aclose()

    async def aclose(self) -> None:
        """Close every runtime the registry holds. Idempotent."""
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            runtimes = list(self._runtimes.values())
            self._runtimes.clear()
            self._refcounts.clear()
        for runtime in runtimes:
            await runtime.aclose()

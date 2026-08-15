"""Tests for CollectionRuntime and CollectionRegistry (ADR-0013).

Async methods are driven with ``asyncio.run()`` inside sync test functions
(pytest-asyncio is not configured in this repo).

The tests that matter here are the ones about *ordering* and *concurrency* —
when the generation is sampled relative to the rebuild, and what a second
request sees while a rebuild is in flight. Those paths are unreachable on a
non-failing path, so neither coverage nor a green suite distinguishes a real
test of them from a decorative one; each states below what it would catch.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from groundkit.contracts import Chunk, CollectionManifest, EmbeddingIdentity
from groundkit.errors import ConfigurationError, StorageError
from groundkit.index.metadata import SQLiteMetadataStore
from groundkit.providers.embeddings import InMemoryEmbedder
from groundkit.runtime import AcquiredRetriever, CollectionRegistry, CollectionRuntime


def _chunk(chunk_id: str, document_id: str, content: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=0,
        content=content,
        start_offset=0,
        end_offset=len(content),
    )


async def _write(index_dir: Path, collection: str, source: str, doc_id: str, text: str) -> None:
    """Write one document + chunk through a fresh store handle.

    A *separate* handle on purpose: this is what an out-of-process ``grk
    ingest`` looks like to a runtime that already has the collection open.
    """
    store = await SQLiteMetadataStore.open(index_dir, collection)
    try:
        await store.replace_document(
            source, doc_id, f"hash-{doc_id}", [_chunk(doc_id, doc_id, text)]
        )
    finally:
        await store.close()


class _CountingStore:
    """Delegating store wrapper that counts rebuilds and can stall one.

    Counts ``get_chunks`` because that is what ``BM25Index.from_store`` calls:
    one call per rebuild, so the count is the rebuild count.
    """

    def __init__(self, inner: SQLiteMetadataStore) -> None:
        self._inner = inner
        self.get_chunks_calls = 0
        self.gate: asyncio.Event | None = None
        self.entered: asyncio.Event | None = None

    @property
    def db_path(self) -> Path:
        return self._inner.db_path

    async def get_chunks(self) -> list[Chunk]:
        """Read first, *then* stall.

        The ordering here is load-bearing for the stamp test. Stalling before
        delegating would make the rebuild read whatever landed during the
        stall, so the artifact would contain the interleaved write either way
        and the test would pass against both orderings — i.e. prove nothing.
        Reading first pins the artifact to the pre-write snapshot, which is
        what makes the stamp the only thing that decides whether the next
        acquire rebuilds.
        """
        self.get_chunks_calls += 1
        chunks = await self._inner.get_chunks()
        if self.entered is not None:
            self.entered.set()
        if self.gate is not None:
            await self.gate.wait()
        return chunks

    async def upsert_document(self, source: str, document_id: str, content_hash: str) -> None:
        await self._inner.upsert_document(source, document_id, content_hash)

    async def get_document_hash(self, source: str) -> str | None:
        return await self._inner.get_document_hash(source)

    async def get_document_id(self, source: str) -> str | None:
        return await self._inner.get_document_id(source)

    async def get_document_sources(self) -> dict[str, str]:
        return await self._inner.get_document_sources()

    async def add_chunks(self, chunks: list[Chunk], source: str) -> None:
        await self._inner.add_chunks(chunks, source)

    async def replace_document(
        self, source: str, document_id: str, content_hash: str, chunks: list[Chunk]
    ) -> None:
        await self._inner.replace_document(source, document_id, content_hash, chunks)

    async def get_chunk(self, chunk_id: str) -> Chunk | None:
        return await self._inner.get_chunk(chunk_id)

    async def delete_document(self, document_id: str) -> int:
        return await self._inner.delete_document(document_id)

    async def write_manifest(self, identity: EmbeddingIdentity) -> None:
        await self._inner.write_manifest(identity)

    async def verify_manifest(self, identity: EmbeddingIdentity) -> CollectionManifest | None:
        return await self._inner.verify_manifest(identity)

    async def get_manifest(self) -> CollectionManifest | None:
        return await self._inner.get_manifest()

    async def get_generation(self) -> int | None:
        return await self._inner.get_generation()

    async def close(self) -> None:
        await self._inner.close()


def test_cached_retriever_observes_an_out_of_process_ingest(tmp_path: Path) -> None:
    """THE defining test: a write from another handle is visible on the next acquire.

    Fail-first: make ``acquire`` return ``self._cached`` unconditionally once
    populated and the second search returns zero results — silently, which is
    exactly the ADR-0002 staleness direction that fails *open* and is why this
    runtime exists at all.
    """

    async def run() -> None:
        await _write(tmp_path, "col", "/a.md", "doc-a", "alpha content about turbines")
        runtime = await CollectionRuntime.open(tmp_path, "col")
        try:
            first = await runtime.acquire()
            assert (await first.retriever.search("turbines")).results

            # Another process ingests. The runtime is not told.
            await _write(tmp_path, "col", "/b.md", "doc-b", "beta content about hydrofoils")

            second = await runtime.acquire()
            response = await second.retriever.search("hydrofoils")
            assert response.results, "content ingested after open() was invisible"
            assert second.generation != first.generation
        finally:
            await runtime.aclose()

    asyncio.run(run())


def test_unchanged_store_reuses_the_cached_retriever(tmp_path: Path) -> None:
    """The point of the cache: no write, no rebuild, same object identity."""

    async def run() -> None:
        await _write(tmp_path, "col", "/a.md", "doc-a", "alpha")
        inner = await SQLiteMetadataStore.open(tmp_path, "col")
        counting = _CountingStore(inner)
        runtime = CollectionRuntime(counting)  # type: ignore[arg-type]
        try:
            first = await runtime.acquire()
            second = await runtime.acquire()
            assert first is second
            assert counting.get_chunks_calls == 1
        finally:
            await runtime.aclose()

    asyncio.run(run())


def test_artifact_is_stamped_with_the_pre_build_generation(tmp_path: Path) -> None:
    """A write landing DURING a rebuild is not swallowed by the artifact it raced.

    This is the sharpest ordering property in ADR-0013. The generation is
    sampled before the rebuild, so an interleaved write leaves
    ``current > stamp`` and the next acquire rebuilds.

    Fail-first: sample the generation *after* ``Retriever.open`` returns and
    this fails — the artifact is stamped with the post-write generation while
    its content was read before that write, the equality predicate then holds
    indefinitely, and the interleaved document stays invisible until some
    later unrelated write happens to bump the counter again.
    """

    async def run() -> None:
        await _write(tmp_path, "col", "/a.md", "doc-a", "alpha content")
        inner = await SQLiteMetadataStore.open(tmp_path, "col")
        counting = _CountingStore(inner)
        counting.gate = asyncio.Event()
        counting.entered = asyncio.Event()
        runtime = CollectionRuntime(counting)  # type: ignore[arg-type]
        try:
            acquiring = asyncio.ensure_future(runtime.acquire())
            await counting.entered.wait()

            # Interleave a write while the rebuild is mid-flight.
            await _write(tmp_path, "col", "/b.md", "doc-b", "beta content about hydrofoils")

            counting.gate.set()
            await acquiring

            # The next acquire must see the interleaved write.
            nxt = await runtime.acquire()
            assert (await nxt.retriever.search("hydrofoils")).results
        finally:
            await runtime.aclose()

    asyncio.run(run())


def test_concurrent_stale_requests_rebuild_once(tmp_path: Path) -> None:
    """Single-flight: N waiters observing one generation produce one rebuild, not N."""

    async def run() -> None:
        await _write(tmp_path, "col", "/a.md", "doc-a", "alpha")
        inner = await SQLiteMetadataStore.open(tmp_path, "col")
        counting = _CountingStore(inner)
        runtime = CollectionRuntime(counting)  # type: ignore[arg-type]
        try:
            results = await asyncio.gather(*(runtime.acquire() for _ in range(8)))
            assert counting.get_chunks_calls == 1, "each waiter rebuilt instead of sharing one"
            assert all(r is results[0] for r in results)
        finally:
            await runtime.aclose()

    asyncio.run(run())


def test_request_during_a_rebuild_is_not_served_the_stale_index(tmp_path: Path) -> None:
    """A second request waits for the rebuild rather than being handed the stale artifact.

    Fail-first: return the cached artifact whenever the rebuild lock is held
    and this fails — the second caller gets a retriever that cannot see the
    document written before it ever called.
    """

    async def run() -> None:
        await _write(tmp_path, "col", "/a.md", "doc-a", "alpha")
        inner = await SQLiteMetadataStore.open(tmp_path, "col")
        counting = _CountingStore(inner)
        runtime = CollectionRuntime(counting)  # type: ignore[arg-type]
        try:
            warm = await runtime.acquire()
            assert not (await warm.retriever.search("hydrofoils")).results

            await _write(tmp_path, "col", "/b.md", "doc-b", "beta about hydrofoils")

            counting.gate = asyncio.Event()
            counting.entered = asyncio.Event()
            first = asyncio.ensure_future(runtime.acquire())
            await counting.entered.wait()
            second = asyncio.ensure_future(runtime.acquire())
            await asyncio.sleep(0)  # let it reach the lock

            counting.gate.set()
            got_first, got_second = await asyncio.gather(first, second)

            for acquired in (got_first, got_second):
                assert (await acquired.retriever.search("hydrofoils")).results
        finally:
            await runtime.aclose()

    asyncio.run(run())


def test_a_legacy_store_rebuilds_on_every_acquire(tmp_path: Path) -> None:
    """No marker means no caching — correct and slow, never cached-and-stale.

    Fail-first: have ``get_generation`` report ``0`` instead of ``None`` for a
    store that cannot answer, and the cache engages over a collection nothing
    can vouch for.
    """

    class _Unanswerable(_CountingStore):
        async def get_generation(self) -> int | None:
            return None

    async def run() -> None:
        await _write(tmp_path, "col", "/a.md", "doc-a", "alpha")
        inner = await SQLiteMetadataStore.open(tmp_path, "col")
        store = _Unanswerable(inner)
        runtime = CollectionRuntime(store)  # type: ignore[arg-type]
        try:
            first = await runtime.acquire()
            second = await runtime.acquire()
            assert first.generation is None and second.generation is None
            assert first is not second
            assert store.get_chunks_calls == 2
        finally:
            await runtime.aclose()

    asyncio.run(run())


def test_a_failed_rebuild_does_not_publish_a_partial_artifact(tmp_path: Path) -> None:
    """A rebuild that raises leaves nothing cached and propagates the error."""

    class _Exploding(_CountingStore):
        async def get_chunks(self) -> list[Chunk]:
            raise StorageError("simulated backend failure")

    async def run() -> None:
        await _write(tmp_path, "col", "/a.md", "doc-a", "alpha")
        inner = await SQLiteMetadataStore.open(tmp_path, "col")
        store = _Exploding(inner)
        runtime = CollectionRuntime(store)  # type: ignore[arg-type]
        try:
            with pytest.raises(StorageError, match="simulated"):
                await runtime.acquire()
            assert runtime._cached is None
            # The lock was released, so a later acquire is not deadlocked.
            with pytest.raises(StorageError, match="simulated"):
                await runtime.acquire()
        finally:
            await runtime.aclose()

    asyncio.run(run())


def test_acquire_after_close_is_refused(tmp_path: Path) -> None:
    """A closed runtime refuses rather than handing back a retriever over a dead connection.

    Fail-first: drop the closed-flag guard and this surfaces as a sqlite
    ProgrammingError from deep inside the rebuild instead of a typed
    GroundkitError the CLI and the service both already handle.
    """

    async def run() -> None:
        await _write(tmp_path, "col", "/a.md", "doc-a", "alpha")
        runtime = await CollectionRuntime.open(tmp_path, "col")
        await runtime.acquire()
        await runtime.aclose()
        await runtime.aclose()  # idempotent
        with pytest.raises(StorageError, match="closed"):
            await runtime.acquire()

    asyncio.run(run())


def test_runtime_refuses_half_a_dense_pair(tmp_path: Path) -> None:
    """Both or neither, via the shared identity helper rather than a local copy."""

    async def run() -> None:
        store = await SQLiteMetadataStore.open(tmp_path, "col")
        try:
            with pytest.raises(ConfigurationError, match="inseparable"):
                CollectionRuntime(store, vector_store_factory=lambda: asyncio.sleep(0))  # type: ignore[arg-type,return-value]
        finally:
            await store.close()

    asyncio.run(run())


def test_registry_refuses_an_unknown_collection_without_creating_it(tmp_path: Path) -> None:
    """Reading a collection must never create one.

    Fail-first: drop the existence pre-check so ``SQLiteMetadataStore.open``
    runs, and the directory-contents assertion fails — an unauthenticated read
    surface would have become a disk-fill primitive, one empty SQLite file per
    distinct name asked for.
    """

    async def run() -> None:
        registry = CollectionRegistry(tmp_path)
        try:
            before = sorted(p.name for p in tmp_path.iterdir())
            with pytest.raises(ConfigurationError, match="does not exist"):
                async with registry.acquire("nope"):
                    pass
            assert sorted(p.name for p in tmp_path.iterdir()) == before
        finally:
            await registry.aclose()

    asyncio.run(run())


def test_registry_reuses_one_runtime_per_collection(tmp_path: Path) -> None:
    """Two acquires of the same collection share a runtime, so they share its cache."""

    async def run() -> None:
        await _write(tmp_path, "col", "/a.md", "doc-a", "alpha")
        registry = CollectionRegistry(tmp_path)
        try:
            async with registry.acquire("col") as runtime_a:
                first = await runtime_a.acquire()
            async with registry.acquire("col") as runtime_b:
                second = await runtime_b.acquire()
            assert runtime_a is runtime_b
            assert isinstance(first, AcquiredRetriever)
            assert first is second, "the second acquire rebuilt instead of reusing the cache"
        finally:
            await registry.aclose()

    asyncio.run(run())


def test_registry_evicts_idle_runtimes_over_the_bound(tmp_path: Path) -> None:
    """The bound is enforced against idle runtimes."""

    async def run() -> None:
        for name in ("a", "b", "c"):
            await _write(tmp_path, name, f"/{name}.md", f"doc-{name}", "text")
        registry = CollectionRegistry(tmp_path, max_open_collections=2)
        try:
            for name in ("a", "b", "c"):
                async with registry.acquire(name):
                    pass
            assert len(registry._runtimes) <= 2
        finally:
            await registry.aclose()

    asyncio.run(run())


def test_registry_holds_open_rather_than_closing_a_runtime_in_use(tmp_path: Path) -> None:
    """A soft bound: exceeding it beats closing a store out from under a live request.

    Overshooting a memory target is recoverable. Closing a connection
    mid-request hands a failure to a caller who did nothing wrong, so the
    registry logs and exceeds instead.
    """

    async def run() -> None:
        for name in ("a", "b", "c"):
            await _write(tmp_path, name, f"/{name}.md", f"doc-{name}", "text")
        registry = CollectionRegistry(tmp_path, max_open_collections=1)
        try:
            async with (
                registry.acquire("a"),
                registry.acquire("b"),
                registry.acquire("c"),
            ):
                assert len(registry._runtimes) == 3
        finally:
            await registry.aclose()

    asyncio.run(run())


def test_each_collection_gets_its_own_vector_store(tmp_path: Path) -> None:
    """The registry's factory is called with each collection's OWN name.

    This is a real regression test, shown to fail against the unfixed source
    (SPEC.md §8): before ``_bind_factory``, ``CollectionRegistry`` held one
    zero-argument factory and handed the SAME awaited handle to every runtime
    it opened.

    The failure that produces is the nastiest kind this repo tracks. The dense
    store is laid out per collection (``<index_dir>/<collection>.lance``), so a
    shared handle searches one collection's vectors and then joins the
    resulting chunk ids against a DIFFERENT collection's SQLite. Those ids do
    not match, so nothing raises — the caller gets a silently empty or
    under-populated result on a collection that is perfectly healthy. That is
    the same silent-zero-results class ADR-0013 exists to close, reintroduced
    one level up in the registry.

    Asserting on the ARGUMENT rather than on returned results is deliberate:
    it fails for the right reason and does not need a real dense corpus, whose
    absence would make an empty result indistinguishable from the bug.
    """
    requested: list[str] = []

    class _FakeVectorStore:
        """Minimal stand-in; the runtime only stores the handle here."""

        async def search(self, *args: object, **kwargs: object) -> list[object]:
            return []

    async def factory(collection: str) -> _FakeVectorStore:
        requested.append(collection)
        return _FakeVectorStore()

    embedder = InMemoryEmbedder(dimensions=8)

    async def run() -> None:
        for name in ("alpha", "beta"):
            # A manifest so the collection is dense-legal (ADR-0008), and
            # deliberately NO documents: with documents present, `Retriever.open`
            # fails closed because a stand-in vector store holds no vectors for
            # them. That guard is correct and is not what this test is about —
            # the factory is called during the rebuild either way, and the
            # argument it receives is the whole assertion.
            store = await SQLiteMetadataStore.open(tmp_path, name)
            try:
                await store.write_manifest(
                    EmbeddingIdentity(
                        provider=embedder.provider,
                        model_name=embedder.model_name,
                        dimensions=embedder.dimensions,
                    )
                )
            finally:
                await store.close()

        registry = CollectionRegistry(
            tmp_path,
            embedder=embedder,
            vector_store_factory=factory,  # type: ignore[arg-type]
        )
        try:
            async with registry.acquire("alpha") as runtime:
                await runtime.acquire()
            async with registry.acquire("beta") as runtime:
                await runtime.acquire()
        finally:
            await registry.aclose()

    asyncio.run(run())

    assert requested == ["alpha", "beta"], (
        "each collection must get a vector store built for ITS OWN name; "
        f"the factory was asked for {requested!r}"
    )

"""Retriever and citation-resolution tests.

Phase 1's BM25-only path lives in the first half of this module. Phase 3
Wave C's dense and hybrid read path (ADR-0004, ADR-0005, ADR-0006) is
appended in its own section below, extending this file without touching
any pre-existing test.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import pytest

from groundkit.config import RetrievalConfig
from groundkit.contracts import (
    Chunk,
    Citation,
    CollectionManifest,
    Document,
    EmbeddingIdentity,
)
from groundkit.errors import ConfigurationError, IndexIdentityError, RetrievalError
from groundkit.index.bm25 import BM25Index
from groundkit.index.dense import InMemoryVectorStore
from groundkit.index.metadata import SQLiteMetadataStore
from groundkit.index.protocols import MetadataStoreProtocol
from groundkit.indexer import Indexer
from groundkit.ingestion.loaders import FileLoader
from groundkit.providers.embeddings import InMemoryEmbedder
from groundkit.providers.protocols import EmbeddingProtocol
from groundkit.retrieval.citations import resolve_citation, verify_citation
from groundkit.retrieval.fusion import reciprocal_rank_fusion
from groundkit.retrieval.search import MAX_TOP_K, Retriever, SearchMode

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

DOC_TEXTS = {
    "alpha.md": "Retrieval systems rank documents. BM25 is a lexical ranking function.",
    "beta.md": "Cooking pasta requires boiling water and a generous pinch of salt.",
}


def _chunks_for(doc: Document) -> list[Chunk]:
    return [
        Chunk(
            document_id=doc.document_id,
            chunk_index=0,
            content=doc.content,
            start_offset=0,
            end_offset=len(doc.content),
            metadata={"source": doc.source},
        )
    ]


async def _populated_store(tmp_path: Path) -> SQLiteMetadataStore:
    store = await SQLiteMetadataStore.open(tmp_path / "idx", "default")
    for name, text in DOC_TEXTS.items():
        doc = Document(source=str(tmp_path / name), content=text)
        await store.upsert_document(
            source=doc.source, document_id=doc.document_id, content_hash="h"
        )
        await store.add_chunks(_chunks_for(doc), source=doc.source)
    return store


class TestRetriever:
    def test_search_returns_ranked_citation_bearing_results(self, tmp_path: Path) -> None:
        async def run() -> None:
            store = await _populated_store(tmp_path)
            try:
                retriever = await Retriever.open(store)
                response = await retriever.search("BM25 lexical ranking")
            finally:
                await store.close()
            assert response.total_results == 1
            top = response.results[0]
            assert top.source.endswith("alpha.md")
            assert top.start_offset == 0
            assert top.end_offset == len(DOC_TEXTS["alpha.md"])
            assert top.score > 0
            assert top.citation.chunk_id == top.chunk_id
            assert response.metadata["stage"] == "bm25"

        asyncio.run(run())

    def test_empty_query_raises(self, tmp_path: Path) -> None:
        async def run() -> None:
            store = await _populated_store(tmp_path)
            try:
                retriever = await Retriever.open(store)
                with pytest.raises(RetrievalError, match="empty"):
                    await retriever.search("   ")
            finally:
                await store.close()

        asyncio.run(run())

    @pytest.mark.parametrize("bad_k", [0, -1, MAX_TOP_K + 1])
    def test_out_of_range_top_k_raises(self, tmp_path: Path, bad_k: int) -> None:
        async def run() -> None:
            store = await _populated_store(tmp_path)
            try:
                retriever = await Retriever.open(store)
                with pytest.raises(RetrievalError, match="top_k"):
                    await retriever.search("pasta", top_k=bad_k)
            finally:
                await store.close()

        asyncio.run(run())

    def test_score_threshold_filters(self, tmp_path: Path) -> None:
        async def run() -> None:
            store = await _populated_store(tmp_path)
            try:
                strict = await Retriever.open(store, RetrievalConfig(score_threshold=1_000_000.0))
                response = await strict.search("pasta")
            finally:
                await store.close()
            assert response.total_results == 0

        asyncio.run(run())

    def test_missing_source_mapping_fails_closed(self) -> None:
        doc = Document(source="ghost.md", content="orphaned chunk content")

        class OrphanStore:
            async def upsert_document(
                self, source: str, document_id: str, content_hash: str
            ) -> None:
                raise NotImplementedError

            async def get_document_hash(self, source: str) -> str | None:
                return None

            async def get_document_id(self, source: str) -> str | None:
                return None

            async def get_document_sources(self) -> dict[str, str]:
                return {}

            async def add_chunks(self, chunks: list[Chunk], source: str) -> None:
                raise NotImplementedError

            async def replace_document(
                self, source: str, document_id: str, content_hash: str, chunks: list[Chunk]
            ) -> None:
                raise NotImplementedError

            async def get_chunks(self) -> list[Chunk]:
                return _chunks_for(doc)

            async def get_chunk(self, chunk_id: str) -> Chunk | None:
                return None

            async def delete_document(self, document_id: str) -> int:
                return 0

            async def write_manifest(self, identity: EmbeddingIdentity) -> None:
                raise NotImplementedError

            async def verify_manifest(
                self, identity: EmbeddingIdentity
            ) -> CollectionManifest | None:
                raise NotImplementedError

            async def get_manifest(self) -> CollectionManifest | None:
                return None

            async def get_generation(self) -> int | None:
                # None means "freshness unanswerable" (ADR-0013), which is the
                # honest answer for a hand-built fake with no durable state.
                return None

        store = OrphanStore()
        assert isinstance(store, MetadataStoreProtocol)

        async def run() -> None:
            retriever = await Retriever.open(store)
            with pytest.raises(RetrievalError, match="inconsistency"):
                await retriever.search("orphaned chunk")

        asyncio.run(run())


class TestStaleRetriever:
    """ADR-0002: ``Retriever.open`` snapshots BM25 once and never refreshes.

    Pins both halves of that deliberate, currently-undocumented-in-code
    behavior with a real end-to-end re-ingest (``Indexer`` + a persisted
    ``SQLiteMetadataStore``, not a hand-built stub store):

    - a retriever holding a chunk whose document was re-ingested (so its
      old ``document_id`` no longer resolves to a source) fails closed —
      ``RetrievalError`` — exactly like the pre-existing orphaned-chunk case
      above, just reached via a real re-ingest instead of a stub.
    - a retriever queried for content ingested *after* it was opened finds
      nothing and raises nothing: BM25 was never rebuilt, so the new
      chunks were never tokenized into it. There is no signal that the
      index is stale — this is the surprising half.

    ``src/groundkit/retrieval/search.py`` is not owned by this change; its
    class docstring still needs a staleness note (see the accompanying
    report — that file is out of scope here).
    """

    def test_stale_retriever_raises_on_document_modified_after_open(self, tmp_path: Path) -> None:
        """A retriever opened before a re-ingest fails closed on a doc changed underneath it."""

        async def run() -> None:
            docs_dir = tmp_path / "docs"
            docs_dir.mkdir()
            target = docs_dir / "doc.md"
            target.write_text("Retrieval systems rank documents by relevance.", encoding="utf-8")

            store = await SQLiteMetadataStore.open(tmp_path / "idx", "default")
            try:
                indexer = Indexer(store, FileLoader(allowed_base_dir=docs_dir))
                await indexer.index_directory(str(docs_dir))

                retriever = await Retriever.open(store)

                # Re-ingest with different content: replace_document deletes
                # the old document row (and its document_id) and inserts a
                # fresh one — the stale retriever's BM25 snapshot still
                # holds a chunk pointing at the now-gone document_id.
                target.write_text("Something completely unrelated now.", encoding="utf-8")
                await indexer.index_directory(str(docs_dir))

                with pytest.raises(RetrievalError, match="inconsistency"):
                    await retriever.search("relevance")
            finally:
                await store.close()

        asyncio.run(run())

    def test_stale_retriever_returns_zero_results_for_content_ingested_after_open(
        self, tmp_path: Path
    ) -> None:
        """A retriever opened before a new document is ingested silently misses it: zero
        results, no error — no signal at all that the index is stale."""

        async def run() -> None:
            docs_dir = tmp_path / "docs"
            docs_dir.mkdir()
            (docs_dir / "alpha.md").write_text(
                "Retrieval systems rank documents by relevance.", encoding="utf-8"
            )

            store = await SQLiteMetadataStore.open(tmp_path / "idx", "default")
            try:
                indexer = Indexer(store, FileLoader(allowed_base_dir=docs_dir))
                await indexer.index_directory(str(docs_dir))

                retriever = await Retriever.open(store)

                (docs_dir / "beta.md").write_text(
                    "Zebras migrate across the savanna every dry season.", encoding="utf-8"
                )
                await indexer.index_directory(str(docs_dir))

                response = await retriever.search("zebras savanna")

                assert response.total_results == 0
                assert response.results == []
            finally:
                await store.close()

        asyncio.run(run())


class TestCitations:
    def _write_source(self, tmp_path: Path) -> tuple[Citation, str]:
        text = "Alpha beta gamma delta epsilon."
        path = tmp_path / "doc.md"
        path.write_text(text, encoding="utf-8")
        span = (6, 16)
        citation = Citation(
            document_id="d",
            chunk_id="c",
            source=str(path),
            start_offset=span[0],
            end_offset=span[1],
        )
        return citation, text[span[0] : span[1]]

    def test_resolve_returns_exact_span(self, tmp_path: Path) -> None:
        citation, expected = self._write_source(tmp_path)
        assert asyncio.run(resolve_citation(citation, tmp_path)) == expected

    def test_verify_roundtrip(self, tmp_path: Path) -> None:
        citation, expected = self._write_source(tmp_path)
        assert asyncio.run(verify_citation(citation, expected, tmp_path))
        assert not asyncio.run(verify_citation(citation, "tampered", tmp_path))

    def test_source_escape_rejected(self, tmp_path: Path) -> None:
        self._write_source(tmp_path)
        outside = Citation(
            document_id="d",
            chunk_id="c",
            source=str(tmp_path / ".." / "outside.md"),
            start_offset=0,
            end_offset=4,
        )
        with pytest.raises(RetrievalError, match="escapes"):
            asyncio.run(resolve_citation(outside, tmp_path))

    def test_changed_source_detected(self, tmp_path: Path) -> None:
        citation, _ = self._write_source(tmp_path)
        Path(citation.source).write_text("tiny", encoding="utf-8")
        with pytest.raises(RetrievalError, match="source changed"):
            asyncio.run(resolve_citation(citation, tmp_path))

    def test_missing_source_raises(self, tmp_path: Path) -> None:
        citation = Citation(
            document_id="d",
            chunk_id="c",
            source=str(tmp_path / "never.md"),
            start_offset=0,
            end_offset=4,
        )
        with pytest.raises(RetrievalError, match="Cannot read"):
            asyncio.run(resolve_citation(citation, tmp_path))


class TestCitationsInvalidUtf8:
    """A source that becomes invalid UTF-8 after indexing must surface the
    typed ``RetrievalError`` (SPEC.md §2), not a raw ``UnicodeDecodeError`` —
    ``UnicodeDecodeError`` is a ``ValueError`` subclass, not an ``OSError``,
    so a naive ``except OSError`` around the read lets it escape uncaught."""

    def _write_invalid_utf8_source(self, tmp_path: Path) -> Citation:
        path = tmp_path / "invalid.md"
        path.write_bytes(b"\xff\xfe\x00invalid")
        return Citation(
            document_id="d",
            chunk_id="c",
            source=str(path),
            start_offset=0,
            end_offset=4,
        )

    def test_resolve_raises_retrieval_error_not_unicode_decode_error(self, tmp_path: Path) -> None:
        citation = self._write_invalid_utf8_source(tmp_path)
        with pytest.raises(RetrievalError, match="not valid UTF-8") as exc_info:
            asyncio.run(resolve_citation(citation, tmp_path))
        assert isinstance(exc_info.value.__cause__, UnicodeDecodeError)

    def test_verify_raises_retrieval_error_not_unicode_decode_error(self, tmp_path: Path) -> None:
        citation = self._write_invalid_utf8_source(tmp_path)
        with pytest.raises(RetrievalError, match="not valid UTF-8") as exc_info:
            asyncio.run(verify_citation(citation, "anything", tmp_path))
        assert isinstance(exc_info.value.__cause__, UnicodeDecodeError)


# ── Phase 3 Wave C: dense + hybrid retrieval ────────────────────────────────
#
# ADR-0004 (embedding-identity binding), ADR-0005 (RRF fusion scoring), and
# ADR-0006 (the dense seam returns (Chunk, score) pairs, joined once by
# Retriever._resolve) are the authorities for everything below.

#: Vector width for every dense-path fixture in this section — small and
#: readable, exercising the (provider, model_name, dimensions) identity
#: triple's *comparison* logic, never its arithmetic.
_DENSE_DIMS: Final[int] = 8


class _IdentityEmbedder:
    """Fake embedder exposing a caller-chosen ``(provider, model_name, dimensions)``.

    Wraps :class:`InMemoryEmbedder` for its actual vector math and overrides
    only the identity triple ADR-0004's manifest binds to, reproducing
    locally the same "same width, different model" swap
    ``tests/test_indexer.py``'s ``_CountingEmbedder`` makes representable —
    not imported from there, since retrieval tests should not reach into
    another test module's private fixtures.

    Satisfies :class:`EmbeddingProtocol` structurally.
    """

    def __init__(
        self,
        *,
        provider: str = "inmemory",
        model_name: str = "inmemory-hash-v1",
        dimensions: int = _DENSE_DIMS,
    ) -> None:
        self._inner: InMemoryEmbedder = InMemoryEmbedder(dimensions=dimensions)
        self._provider: str = provider
        self._model_name: str = model_name

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await self._inner.embed(texts)


class _ScriptedEmbedder:
    """Fake embedder returning caller-assigned vectors, keyed by exact text.

    Lets a test pin the *dense* ranking exactly, independent of whatever the
    real BM25 ranking for the same corpus turns out to be: the two rankings
    only need to differ for the fusion tests to exercise RRF's
    rank-combination behavior rather than any embedding model's quality. An
    unscripted text is a fixture bug (corpus or query drifted out of sync
    with the vector table), so it raises ``KeyError`` immediately rather
    than embedding it some other way.

    Satisfies :class:`EmbeddingProtocol` structurally.
    """

    def __init__(self, vectors: dict[str, list[float]], *, dimensions: int) -> None:
        self._vectors: dict[str, list[float]] = vectors
        self._dimensions: int = dimensions

    @property
    def provider(self) -> str:
        return "scripted"

    @property
    def model_name(self) -> str:
        return "scripted-v1"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors[text] for text in texts]


async def _dense_ingest(
    docs_dir: Path,
    store: SQLiteMetadataStore,
    embedder: EmbeddingProtocol,
    vector_store: InMemoryVectorStore,
    texts: dict[str, str],
) -> Indexer:
    """Write ``texts`` (filename -> content) under ``docs_dir`` and dense-ingest them.

    Mirrors ``tests/test_indexer.py``'s dense-enabled ``Indexer`` construction
    convention (a real ``FileLoader`` over real files — there is no
    in-memory document injection path for the dense write side). Returns
    the constructed :class:`Indexer` so a test can call ``index_directory``
    again later against the exact same store and vector_store instances
    (the snapshot-staleness tests need this).
    """
    docs_dir.mkdir(parents=True, exist_ok=True)
    for name, text in texts.items():
        (docs_dir / name).write_text(text, encoding="utf-8")
    indexer = Indexer(
        store, FileLoader(allowed_base_dir=docs_dir), embedder=embedder, vector_store=vector_store
    )
    await indexer.index_directory(str(docs_dir))
    return indexer


class TestDenseConfiguration:
    """Pair validation (WAVE_C_INTERFACES.md §2): the dense embedder/
    vector_store pair is both-or-neither, checked in ``__init__`` so direct
    construction is exactly as safe as ``Retriever.open()``, and a supplied
    pair requires ``documents_at_open`` — the snapshot its staleness filter
    is defined over."""

    def test_open_with_embedder_but_no_vector_store_raises_configuration_error(
        self, tmp_path: Path
    ) -> None:
        async def run() -> None:
            store = await _populated_store(tmp_path)
            try:
                with pytest.raises(ConfigurationError, match="vector_store"):
                    await Retriever.open(store, embedder=InMemoryEmbedder(dimensions=_DENSE_DIMS))
            finally:
                await store.close()

        asyncio.run(run())

    def test_open_with_vector_store_but_no_embedder_raises_configuration_error(
        self, tmp_path: Path
    ) -> None:
        async def run() -> None:
            store = await _populated_store(tmp_path)
            try:
                with pytest.raises(ConfigurationError, match="embedder"):
                    await Retriever.open(store, vector_store=InMemoryVectorStore())
            finally:
                await store.close()

        asyncio.run(run())

    def test_direct_construction_with_dense_pair_but_no_snapshot_raises_configuration_error(
        self, tmp_path: Path
    ) -> None:
        """Constructing a ``Retriever`` directly (bypassing ``open()``) with
        both halves of the dense pair but ``documents_at_open=None`` must
        refuse: the snapshot is not optional plumbing, it is what the dense
        staleness filter is defined over."""

        async def run() -> None:
            store = await _populated_store(tmp_path)
            try:
                bm25 = await BM25Index.from_store(store)
                with pytest.raises(ConfigurationError, match="documents_at_open"):
                    Retriever(
                        store,
                        bm25,
                        embedder=InMemoryEmbedder(dimensions=_DENSE_DIMS),
                        vector_store=InMemoryVectorStore(),
                        documents_at_open=None,
                    )
            finally:
                await store.close()

        asyncio.run(run())


class TestManifestVerificationAtOpen:
    """ADR-0004 decision 3's second boundary: ``Retriever.open()`` verifies
    the collection manifest before the BM25 rebuild, closing the read-side
    half Wave B deliberately left open (KNOWN_LIMITATIONS.md)."""

    def test_open_with_mismatched_model_name_same_dimensions_raises_index_identity_error(
        self, tmp_path: Path
    ) -> None:
        """The 768-vs-768 class: two embedders sharing a width but not a
        model name must never both be allowed to open the same collection —
        width alone is not identity (ADR-0004 decision 2). The mismatch
        must surface from ``open()`` itself, before any search runs."""

        async def run() -> None:
            docs_dir = tmp_path / "docs"
            vector_store = InMemoryVectorStore()
            bound_embedder = _IdentityEmbedder(model_name="model-a", dimensions=_DENSE_DIMS)
            store = await SQLiteMetadataStore.open(tmp_path / "idx", "default")
            try:
                await _dense_ingest(
                    docs_dir,
                    store,
                    bound_embedder,
                    vector_store,
                    {"alpha.md": "Retrieval systems rank documents by relevance."},
                )

                swapped_embedder = _IdentityEmbedder(model_name="model-b", dimensions=_DENSE_DIMS)
                with pytest.raises(IndexIdentityError):
                    await Retriever.open(
                        store, embedder=swapped_embedder, vector_store=vector_store
                    )
            finally:
                await store.close()

        asyncio.run(run())

    def test_collection_bound_after_the_identity_check_is_never_treated_as_dense_bound(
        self, tmp_path: Path
    ) -> None:
        """The manifest TOCTOU: one read must decide identity *and* dense-bound.

        ``open()`` used to check identity against one manifest read and then
        decide "is this collection dense-bound" from a second, later one.
        An unbound collection passes the first check trivially, so a dense
        ingest landing in between — another task, or another process, which
        the store's ``asyncio.Lock`` does not span — left this retriever
        answering ``dense`` and ``hybrid`` queries against *another
        provider's* vectors with this embedder's query vectors. Mixed
        embedding spaces, no error: exactly what ADR-0004 exists to make
        unrepresentable.

        The store here fires that concurrent ingest from inside
        ``verify_manifest``, which is precisely the interleaving the two-read
        version could not survive. The requirement is only that the two
        answers cannot disagree, so either outcome is acceptable — refuse
        the search (the collection was unbound when read, which is what this
        asserts) or raise ``IndexIdentityError`` — while returning provider
        B's hits is not.
        """

        async def run() -> None:
            docs_dir = tmp_path / "docs"
            docs_dir.mkdir()
            (docs_dir / "alpha.md").write_text(
                "Retrieval systems rank documents by relevance.", encoding="utf-8"
            )
            vector_store = InMemoryVectorStore()
            embedder_b = _IdentityEmbedder(provider="provider-b", model_name="model-b")
            store = await SQLiteMetadataStore.open(tmp_path / "idx", "default")

            async def concurrent_dense_ingest() -> None:
                indexer = Indexer(
                    store,
                    FileLoader(allowed_base_dir=docs_dir),
                    embedder=embedder_b,
                    vector_store=vector_store,
                )
                await indexer.index_directory(str(docs_dir))

            racing_store = _BindsManifestDuringVerify(store, concurrent_dense_ingest)
            embedder_a = _IdentityEmbedder(provider="provider-a", model_name="model-a")

            try:
                retriever = await Retriever.open(
                    racing_store, embedder=embedder_a, vector_store=vector_store
                )
                # The race really did bind the collection to the other provider.
                bound = await store.get_manifest()
                assert bound is not None
                assert (bound.provider, bound.model_name) == ("provider-b", "model-b")

                modes: tuple[SearchMode, ...] = ("dense", "hybrid")
                for mode in modes:
                    with pytest.raises(ConfigurationError, match="no embedding-identity manifest"):
                        await retriever.search("retrieval", mode=mode)
            finally:
                await store.close()

        asyncio.run(run())


class _BindsManifestDuringVerify:
    """Metadata store that runs a concurrent dense ingest inside ``verify_manifest``.

    Delegates everything to a real :class:`SQLiteMetadataStore`; the only
    behavior it adds is firing ``on_verify`` once, immediately after the
    first ``verify_manifest`` call returns. That is the exact instant a
    second reader/writer can slip between "identity checked" and any later
    manifest read, so it makes the TOCTOU window deterministic instead of
    hoping a real thread race lands in it.

    Satisfies :class:`MetadataStoreProtocol` structurally, by delegation.
    """

    def __init__(
        self, inner: SQLiteMetadataStore, on_verify: Callable[[], Awaitable[None]]
    ) -> None:
        self._inner = inner
        self._on_verify = on_verify
        self._fired = False

    async def verify_manifest(self, identity: EmbeddingIdentity) -> CollectionManifest | None:
        manifest = await self._inner.verify_manifest(identity)
        if not self._fired:
            self._fired = True
            await self._on_verify()
        return manifest

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class TestNoBackfillDenseRead:
    """ADR-0008: a dense or hybrid search refuses a never-dense collection.

    **This class asserted the opposite through Wave C, deliberately
    reversed by ADR-0008.** Wave C held that opening a dense pair over a
    BM25-only collection should succeed and the dense side should then
    "behave as an honestly empty index, never an error". Opening still
    succeeds — that half was right, and a caller may open with a pair and
    search ``"bm25"`` only. What was wrong is what the caller got back from
    the other modes: ``hybrid`` fused one non-empty ranking with an empty
    one and returned *BM25's ordering* stamped ``metadata["stage"] =
    "fusion"``, and ``dense`` returned zero results, which reads as "nothing
    matched" when the truth is "there is no dense index". Both are the
    silent-absence class SPEC.md §2 fails closed against, and the first is
    the same defect shape already closed for an unrecognised ``mode``.
    """

    def test_dense_mode_over_never_dense_ingested_collection_raises(self, tmp_path: Path) -> None:
        """Zero results would read as "nothing matched", not "no dense index"."""

        async def run() -> None:
            store = await _populated_store(tmp_path)
            try:
                retriever = await Retriever.open(
                    store,
                    embedder=InMemoryEmbedder(dimensions=_DENSE_DIMS),
                    vector_store=InMemoryVectorStore(),
                )
                with pytest.raises(ConfigurationError, match="no embedding-identity manifest"):
                    await retriever.search("pasta", mode="dense")
            finally:
                await store.close()

        asyncio.run(run())

    def test_hybrid_mode_over_never_dense_ingested_collection_raises(self, tmp_path: Path) -> None:
        """The sharper half: this used to return BM25 results labelled "fusion"."""

        async def run() -> None:
            store = await _populated_store(tmp_path)
            try:
                retriever = await Retriever.open(
                    store,
                    embedder=InMemoryEmbedder(dimensions=_DENSE_DIMS),
                    vector_store=InMemoryVectorStore(),
                )
                with pytest.raises(ConfigurationError, match="no embedding-identity manifest"):
                    await retriever.search("BM25 lexical ranking", top_k=5, mode="hybrid")
            finally:
                await store.close()

        asyncio.run(run())

    def test_the_error_points_at_the_remedy_that_actually_works(self, tmp_path: Path) -> None:
        """A bare "no manifest" would send the caller to a silent no-op.

        Enabling ``--dense`` over an existing collection backfills nothing —
        the content-hash gate runs before embedding — so the message has to
        name re-ingestion, or it routes the reader to the exact command that
        appears to fix this and does not.
        """

        async def run() -> None:
            store = await _populated_store(tmp_path)
            try:
                retriever = await Retriever.open(
                    store,
                    embedder=InMemoryEmbedder(dimensions=_DENSE_DIMS),
                    vector_store=InMemoryVectorStore(),
                )
                with pytest.raises(ConfigurationError) as excinfo:
                    await retriever.search("pasta", mode="dense")
            finally:
                await store.close()
            message = str(excinfo.value)
            assert "does not backfill" in message
            assert "grk ingest --dense" in message

        asyncio.run(run())

    def test_bm25_mode_still_works_on_a_dense_paired_retriever(self, tmp_path: Path) -> None:
        """ADR-0008 decision 1: the refusal is per-mode, never at ``open()``.

        Opening with a dense pair and searching only ``"bm25"`` is valid and
        must keep working — failing at ``open()`` would have been simpler and
        would have broken this for a mode the caller never used.
        """

        async def run() -> None:
            store = await _populated_store(tmp_path)
            try:
                retriever = await Retriever.open(
                    store,
                    embedder=InMemoryEmbedder(dimensions=_DENSE_DIMS),
                    vector_store=InMemoryVectorStore(),
                )
                response = await retriever.search("BM25 lexical ranking", top_k=5)
            finally:
                await store.close()
            assert response.total_results > 0
            assert response.metadata["stage"] == "bm25"

        asyncio.run(run())


class TestDenseSearch:
    """Dense mode returns citation-bearing results resolved against the
    metadata store's live source mapping (ADR-0006)."""

    def test_dense_mode_returns_citation_bearing_results(self, tmp_path: Path) -> None:
        async def run() -> None:
            docs_dir = tmp_path / "docs"
            embedder = InMemoryEmbedder(dimensions=_DENSE_DIMS)
            vector_store = InMemoryVectorStore()
            store = await SQLiteMetadataStore.open(tmp_path / "idx", "default")
            try:
                await _dense_ingest(
                    docs_dir,
                    store,
                    embedder,
                    vector_store,
                    {
                        "alpha.md": "Retrieval systems rank documents by relevance.",
                        "beta.md": "Cooking pasta requires boiling water and salt.",
                    },
                )
                sources = await store.get_document_sources()

                retriever = await Retriever.open(
                    store, embedder=embedder, vector_store=vector_store
                )
                response = await retriever.search("retrieval ranking", top_k=5, mode="dense")

                assert response.total_results == 2
                assert response.metadata["stage"] == "dense"
                for result in response.results:
                    assert result.source == sources[result.document_id]
                    assert result.citation.chunk_id == result.chunk_id
                    assert result.score >= 0.0
                assert {r.document_id for r in response.results} == set(sources.keys())
            finally:
                await store.close()

        asyncio.run(run())


#: Corpus + query + vectors for the hybrid-fusion tests: dense ranking is
#: fully controlled via `_ScriptedEmbedder`, chosen to be the near-reverse
#: of the lexical overlap so bm25 and dense disagree on ordering.
_HYBRID_DOC_TEXTS: Final[dict[str, str]] = {
    "alpha.md": "sierra tango sierra tango uniform sierra",
    "beta.md": "sierra kilo lima mike november oscar papa",
    "gamma.md": "tango uniform kilo lima mike november",
}
_HYBRID_QUERY: Final[str] = "sierra tango uniform"
_HYBRID_VECTORS: Final[dict[str, list[float]]] = {
    _HYBRID_QUERY: [1.0, 0.0, 0.0],
    _HYBRID_DOC_TEXTS["alpha.md"]: [0.0, 1.0, 0.0],  # orthogonal to the query: sim 0.0
    _HYBRID_DOC_TEXTS["beta.md"]: [1.0, 0.0, 0.0],  # identical to the query: sim 1.0
    _HYBRID_DOC_TEXTS["gamma.md"]: [0.6, 0.8, 0.0],  # sim 0.6
}


async def _open_scripted_hybrid_retriever(
    tmp_path: Path,
) -> tuple[SQLiteMetadataStore, Retriever, EmbeddingProtocol, InMemoryVectorStore]:
    """Dense-ingest ``_HYBRID_DOC_TEXTS`` under a :class:`_ScriptedEmbedder`
    and open a dense-paired retriever over the result.

    Shared by the hybrid-ordering and hybrid-determinism tests, which need
    the identical fully-controlled corpus, just exercised differently.
    """
    docs_dir = tmp_path / "docs"
    embedder = _ScriptedEmbedder(_HYBRID_VECTORS, dimensions=3)
    vector_store = InMemoryVectorStore()
    store = await SQLiteMetadataStore.open(tmp_path / "idx", "default")
    await _dense_ingest(docs_dir, store, embedder, vector_store, _HYBRID_DOC_TEXTS)
    retriever = await Retriever.open(store, embedder=embedder, vector_store=vector_store)
    return store, retriever, embedder, vector_store


class TestHybridSearch:
    """RRF fusion end-to-end: hand-computed ordering, determinism, and the
    threshold razor (ADR-0005, ADR-0006)."""

    def test_hybrid_order_matches_hand_computed_rrf_of_the_two_candidate_lists(
        self, tmp_path: Path
    ) -> None:
        async def run() -> None:
            store, retriever, embedder, vector_store = await _open_scripted_hybrid_retriever(
                tmp_path
            )
            try:
                top_k = 3
                cfg = RetrievalConfig()

                # Independently reproduce the exact candidate lists
                # Retriever.search builds internally for hybrid mode: a
                # fresh BM25 rebuild from the same (unmutated) store, and a
                # direct dense-store search with the same query embedding.
                bm25_index = await BM25Index.from_store(store, k1=cfg.bm25_k1, b=cfg.bm25_b)
                bm25_pairs = bm25_index.search(_HYBRID_QUERY, top_k=top_k)
                query_embedding = (await embedder.embed([_HYBRID_QUERY]))[0]
                dense_pairs = await vector_store.search(query_embedding, top_k=top_k)

                bm25_order = [chunk.chunk_id for chunk, _ in bm25_pairs]
                dense_order = [chunk.chunk_id for chunk, _ in dense_pairs]
                assert bm25_order != dense_order  # the fixture's whole premise

                expected = reciprocal_rank_fusion(
                    [bm25_pairs, dense_pairs], rrf_k=cfg.rrf_k, top_k=top_k
                )
                expected_pairs = [(chunk.chunk_id, score) for chunk, score in expected]

                response = await retriever.search(_HYBRID_QUERY, top_k=top_k, mode="hybrid")
                actual_pairs = [(r.chunk_id, r.score) for r in response.results]

                assert actual_pairs == expected_pairs
                # The fused order must differ from at least one pure
                # ranking (WAVE_C_INTERFACES.md §5) — here, from both.
                fused_order = [chunk_id for chunk_id, _ in actual_pairs]
                assert fused_order != bm25_order
                assert fused_order != dense_order
                assert response.metadata["stage"] == "fusion"
                assert response.metadata["rrf_k"] == cfg.rrf_k
            finally:
                await store.close()

        asyncio.run(run())

    def test_hybrid_search_is_deterministic_across_repeated_calls(self, tmp_path: Path) -> None:
        async def run() -> None:
            store, retriever, _embedder, _vector_store = await _open_scripted_hybrid_retriever(
                tmp_path
            )
            try:
                first = await retriever.search(_HYBRID_QUERY, top_k=3, mode="hybrid")
                second = await retriever.search(_HYBRID_QUERY, top_k=3, mode="hybrid")

                first_pairs = [(r.chunk_id, r.score) for r in first.results]
                second_pairs = [(r.chunk_id, r.score) for r in second.results]
                assert first_pairs == second_pairs
                assert len(first_pairs) > 0
            finally:
                await store.close()

        asyncio.run(run())

    def test_impossible_score_threshold_zeroes_bm25_and_dense_but_not_hybrid(
        self, tmp_path: Path
    ) -> None:
        """ADR-0005 decision 6: an absolute threshold is meaningless against
        rank-derived fused scores, so hybrid must never apply it — proven
        with a threshold no producer score could ever clear."""

        async def run() -> None:
            docs_dir = tmp_path / "docs"
            embedder = InMemoryEmbedder(dimensions=_DENSE_DIMS)
            vector_store = InMemoryVectorStore()
            store = await SQLiteMetadataStore.open(tmp_path / "idx", "default")
            try:
                await _dense_ingest(
                    docs_dir,
                    store,
                    embedder,
                    vector_store,
                    {
                        "alpha.md": "Retrieval systems rank documents by relevance.",
                        "beta.md": "Cooking pasta requires boiling water and salt.",
                    },
                )
                impossible = RetrievalConfig(score_threshold=1_000_000.0)
                retriever = await Retriever.open(
                    store, impossible, embedder=embedder, vector_store=vector_store
                )

                bm25_response = await retriever.search("retrieval relevance", mode="bm25")
                dense_response = await retriever.search("retrieval relevance", mode="dense")
                hybrid_response = await retriever.search("retrieval relevance", mode="hybrid")

                assert bm25_response.total_results == 0
                assert dense_response.total_results == 0
                assert hybrid_response.total_results > 0
            finally:
                await store.close()

        asyncio.run(run())


class TestDenseSnapshotSemantics:
    """ADR-0002 staleness extended to the dense side: a ``Retriever``
    reflects the store only as of ``open()``, enforced on the dense path by
    the ``documents_at_open`` filter rather than by rebuild (the class
    docstring in ``search.py``)."""

    def test_stale_dense_retriever_silently_misses_content_ingested_after_open(
        self, tmp_path: Path
    ) -> None:
        async def run() -> None:
            docs_dir = tmp_path / "docs"
            embedder = InMemoryEmbedder(dimensions=_DENSE_DIMS)
            vector_store = InMemoryVectorStore()
            store = await SQLiteMetadataStore.open(tmp_path / "idx", "default")
            try:
                indexer = await _dense_ingest(
                    docs_dir,
                    store,
                    embedder,
                    vector_store,
                    {"alpha.md": "Retrieval systems rank documents by relevance."},
                )

                stale = await Retriever.open(store, embedder=embedder, vector_store=vector_store)

                (docs_dir / "zephyr.md").write_text(
                    "Zephyr breezes flow through canyon valleys.", encoding="utf-8"
                )
                await indexer.index_directory(str(docs_dir))

                # BM25 has no token overlap with the old corpus, so the
                # lexical path returns nothing at all. The dense path is a
                # nearest-neighbour search: it still returns the *old*
                # document (hash-vector similarity is arbitrary but valid) —
                # what the snapshot filter guarantees is that the post-open
                # document never appears, in any mode, silently.
                bm25_response = await stale.search("zephyr breezes", mode="bm25")
                assert bm25_response.total_results == 0
                dense_response = await stale.search("zephyr breezes", mode="dense")
                hybrid_response = await stale.search("zephyr breezes", mode="hybrid")
                for response in (bm25_response, dense_response, hybrid_response):
                    assert all(not r.source.endswith("zephyr.md") for r in response.results)
                    assert all("Zephyr" not in r.content for r in response.results)

                fresh = await Retriever.open(store, embedder=embedder, vector_store=vector_store)
                fresh_response = await fresh.search("zephyr breezes", mode="dense")
                assert fresh_response.total_results >= 1
                assert any("Zephyr" in r.content for r in fresh_response.results)
            finally:
                await store.close()

        asyncio.run(run())


class TestDenseFailClosed:
    """Both varieties of orphaned-vector residue KNOWN_LIMITATIONS.md
    documents: a document deleted after ``open()`` (still in the snapshot,
    gone from live ``sources``) and a document already gone before
    ``open()`` (absent from both). Both must fail closed rather than emit
    an unverifiable citation."""

    def test_document_deleted_after_open_fails_closed_on_dense_hit(self, tmp_path: Path) -> None:
        async def run() -> None:
            docs_dir = tmp_path / "docs"
            embedder = InMemoryEmbedder(dimensions=_DENSE_DIMS)
            vector_store = InMemoryVectorStore()
            store = await SQLiteMetadataStore.open(tmp_path / "idx", "default")
            try:
                await _dense_ingest(
                    docs_dir,
                    store,
                    embedder,
                    vector_store,
                    {"alpha.md": "Retrieval systems rank documents by relevance."},
                )
                sources = await store.get_document_sources()
                (document_id,) = sources.keys()

                retriever = await Retriever.open(
                    store, embedder=embedder, vector_store=vector_store
                )

                # SQLite-only delete, bypassing the indexer entirely: the
                # document was in documents_at_open, so it survives the
                # dense candidates filter and can only fail in the join.
                await store.delete_document(document_id)

                with pytest.raises(RetrievalError, match="inconsistency"):
                    await retriever.search("retrieval relevance", mode="dense")
            finally:
                await store.close()

        asyncio.run(run())

    def test_document_deleted_before_open_fails_closed_with_orphan_message(
        self, tmp_path: Path
    ) -> None:
        """The KNOWN_LIMITATIONS.md pre-open orphan read path: a fresh
        retriever whose ``documents_at_open`` snapshot never included the
        document at all must still fail closed on its orphaned vectors,
        distinctly named as such in the error."""

        async def run() -> None:
            docs_dir = tmp_path / "docs"
            embedder = InMemoryEmbedder(dimensions=_DENSE_DIMS)
            vector_store = InMemoryVectorStore()
            store = await SQLiteMetadataStore.open(tmp_path / "idx", "default")
            try:
                await _dense_ingest(
                    docs_dir,
                    store,
                    embedder,
                    vector_store,
                    {"alpha.md": "Retrieval systems rank documents by relevance."},
                )
                sources = await store.get_document_sources()
                (document_id,) = sources.keys()

                # SQLite-only delete: leaves the dense vectors behind as
                # orphans (a BM25-only mutation over a vector-bearing
                # collection — KNOWN_LIMITATIONS.md).
                await store.delete_document(document_id)

                fresh = await Retriever.open(store, embedder=embedder, vector_store=vector_store)

                with pytest.raises(RetrievalError, match="orphan"):
                    await fresh.search("retrieval relevance", mode="dense")
            finally:
                await store.close()

        asyncio.run(run())


class TestModeRequiresDensePair:
    """A retriever opened without a dense pair refuses dense/hybrid search
    (WAVE_C_INTERFACES.md §2): ``_require_dense`` is the single gate for
    both modes."""

    @pytest.mark.parametrize("mode", ["dense", "hybrid"])
    def test_dense_or_hybrid_mode_on_bm25_only_retriever_raises_configuration_error(
        self, tmp_path: Path, mode: SearchMode
    ) -> None:
        async def run() -> None:
            store = await _populated_store(tmp_path)
            try:
                retriever = await Retriever.open(store)
                with pytest.raises(ConfigurationError, match="dense"):
                    await retriever.search("pasta", mode=mode)
            finally:
                await store.close()

        asyncio.run(run())


class TestValidationAppliesToEveryMode:
    """Query/top_k validation runs before the mode dispatch in ``search()``,
    so it must reject the same inputs regardless of which mode was
    requested."""

    @pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
    @pytest.mark.parametrize("query", ["", "   "])
    def test_empty_or_whitespace_query_raises_in_every_mode(
        self, tmp_path: Path, query: str, mode: SearchMode
    ) -> None:
        async def run() -> None:
            docs_dir = tmp_path / "docs"
            embedder = InMemoryEmbedder(dimensions=_DENSE_DIMS)
            vector_store = InMemoryVectorStore()
            store = await SQLiteMetadataStore.open(tmp_path / "idx", "default")
            try:
                await _dense_ingest(
                    docs_dir,
                    store,
                    embedder,
                    vector_store,
                    {"alpha.md": "Retrieval systems rank documents by relevance."},
                )
                retriever = await Retriever.open(
                    store, embedder=embedder, vector_store=vector_store
                )
                with pytest.raises(RetrievalError, match="empty"):
                    await retriever.search(query, mode=mode)
            finally:
                await store.close()

        asyncio.run(run())

    @pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
    @pytest.mark.parametrize("bad_k", [0, -1, MAX_TOP_K + 1])
    def test_out_of_range_top_k_raises_in_every_mode(
        self, tmp_path: Path, bad_k: int, mode: SearchMode
    ) -> None:
        async def run() -> None:
            docs_dir = tmp_path / "docs"
            embedder = InMemoryEmbedder(dimensions=_DENSE_DIMS)
            vector_store = InMemoryVectorStore()
            store = await SQLiteMetadataStore.open(tmp_path / "idx", "default")
            try:
                await _dense_ingest(
                    docs_dir,
                    store,
                    embedder,
                    vector_store,
                    {"alpha.md": "Retrieval systems rank documents by relevance."},
                )
                retriever = await Retriever.open(
                    store, embedder=embedder, vector_store=vector_store
                )
                with pytest.raises(RetrievalError, match="top_k"):
                    await retriever.search("retrieval", top_k=bad_k, mode=mode)
            finally:
                await store.close()

        asyncio.run(run())


class _ScriptedVectorStore:
    """Returns a fixed ranking truncated to ``top_k``, recording each ``top_k`` asked for.

    The snapshot filter's behaviour depends on *what order* the live store
    returns hits in, which a hash-vector embedder cannot be made to control.
    Scripting the ranking is the only way to put post-open content above
    eligible content deterministically. ``requested`` exposes the widening.

    Satisfies :class:`~groundkit.index.protocols.VectorStoreProtocol`
    structurally.
    """

    def __init__(self) -> None:
        self.ranked: list[tuple[Chunk, float]] = []
        self.requested: list[int] = []

    async def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        return None

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filter: dict[str, object] | None = None,
    ) -> list[tuple[Chunk, float]]:
        self.requested.append(top_k)
        return self.ranked[:top_k]

    async def delete(self, document_id: str) -> int:
        return 0


def test_post_open_documents_do_not_displace_eligible_dense_results(tmp_path: Path) -> None:
    """Content ingested after open() must be invisible, not obstructive.

    The dense path reads a *live* vector-store handle, so post-open chunks
    occupy ranking slots that the stale in-memory BM25 index does not even
    contain. Fetching exactly top_k and only then dropping them let new
    content push eligible results off the end: the search returns fewer than
    top_k -- here, nothing at all -- while perfectly good pre-open chunks sat
    immediately below the cut. That contradicts the open()-time snapshot
    semantics the dense path claims, since a search over the old corpus
    would have returned them, and it breaks the filter-then-truncate rule
    ``index/dense.py`` already applies to metadata filters.
    """

    async def run() -> None:
        store = await _populated_store(tmp_path)
        pre_open_chunks = await store.get_chunks()
        assert len(pre_open_chunks) == 2

        scripted = _ScriptedVectorStore()
        embedder = InMemoryEmbedder(dimensions=8)
        # Two open()-time guards have to be satisfied before this test can
        # reach the behaviour it is actually about (over-fetch ranking).
        # Bind the manifest, or ADR-0008 refuses dense search on a collection
        # that never had vectors; and seed a ranking, or
        # verify_dense_side_present sees a bound manifest over an empty store
        # and calls it a lost dense side. Both are real guards doing their
        # job — the seed is replaced below with the adversarial ordering.
        await store.write_manifest(
            EmbeddingIdentity(
                provider=embedder.provider,
                model_name=embedder.model_name,
                dimensions=embedder.dimensions,
            )
        )
        scripted.ranked = [(chunk, 1.0) for chunk in pre_open_chunks]
        retriever = await Retriever.open(store, embedder=embedder, vector_store=scripted)

        # Ingested after open(): present in sources, absent from the snapshot.
        post_open_chunks: list[Chunk] = []
        for name in ("gamma.md", "delta.md"):
            doc = Document(source=str(tmp_path / name), content=f"{name} added after open")
            await store.upsert_document(
                source=doc.source, document_id=doc.document_id, content_hash="h-post"
            )
            chunks = _chunks_for(doc)
            await store.add_chunks(chunks, source=doc.source)
            post_open_chunks.extend(chunks)

        # Both post-open chunks outrank everything eligible.
        scripted.ranked = [(chunk, 9.9) for chunk in post_open_chunks] + [
            (chunk, 1.0) for chunk in pre_open_chunks
        ]
        # Drop the open()-time probe verify_dense_side_present made
        # (search(top_k=1)); `requested` below is about the search's own
        # widening, and counting the probe as its first fetch would make the
        # assertion measure the wrong thing.
        scripted.requested.clear()

        response = await retriever.search("anything", top_k=2, mode="dense")

        # Both eligible chunks come back; neither post-open document appears.
        assert len(response.results) == 2
        returned = {result.chunk_id for result in response.results}
        assert returned == {chunk.chunk_id for chunk in pre_open_chunks}

        # It got there by widening the fetch, not by asking for top_k once.
        assert scripted.requested[0] == 2
        assert max(scripted.requested) > 2

    asyncio.run(run())


def test_unknown_search_mode_raises_instead_of_silently_fusing(tmp_path: Path) -> None:
    """An unrecognised mode is a caller bug, not a request for hybrid.

    ``SearchMode`` is a type hint, not a runtime guard. Dispatching the
    default branch to fusion meant a typo'd mode returned fused results
    stamped ``metadata["stage"] = "fusion"`` -- a wrong answer presented as a
    valid one, which SPEC.md §2's fail-closed rule forbids.
    """

    async def run() -> None:
        store = await _populated_store(tmp_path)
        retriever = await Retriever.open(
            store,
            embedder=InMemoryEmbedder(dimensions=8),
            vector_store=InMemoryVectorStore(),
        )

        with pytest.raises(RetrievalError, match="Unknown search mode"):
            await retriever.search("anything", mode="sparse")  # type: ignore[arg-type]

        # The three real modes still dispatch.
        assert (await retriever.search("ranking", mode="bm25")).metadata["stage"] == "bm25"

    asyncio.run(run())

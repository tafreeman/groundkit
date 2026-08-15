"""Persisted ingestion: load -> chunk -> store, with incremental re-index.

This is the wiring ARP never had (ADR-0001 gap #1: a persistence-capable
store existed but no entry point used it). The :class:`Indexer` connects the
loader/chunker to the metadata store and skips sources whose processing
fingerprint is unchanged (ADR-0002 incremental re-index) — content *plus*
the chunker and chunking configuration that decide what rows the content
becomes, so a settings change re-indexes instead of hash-matching its way
into a collection of stale chunks (see :func:`_processing_fingerprint`).

Phase 3 Wave B adds an optional dense write path: constructed with an
embedder *and* a vector store, the same pipeline also embeds every changed
document's chunks and keeps the vector store in lockstep with SQLite —
replacing or pruning a document deletes its vectors in the same logical
operation, the ADR-0004 embedding-identity manifest is verified before any
dense mutation and bound on the first dense write, and the existing
fingerprint gate doubles as the re-embed gate. With neither supplied, the
behaviour is exactly the Phase 1 BM25-only path.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from groundkit.config import ChunkingConfig
from groundkit.contracts import EmbeddingIdentity
from groundkit.errors import ConfigurationError, IngestionError
from groundkit.index.dense import verify_dense_side_present
from groundkit.ingestion.chunking import RecursiveChunker
from groundkit.ingestion.pipeline import DEFAULT_MAX_CONCURRENT, discover_files
from groundkit.utils.path_safety import is_within_base

if TYPE_CHECKING:
    from groundkit.contracts import Chunk, Document
    from groundkit.index.protocols import MetadataStoreProtocol, VectorStoreProtocol
    from groundkit.ingestion.protocols import ChunkerProtocol, LoaderProtocol
    from groundkit.providers.protocols import EmbeddingProtocol

logger = logging.getLogger(__name__)


class IndexReport(BaseModel):
    """Outcome of an indexing run.

    Attributes:
        files_seen: Files considered (matched a supported extension).
        documents_indexed: Documents newly written or replaced.
        documents_skipped: Documents skipped because their processing
            fingerprint — content, chunker, and chunking configuration —
            was unchanged since the last run.
        chunks_written: Total chunks persisted this run.
        documents_pruned: Stored documents deleted because their source no
            longer has anything to index. Two distinct cases feed this
            count: the source vanished from disk entirely (a renamed or
            deleted file — :meth:`Indexer.index_directory` only, detected by
            its post-pass sweep), and the source still exists but its
            content was emptied to nothing or whitespace, so the loader
            yields no documents for it (both :meth:`Indexer.index_directory`
            and :meth:`Indexer.index_source`). No longer always ``0`` for
            :meth:`Indexer.index_source`: it counts the emptied-content case
            for the exact source it was given, but never a vanished one.
        vectors_written: Embedding vectors persisted to the dense store this
            run — one per chunk of every document written through the dense
            path. Always ``0`` on a BM25-only indexer (no embedder/vector
            store configured).
        vectors_deleted: Vectors removed from the dense store this run, by
            document replacement and by both prune cases. Surfaced rather
            than swallowed because ADR-0004 decision 6 makes dense deletion
            verified-by-count, not assumed: the total lets a caller
            reconcile what the dense store reported removing against what
            SQLite deleted. Always ``0`` on a BM25-only indexer.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    files_seen: int = Field(ge=0)
    documents_indexed: int = Field(ge=0)
    documents_skipped: int = Field(ge=0)
    chunks_written: int = Field(ge=0)
    documents_pruned: int = Field(default=0, ge=0)
    vectors_written: int = Field(default=0, ge=0)
    vectors_deleted: int = Field(default=0, ge=0)


@dataclass(frozen=True)
class _SourceOutcome:
    """Per-source tally returned by :meth:`Indexer._process` (module-private).

    Six counters outgrew the positional tuple ``_process`` used to return;
    named fields keep :meth:`Indexer.index_directory`'s aggregation
    self-describing. Mirrors :class:`IndexReport` minus ``files_seen``,
    which only the caller can know.
    """

    documents_indexed: int = 0
    documents_skipped: int = 0
    chunks_written: int = 0
    documents_pruned: int = 0
    vectors_written: int = 0
    vectors_deleted: int = 0


class Indexer:
    """Ingest sources into the persisted index, incrementally.

    Constructed with only a metadata store, this is the Phase 1 BM25-only
    pipeline, unchanged. Constructed additionally with an ``embedder`` and
    a ``vector_store`` (keyword-only, both or neither), every mutation also
    maintains the dense side, under one ordering invariant:

    **The dense store is updated before SQLite commits, in every mutation.
    SQLite may never be ahead of the dense store.**

    The two stores share no transaction, so ordering alone decides which
    residue a failure between them leaves behind. SQLite's ``content_hash``
    is the incremental skip key: were SQLite to commit first and the dense
    write then fail, the document would carry its new hash with no vectors,
    and every later run would hash-skip it — permanently absent from dense
    results with no error ever raised, exactly the silent gap SPEC.md §2
    fail-closed forbids. The residue the chosen order risks instead — the
    dense store ahead of SQLite — is *detectable*: ``Retriever.search``
    raises ``RetrievalError("Index inconsistency")`` for a dense hit whose
    document has no stored source, and an interrupted deletion leaves the
    SQLite row that keeps the document on the next run's prune list, so
    cleanup retries to completion. Prefer the loud residue over the silent
    one. (A BM25-only ``Indexer`` mutating a collection that *does* hold
    vectors cannot honor the dense half at all — it has no handle to the
    vector store — and the residue it can leave, orphaned vectors, is again
    the loud kind, never the silent kind.)

    Args:
        store: The collection's metadata store (durable truth, ADR-0002).
            Typed as ``MetadataStoreProtocol``, not a concrete store: the
            indexer writes through
            :meth:`~groundkit.index.protocols.MetadataStoreProtocol.replace_document`,
            whose one-commit atomicity is part of the protocol contract
            precisely because it is the guarantee the indexer depends on.
        loader: Document loader (containment enforced by the loader itself).
        chunker: Chunker; defaults to :class:`RecursiveChunker`.
        chunking_config: Chunking settings passed to the chunker.
        embedder: Optional embedding provider for the dense write path
            (keyword-only). The collection's ADR-0004 identity triple is
            read straight off this object — ``provider`` / ``model_name`` /
            ``dimensions`` — never off a config object, so the manifest can
            only ever describe the model that actually embedded.
        vector_store: Optional dense vector store (keyword-only), mutated
            in lockstep with the metadata store per the invariant above.

    Raises:
        ConfigurationError: Exactly one of ``embedder`` / ``vector_store``
            was supplied. The pair is inseparable — an embedder with no
            store silently discards every vector it produces, and a store
            with no embedder can never be written to.
    """

    def __init__(
        self,
        store: MetadataStoreProtocol,
        loader: LoaderProtocol,
        chunker: ChunkerProtocol | None = None,
        chunking_config: ChunkingConfig | None = None,
        *,
        embedder: EmbeddingProtocol | None = None,
        vector_store: VectorStoreProtocol | None = None,
    ) -> None:
        # The dense pair is validated at construction, not discovered
        # mid-ingest: half a dense path is always a caller bug, and each
        # half fails differently (see the messages), so name the missing one.
        if embedder is not None and vector_store is None:
            raise ConfigurationError(
                "Indexer was given an embedder but no vector_store. The pair is "
                "inseparable: with no store to receive them, every vector the "
                "embedder produced would be silently discarded. Pass both or neither."
            )
        if vector_store is not None and embedder is None:
            raise ConfigurationError(
                "Indexer was given a vector_store but no embedder. The pair is "
                "inseparable: with nothing to produce vectors, the store could "
                "never be written to. Pass both or neither."
            )
        self._store = store
        self._loader = loader
        self._chunker: ChunkerProtocol
        if chunker is not None:
            self._chunker = chunker
        else:
            self._chunker = RecursiveChunker()
        self._chunking_config = chunking_config
        self._embedder = embedder
        self._vector_store = vector_store
        # True once this instance has successfully bound the collection
        # manifest; see _ensure_manifest for why binding waits for the
        # first real dense write instead of happening here or at run start.
        self._manifest_bound: bool = False

    async def index_source(self, source: str) -> IndexReport:
        """Ingest one file into the store (skipping if unchanged).

        If ``source`` now loads to no content (empty or whitespace-only),
        any document previously stored for it is deleted and counted in
        ``documents_pruned`` — the loader yields ``[]`` in that case, so
        without this the stale document and its chunks would remain
        searchable forever even though the source no longer contains them.
        This is narrower than :meth:`index_directory`'s pruning: it still
        never chases a source that has *vanished* from disk (that requires
        the full set of sources under a root, which only a directory walk
        provides) — it only forgets the exact source it was explicitly given
        once that source's content disappears.

        On a dense-enabled indexer, the collection's embedding-identity
        manifest is verified before anything is loaded, chunked, embedded,
        or deleted — see :meth:`_verify_identity` for why it must come
        first.

        Raises:
            IngestionError: If loading or chunking fails.
            IndexIdentityError: The dense path is enabled and the
                collection is bound to a different embedding identity
                (ADR-0004 — never a re-embed, never a fallback).
            EmbeddingError: A dense-enabled run failed to embed.
            StorageError: If persisting fails.
        """
        await self._verify_identity()
        outcome = await self._process(source)
        return IndexReport(
            files_seen=1,
            documents_indexed=outcome.documents_indexed,
            documents_skipped=outcome.documents_skipped,
            chunks_written=outcome.chunks_written,
            documents_pruned=outcome.documents_pruned,
            vectors_written=outcome.vectors_written,
            vectors_deleted=outcome.vectors_deleted,
        )

    async def index_directory(
        self, source_dir: str, max_concurrent: int = DEFAULT_MAX_CONCURRENT
    ) -> IndexReport:
        """Walk ``source_dir`` and ingest every supported file, incrementally.

        Discovery mirrors ``IngestionPipeline.ingest_directory`` (hidden
        directories skipped, deterministic path order). Loading and chunking
        run with bounded concurrency; store writes are serialized by the
        store itself.

        On a dense-enabled indexer, the collection's embedding-identity
        manifest is verified up front, before discovery even runs — only
        the plain ``max_concurrent`` argument check precedes it. See
        :meth:`_verify_identity` for why it must come first.

        After ingesting, prunes stored documents whose source no longer
        exists on disk (a renamed or deleted file) — otherwise a rename
        leaves both the old and new document rows in the store forever
        (duplicate search results, and a citation that resolves against a
        source that no longer exists). A second, narrower case is pruned
        per-source inside :meth:`_process` itself, before this sweep even
        runs: a file that still exists but whose content was emptied to
        nothing or whitespace yields no documents from the loader, so it
        never appears here as a "missing" source (it was discovered on
        disk just fine) and would otherwise never be pruned at all. Pruning
        is scoped to ``source_dir``: only stored documents whose source
        resolves *under* ``source_dir`` are candidates, so indexing a
        subdirectory never touches documents ingested from outside it (e.g.
        a sibling directory, or a prior :meth:`index_source` call
        elsewhere). :meth:`index_source` never prunes a *vanished* source —
        that requires knowing the full set of sources that should currently
        exist under a root, which only a directory walk provides — but it
        does prune the emptied-content case for the exact source it is
        given, via the same :meth:`_process` path used here. When the dense
        path is enabled, every prune also deletes the document's vectors —
        see :meth:`_delete_document_everywhere` for the ordering and the
        count reconciliation.

        Raises:
            IngestionError: The directory is missing, cannot be walked, or
                any file fails to load/chunk.
            IndexIdentityError: The dense path is enabled and the
                collection is bound to a different embedding identity
                (ADR-0004 — never a re-embed, never a fallback).
            EmbeddingError: A dense-enabled run failed to embed.
            StorageError: If persisting fails.
            ValueError: ``max_concurrent`` is less than 1.
        """
        if max_concurrent < 1:
            raise ValueError(f"max_concurrent must be >= 1, got {max_concurrent}")
        await self._verify_identity()

        root = Path(source_dir)
        exists, is_dir = await asyncio.to_thread(lambda: (root.exists(), root.is_dir()))
        if not exists or not is_dir:
            raise IngestionError(f"Directory not found: {source_dir!r}")

        extensions = tuple(ext.lower() for ext in self._loader.supported_extensions)
        try:
            files = await asyncio.to_thread(discover_files, root, extensions)
        except OSError as exc:
            raise IngestionError(f"Failed to walk directory {source_dir!r}: {exc}") from exc

        semaphore = asyncio.Semaphore(max_concurrent)

        async def _one(path: Path) -> _SourceOutcome:
            async with semaphore:
                return await self._process(str(path))

        # return_exceptions=True: a bare gather propagates the first failure
        # immediately and cancels every in-flight sibling, which can land a
        # CancelledError between a sibling's dense write and its SQLite
        # commit — the one window the ordering invariant above cannot make
        # self-healing, since the interrupted document never gets to record
        # that it is mid-write. Letting every _process finish or fail on its
        # own costs nothing (the failure still propagates, below) and leaves
        # no torn write behind.
        settled = await asyncio.gather(*(_one(path) for path in files), return_exceptions=True)

        outcomes: list[_SourceOutcome] = []
        first_error: BaseException | None = None
        for result in settled:
            if isinstance(result, BaseException):
                # files is path-sorted and gather resolves in argument order,
                # so "first" is deterministic across runs.
                if first_error is None:
                    first_error = result
            else:
                outcomes.append(result)
        if first_error is not None:
            raise first_error

        totals = _aggregate(outcomes)

        # Runs after every _process call has completed, so it can never see
        # (and re-count) a document _process already deleted above.
        missing_pruned, missing_vectors_deleted = await self._prune_missing(root, files)
        pruned = totals.documents_pruned + missing_pruned
        vectors_deleted = totals.vectors_deleted + missing_vectors_deleted
        logger.info(
            "Indexed %s: %d files, %d indexed, %d skipped, %d pruned, %d chunks, "
            "%d vectors written, %d vectors deleted",
            source_dir,
            len(files),
            totals.documents_indexed,
            totals.documents_skipped,
            pruned,
            totals.chunks_written,
            totals.vectors_written,
            vectors_deleted,
        )
        return IndexReport(
            files_seen=len(files),
            documents_indexed=totals.documents_indexed,
            documents_skipped=totals.documents_skipped,
            chunks_written=totals.chunks_written,
            documents_pruned=pruned,
            vectors_written=totals.vectors_written,
            vectors_deleted=vectors_deleted,
        )

    async def _verify_identity(self) -> None:
        """Verify the collection manifest before a dense-enabled run does any work.

        No-op on a BM25-only indexer. On a dense-enabled one this is the
        first thing every run does — before loading, chunking, embedding,
        or deleting — for two reasons: it fails fast rather than after the
        expensive walk-and-embed work, and, decisively, an identity
        mismatch is discovered *before* this run has deleted anybody's
        vectors. Both prune paths and every replacement delete vectors, so
        a misconfigured process must be stopped here, not after it has
        already destroyed part of a healthy collection's dense side on the
        way to its error.

        The same reasoning covers the dense-side integrity check that
        follows it: a collection that kept its SQLite file but lost its
        vectors must be refused *before* this run hash-skips every document
        and writes nothing, not after. Identity is checked first — a
        mismatched embedder is a misconfiguration, and reporting a missing
        dense side to a process that was never entitled to touch this
        collection would name the wrong problem.

        Raises:
            IndexIdentityError: The collection is bound to a different
                embedding identity (never a re-embed, never a fallback).
            StorageError: The collection is manifest-bound and holds
                documents, but its vector store is empty — see
                :func:`~groundkit.index.dense.verify_dense_side_present`.
        """
        if self._embedder is None:
            return
        manifest = await self._store.verify_manifest(_identity_of(self._embedder))
        if self._vector_store is not None:
            await verify_dense_side_present(
                self._store, self._vector_store, self._embedder.dimensions, manifest=manifest
            )

    async def _ensure_manifest(self, embedder: EmbeddingProtocol) -> None:
        """Bind the collection to this embedder's identity on the first real write.

        ADR-0004 writes the manifest on the collection's *first dense
        write*. Binding on intent instead — at construction or at run start
        — would let an ingest that turns out to be a complete no-op (every
        document hash-skipped) bind the collection to an identity that
        never produced a single vector. So this is called immediately
        before every ``vector_store.add``, and flag-guarded so it costs one
        store call per indexer instance rather than one per document: the
        identity is fixed for the instance's lifetime and the manifest is
        immutable once written, so after one successful bind there is
        nothing left to write — including on later runs of the same
        instance, whose run-start :meth:`_verify_identity` already proved
        the stored manifest matches. The flag is set only after the write
        succeeds, so a failed bind is retried on the next dense write. Two
        documents racing to be the run's first write may both reach the
        store call, which is safe without a lock: ``write_manifest`` is a
        no-op for a matching triple, so the duplicate is absorbed.
        """
        if self._manifest_bound:
            return
        await self._store.write_manifest(_identity_of(embedder))
        self._manifest_bound = True

    async def _persist_document(
        self, doc: Document, doc_hash: str, chunks: list[Chunk]
    ) -> tuple[int, int]:
        """Write one changed document to every store — dense first, SQLite last.

        The class ordering invariant applied to the replace path.
        ``replace_document`` — the SQLite commit that installs the new
        ``content_hash`` — must come last: committed first, a subsequent
        dense failure would leave the new hash with no vectors, and every
        later run would hash-skip the document (a permanent, silent dense
        gap). The dense-first residue — vectors for a commit that never
        landed — is loud instead: ``Retriever.search`` raises
        ``RetrievalError`` on a dense hit it cannot join to a stored
        document, and the un-advanced hash means the next run redoes the
        whole document.

        Within the dense half the order is also load-bearing:

        - The old document id is read *before* ``replace_document`` runs,
          because ``replace_document`` destroys the row
          ``get_document_id`` reads it from.
        - The new vectors are added *before* the old ones are deleted.
          Deleting first opens a window in which the document has no
          vectors at all: a crash there, followed by a content reversion
          (``git checkout``), leaves SQLite's ``content_hash`` matching
          the restored bytes, so every later run hash-skips the document
          and it stays silently absent from dense results forever. The
          add-first residue is the benign one — both versions' vectors
          present, the un-advanced hash guaranteeing the next run redoes
          the document and deletes the stale set.
        - The manifest is bound after embedding succeeds and before the
          first ``add`` (see :meth:`_ensure_manifest`).

        Replace-path reconciliation: unlike the prune paths, there is no
        chunk-delete count to reconcile the dense delete against —
        ``replace_document`` returns ``None``, and the previous version's
        chunk count is not obtainable without inventing an extra store
        query, which ADR-0004 decision 6 does not justify. The
        deleted-vector count is therefore surfaced in the report totals,
        and per-document reconciliation happens only on the prune paths,
        where ``delete_document`` does return the chunk count.

        Returns:
            ``(vectors_written, vectors_deleted)`` — both ``0`` on the
            BM25-only path.
        """
        vectors_written = vectors_deleted = 0
        embedder = self._embedder
        vector_store = self._vector_store
        if embedder is not None and vector_store is not None:
            embeddings = await embedder.embed([chunk.content for chunk in chunks])
            await self._ensure_manifest(embedder)
            old_id = await self._store.get_document_id(doc.source)
            if old_id is not None and old_id == doc.document_id:
                # Ids collide, so rows added under the new id are
                # indistinguishable from the old ones and the delete cannot
                # come second — it would take the new vectors with it. Not
                # reachable while Document.document_id defaults to a fresh
                # uuid4 per load (contracts.py), but a content-derived id
                # would make it the normal path, and the failure it guards
                # against is silent data loss rather than an error.
                vectors_deleted = await vector_store.delete(old_id)
                await vector_store.add(chunks, embeddings)
            else:
                await vector_store.add(chunks, embeddings)
                if old_id is not None:
                    vectors_deleted = await vector_store.delete(old_id)
            vectors_written = len(embeddings)
        await self._store.replace_document(
            source=doc.source,
            document_id=doc.document_id,
            content_hash=doc_hash,
            chunks=chunks,
        )
        return vectors_written, vectors_deleted

    async def _delete_document_everywhere(self, document_id: str) -> tuple[int, int]:
        """Delete one document's vectors, then its SQLite row, reconciling counts.

        The class ordering invariant applied to deletion, where it earns
        its keep a second way: the prune sweep builds its candidate list
        from SQLite (``get_document_sources``), so deleting the SQLite row
        first and then failing the dense delete would remove the document
        from the very list that could ever retry the cleanup — stranding
        its vectors as permanent orphans that turn every dense hit on them
        into a ``RetrievalError``. Dense-first instead leaves the SQLite
        row (and with it the document's prune candidacy) intact on failure,
        so the next run simply retries the deletion to completion.

        Reconciliation (ADR-0004 decision 6: deletion is verified by
        count, not assumed) is a logged warning naming both counts —
        deliberately not a raised error: a collection first ingested
        BM25-only and later worked on with the dense path enabled
        legitimately deletes N chunks and 0 vectors, because its documents
        never had vectors, so strict equality would fail a completely
        healthy upgrade. No reconciliation is attempted on a BM25-only
        indexer: no dense delete happened, so there is no count to verify.

        Returns:
            ``(chunks_deleted, vectors_deleted)``.
        """
        vector_store = self._vector_store
        vectors_deleted = 0
        if vector_store is not None:
            vectors_deleted = await vector_store.delete(document_id)
        chunks_deleted = await self._store.delete_document(document_id)
        if vector_store is not None and vectors_deleted != chunks_deleted:
            logger.warning(
                "Delete count mismatch for document %s: dense store removed %d "
                "vectors, SQLite removed %d chunks. Benign when the document "
                "predates the dense path (it never had vectors); otherwise the "
                "two stores had already drifted.",
                document_id,
                vectors_deleted,
                chunks_deleted,
            )
        return chunks_deleted, vectors_deleted

    async def _prune_missing(self, root: Path, files: list[Path]) -> tuple[int, int]:
        """Delete stored documents under ``root`` whose source no longer exists on disk.

        Each deletion goes through :meth:`_delete_document_everywhere`, so
        a pruned document's vectors are removed first and reconciled by
        count (ADR-0004 decision 6) — the sweep that forgets a document in
        SQLite must not strand its dense rows.

        Args:
            root: The directory just walked — the prune scope. A stored
                document is only a deletion candidate when its source
                resolves under ``root``.
            files: The files just discovered under ``root``, i.e. the
                current, authoritative set of sources that should exist.

        Returns:
            ``(documents_pruned, vectors_deleted)``: the number of
            documents pruned, and the vectors deleted with them (always
            ``0`` on a BM25-only indexer).
        """

        def _current_sources() -> set[str]:
            return {os.path.realpath(str(path)) for path in files}

        current = await asyncio.to_thread(_current_sources)
        sources = await self._store.get_document_sources()

        pruned = 0
        vectors_deleted = 0
        for document_id, source in sources.items():
            if source in current or not is_within_base(source, root):
                continue
            _, vectors = await self._delete_document_everywhere(document_id)
            pruned += 1
            vectors_deleted += vectors
        return pruned, vectors_deleted

    async def _process(self, source: str) -> _SourceOutcome:
        """Load, hash-compare, chunk, embed (dense path), and persist one source.

        When the loader returns no documents for ``source`` (empty or
        whitespace-only content), any document previously stored for that
        exact source is deleted via :meth:`_prune_emptied_source` — the
        ``for doc in documents`` loop below would otherwise never execute,
        so the stale document and its chunks would remain in the store
        forever even though the source no longer contains them.

        The fingerprint gate (:func:`_processing_fingerprint`) is also the
        dense path's re-embed gate: an unchanged document ``continue``s
        before chunking, so it is never embedded either. Incremental
        re-embedding is therefore a property of the one existing skip, not a
        second mechanism that could drift from it — which is also why the
        fingerprint has to cover chunking settings. A chunk-boundary change
        that skipped here would have kept the *old* vectors as well, since
        they are written per chunk.

        Returns:
            A :class:`_SourceOutcome` tallying this source's documents,
            chunks, and (dense path only) vectors.
        """
        try:
            documents = await self._loader.load(source)
        except IngestionError:
            raise
        except Exception as exc:
            raise IngestionError(f"Loader failed for {source!r}: {exc}") from exc

        if not documents:
            pruned, vectors_deleted = await self._prune_emptied_source(source)
            return _SourceOutcome(documents_pruned=pruned, vectors_deleted=vectors_deleted)

        indexed = skipped = chunks_written = vectors_written = vectors_deleted = 0
        for doc in documents:
            doc_hash = _processing_fingerprint(doc, self._chunker, self._chunking_config)
            stored = await self._store.get_document_hash(doc.source)
            if stored == doc_hash:
                logger.debug("Unchanged, skipping: %s", doc.source)
                skipped += 1
                continue

            try:
                chunks = self._chunker.chunk(doc, config=self._chunking_config)
            except Exception as exc:
                raise IngestionError(
                    f"Chunking failed for document {doc.document_id}: {exc}"
                ) from exc

            written, deleted = await self._persist_document(doc, doc_hash, chunks)
            indexed += 1
            chunks_written += len(chunks)
            vectors_written += written
            vectors_deleted += deleted
        return _SourceOutcome(
            documents_indexed=indexed,
            documents_skipped=skipped,
            chunks_written=chunks_written,
            vectors_written=vectors_written,
            vectors_deleted=vectors_deleted,
        )

    async def _prune_emptied_source(self, source: str) -> tuple[int, int]:
        """Delete the stored document for ``source`` if one exists.

        Called only when the loader just returned no documents for
        ``source`` (empty or whitespace-only content) — the file still
        exists on disk, so it is not a "missing source" :meth:`_prune_missing`
        would ever catch, but it now has nothing to index. Deletion goes
        through :meth:`_delete_document_everywhere`: vectors first, SQLite
        second, counts reconciled (ADR-0004 decision 6).

        Stored sources are realpath-normalized (:class:`FileLoader` resolves
        every path via
        :func:`~groundkit.utils.path_safety.ensure_within_base` before
        storing it), so ``source`` must be realpath-resolved the same way
        before comparing — done off the event loop, consistent with every
        other blocking filesystem call in this module.

        Args:
            source: The path this run was asked to index, as given by the
                caller (relative or absolute, not yet resolved).

        Returns:
            ``(1, vectors_deleted)`` if a stored document was found and
            deleted — ``vectors_deleted`` being the dense rows removed with
            it — or ``(0, 0)`` if ``source`` was never indexed (nothing to
            prune).
        """
        resolved = await asyncio.to_thread(os.path.realpath, source)
        sources = await self._store.get_document_sources()
        for document_id, stored_source in sources.items():
            if stored_source == resolved:
                _, vectors_deleted = await self._delete_document_everywhere(document_id)
                logger.info("Emptied source, pruning stored document: %s", source)
                return 1, vectors_deleted
        return 0, 0


def _identity_of(embedder: EmbeddingProtocol) -> EmbeddingIdentity:
    """Read the ADR-0004 identity triple off the embedder itself.

    Never derived from a config object: sourcing all three fields from the
    object that actually produces the vectors makes "the manifest describes
    a different model than the one that embedded" unrepresentable, rather
    than a divergence a caller has to be trusted not to introduce.
    """
    return EmbeddingIdentity(
        provider=embedder.provider,
        model_name=embedder.model_name,
        dimensions=embedder.dimensions,
    )


def _aggregate(outcomes: list[_SourceOutcome]) -> _SourceOutcome:
    """Sum per-source outcomes into a new combined outcome (inputs untouched)."""
    return _SourceOutcome(
        documents_indexed=sum(o.documents_indexed for o in outcomes),
        documents_skipped=sum(o.documents_skipped for o in outcomes),
        chunks_written=sum(o.chunks_written for o in outcomes),
        documents_pruned=sum(o.documents_pruned for o in outcomes),
        vectors_written=sum(o.vectors_written for o in outcomes),
        vectors_deleted=sum(o.vectors_deleted for o in outcomes),
    )


def _processing_fingerprint(
    document: Document, chunker: ChunkerProtocol, chunking_config: ChunkingConfig | None
) -> str:
    """SHA-256 over every input that decides a document's stored chunks.

    The incremental re-index skip key, and deliberately *not* a hash of the
    content alone. Content-only, the skip answered "have these bytes
    changed?" when the question it is actually asked is "would re-processing
    this source produce different rows?" — so re-ingesting with a different
    ``ChunkingConfig`` (or a different chunker) hash-matched, skipped, and
    left the collection holding chunks built to the *previous* configuration
    with nothing anywhere recording that. Silent, permanent, and invisible
    to every later run: exactly the shape SPEC.md §2's fail-closed rule
    exists to keep out. It is also the dense path's re-embed gate, so those
    stale chunks kept their stale vectors too.

    Three inputs, all of them things the indexer can actually see:

    - the document content;
    - the chunking configuration, normalized through ``ChunkingConfig()``
      so a caller passing ``None`` and a caller passing the defaults
      explicitly agree — ``RecursiveChunker`` resolves ``None`` to exactly
      that, and a fingerprint that disagreed would re-index a corpus for a
      no-op change;
    - the chunker's type name, because swapping the chunker changes the
      output as surely as changing its settings does. Type name rather than
      anything richer: ``ChunkerProtocol`` exposes no identity, and a name
      is the strongest signal available without widening that seam.

    The consequence on an existing collection is one full re-index the
    first time this runs — the stored hashes were computed the old way, so
    nothing matches. That is correct rather than merely tolerable: those
    chunks were produced under a configuration this store never recorded,
    and re-deriving them is seconds of work (ADR-0004 decision 5's
    reasoning, applied to chunks rather than vectors).

    Args:
        document: The freshly loaded document.
        chunker: The chunker that will split it.
        chunking_config: The configuration it will be split under, or
            ``None`` to accept the chunker's defaults.

    Returns:
        Hex-encoded SHA-256 digest over all three, domain-separated by NULs
        so no concatenation of one field can impersonate another.
    """
    config = chunking_config if chunking_config is not None else ChunkingConfig()
    hasher = hashlib.sha256()
    for part in (
        document.content.encode(),
        type(chunker).__qualname__.encode(),
        config.model_dump_json().encode(),
    ):
        hasher.update(part)
        hasher.update(b"\0")
    return hasher.hexdigest()

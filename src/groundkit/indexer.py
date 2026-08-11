"""Persisted ingestion: load -> chunk -> store, with incremental re-index.

This is the wiring ARP never had (ADR-0001 gap #1: a persistence-capable
store existed but no entry point used it). The :class:`Indexer` connects the
loader/chunker to the metadata store and skips sources whose content hash is
unchanged (ADR-0002 incremental re-index).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from groundkit.errors import IngestionError
from groundkit.ingestion.chunking import RecursiveChunker
from groundkit.ingestion.pipeline import DEFAULT_MAX_CONCURRENT, discover_files
from groundkit.utils.path_safety import is_within_base

if TYPE_CHECKING:
    from groundkit.config import ChunkingConfig
    from groundkit.contracts import Document
    from groundkit.index.protocols import MetadataStoreProtocol
    from groundkit.ingestion.protocols import ChunkerProtocol, LoaderProtocol

logger = logging.getLogger(__name__)


class IndexReport(BaseModel):
    """Outcome of an indexing run.

    Attributes:
        files_seen: Files considered (matched a supported extension).
        documents_indexed: Documents newly written or replaced.
        documents_skipped: Documents skipped because their content hash was
            unchanged since the last run.
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
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    files_seen: int = Field(ge=0)
    documents_indexed: int = Field(ge=0)
    documents_skipped: int = Field(ge=0)
    chunks_written: int = Field(ge=0)
    documents_pruned: int = Field(default=0, ge=0)


class Indexer:
    """Ingest sources into the persisted index, incrementally.

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
    """

    def __init__(
        self,
        store: MetadataStoreProtocol,
        loader: LoaderProtocol,
        chunker: ChunkerProtocol | None = None,
        chunking_config: ChunkingConfig | None = None,
    ) -> None:
        self._store = store
        self._loader = loader
        self._chunker: ChunkerProtocol
        if chunker is not None:
            self._chunker = chunker
        else:
            self._chunker = RecursiveChunker()
        self._chunking_config = chunking_config

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

        Raises:
            IngestionError: If loading or chunking fails.
            StorageError: If persisting fails.
        """
        indexed, skipped, chunks_written, pruned = await self._process(source)
        return IndexReport(
            files_seen=1,
            documents_indexed=indexed,
            documents_skipped=skipped,
            chunks_written=chunks_written,
            documents_pruned=pruned,
        )

    async def index_directory(
        self, source_dir: str, max_concurrent: int = DEFAULT_MAX_CONCURRENT
    ) -> IndexReport:
        """Walk ``source_dir`` and ingest every supported file, incrementally.

        Discovery mirrors ``IngestionPipeline.ingest_directory`` (hidden
        directories skipped, deterministic path order). Loading and chunking
        run with bounded concurrency; store writes are serialized by the
        store itself.

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
        given, via the same :meth:`_process` path used here.

        Raises:
            IngestionError: The directory is missing, cannot be walked, or
                any file fails to load/chunk.
            StorageError: If persisting fails.
            ValueError: ``max_concurrent`` is less than 1.
        """
        if max_concurrent < 1:
            raise ValueError(f"max_concurrent must be >= 1, got {max_concurrent}")

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

        async def _one(path: Path) -> tuple[int, int, int, int]:
            async with semaphore:
                return await self._process(str(path))

        outcomes = await asyncio.gather(*(_one(path) for path in files))

        indexed = sum(o[0] for o in outcomes)
        skipped = sum(o[1] for o in outcomes)
        chunks_written = sum(o[2] for o in outcomes)
        emptied_pruned = sum(o[3] for o in outcomes)
        # Runs after every _process call has completed, so it can never see
        # (and re-count) a document _process already deleted above.
        missing_pruned = await self._prune_missing(root, files)
        pruned = emptied_pruned + missing_pruned
        logger.info(
            "Indexed %s: %d files, %d indexed, %d skipped, %d pruned, %d chunks",
            source_dir,
            len(files),
            indexed,
            skipped,
            pruned,
            chunks_written,
        )
        return IndexReport(
            files_seen=len(files),
            documents_indexed=indexed,
            documents_skipped=skipped,
            chunks_written=chunks_written,
            documents_pruned=pruned,
        )

    async def _prune_missing(self, root: Path, files: list[Path]) -> int:
        """Delete stored documents under ``root`` whose source no longer exists on disk.

        Args:
            root: The directory just walked — the prune scope. A stored
                document is only a deletion candidate when its source
                resolves under ``root``.
            files: The files just discovered under ``root``, i.e. the
                current, authoritative set of sources that should exist.

        Returns:
            The number of documents pruned.
        """

        def _current_sources() -> set[str]:
            return {os.path.realpath(str(path)) for path in files}

        current = await asyncio.to_thread(_current_sources)
        sources = await self._store.get_document_sources()

        pruned = 0
        for document_id, source in sources.items():
            if source in current or not is_within_base(source, root):
                continue
            await self._store.delete_document(document_id)
            pruned += 1
        return pruned

    async def _process(self, source: str) -> tuple[int, int, int, int]:
        """Load, hash-compare, chunk, and persist one source.

        When the loader returns no documents for ``source`` (empty or
        whitespace-only content), any document previously stored for that
        exact source is deleted via :meth:`_prune_emptied_source` — the
        ``for doc in documents`` loop below would otherwise never execute,
        so the stale document and its chunks would remain in the store
        forever even though the source no longer contains them.

        Returns:
            ``(documents_indexed, documents_skipped, chunks_written,
            documents_pruned)``.
        """
        try:
            documents = await self._loader.load(source)
        except IngestionError:
            raise
        except Exception as exc:
            raise IngestionError(f"Loader failed for {source!r}: {exc}") from exc

        if not documents:
            pruned = await self._prune_emptied_source(source)
            return 0, 0, 0, pruned

        indexed = skipped = chunks_written = 0
        for doc in documents:
            doc_hash = _content_hash(doc)
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

            await self._store.replace_document(
                source=doc.source,
                document_id=doc.document_id,
                content_hash=doc_hash,
                chunks=chunks,
            )
            indexed += 1
            chunks_written += len(chunks)
        return indexed, skipped, chunks_written, 0

    async def _prune_emptied_source(self, source: str) -> int:
        """Delete the stored document for ``source`` if one exists.

        Called only when the loader just returned no documents for
        ``source`` (empty or whitespace-only content) — the file still
        exists on disk, so it is not a "missing source" :meth:`_prune_missing`
        would ever catch, but it now has nothing to index.

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
            ``1`` if a stored document was found and deleted, ``0`` if
            ``source`` was never indexed (nothing to prune).
        """
        resolved = await asyncio.to_thread(os.path.realpath, source)
        sources = await self._store.get_document_sources()
        for document_id, stored_source in sources.items():
            if stored_source == resolved:
                await self._store.delete_document(document_id)
                logger.info("Emptied source, pruning stored document: %s", source)
                return 1
        return 0


def _content_hash(document: Document) -> str:
    """SHA-256 of the document content — the incremental re-index skip key."""
    return hashlib.sha256(document.content.encode()).hexdigest()

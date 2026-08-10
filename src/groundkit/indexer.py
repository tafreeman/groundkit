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
            longer exists on disk under the indexed root. Always ``0`` for
            :meth:`Indexer.index_source` — pruning only ever runs for
            :meth:`Indexer.index_directory`.
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

        Raises:
            IngestionError: If loading or chunking fails.
            StorageError: If persisting fails.
        """
        indexed, skipped, chunks_written = await self._process(source)
        return IndexReport(
            files_seen=1,
            documents_indexed=indexed,
            documents_skipped=skipped,
            chunks_written=chunks_written,
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
        source that no longer exists). Pruning is scoped to ``source_dir``:
        only stored documents whose source resolves *under* ``source_dir``
        are candidates, so indexing a subdirectory never touches documents
        ingested from outside it (e.g. a sibling directory, or a prior
        :meth:`index_source` call elsewhere). :meth:`index_source` never
        prunes anything — pruning requires knowing the full set of sources
        that should currently exist under a root, which only a directory
        walk provides.

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

        async def _one(path: Path) -> tuple[int, int, int]:
            async with semaphore:
                return await self._process(str(path))

        outcomes = await asyncio.gather(*(_one(path) for path in files))

        indexed = sum(o[0] for o in outcomes)
        skipped = sum(o[1] for o in outcomes)
        chunks_written = sum(o[2] for o in outcomes)
        pruned = await self._prune_missing(root, files)
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

    async def _process(self, source: str) -> tuple[int, int, int]:
        """Load, hash-compare, chunk, and persist one source.

        Returns:
            ``(documents_indexed, documents_skipped, chunks_written)``.
        """
        try:
            documents = await self._loader.load(source)
        except IngestionError:
            raise
        except Exception as exc:
            raise IngestionError(f"Loader failed for {source!r}: {exc}") from exc

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
        return indexed, skipped, chunks_written


def _content_hash(document: Document) -> str:
    """SHA-256 of the document content — the incremental re-index skip key."""
    return hashlib.sha256(document.content.encode()).hexdigest()

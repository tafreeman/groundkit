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
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from groundkit.errors import IngestionError
from groundkit.ingestion.chunking import RecursiveChunker
from groundkit.ingestion.pipeline import DEFAULT_MAX_CONCURRENT, discover_files

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
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    files_seen: int = Field(ge=0)
    documents_indexed: int = Field(ge=0)
    documents_skipped: int = Field(ge=0)
    chunks_written: int = Field(ge=0)


class Indexer:
    """Ingest sources into the persisted index, incrementally.

    Args:
        store: The collection's metadata store (durable truth, ADR-0002).
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
        logger.info(
            "Indexed %s: %d files, %d indexed, %d skipped, %d chunks",
            source_dir,
            len(files),
            indexed,
            skipped,
            chunks_written,
        )
        return IndexReport(
            files_seen=len(files),
            documents_indexed=indexed,
            documents_skipped=skipped,
            chunks_written=chunks_written,
        )

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

            await self._store.upsert_document(
                source=doc.source, document_id=doc.document_id, content_hash=doc_hash
            )
            await self._store.add_chunks(chunks, source=doc.source)
            indexed += 1
            chunks_written += len(chunks)
        return indexed, skipped, chunks_written


def _content_hash(document: Document) -> str:
    """SHA-256 of the document content — the incremental re-index skip key."""
    return hashlib.sha256(document.content.encode()).hexdigest()

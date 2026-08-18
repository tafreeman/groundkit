"""Ingestion pipeline — orchestrates load -> chunk, single-source and directory-scale.

Ports ARP's ``agentic_v2/rag/ingestion.py`` (promote decision, ADR-0001) and
adds the directory-scale entry point ARP never had (ADR-0001 gap #2):
``ingest()`` alone cannot meet the "directory-scale ingestion" v1
requirement (SPEC.md §4).

**This is not the ingest path** (ADR-0010). It loads and chunks; it never
touches a collection. Writing to a collection is
:class:`~groundkit.indexer.Indexer`'s alone, and the two must not be merged:
``Indexer`` skips an unchanged document on its processing fingerprint
*before* chunking (ADR-0009), and that same gate is what keeps an unchanged
document from being re-embedded. :meth:`IngestionPipeline.ingest` always
chunks, so composing it into the indexer would move chunking ahead of the
gate and re-embed an unchanged corpus on every run — billable against a
hosted provider, and silent: nothing would fail, the collection would stay
correct, and only the bill and the wall-clock would move. The resemblance
between the two modules is therefore deliberate and bounded to
:func:`discover_files` and :data:`DEFAULT_MAX_CONCURRENT`, which find files
rather than process them.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from groundkit.config import ChunkingConfig
from groundkit.contracts import Chunk
from groundkit.errors import IngestionError
from groundkit.ingestion.chunking import RecursiveChunker
from groundkit.ingestion.protocols import ChunkerProtocol, LoaderProtocol
from groundkit.utils.url_safety import sanitize_url

logger = logging.getLogger(__name__)

#: Default bound on concurrently in-flight file ingestions for ``ingest_directory``.
DEFAULT_MAX_CONCURRENT: int = 4


class IngestionPipeline:
    """Orchestrate document loading and chunking.

    Containment for directory ingestion is delegated entirely to the loader:
    ``ingest_directory`` only discovers candidate paths by extension and
    dispatches each through the same ``loader.load()`` used by
    :meth:`ingest`, which already enforces the loader's own
    ``allowed_base_dir`` containment before touching the filesystem. The
    pipeline never inspects or duplicates that check itself — there is no
    ``allowed_base_dir`` on :class:`~groundkit.ingestion.protocols.LoaderProtocol`,
    by design, so any loader implementation's own containment rule is what
    governs, not a pipeline-level assumption about paths.

    Usage::

        pipeline = IngestionPipeline(
            loader=FileLoader(allowed_base_dir=Path("./docs")),
            chunker=RecursiveChunker(),
        )
        chunks = await pipeline.ingest("docs/README.md")
        all_chunks = await pipeline.ingest_directory("docs")
    """

    def __init__(
        self,
        loader: LoaderProtocol,
        chunker: ChunkerProtocol | None = None,
        chunking_config: ChunkingConfig | None = None,
    ) -> None:
        self._loader = loader
        self._chunker: ChunkerProtocol
        if chunker is not None:
            self._chunker = chunker
        else:
            self._chunker = RecursiveChunker()
        self._chunking_config = chunking_config

    async def ingest(self, source: str) -> list[Chunk]:
        """Load and chunk a single source.

        Args:
            source: Path or locator for the document, passed to the loader.

        Returns:
            Ordered list of :class:`Chunk` objects.

        Raises:
            IngestionError: If loading or chunking fails. Loader-raised
                :class:`IngestionError` propagates unchanged; any other
                exception from the loader or chunker is wrapped with the
                sanitized source (loader failures) or document ID (chunking
                failures) for context.

        Notes:
            ``source`` is caller-supplied and reaches this method *before* any
            loader has looked at it, so every log line and exception message
            below carries ``safe_source`` rather than the raw value. This class
            is exported public API: ``IngestionPipeline(UrlLoader(...))`` is a
            supported construction, and it used to log an unredacted
            ``https://user:password@host/...`` at INFO before ``UrlLoader``
            reached its own userinfo and credential-query refusals -- ADR-0001
            hazard 6 in the one call site of this module family that had not
            adopted the discipline. ``sanitize_url`` also redacts every query
            *value* unconditionally, which the loader's credential-shaped-key
            refusal deliberately does not, so the sanitized form stays the
            right thing to log even after validation has passed. It leaves a
            filesystem path intact, which is the other thing ``source`` can be.
        """
        safe_source = sanitize_url(source)
        logger.info("Ingesting source: %s", safe_source)

        try:
            documents = await self._loader.load(source)
        except IngestionError:
            raise
        except Exception as exc:
            raise IngestionError(f"Loader failed for {safe_source!r}: {exc}") from exc

        all_chunks: list[Chunk] = []
        for doc in documents:
            try:
                chunks = self._chunker.chunk(doc, config=self._chunking_config)
            except Exception as exc:
                raise IngestionError(
                    f"Chunking failed for document {doc.document_id}: {exc}"
                ) from exc
            all_chunks.extend(chunks)
            logger.debug(
                "Chunked %s: %d chunks from document %s",
                safe_source,
                len(chunks),
                doc.document_id,
            )

        logger.info(
            "Ingestion complete: %s -> %d documents -> %d chunks",
            safe_source,
            len(documents),
            len(all_chunks),
        )
        return all_chunks

    async def ingest_directory(
        self, source_dir: str, max_concurrent: int = DEFAULT_MAX_CONCURRENT
    ) -> list[Chunk]:
        """Walk a directory tree and ingest every file the loader supports.

        Files are discovered by matching ``loader.supported_extensions``
        (case-insensitively) and dispatched in deterministic path-sorted
        order with bounded concurrency; the returned chunk list preserves
        that order (``asyncio.gather`` resolves in argument order regardless
        of completion order).

        Args:
            source_dir: Directory to walk. Hidden subdirectories (name
                starting with ``.``) are skipped entirely.
            max_concurrent: Maximum number of files ingested concurrently.

        Returns:
            Ordered list of :class:`Chunk` objects from every matching file.
            ``[]`` if the directory contains no matching files.

        Raises:
            IngestionError: ``source_dir`` does not exist or is not a
                directory, the directory cannot be walked, or ingesting any
                discovered file fails (see :meth:`ingest`).
            ValueError: ``max_concurrent`` is less than 1.
        """
        if max_concurrent < 1:
            raise ValueError(f"max_concurrent must be >= 1, got {max_concurrent}")

        root = Path(source_dir)
        exists, is_dir = await asyncio.to_thread(self._stat_dir, root)
        if not exists or not is_dir:
            raise IngestionError(f"Directory not found: {source_dir!r}")

        extensions = tuple(ext.lower() for ext in self._loader.supported_extensions)
        try:
            files = await asyncio.to_thread(self._discover_files, root, extensions)
        except OSError as exc:
            raise IngestionError(f"Failed to walk directory {source_dir!r}: {exc}") from exc

        if not files:
            logger.info("No matching files under %s", source_dir)
            return []

        semaphore = asyncio.Semaphore(max_concurrent)

        async def _ingest_one(path: Path) -> list[Chunk]:
            async with semaphore:
                return await self.ingest(str(path))

        results = await asyncio.gather(*(_ingest_one(path) for path in files))

        all_chunks: list[Chunk] = []
        for chunks in results:
            all_chunks.extend(chunks)
        return all_chunks

    @staticmethod
    def _stat_dir(path: Path) -> tuple[bool, bool]:
        """Return ``(exists, is_dir)`` for ``path`` (runs off the event loop)."""
        return path.exists(), path.is_dir()

    @staticmethod
    def _discover_files(root: Path, extensions: tuple[str, ...]) -> list[Path]:
        """Walk ``root`` for matching files (see :func:`discover_files`)."""
        return discover_files(root, extensions)


def discover_files(root: Path, extensions: tuple[str, ...]) -> list[Path]:
    """Walk ``root``, skipping hidden directories, for files matching ``extensions``.

    Blocking — callers on the event loop wrap this in ``asyncio.to_thread``.

    Returns paths sorted for deterministic dispatch order.
    """
    matches: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for filename in filenames:
            candidate = Path(dirpath) / filename
            if candidate.suffix.lower() in extensions:
                matches.append(candidate)
    return sorted(matches, key=str)

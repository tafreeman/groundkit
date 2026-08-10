"""File loading for ingestion: one parametrized loader for text-like formats.

Collapses ARP's ~90%-duplicated ``MarkdownLoader``/``TextLoader`` pair
(``agentic_v2/rag/loaders.py``) into a single class parametrized over
supported extensions (ADR-0001, loaders.py row). Directory- and URL-scale
ingestion are out of scope for this module: :class:`FileLoader` only ever
reads one file per call; walking a tree belongs to
:class:`~groundkit.ingestion.pipeline.IngestionPipeline`.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from groundkit.contracts import Document
from groundkit.errors import IngestionError
from groundkit.utils.path_safety import ensure_within_base

logger = logging.getLogger(__name__)

#: Extensions this loader accepts when the caller does not override them.
DEFAULT_EXTENSIONS: tuple[str, ...] = (".md", ".markdown", ".txt")

#: Default per-file size cap (10 MiB) — guards against accidentally ingesting
#: a huge or misnamed-binary file that would blow up chunking/memory.
DEFAULT_MAX_BYTES: int = 10 * 1024 * 1024

#: Containment root used when the caller supplies none.
_DEFAULT_BASE_DIR: Path = Path(".")


class FileLoader:
    """Load a single text-like file (markdown, plain text, ...) as a Document.

    Satisfies :class:`~groundkit.ingestion.protocols.LoaderProtocol`.

    Args:
        allowed_base_dir: Directory every loaded path must resolve within.
            Defaults to the current working directory. Checked via
            :func:`groundkit.utils.path_safety.ensure_within_base` before any
            filesystem access — including ``stat`` and existence checks.
        extensions: Lowercase, dot-prefixed extensions this loader accepts.
        max_bytes: Maximum file size in bytes; larger files raise
            :class:`IngestionError` instead of being read.
    """

    def __init__(
        self,
        allowed_base_dir: Path | None = None,
        extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        self._base_dir = (allowed_base_dir or _DEFAULT_BASE_DIR).resolve()
        self._extensions = tuple(ext.lower() for ext in extensions)
        self._max_bytes = max_bytes

    @property
    def supported_extensions(self) -> list[str]:
        """Lowercase, dot-prefixed extensions this loader accepts."""
        return list(self._extensions)

    @property
    def allowed_base_dir(self) -> Path:
        """Resolved containment root every loaded path must fall under."""
        return self._base_dir

    async def load(self, source: str) -> list[Document]:
        """Load a single file and return it as zero or one Document.

        Empty and whitespace-only files return ``[]`` (logged as a warning)
        regardless of extension — ARP logged this only for markdown and
        silently dropped it for text; groundkit treats every extension the
        same way.

        Args:
            source: Path to the file, relative or absolute.

        Returns:
            A list containing zero or one :class:`Document`.

        Raises:
            IngestionError: ``source`` escapes ``allowed_base_dir``, has an
                extension this loader does not accept, does not exist, is not
                a regular file, exceeds ``max_bytes``, or cannot be decoded
                as UTF-8 text.
        """
        try:
            path = ensure_within_base(source, self._base_dir)
        except ValueError as exc:
            raise IngestionError(str(exc)) from exc

        if path.suffix.lower() not in self._extensions:
            raise IngestionError(
                f"Unsupported extension {path.suffix!r} for {source!r} "
                f"(expected one of {self._extensions})"
            )

        try:
            content = await asyncio.to_thread(self._read_text, path)
        except IngestionError:
            raise
        except OSError as exc:
            raise IngestionError(f"Failed to read {source!r}: {exc}") from exc

        if not content.strip():
            logger.warning("Empty or whitespace-only file: %s", source)
            return []

        return [
            Document(
                source=str(path),
                content=content,
                metadata={
                    "file_name": path.name,
                    "file_extension": path.suffix,
                },
            )
        ]

    def _read_text(self, path: Path) -> str:
        """Validate and read ``path`` synchronously (runs off the event loop).

        Raises:
            IngestionError: If the file is larger than ``max_bytes``.
            OSError: If the file is missing, is a directory, or unreadable.
        """
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise IsADirectoryError(f"Not a file: {path}")
        size = path.stat().st_size
        if size > self._max_bytes:
            raise IngestionError(
                f"File exceeds max size ({size} > {self._max_bytes} bytes): {path}"
            )
        return path.read_text(encoding="utf-8")

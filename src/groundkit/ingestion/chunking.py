"""Offset-preserving document chunking.

:class:`RecursiveChunker` ports ARP's separator-cascade strategy
(``agentic_v2/rag/chunking.py``) — try paragraph, then line, then sentence,
then word, then character boundaries — but tracks **character offsets into
the original document** throughout the recursion instead of splitting and
rejoining strings. Every emitted :class:`~groundkit.contracts.Chunk` is
constructed as ``document.content[start:end]``, so the substring invariant
:class:`~groundkit.contracts.Chunk` itself enforces (``end - start ==
len(content)``) holds by construction, not by a downstream reassembly step
that could silently drift.

This also fixes ADR-0001 hazard 1: ARP's ``_hard_split`` recomputed
``start = end - overlap`` every iteration, which pins both ``start`` and
``end`` to the same values forever once ``end`` saturates at ``len(text)``
(``end - overlap < end`` is always true when ``overlap > 0``, so the
``start >= end`` guard never fires) — an infinite loop on any
separator-free text (base64, long URLs, minified code) once one exists.
Here the fallback advances by a fixed, strictly positive step
(``max(1, chunk_size - overlap)``) instead of recomputing from ``end``, so
progress is guaranteed regardless of overlap.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from groundkit.config import ChunkingConfig
from groundkit.contracts import Chunk, Document
from groundkit.errors import ChunkingError

#: Hard-split always advances by at least this many characters per window,
#: even if ``chunk_size - overlap`` would otherwise be <= 0 — this is what
#: makes the fallback's progress strictly monotonic (ADR-0001 hazard 1).
_MIN_HARD_SPLIT_STEP: int = 1


class RecursiveChunker:
    """Split documents into offset-addressed chunks via a separator cascade.

    Tries each separator in ``config.separators`` in order (default
    ``\\n\\n``, ``\\n``, ``. ``, `` ``, ``""``), merging split parts back into
    chunk-sized segments with overlap, and recursing with finer separators
    when a merged segment is still oversized. Falls back to a fixed-step
    character split when no separator is present in a segment at all.

    Satisfies :class:`~groundkit.ingestion.protocols.ChunkerProtocol`.
    """

    def chunk(self, document: Document, **kwargs: Any) -> list[Chunk]:
        """Split *document* into chunks.

        Args:
            document: Source document to split.
            **kwargs: Accepts an optional ``config`` keyword — a
                :class:`ChunkingConfig` (defaults are used when omitted or
                ``None``). Matches
                :class:`~groundkit.ingestion.protocols.ChunkerProtocol`'s
                ``**kwargs`` signature; no other keyword is recognized.

        Returns:
            Ordered list of :class:`Chunk` objects, ``chunk_index`` sequential
            from 0. Every chunk's ``content`` is exactly
            ``document.content[chunk.start_offset:chunk.end_offset]``.

        Raises:
            ChunkingError: If the ``config`` keyword is present but is not a
                :class:`ChunkingConfig`, or a computed chunk violates the
                offset/content invariant :class:`Chunk` enforces — the latter
                is a bug in this chunker, not a caller error, but reported as
                a typed error rather than a raw ``ValidationError`` (SPEC.md
                §2, fail closed).
        """
        cfg = self._resolve_config(kwargs)
        text = document.content
        spans = self._split_range(
            text, 0, len(text), cfg.separators, cfg.chunk_size, cfg.chunk_overlap
        )

        chunks: list[Chunk] = []
        for idx, (start, end) in enumerate(spans):
            try:
                chunks.append(
                    Chunk(
                        document_id=document.document_id,
                        chunk_index=idx,
                        content=text[start:end],
                        start_offset=start,
                        end_offset=end,
                        metadata={"source": document.source, **document.metadata},
                    )
                )
            except ValidationError as exc:
                raise ChunkingError(
                    f"Invalid chunk span [{start}:{end}] for document {document.document_id}: {exc}"
                ) from exc
        return chunks

    @staticmethod
    def _resolve_config(kwargs: dict[str, Any]) -> ChunkingConfig:
        """Extract and validate the ``config`` keyword, defaulting when absent."""
        config = kwargs.get("config")
        if config is None:
            return ChunkingConfig()
        if not isinstance(config, ChunkingConfig):
            raise ChunkingError(
                f"chunk() 'config' keyword must be a ChunkingConfig, got {type(config).__name__}"
            )
        return config

    def _split_range(
        self,
        text: str,
        start: int,
        end: int,
        separators: list[str],
        chunk_size: int,
        overlap: int,
    ) -> list[tuple[int, int]]:
        """Return offset spans covering ``text[start:end]``.

        Each span is <= ``chunk_size`` characters except when a single
        indivisible unit (one separator-delimited part, or the un-splittable
        remainder after all separators are exhausted) is itself larger.
        Blank/whitespace-only spans are dropped.
        """
        if end <= start:
            return []
        if end - start <= chunk_size:
            return [(start, end)] if text[start:end].strip() else []

        for i, sep in enumerate(separators):
            if sep and text.find(sep, start, end) != -1:
                parts = self._part_offsets(text, start, end, sep)
                return self._merge_parts(
                    text, parts, len(sep), separators[i + 1 :], chunk_size, overlap
                )

        return self._hard_split(text, start, end, chunk_size, overlap)

    @staticmethod
    def _part_offsets(text: str, start: int, end: int, sep: str) -> list[tuple[int, int]]:
        """Return the offset span of each ``sep``-delimited part in ``[start, end)``.

        Equivalent to locating ``text[start:end].split(sep)`` positions
        without materializing the intermediate substring or the split parts:
        each returned span is contiguous with its neighbor separated only by
        ``sep``, so joining any consecutive run reconstructs the exact
        original substring.
        """
        parts: list[tuple[int, int]] = []
        pos = start
        sep_len = len(sep)
        while True:
            idx = text.find(sep, pos, end)
            if idx == -1:
                parts.append((pos, end))
                return parts
            parts.append((pos, idx))
            pos = idx + sep_len

    def _merge_parts(
        self,
        text: str,
        parts: list[tuple[int, int]],
        sep_len: int,
        next_separators: list[str],
        chunk_size: int,
        overlap: int,
    ) -> list[tuple[int, int]]:
        """Greedily merge consecutive parts into chunk_size-sized spans, with overlap."""
        results: list[tuple[int, int]] = []
        current: list[tuple[int, int]] = []

        for part in parts:
            part_len = (part[1] - part[0]) + (sep_len if current else 0)
            if current and self._span_len(current) + part_len > chunk_size:
                self._flush(text, current, next_separators, chunk_size, overlap, results)
                current = self._carry_overlap(current, sep_len, overlap)
            current.append(part)

        if current:
            self._flush(text, current, next_separators, chunk_size, overlap, results)

        return results

    def _flush(
        self,
        text: str,
        current: list[tuple[int, int]],
        next_separators: list[str],
        chunk_size: int,
        overlap: int,
        results: list[tuple[int, int]],
    ) -> None:
        """Emit the accumulated ``current`` span, recursing if still oversized."""
        seg_start, seg_end = current[0][0], current[-1][1]
        if seg_end - seg_start > chunk_size:
            results.extend(
                self._split_range(text, seg_start, seg_end, next_separators, chunk_size, overlap)
            )
        elif text[seg_start:seg_end].strip():
            results.append((seg_start, seg_end))

    @staticmethod
    def _span_len(parts: list[tuple[int, int]]) -> int:
        """Length of the contiguous span covered by a run of adjacent parts."""
        return parts[-1][1] - parts[0][0]

    @staticmethod
    def _carry_overlap(
        current: list[tuple[int, int]], sep_len: int, overlap: int
    ) -> list[tuple[int, int]]:
        """Return the trailing parts of ``current`` to retain as overlap context."""
        if overlap <= 0 or not current:
            return []
        kept: list[tuple[int, int]] = []
        total = 0
        for part in reversed(current):
            total += (part[1] - part[0]) + sep_len
            if total > overlap:
                break
            kept.insert(0, part)
        return kept

    @staticmethod
    def _hard_split(
        text: str, start: int, end: int, chunk_size: int, overlap: int
    ) -> list[tuple[int, int]]:
        """Split ``text[start:end]`` by fixed-size windows when no separator applies.

        Advances by a strictly positive, fixed step
        (``max(1, chunk_size - overlap)``) each iteration instead of
        recomputing ``start`` from the previous window's ``end`` — the fix
        for ADR-0001 hazard 1 (see module docstring). This guarantees
        termination in ``O((end - start) / step)`` iterations regardless of
        ``overlap``, and covers ``[start, end)`` fully since ``step <=
        chunk_size`` leaves no gap between consecutive windows.
        """
        step = max(_MIN_HARD_SPLIT_STEP, chunk_size - overlap)
        spans: list[tuple[int, int]] = []
        pos = start
        while pos < end:
            window_end = min(pos + chunk_size, end)
            if text[pos:window_end].strip():
                spans.append((pos, window_end))
            if window_end >= end:
                break
            pos += step
        return spans

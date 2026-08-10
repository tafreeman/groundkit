"""Citation resolution: verify a retrieved span against its source (Phase 1).

A citation is *verifiable*, not vibes (SPEC.md §2): given the source file and
character offsets, these helpers read the span back out of the source and
compare it to what retrieval returned. Pure code, no LLM.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from groundkit.errors import RetrievalError
from groundkit.utils.path_safety import ensure_within_base

if TYPE_CHECKING:
    from groundkit.contracts import Citation


async def resolve_citation(citation: Citation, allowed_base_dir: Path) -> str:
    """Return the exact text span a citation points at, read from its source.

    Args:
        citation: The citation to resolve.
        allowed_base_dir: Containment root the citation's source must
            resolve within (same path-safety barrier as ingestion).

    Returns:
        ``source_text[start_offset:end_offset]``.

    Raises:
        RetrievalError: The source escapes ``allowed_base_dir``, cannot be
            read, or is shorter than the cited offsets (source changed since
            indexing).
    """
    try:
        path = ensure_within_base(citation.source, allowed_base_dir)
    except ValueError as exc:
        raise RetrievalError(str(exc)) from exc

    try:
        text = await asyncio.to_thread(path.read_text, "utf-8")
    except OSError as exc:
        raise RetrievalError(f"Cannot read cited source {citation.source!r}: {exc}") from exc

    if citation.end_offset > len(text):
        raise RetrievalError(
            f"Cited span [{citation.start_offset}:{citation.end_offset}] exceeds "
            f"source length ({len(text)}) — source changed since indexing"
        )
    return text[citation.start_offset : citation.end_offset]


async def verify_citation(
    citation: Citation, expected_content: str, allowed_base_dir: Path
) -> bool:
    """True when the cited span in the source equals ``expected_content``.

    Raises:
        RetrievalError: As :func:`resolve_citation`.
    """
    return await resolve_citation(citation, allowed_base_dir) == expected_content

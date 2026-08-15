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

    Dispatches on ``citation.source_class`` (ADR-0016), because "read the
    source and slice" is only correct for one of the three classes. It is
    correct for ``text`` precisely because a text loader makes
    ``Document.content`` the file's decoded bytes, so offsets into content are
    offsets into the file.

    For the other two it is wrong rather than merely incomplete, which is why
    they are refused here instead of falling through:

    - ``extracted`` — offsets index deterministic extractor output. Reading a
      PDF as UTF-8 raises; reading an HTML file returns markup the offsets were
      never measured against. Resolving one needs the *same* extractor that
      produced it, and re-running a different version silently shifts every
      offset after the first difference.
    - ``snapshot`` — ``source`` is a URL. It is refused before it can reach
      :func:`~groundkit.utils.path_safety.ensure_within_base`, and that
      ordering is deliberate: ``ensure_within_base`` validates only that its
      input is non-empty and null-byte-free, then hands it to
      ``os.path.realpath``, which resolves ``"https://example.com/x"`` as a
      *relative path* under the current directory. The containment check can
      therefore pass and the failure surfaces later as a confusing
      file-not-found. A URL is not a path, and the class — not a URL-sniffing
      path helper — is what keeps that boundary sharp.

    Args:
        citation: The citation to resolve.
        allowed_base_dir: Containment root the citation's source must
            resolve within (same path-safety barrier as ingestion).

    Returns:
        ``source_text[start_offset:end_offset]``.

    Raises:
        RetrievalError: The source escapes ``allowed_base_dir``, cannot be
            read, is not valid UTF-8, is shorter than the cited offsets
            (source changed since indexing), or belongs to a source class this
            build cannot resolve.
    """
    if citation.source_class != "text":
        raise RetrievalError(
            f"cannot resolve a {citation.source_class!r} citation for "
            f"{citation.source!r}: re-deriving its text needs the loader that "
            "produced it, which this build does not have wired. Refused rather "
            "than read as plain text, which would compare the cited offsets "
            "against bytes they were never measured against (ADR-0016)."
        )

    try:
        path = ensure_within_base(citation.source, allowed_base_dir)
    except ValueError as exc:
        raise RetrievalError(str(exc)) from exc

    try:
        text = await asyncio.to_thread(path.read_text, "utf-8")
    except OSError as exc:
        raise RetrievalError(f"Cannot read cited source {citation.source!r}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise RetrievalError(f"Cited source {citation.source!r} is not valid UTF-8: {exc}") from exc

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

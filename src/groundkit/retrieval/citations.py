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


#: Identity strings of every extractor this build can re-run a citation
#: through. Empty in Waves 1-2 by design, not a TODO: no PDF/HTML extractor
#: is wired in yet (Wave 3, ADR-0016 decisions 3/5), so no extractor
#: identity is ever active, and every `extracted` citation correctly refuses
#: regardless of what it recorded. Wave 3 populates this per extractor it
#: ships; the membership check below does not change shape when it does.
_ACTIVE_EXTRACTOR_IDENTITIES: frozenset[str] = frozenset()


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
      offset after the first difference — so this compares the citation's
      recorded extractor identity against :data:`_ACTIVE_EXTRACTOR_IDENTITIES`,
      the set this build can actually re-run a citation through (ADR-0016
      decision 2). That set is empty until Wave 3 ships a real extractor, so
      every ``extracted`` citation refuses today regardless of what it
      recorded — a correct answer for a build with zero extractors, not a
      placeholder.
    - ``snapshot`` — ``source`` is a URL. It is refused before it can reach
      :func:`~groundkit.utils.path_safety.ensure_within_base`, and that
      ordering is deliberate: ``ensure_within_base`` validates only that its
      input is non-empty and null-byte-free, then hands it to
      ``os.path.realpath``, which resolves ``"https://example.com/x"`` as a
      *relative path* under the current directory. The containment check can
      therefore pass and the failure surfaces later as a confusing
      file-not-found. A URL is not a path, and the class — not a URL-sniffing
      path helper — is what keeps that boundary sharp. Wave 4 builds the local
      snapshot this class verifies against; until then the refusal states that
      reason specifically rather than sharing wording with ``extracted``.

    Every failure here sets :attr:`~groundkit.errors.RetrievalError.verdict`
    explicitly, at the raise site that already knows it best, so
    ``service.tools.handle_fetch_chunk`` can classify the outcome by reading
    an attribute instead of pattern-matching the message (ADR-0016 decision
    6). ``drifted`` means the source was read and no longer matches;
    ``unresolvable`` means it could not be checked at all — a distinction an
    extractor-identity mismatch falls on the ``unresolvable`` side of, because
    nothing was re-read to disagree with.

    Args:
        citation: The citation to resolve.
        allowed_base_dir: Containment root the citation's source must
            resolve within (same path-safety barrier as ingestion).

    Returns:
        ``source_text[start_offset:end_offset]``.

    Raises:
        RetrievalError: The source escapes ``allowed_base_dir``, cannot be
            read, is not valid UTF-8, is shorter than the cited offsets
            (source changed since indexing — ``verdict="drifted"``), or
            belongs to a source class or extractor identity this build cannot
            resolve (``verdict="unresolvable"`` in every other case).
    """
    if citation.source_class == "extracted":
        if citation.extractor not in _ACTIVE_EXTRACTOR_IDENTITIES:
            raise RetrievalError(
                f"cannot resolve an 'extracted' citation for {citation.source!r}: "
                f"the extractor identity recorded at ingest ({citation.extractor!r}) "
                "is not active in this build "
                f"({sorted(_ACTIVE_EXTRACTOR_IDENTITIES) or 'none registered'}). "
                "Re-deriving these offsets needs the exact extractor that produced "
                "them; refused rather than guessing (ADR-0016 decision 2).",
                verdict="unresolvable",
            )
        # Unreachable while _ACTIVE_EXTRACTOR_IDENTITIES is empty (Waves 1-2).
        # Wave 3 adds the re-extraction call here: re-run the active extractor
        # over citation.source, slice [start_offset:end_offset], and return it
        # — the same shape resolve_citation already has for `text` below.
    elif citation.source_class == "snapshot":
        raise RetrievalError(
            f"cannot resolve a 'snapshot' citation for {citation.source!r}: "
            "verification requires the local snapshot URL ingestion stores at "
            "ingest time (ADR-0016 decision 4), and no loader that builds one "
            "is wired into this build yet (Wave 4). source is a URL, not a "
            "path, so it is refused here before it could reach "
            "ensure_within_base, which would resolve it as a relative path "
            "under the current directory rather than reject it as not-a-path.",
            verdict="unresolvable",
        )

    try:
        path = ensure_within_base(citation.source, allowed_base_dir)
    except ValueError as exc:
        raise RetrievalError(str(exc), verdict="unresolvable") from exc

    try:
        text = await asyncio.to_thread(path.read_text, "utf-8")
    except OSError as exc:
        raise RetrievalError(
            f"Cannot read cited source {citation.source!r}: {exc}", verdict="unresolvable"
        ) from exc
    except UnicodeDecodeError as exc:
        raise RetrievalError(
            f"Cited source {citation.source!r} is not valid UTF-8: {exc}", verdict="unresolvable"
        ) from exc

    if citation.end_offset > len(text):
        raise RetrievalError(
            f"Cited span [{citation.start_offset}:{citation.end_offset}] exceeds "
            f"source length ({len(text)}) — source changed since indexing",
            verdict="drifted",
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

"""Path safety helpers for validating locally-contained file paths.

Ported from ARP (`agentic_v2/utils/path_safety.py`) per ADR-0001.

Containment is decided with ``os.path.commonpath`` over fully realpath-ed
operands. That is deliberate: it is the barrier pattern CodeQL recognizes for
``py/path-injection``. ``Path.is_relative_to`` (and the ``in .parents`` check
before it) are equivalent in behavior but are not treated as sanitizers.

The check is inlined in both public helpers rather than shared through a third
function so each one carries the barrier in the same frame as the value it
returns.

Containment alone is not the whole barrier, which is why the ``O_NOFOLLOW``
constants below live here too. :func:`ensure_within_base` resolves symlinks
and then *returns*; whoever opens the returned path issues a second syscall,
and anything able to create a file in the containment root can win the gap
between the two. Closing that needs the open itself to refuse a symlink, and
both sides of the snapshot round trip need the identical answer to "which
errno means the final component was a link" -- so it is stated once, here,
rather than twice in modules that could drift apart
(:mod:`groundkit.ingestion.url_loader` writes, then
:mod:`groundkit.retrieval.citations` reads).
"""

from __future__ import annotations

import errno
import os
import sys
from pathlib import Path

#: ``os.O_NOFOLLOW`` where the platform defines it, ``0`` (a no-op in a flag
#: mask) where it does not. Windows has no such flag, and CI runs Linux while
#: this repo is developed on Windows, so the guard must degrade rather than
#: raise ``AttributeError`` at import on the maintainer's own machine. The
#: consequence is stated plainly rather than hidden: on Windows the snapshot
#: write and read are exactly as racy as they were before, and the refusals
#: that use this can only fire where the flag exists.
O_NOFOLLOW: int = getattr(os, "O_NOFOLLOW", 0)

#: ``os.O_BINARY`` on Windows, ``0`` elsewhere. Required because ``os.open``
#: leaves the descriptor in the C runtime's default (text) mode on Windows,
#: which translates line endings underneath whatever text wrapper is layered
#: over it. Both snapshot sides want the file to be exactly
#: ``content.encode("utf-8")``, in both directions, because citation offsets
#: are measured against the decoded string and a translated CRLF is one
#: character shorter than what was indexed.
O_BINARY: int = getattr(os, "O_BINARY", 0)

#: Platforms whose ``open(..., O_NOFOLLOW)`` has historically reported
#: ``EMLINK`` rather than POSIX's ``ELOOP`` for a symlinked final component.
_EMLINK_MEANS_SYMLINK_PLATFORMS: tuple[str, ...] = (
    "freebsd",
    "netbsd",
    "openbsd",
    "dragonfly",
)

#: Errnos a POSIX ``open(..., O_NOFOLLOW)`` reports when the final path
#: component is a symbolic link.
#:
#: ``ELOOP`` everywhere, per POSIX. ``EMLINK`` **only** on the BSDs above,
#: and the narrowing is the point: on Linux -- which is what CI runs --
#: ``EMLINK`` is "Too many links", an unrelated filesystem limit. Treating it
#: as a symlink refusal there would tell an operator that someone replaced
#: the path between the containment check and the open, i.e. report a TOCTOU
#: attack, when what actually happened is that a directory hit its link
#: ceiling. A security refusal that fires on an unrelated I/O failure costs
#: more than the exotic case it was meant to cover: it trains the reader to
#: disbelieve the message.
SYMLINK_ERRNOS: frozenset[int] = (
    frozenset({errno.ELOOP, errno.EMLINK})
    if sys.platform.startswith(_EMLINK_MEANS_SYMLINK_PLATFORMS)
    else frozenset({errno.ELOOP})
)


def _validate_path_input(path: str | Path) -> str:
    """Return ``path`` as a string, rejecting empty and null-byte inputs.

    The value is returned unmodified — leading and trailing whitespace can be
    legal in a filename, so it is only inspected, never stripped.

    Raises:
        ValueError: If the path is empty, whitespace-only, or contains a null
            byte (which would truncate the path inside the C layer).
    """
    path_str = str(path)
    if not path_str.strip():
        raise ValueError("Path must not be empty")
    if "\0" in path_str:
        raise ValueError("Path must not contain null bytes")
    return path_str


def is_within_base(path: str | Path, base_dir: str | Path) -> bool:
    """Return True when ``path`` resolves under ``base_dir``."""
    resolved_path = os.path.realpath(_validate_path_input(path))
    resolved_base = os.path.realpath(base_dir)

    try:
        return os.path.commonpath([resolved_base, resolved_path]) == resolved_base
    except ValueError:
        # Raised when the operands share no prefix at all — on Windows that
        # means separate drives. Treat it as "outside the base" rather than
        # letting it surface to the caller as an unexpected error.
        return False


def ensure_within_base(path: str | Path, base_dir: str | Path) -> Path:
    """Resolve ``path`` and raise ValueError when it escapes ``base_dir``."""
    resolved_path = os.path.realpath(_validate_path_input(path))
    resolved_base = os.path.realpath(base_dir)

    try:
        within = os.path.commonpath([resolved_base, resolved_path]) == resolved_base
    except ValueError:
        within = False

    if not within:
        raise ValueError(f"Path escapes base directory: {path}")
    return Path(resolved_path)

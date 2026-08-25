"""Where a URL-ingested document's local snapshot lives on disk (ADR-0016
decision 4; docs/specs/loaders-extracted-and-remote-sources.md §10.1). Pure
path arithmetic, no I/O -- shared by the URL loader
(:class:`~groundkit.ingestion.url_loader.UrlLoader`, which writes) and
:func:`groundkit.retrieval.citations.resolve_citation` (which reads), so the
naming convention is asserted from exactly one place and the two sides
cannot independently drift on what a snapshot path means.

Placement follows :mod:`groundkit.identity`'s precedent (see
:mod:`groundkit.extraction`'s docstring, which quotes the same reasoning): a
module outside whichever caller happened to need it first, so sharing it
creates no dependency between ingest and retrieval, and nothing imports it
back.
"""

from __future__ import annotations

import os
from pathlib import Path

_SEPARATORS = frozenset(sep for sep in (os.sep, os.altsep) if sep)


def snapshot_dir_for(index_dir: Path, collection: str) -> Path:
    """The containment root for one collection's stored snapshots.

    Sibling of ``<index_dir>/<collection>.sqlite3`` and ``.lance``, following
    the same per-collection-suffix convention. Not resolved or created here
    -- the caller (:class:`~groundkit.ingestion.url_loader.UrlLoader` on the
    write side, :func:`~groundkit.retrieval.citations.resolve_citation` on
    the read side) is responsible for that, exactly as both already are for
    ``allowed_base_dir``.
    """
    return index_dir / f"{collection}.snapshots"


def snapshot_path_for(snapshot_dir: Path, document_id: str) -> Path:
    """The snapshot file for one document within its collection's ``snapshot_dir``.

    ``document_id`` is attacker-influenced in principle (it is a plain string
    field with no character-class restriction), so it is required here to be a
    single path component -- one name directly inside ``snapshot_dir``, never a
    nested path.

    That restriction is what makes ``O_NOFOLLOW`` at the callers' opens
    sufficient. ``O_NOFOLLOW`` refuses a symlink at the *final* component only;
    every intermediate directory is still resolved normally. A ``document_id``
    of ``"sub/evil"`` would pass an
    :func:`~groundkit.utils.path_safety.ensure_within_base` check (it escapes
    nothing) while introducing exactly such an intermediate component, which
    the write side would then create with ``mkdir(parents=True)`` -- reopening
    the containment-check-to-open race that ``O_NOFOLLOW`` closes, one level
    up where the flag cannot see it. Refusing separators outright removes the
    intermediate component instead of trying to guard it, so no caller has to
    reason about the ordering of ``mkdir`` against the check.

    Callers must still run the result through ``ensure_within_base`` (or
    ``is_within_base``): this function refuses *shapes*, and containment
    against a specific root remains the caller's check to make. In production
    ``document_id`` is always ``Document``'s own ``uuid.uuid4().hex`` default,
    which is 32 hex characters and can never contain a separator -- this guard
    exists so that property is enforced rather than assumed.

    What counts as a separator is what the **running platform** treats as one
    (``os.sep``/``os.altsep``, plus a drive prefix). A backslash is an
    ordinary filename character on POSIX and a separator on Windows, so the
    same ``document_id`` is legitimately a single component on one and a
    nested path on the other; branching on the platform's own notion states
    that rather than hard-coding one platform's answer.

    Raises:
        ValueError: ``document_id`` is empty, is ``.`` or ``..``, contains a
            null byte, contains a path separator, or carries a drive prefix.
            ``ValueError`` specifically because that is what
            ``ensure_within_base`` raises for a rejected path, so the callers'
            existing handlers cover this without a second except clause.
    """
    if not document_id:
        raise ValueError("a snapshot document_id must not be empty")
    if document_id in {".", ".."}:
        raise ValueError(
            f"snapshot document_id {document_id!r} is a directory reference, not a file name"
        )
    if "\x00" in document_id:
        raise ValueError("a snapshot document_id must not contain a null byte")
    if any(separator in document_id for separator in _SEPARATORS):
        raise ValueError(
            f"snapshot document_id {document_id!r} contains a path separator; it must name "
            "a single file directly inside the snapshot directory, so that no intermediate "
            "path component exists for a symlink to be planted at"
        )
    if os.path.splitdrive(document_id)[0]:
        raise ValueError(
            f"snapshot document_id {document_id!r} carries a drive prefix; it must name a "
            "single file directly inside the snapshot directory"
        )
    return snapshot_dir / document_id

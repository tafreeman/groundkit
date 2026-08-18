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

from pathlib import Path


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
    field with no character-class restriction), so a caller must still run
    the result through
    :func:`~groundkit.utils.path_safety.ensure_within_base` against
    ``snapshot_dir`` before touching the filesystem -- this function performs
    no containment check of its own, matching how ``citation.source`` is
    handled for the ``text``/``extracted`` classes.
    """
    return snapshot_dir / document_id

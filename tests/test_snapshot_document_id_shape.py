"""A snapshot ``document_id`` must name a single component, not a nested path.

``O_NOFOLLOW`` (GK-030) refuses a symlink at the *final* path component only;
every intermediate directory is resolved normally. A ``document_id`` such as
``"sub/evil"`` escapes nothing -- ``ensure_within_base`` passes it -- while
introducing exactly such an intermediate component, which the write side then
creates with ``mkdir(parents=True)``. That reopens the
containment-check-to-open race one level above where ``O_NOFOLLOW`` can see
it, on both the write and the read side.

Production never produces such an id (``Document.document_id`` defaults to
``uuid.uuid4().hex``, 32 hex characters), so these tests drive the private
helpers directly, exactly as ``TestUrlLoaderSnapshotContainment`` does and for
the same stated reason.

Platform note, and the reason the nested case uses a forward slash: what
counts as a separator is what the *running* platform treats as one. A
backslash is an ordinary filename character on POSIX and a separator on
Windows, so ``"sub\\evil"`` is a legitimate single component on Linux CI. A
forward slash is a separator everywhere, so it is the shape that pins the
property on every platform without branching.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from groundkit import snapshots
from groundkit.errors import IngestionError
from groundkit.index.metadata import SQLiteMetadataStore
from groundkit.indexer import Indexer
from groundkit.ingestion.url_loader import UrlLoader


def _client(handler: object) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


class TestSnapshotPathForRefusesNonComponents:
    @pytest.mark.parametrize(
        "document_id",
        ["sub/evil", "a/b/c", "", ".", "..", "with\x00null"],
    )
    def test_refused_shapes(self, tmp_path: Path, document_id: str) -> None:
        with pytest.raises(ValueError):
            snapshots.snapshot_path_for(tmp_path, document_id)

    def test_a_plain_name_is_still_accepted(self, tmp_path: Path) -> None:
        assert snapshots.snapshot_path_for(tmp_path, "abc123") == tmp_path / "abc123"

    def test_a_uuid4_hex_default_is_always_accepted(self, tmp_path: Path) -> None:
        """The only shape production actually produces must never be refused."""
        import uuid

        for _ in range(50):
            document_id = uuid.uuid4().hex
            assert snapshots.snapshot_path_for(tmp_path, document_id).name == document_id


class TestNestedDocumentIdNeverCreatesAnIntermediateDirectory:
    """The write side must refuse before ``mkdir(parents=True)`` runs.

    Asserting on the absence of the created directory, not merely on the
    raise: a guard that refused *after* creating the intermediate component
    would still leave the planting surface behind.
    """

    def test_write_refuses_and_creates_nothing(self, tmp_path: Path) -> None:
        snapshot_dir = tmp_path / "col.snapshots"
        loader = UrlLoader(snapshot_dir)

        with pytest.raises(IngestionError, match="escapes"):
            loader._write_snapshot("sub/evil", "content")

        assert not (snapshot_dir / "sub").exists()
        # `_write_snapshot` mkdirs only after the check passes, so a refused
        # write leaves not even the containment root behind.
        assert not snapshot_dir.exists()


class TestNestedDocumentIdIsNeverUnlinked:
    """Removal is best-effort and never fatal, so the refusal must be logged
    rather than raised -- but it must still refuse."""

    def test_remove_is_inert_and_does_not_raise(self, tmp_path: Path) -> None:
        index_dir = tmp_path / ".groundkit"
        snapshot_dir = snapshots.snapshot_dir_for(index_dir, "default")
        snapshot_dir.mkdir(parents=True)
        victim_dir = snapshot_dir / "sub"
        victim_dir.mkdir()
        victim = victim_dir / "evil"
        victim.write_text("must survive", encoding="utf-8")

        async def attempt() -> None:
            store = await SQLiteMetadataStore.open(index_dir, "default")
            try:
                indexer = Indexer(
                    store,
                    UrlLoader(snapshot_dir, client=_client(lambda _r: httpx.Response(200))),
                    collection="default",
                    snapshot_dir=snapshot_dir,
                )
                await indexer._remove_snapshot("sub/evil")
            finally:
                await store.close()

        asyncio.run(attempt())

        assert victim.exists()
        assert victim.read_text(encoding="utf-8") == "must survive"

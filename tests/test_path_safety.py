"""Path-safety tests — including the traversal rejection ARP never tested
(ADR-0001, loaders.py row)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from groundkit.utils.path_safety import ensure_within_base, is_within_base


def _can_create_symlinks() -> bool:
    """Probe whether this runtime can create filesystem symlinks.

    Windows requires Developer Mode or an elevated process to create
    symlinks without extra privilege, so this is a runtime capability probe
    rather than a bare ``sys.platform`` check — it skips nowhere it doesn't
    have to (e.g. Windows CI with Developer Mode enabled) and always runs on
    Linux/macOS CI, where the symlink-escape hazard this guards against
    actually lives. The probe directory is self-cleaning and never touches a
    test's own ``tmp_path``.
    """
    with tempfile.TemporaryDirectory() as probe_dir:
        target = Path(probe_dir) / "target.txt"
        target.write_text("x", encoding="utf-8")
        link = Path(probe_dir) / "link.txt"
        try:
            link.symlink_to(target)
        except (OSError, NotImplementedError):
            return False
        return True


requires_symlinks = pytest.mark.skipif(
    not _can_create_symlinks(), reason="runtime cannot create filesystem symlinks"
)


def test_contained_path_is_within(tmp_path: Path) -> None:
    inner = tmp_path / "docs" / "a.md"
    inner.parent.mkdir()
    inner.write_text("x", encoding="utf-8")
    assert is_within_base(inner, tmp_path)


def test_traversal_escapes_base(tmp_path: Path) -> None:
    escape = tmp_path / "docs" / ".." / ".." / "etc" / "passwd"
    assert not is_within_base(escape, tmp_path / "docs")


def test_ensure_within_base_raises_on_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes base"):
        ensure_within_base(tmp_path / ".." / "outside.txt", tmp_path)


def test_ensure_within_base_returns_resolved_path(tmp_path: Path) -> None:
    inner = tmp_path / "a.txt"
    inner.write_text("x", encoding="utf-8")
    resolved = ensure_within_base(inner, tmp_path)
    assert resolved.is_absolute()
    assert resolved.name == "a.txt"


def test_empty_path_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        is_within_base("", tmp_path)


def test_null_byte_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="null"):
        is_within_base("a\0b", tmp_path)


def test_disjoint_roots_are_outside_not_an_error(tmp_path: Path) -> None:
    # On Windows different drives make commonpath raise ValueError internally;
    # the helper must translate that to "outside", never propagate.
    assert is_within_base("Q:/nonexistent/elsewhere", tmp_path) is False


@requires_symlinks
def test_symlink_escaping_base_is_rejected(tmp_path: Path) -> None:
    """A symlink that lives INSIDE the base dir but points OUTSIDE it must
    still be rejected — ``os.path.realpath`` resolves the symlink on both
    operands before ``commonpath`` compares them, so the on-disk location of
    the symlink itself is irrelevant; only where it resolves to matters."""
    base = tmp_path / "base"
    base.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("secret", encoding="utf-8")

    escape_link = base / "escape.txt"
    escape_link.symlink_to(secret)

    assert is_within_base(escape_link, base) is False
    with pytest.raises(ValueError, match="escapes base"):
        ensure_within_base(escape_link, base)


@requires_symlinks
def test_symlink_within_base_is_accepted(tmp_path: Path) -> None:
    """A symlink inside base pointing at another location still inside base
    resolves to a path under base and must be accepted."""
    base = tmp_path / "base"
    base.mkdir()
    real = base / "real.txt"
    real.write_text("x", encoding="utf-8")

    alias_link = base / "alias.txt"
    alias_link.symlink_to(real)

    assert is_within_base(alias_link, base) is True
    resolved = ensure_within_base(alias_link, base)
    assert resolved == real.resolve()


@requires_symlinks
def test_symlink_directory_escaping_base_is_rejected(tmp_path: Path) -> None:
    """A directory symlink inside base pointing at a directory outside base
    must not let a path built through it escape detection."""
    base = tmp_path / "base"
    base.mkdir()
    outside_dir = tmp_path / "outside_dir"
    outside_dir.mkdir()
    secret = outside_dir / "secret.txt"
    secret.write_text("secret", encoding="utf-8")

    escape_dir_link = base / "escape_dir"
    escape_dir_link.symlink_to(outside_dir, target_is_directory=True)
    escaped_path = escape_dir_link / "secret.txt"

    assert is_within_base(escaped_path, base) is False
    with pytest.raises(ValueError, match="escapes base"):
        ensure_within_base(escaped_path, base)

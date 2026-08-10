"""Path-safety tests — including the traversal rejection ARP never tested
(ADR-0001, loaders.py row)."""

from __future__ import annotations

from pathlib import Path

import pytest

from groundkit.utils.path_safety import ensure_within_base, is_within_base


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

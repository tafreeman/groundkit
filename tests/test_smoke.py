"""Phase-0 smoke tests: package imports, version wiring, CLI entry point."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from groundkit import __version__
from groundkit.cli import main


def test_version_is_set() -> None:
    assert __version__


def test_version_matches_pyproject() -> None:
    """``__version__`` and ``pyproject.toml`` must agree.

    The version lives in two independent places: ``pyproject.toml``'s
    ``project.version`` (what the wheel and PyPI see) and this package's
    ``__version__`` (what ``grk --version`` prints and what
    ``RunMetadata.groundkit_version`` stamps into every eval artifact).
    Nothing else enforces that they match -- ``release-gates.yml``'s
    tag/version parity step compares the release tag against
    ``pyproject.toml`` alone and never reads ``__version__``, so a bump to
    one and not the other ships a package whose CLI misreports its own
    version, past every gate, with a green suite.

    Skipped rather than failed when ``pyproject.toml`` is absent, which is
    the installed-wheel case: the file is not packaged, and this check is a
    source-tree invariant, not a runtime one.
    """
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("pyproject.toml is not present (installed-wheel layout)")
    with pyproject.open("rb") as handle:
        declared = tomllib.load(handle)["project"]["version"]
    assert __version__ == declared, (
        f"__version__ is {__version__!r} but pyproject.toml declares {declared!r}; "
        "bump both or neither"
    )


def test_cli_main_returns_zero() -> None:
    assert main([]) == 0


def test_cli_version_flag_exits_zero() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0

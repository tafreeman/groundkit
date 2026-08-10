"""Phase-0 smoke tests: package imports, version wiring, CLI entry point."""

from __future__ import annotations

import pytest

from groundkit import __version__
from groundkit.cli import main


def test_version_is_set() -> None:
    assert __version__


def test_cli_main_returns_zero() -> None:
    assert main([]) == 0


def test_cli_version_flag_exits_zero() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0

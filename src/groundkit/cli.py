"""Command-line entry point (``grk``).

Subcommands (``ingest``, ``search``, ``eval``, ``serve``, ``serve-mcp``) land in
their phases per SPEC.md; Phase 0 ships only version wiring so the entry point
is installable and testable.
"""

from __future__ import annotations

import argparse

from groundkit import __version__


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="grk",
        description="groundkit: grounded, citation-verifiable hybrid retrieval.",
    )
    parser.add_argument("--version", action="version", version=f"grk {__version__}")
    parser.parse_args(argv)
    return 0

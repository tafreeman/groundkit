"""Command-line entry point (``grk``).

Phase 1 ships ``ingest`` and ``search`` end-to-end against the persisted
local index with zero cloud credentials. ``eval``, ``serve``, and
``serve-mcp`` land in their phases per SPEC.md §9.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from groundkit import __version__
from groundkit.config import RetrievalConfig
from groundkit.errors import GroundkitError
from groundkit.index.metadata import SQLiteMetadataStore
from groundkit.indexer import Indexer
from groundkit.ingestion.loaders import FileLoader
from groundkit.retrieval.search import MAX_TOP_K, Retriever

#: Characters of chunk content shown per result in text output.
_SNIPPET_CHARS: int = 160


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        result: int = asyncio.run(args.func(args))
    except GroundkitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grk",
        description="groundkit: grounded, citation-verifiable hybrid retrieval.",
    )
    parser.add_argument("--version", action="version", version=f"grk {__version__}")
    sub = parser.add_subparsers(dest="command")

    ingest = sub.add_parser("ingest", help="Ingest a file or directory into the local index.")
    ingest.add_argument("path", help="File or directory to ingest.")
    ingest.add_argument("--index-dir", default=".groundkit", help="Index directory.")
    ingest.add_argument("--collection", default="default", help="Collection name.")
    ingest.add_argument(
        "--base-dir",
        default=None,
        help="Containment root for loads (default: the ingested directory, or the file's parent).",
    )
    ingest.set_defaults(func=_cmd_ingest)

    search = sub.add_parser("search", help="Search the local index.")
    search.add_argument("query", help="Query text.")
    search.add_argument("--index-dir", default=".groundkit", help="Index directory.")
    search.add_argument("--collection", default="default", help="Collection name.")
    search.add_argument(
        "--top-k", type=int, default=None, help=f"Results to return (1-{MAX_TOP_K})."
    )
    search.add_argument("--json", action="store_true", help="Emit the full response as JSON.")
    search.set_defaults(func=_cmd_search)

    return parser


async def _cmd_ingest(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if args.base_dir is not None:
        base_dir = Path(args.base_dir)
    elif path.is_dir():
        base_dir = path
    else:
        base_dir = path.parent

    store = await SQLiteMetadataStore.open(Path(args.index_dir), args.collection)
    try:
        indexer = Indexer(store, FileLoader(allowed_base_dir=base_dir))
        if path.is_dir():
            report = await indexer.index_directory(str(path))
        else:
            report = await indexer.index_source(str(path))
    finally:
        await store.close()

    print(
        f"ingested: {report.files_seen} files seen, "
        f"{report.documents_indexed} indexed, "
        f"{report.documents_skipped} unchanged, "
        f"{report.chunks_written} chunks written"
    )
    return 0


async def _cmd_search(args: argparse.Namespace) -> int:
    store = await SQLiteMetadataStore.open(Path(args.index_dir), args.collection)
    try:
        retriever = await Retriever.open(store, RetrievalConfig())
        response = await retriever.search(args.query, top_k=args.top_k)
    finally:
        await store.close()

    if args.json:
        print(json.dumps(response.model_dump(), indent=2))
        return 0

    if not response.results:
        print("no results")
        return 0

    for rank, result in enumerate(response.results, start=1):
        snippet = " ".join(result.content.split())[:_SNIPPET_CHARS]
        print(
            f"{rank}. [{result.score:.3f}] {result.source}"
            f"#{result.start_offset}-{result.end_offset}\n   {snippet}"
        )
    return 0

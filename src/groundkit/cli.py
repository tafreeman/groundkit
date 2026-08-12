"""Command-line entry point (``grk``).

Phase 1 shipped ``ingest`` and ``search`` end-to-end against the persisted
local index with zero cloud credentials. Phase 2 adds ``eval``, running the
BM25-baseline retrieval harness (``groundkit.evals.runner``) against a
golden corpus and judgment set. ``serve`` and ``serve-mcp`` land in their
phases per SPEC.md §9.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from groundkit import __version__
from groundkit.config import RetrievalConfig
from groundkit.errors import EvalError, GroundkitError
from groundkit.evals.runner import run_eval, write_report
from groundkit.evals.schema import EvalReport
from groundkit.index.metadata import SQLiteMetadataStore
from groundkit.indexer import Indexer
from groundkit.ingestion.loaders import FileLoader
from groundkit.retrieval.search import MAX_TOP_K, Retriever

#: Characters of chunk content shown per result in text output.
_SNIPPET_CHARS: int = 160

#: Floor on ``grk eval --top-k``: recall@10 cannot be computed from fewer
#: than 10 retrieved results per query.
_MIN_EVAL_TOP_K: int = 10


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

    eval_parser = sub.add_parser(
        "eval", help="Run the BM25-baseline retrieval eval against a golden corpus."
    )
    eval_parser.add_argument(
        "--corpus-dir", default="evals/corpus", help="Golden corpus directory."
    )
    eval_parser.add_argument(
        "--judgments", default="evals/judgments.jsonl", help="Judgments JSONL file."
    )
    eval_parser.add_argument(
        "--output",
        default="evals/results/latest.json",
        help="Path to write the JSON eval report.",
    )
    eval_parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help=f"Results retrieved per query (minimum {_MIN_EVAL_TOP_K}).",
    )
    eval_parser.add_argument("--json", action="store_true", help="Emit the full report as JSON.")
    eval_parser.set_defaults(func=_cmd_eval)

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


async def _cmd_eval(args: argparse.Namespace) -> int:
    if args.top_k < _MIN_EVAL_TOP_K:
        raise EvalError(
            f"--top-k must be at least {_MIN_EVAL_TOP_K} (recall@10 cannot be computed "
            f"from fewer than {_MIN_EVAL_TOP_K} retrieved results), got {args.top_k}"
        )

    judgments_path = Path(args.judgments)
    if not judgments_path.is_file():
        # groundkit.evals.corpus.load_judgments reads this path unguarded, so a
        # missing file must be caught here to fail closed with an EvalError
        # rather than an unhandled FileNotFoundError escaping main().
        raise EvalError(f"judgments file not found: {judgments_path}")

    report = await run_eval(Path(args.corpus_dir), judgments_path, top_k=args.top_k)
    output_path = Path(args.output)
    write_report(report, output_path)

    if args.json:
        print(json.dumps(report.model_dump(), indent=2))
        return 0

    _print_eval_summary(report, output_path)
    return 0


def _print_eval_summary(report: EvalReport, output_path: Path) -> None:
    stage = report.stages[0]
    metrics = stage.aggregate
    print(
        f"eval: stage={stage.stage} queries={metrics.query_count} "
        f"recall@1={metrics.recall_at_1:.3f} recall@5={metrics.recall_at_5:.3f} "
        f"recall@10={metrics.recall_at_10:.3f} mrr={metrics.mrr:.3f} "
        f"ndcg@10={metrics.ndcg_at_10:.3f}"
    )
    print(f"no_answer: {stage.no_answer_abstained_count}/{stage.no_answer_query_count} abstained")
    print(f"latency: p50={stage.latency_p50_ms:.1f}ms p95={stage.latency_p95_ms:.1f}ms")
    print(f"report written to {output_path}")

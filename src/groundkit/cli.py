"""Command-line entry point (``grk``).

Phase 1 shipped ``ingest`` and ``search`` end-to-end against the persisted
local index with zero cloud credentials. Phase 2 adds ``eval``, running the
BM25-baseline retrieval harness (``groundkit.evals.runner``) against a
golden corpus and judgment set. Phase 3 Wave C wires the dense write and
read paths into both commands: ``grk ingest --dense`` embeds and writes
vectors alongside the existing SQLite write, and ``grk search --mode
{bm25,dense,hybrid}`` can read them back. Both are opt-in and default off.
``search --mode`` in particular defaults to, and stays, ``"bm25"``. Wave E
measured the delta Q1 was waiting on and hybrid won it on every quality
metric — and ADR-0007 still keeps BM25 as the default, because hybrid cannot
abstain (ADR-0005 decision 6 keeps ``score_threshold`` away from fused
scores, so no configuration lets it return nothing for an unanswerable
question) and because a hybrid default would require an embedding provider
the default install does not ship, against SPEC.md §10. Hybrid is documented
as *recommended where a provider is configured*, which is a tradeoff a caller
opts into rather than one they inherit. ``serve`` and ``serve-mcp`` land in
their phases per SPEC.md §9.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from groundkit import __version__
from groundkit.config import EmbeddingConfig, RetrievalConfig
from groundkit.errors import ConfigurationError, EvalError, GroundkitError
from groundkit.evals.delta import StageDelta, derive_stage_deltas
from groundkit.evals.runner import MIN_EVAL_TOP_K, run_eval, write_report
from groundkit.evals.schema import EvalReport
from groundkit.index.dense import InMemoryVectorStore, LanceDBVectorStore
from groundkit.index.metadata import SQLiteMetadataStore
from groundkit.indexer import Indexer
from groundkit.ingestion.loaders import FileLoader
from groundkit.providers.embeddings import INMEMORY_PROVIDER, build_embedder
from groundkit.retrieval.search import MAX_TOP_K, Retriever

if TYPE_CHECKING:
    from groundkit.index.protocols import VectorStoreProtocol
    from groundkit.providers.protocols import EmbeddingProtocol

#: Characters of chunk content shown per result in text output.
_SNIPPET_CHARS: int = 160

#: ``--embed-*`` flags shared by ``ingest --dense`` and ``search --mode
#: {dense,hybrid}``, and the ``argparse.Namespace`` attribute each maps to.
_EMBED_FLAG_ATTRS: tuple[str, ...] = (
    "embed_provider",
    "embed_model",
    "embed_dimensions",
    "embed_base_url",
)


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
    ingest.add_argument(
        "--dense",
        action="store_true",
        help="Also embed and write vectors to the collection's LanceDB store (opt-in).",
    )
    _add_embedding_args(ingest)
    ingest.set_defaults(func=_cmd_ingest)

    search = sub.add_parser("search", help="Search the local index.")
    search.add_argument("query", help="Query text.")
    search.add_argument("--index-dir", default=".groundkit", help="Index directory.")
    search.add_argument("--collection", default="default", help="Collection name.")
    search.add_argument(
        "--top-k", type=int, default=None, help=f"Results to return (1-{MAX_TOP_K})."
    )
    search.add_argument("--json", action="store_true", help="Emit the full response as JSON.")
    search.add_argument(
        "--mode",
        choices=["bm25", "dense", "hybrid"],
        default="bm25",
        help=(
            "Retrieval mode. Defaults to 'bm25'. On the golden corpus, 'hybrid' measured "
            "better than the BM25 baseline on every retrieval-quality metric -- but it "
            "cannot abstain: unlike bm25, it returns its top-k for a question the corpus "
            "cannot answer, and no score threshold applies to fused scores. It also needs "
            "a running embedding provider. Recommended where you have one and can accept "
            "that tradeoff; see ADR-0007."
        ),
    )
    _add_embedding_args(search)
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
        help=f"Results retrieved per query ({MIN_EVAL_TOP_K}-{MAX_TOP_K}).",
    )
    eval_parser.add_argument("--json", action="store_true", help="Emit the full report as JSON.")
    eval_parser.add_argument(
        "--dense",
        action="store_true",
        help=(
            "Also evaluate the dense and fusion stages, reporting each one's delta "
            "against the BM25 baseline. Requires a working embedding provider; with "
            "--embed-provider inmemory the stages run but their quality numbers are "
            "hash-derived noise, not a measurement (SPEC.md §2)."
        ),
    )
    _add_embedding_args(eval_parser)
    eval_parser.set_defaults(func=_cmd_eval)

    return parser


def _add_embedding_args(parser: argparse.ArgumentParser) -> None:
    """Add the ``--embed-*`` flags shared by ``ingest --dense`` and ``search --mode``.

    All four default to ``None`` and are only resolved into an
    :class:`~groundkit.config.EmbeddingConfig` (falling back to its own
    defaults — ``ollama`` / ``nomic-embed-text`` / 768 dimensions /
    :data:`~groundkit.config.DEFAULT_OLLAMA_BASE_URL`) once the dense path
    is actually active. Supplying any of them while the dense path is
    inactive is a :class:`~groundkit.errors.ConfigurationError` (see
    :func:`_embed_flags_supplied`) rather than a silently ignored flag.
    """
    parser.add_argument(
        "--embed-provider",
        choices=["ollama", "openai_compatible", "inmemory"],
        default=None,
        help=(
            "Embedding provider for the dense path (default: ollama). 'inmemory' is an "
            "offline test double with no semantic signal — never use it to measure "
            "retrieval quality."
        ),
    )
    parser.add_argument(
        "--embed-model", default=None, help="Embedding model name (default: nomic-embed-text)."
    )
    parser.add_argument(
        "--embed-dimensions",
        type=int,
        default=None,
        help="Expected embedding vector width (default: 768).",
    )
    parser.add_argument(
        "--embed-base-url",
        default=None,
        help="Embedding provider endpoint (default: the local Ollama endpoint).",
    )


def _embed_flags_supplied(args: argparse.Namespace) -> bool:
    """True if any ``--embed-*`` flag was explicitly supplied."""
    return any(getattr(args, attr) is not None for attr in _EMBED_FLAG_ATTRS)


def _resolve_embedding_config(args: argparse.Namespace) -> EmbeddingConfig:
    """Build an :class:`EmbeddingConfig` from ``--embed-*`` flags, unsupplied ones defaulted.

    Built via explicit keyword construction rather than a ``dict[str, ...]``
    splat: a heterogeneous override dict (``provider`` is a ``Literal``,
    ``dimensions`` an ``int``, the rest ``str``) cannot type-check cleanly
    against ``EmbeddingConfig``'s typed fields under ``mypy --strict``.
    ``argparse.Namespace`` attributes are typed ``Any``, so passing them
    straight through here type-checks without a cast, and pydantic still
    validates the ``provider`` literal at construction (argparse's ``choices``
    already constrains it, so this is defense in depth, not the only check).
    Defaults come from a fresh :class:`EmbeddingConfig` — one source of truth,
    not a second copy of its field defaults.

    Pydantic's own field invariants (``dimensions`` must be ``> 0``, and any
    other bound :class:`EmbeddingConfig` grows later) are enforced here and
    nowhere else on this path, so the ``ValidationError`` they raise is
    translated into a :class:`~groundkit.errors.ConfigurationError`.
    Untranslated it is not a ``GroundkitError``, so ``main``'s handler does
    not see it and ``grk ingest --dense --embed-dimensions 0`` exits on a
    pydantic traceback rather than the one-line ``error:`` message every
    other bad flag produces. Translating at this single construction site
    rather than re-checking each bound in argparse keeps
    :class:`EmbeddingConfig` the only place a bound is stated.

    Raises:
        ConfigurationError: A supplied ``--embed-*`` value violates an
            :class:`EmbeddingConfig` invariant.
    """
    defaults = EmbeddingConfig()
    try:
        return EmbeddingConfig(
            provider=args.embed_provider if args.embed_provider is not None else defaults.provider,
            model_name=args.embed_model if args.embed_model is not None else defaults.model_name,
            dimensions=(
                args.embed_dimensions if args.embed_dimensions is not None else defaults.dimensions
            ),
            base_url=args.embed_base_url if args.embed_base_url is not None else defaults.base_url,
        )
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        raise ConfigurationError(f"invalid embedding configuration ({details})") from exc


async def _open_dense_deps(
    args: argparse.Namespace,
) -> tuple[EmbeddingProtocol, VectorStoreProtocol]:
    """Build the embedder + LanceDB store pair shared by ``ingest --dense`` and dense/hybrid search.

    Layout is pinned: the LanceDB data for a collection lives at
    ``<index-dir>/<collection>.lance`` (a directory, sibling of
    ``<collection>.sqlite3``), opened with the default table name.
    """
    embedder = build_embedder(_resolve_embedding_config(args))
    vector_store = await LanceDBVectorStore.open(Path(args.index_dir) / f"{args.collection}.lance")
    return embedder, vector_store


async def _maybe_aclose(embedder: EmbeddingProtocol | None) -> None:
    """Close ``embedder``'s underlying resources if it exposes ``aclose``.

    ``aclose`` is not part of :class:`EmbeddingProtocol` — ``InMemoryEmbedder``
    has nothing to release and does not define it, while the HTTP-backed
    providers do (``providers/embeddings.py``'s ``_HttpEmbedder.aclose``).
    Duck-typed rather than an isinstance check against a concrete class, so a
    future embedder with its own ``aclose`` is closed too without this
    function needing to know about it.
    """
    if embedder is None:
        return
    aclose = getattr(embedder, "aclose", None)
    if callable(aclose):
        await aclose()


async def _cmd_ingest(args: argparse.Namespace) -> int:
    if not args.dense and _embed_flags_supplied(args):
        raise ConfigurationError(
            "--embed-provider/--embed-model/--embed-dimensions/--embed-base-url require "
            "--dense; without --dense there is no dense path for them to configure"
        )

    path = Path(args.path)
    if args.base_dir is not None:
        base_dir = Path(args.base_dir)
    elif path.is_dir():
        base_dir = path
    else:
        base_dir = path.parent

    # SQLiteMetadataStore.open validates the collection name before any
    # path (including the LanceDB directory below) is built from it.
    store = await SQLiteMetadataStore.open(Path(args.index_dir), args.collection)
    embedder: EmbeddingProtocol | None = None
    try:
        vector_store: VectorStoreProtocol | None = None
        if args.dense:
            embedder, vector_store = await _open_dense_deps(args)
        indexer = Indexer(
            store,
            FileLoader(allowed_base_dir=base_dir),
            embedder=embedder,
            vector_store=vector_store,
        )
        if path.is_dir():
            report = await indexer.index_directory(str(path))
        else:
            report = await indexer.index_source(str(path))
    finally:
        await store.close()
        await _maybe_aclose(embedder)

    line = (
        f"ingested: {report.files_seen} files seen, "
        f"{report.documents_indexed} indexed, "
        f"{report.documents_skipped} unchanged, "
        f"{report.chunks_written} chunks written"
    )
    if args.dense:
        line += (
            f", {report.vectors_written} vectors written, {report.vectors_deleted} vectors deleted"
        )
    print(line)
    return 0


async def _cmd_search(args: argparse.Namespace) -> int:
    if args.mode == "bm25" and _embed_flags_supplied(args):
        raise ConfigurationError(
            "--embed-provider/--embed-model/--embed-dimensions/--embed-base-url require "
            "--mode dense or --mode hybrid; bm25 mode never touches the dense path"
        )

    # SQLiteMetadataStore.open validates the collection name before any
    # path (including the LanceDB directory below) is built from it.
    store = await SQLiteMetadataStore.open(Path(args.index_dir), args.collection)
    embedder: EmbeddingProtocol | None = None
    try:
        vector_store: VectorStoreProtocol | None = None
        if args.mode != "bm25":
            embedder, vector_store = await _open_dense_deps(args)
        retriever = await Retriever.open(
            store, RetrievalConfig(), embedder=embedder, vector_store=vector_store
        )
        response = await retriever.search(args.query, top_k=args.top_k, mode=args.mode)
    finally:
        await store.close()
        await _maybe_aclose(embedder)

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
    # Both bounds are checked before anything is constructed or ingested. The
    # upper one matters most on the dense path: without it, an out-of-range
    # --top-k embeds the whole corpus and only then gets rejected inside the
    # stage loop, spending real provider work — billable, against a hosted
    # endpoint — on an invocation that could never have produced a report.
    # run_eval enforces both bounds itself; these restate them in --top-k's
    # own vocabulary, and share run_eval's constants so the two cannot drift.
    if args.top_k < MIN_EVAL_TOP_K:
        raise EvalError(
            f"--top-k must be at least {MIN_EVAL_TOP_K} (recall@10 cannot be computed "
            f"from fewer than {MIN_EVAL_TOP_K} retrieved results), got {args.top_k}"
        )
    if args.top_k > MAX_TOP_K:
        raise EvalError(
            f"--top-k must be at most {MAX_TOP_K} (the retrieval cap enforced by "
            f"Retriever.search), got {args.top_k}"
        )
    if not args.dense and _embed_flags_supplied(args):
        raise ConfigurationError(
            "--embed-provider/--embed-model/--embed-dimensions/--embed-base-url require "
            "--dense; without --dense the eval runs the BM25 baseline stage only"
        )

    judgments_path = Path(args.judgments)
    if not judgments_path.is_file():
        # groundkit.evals.corpus.load_judgments reads this path unguarded, so a
        # missing file must be caught here to fail closed with an EvalError
        # rather than an unhandled FileNotFoundError escaping main().
        raise EvalError(f"judgments file not found: {judgments_path}")

    embedder: EmbeddingProtocol | None = None
    try:
        vector_store: VectorStoreProtocol | None = None
        if args.dense:
            # A throwaway in-memory store, not the LanceDB path ingest and
            # search use: run_eval builds its whole index in an OS temp
            # directory per run, so there is no collection for dense vectors
            # to persist into and nothing that would outlive the run.
            embedder = build_embedder(_resolve_embedding_config(args))
            vector_store = InMemoryVectorStore()
        report = await run_eval(
            Path(args.corpus_dir),
            judgments_path,
            top_k=args.top_k,
            embedder=embedder,
            vector_store=vector_store,
        )
    finally:
        await _maybe_aclose(embedder)

    output_path = Path(args.output)
    write_report(report, output_path)

    if args.json:
        print(json.dumps(report.model_dump(), indent=2))
        return 0

    _print_eval_summary(report, output_path)
    return 0


def _print_eval_summary(report: EvalReport, output_path: Path) -> None:
    """Print every stage and, for non-baseline stages, its delta vs baseline.

    Prints stages unconditionally and in report order, including stages that
    lost to the baseline: SPEC.md §6 requires a feature that does not beat
    baseline to be *reported as such*, so there is deliberately no filtering,
    sorting-by-winner, or "only show improvements" mode here.
    """
    for stage in report.stages:
        metrics = stage.aggregate
        marker = " (baseline)" if stage.is_baseline else ""
        print(
            f"eval: stage={stage.stage}{marker} queries={metrics.query_count} "
            f"recall@1={metrics.recall_at_1:.3f} recall@5={metrics.recall_at_5:.3f} "
            f"recall@10={metrics.recall_at_10:.3f} mrr={metrics.mrr:.3f} "
            f"ndcg@10={metrics.ndcg_at_10:.3f}"
        )
        print(
            f"  no_answer: {stage.no_answer_abstained_count}/{stage.no_answer_query_count} "
            f"abstained | latency: p50={stage.latency_p50_ms:.1f}ms "
            f"p95={stage.latency_p95_ms:.1f}ms p99={stage.latency_p99_ms:.1f}ms"
        )

    for delta in derive_stage_deltas(report):
        # Signs are always explicit: a bare "0.040" leaves the reader to
        # infer direction, which is exactly what baseline discipline is
        # meant to remove.
        quality = " ".join(f"{name}={value:+.3f}" for name, value in delta.quality.items())
        print(f"delta[{delta.stage} vs {delta.baseline_stage}]: {quality}")
        print(
            f"  latency: p50={delta.latency_p50_delta_ms:+.1f}ms "
            f"p95={delta.latency_p95_delta_ms:+.1f}ms "
            f"p99={delta.latency_p99_delta_ms:+.1f}ms"
        )
        print(f"  {_delta_verdict(delta)}")

    embedding = report.run.config.embedding
    if embedding is not None:
        print(
            f"embedding: provider={embedding.provider} model={embedding.model_name} "
            f"dimensions={embedding.dimensions}"
        )
        if embedding.provider == INMEMORY_PROVIDER:
            # ASCII only — see the matching note in evals/runner.py: a
            # section sign printed to a cp1252 console becomes a
            # replacement character, in the middle of the one line that
            # most needs to be read literally.
            print(
                "  WARNING: this provider produces hash-derived vectors with no semantic "
                "signal. The dense and fusion numbers above exercise the code paths and "
                "are NOT a retrieval-quality measurement (SPEC.md section 2)."
            )
    print(f"corpus: {report.run.document_count} documents, {report.run.chunk_count} chunks")
    print(f"report written to {output_path}")


def _delta_verdict(delta: StageDelta) -> str:
    """One line naming the direction of a stage's delta, never hiding a loss.

    Reports "mixed" when a stage both gained and lost metrics rather than
    collapsing it to a single winner, and applies no noise threshold — see
    :mod:`groundkit.evals.delta` for why a tolerance band would be an
    invented number on a corpus this size.
    """
    if delta.is_regression and delta.is_improvement:
        return "MIXED vs baseline: some metrics improved, some regressed."
    if delta.is_regression:
        return "REGRESSION vs baseline: this stage does not beat BM25."
    if delta.is_improvement:
        return "improvement vs baseline."
    return "no change vs baseline on any metric."

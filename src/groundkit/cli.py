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
opts into rather than one they inherit.

Phase 5 adds ``answer`` — retrieval composed with the optional LLM boundary
(ADR-0019): optional query rewrite, cited synthesis whose citations can only
be the retrieved results' own (ADR-0018), and the advisory faithfulness
judge. It is a new verb rather than a ``search`` flag because a rewrite makes
"the query" two strings and ``SearchResponse`` has one field for it; it is
CLI-only because ADR-0019 keeps synthesis off the service surface (cost and
egress amplification are not bounded by a loopback bind). ``grk eval
--synthesis`` runs the planted-marker citation-echo check (SPEC.md §2)
against a real chat provider, writing its own artifact — there is
deliberately no offline double for it, because an echo number from one would
be noise presented as a measurement.

Phase 4 adds ``serve`` and ``serve-mcp``, the read-only service surface: one
FastAPI app carrying both the REST routes and the mounted MCP streamable-HTTP
transport, and the MCP stdio transport respectively (ADR-0014 decision 5).
Both build one :class:`~groundkit.service.tools.ServiceContext` over one
:class:`~groundkit.runtime.CollectionRegistry` at serve time, so the index
directory, the containment root, the reranker and every provider setting are
resolved from the operator's own arguments and are unreachable from a request
(ADR-0014 decision 6). ``grk serve`` binds loopback unless the operator
acknowledges the exposure, which — with no authentication anywhere in Phase 4
— is the service's only access control (ADR-0014 decision 7,
:mod:`groundkit.service.binding`).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from groundkit import __version__
from groundkit.answer import AnswerPipeline, AnswerReport
from groundkit.config import (
    DEFAULT_CHAT_MODEL,
    ChatConfig,
    EmbeddingConfig,
    RetrievalConfig,
    resolve_chat_config,
    resolve_embedding_config,
)
from groundkit.contracts import Citation
from groundkit.errors import ConfigurationError, EvalError, GroundkitError
from groundkit.evals.delta import StageDelta, derive_rerank_attribution, derive_stage_deltas
from groundkit.evals.echo import (
    DEFAULT_ECHO_REPORT_PATH,
    EchoReport,
    run_echo_check,
    write_echo_report,
)
from groundkit.evals.judge import FaithfulnessJudge
from groundkit.evals.runner import MIN_EVAL_TOP_K, run_eval, write_report
from groundkit.evals.schema import EvalReport
from groundkit.index.dense import InMemoryVectorStore, LanceDBVectorStore
from groundkit.index.metadata import SQLiteMetadataStore
from groundkit.indexer import Indexer
from groundkit.ingestion.loaders import FileLoader
from groundkit.providers.embeddings import INMEMORY_PROVIDER, build_embedder
from groundkit.providers.llm import build_chat
from groundkit.providers.query_rewrite import QueryRewriter
from groundkit.providers.synthesis import Synthesizer

# Importing this module never imports torch: `retrieval/rerank.py` defers the
# optional dependency to `_import_cross_encoder`, which nothing reaches until a
# reranker actually loads a model. `grk --help` on a base install therefore
# costs nothing, and `--rerank` fails at first use with the typed error that
# names the install command.
from groundkit.retrieval.rerank import DEFAULT_RERANK_MODEL, CrossEncoderReranker
from groundkit.retrieval.search import MAX_TOP_K, Retriever
from groundkit.runtime import CollectionRegistry

# Importing this module never imports FastAPI, uvicorn or the MCP SDK either.
# `binding.py` is stdlib + `errors.py` only, and `service/tools.py` reaches no
# further than pydantic and `contracts.py`; the web framework, the ASGI server
# and the SDK are imported inside `_serve_http`/`_build_mcp_mount`/`_cmd_serve_mcp`
# instead. They are base dependencies (ADR-0015), so the deferral is about
# startup cost — `grk search` must not pay to import a server it never runs —
# and never about availability.
from groundkit.service.binding import DEFAULT_SERVE_HOST, DEFAULT_SERVE_PORT, ensure_bindable_host
from groundkit.service.tools import ServiceContext

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from starlette.types import Receive, Scope, Send

    from groundkit.index.protocols import VectorStoreProtocol
    from groundkit.providers.protocols import ChatProtocol, EmbeddingProtocol
    from groundkit.service.api import McpMount

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

#: ``--chat-*`` flags shared by ``answer`` and ``eval --synthesis``, and the
#: ``argparse.Namespace`` attribute each maps to.
_CHAT_FLAG_ATTRS: tuple[str, ...] = (
    "chat_provider",
    "chat_model",
    "chat_base_url",
    "chat_api_key_env",
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

    answer = sub.add_parser(
        "answer",
        help=(
            "Retrieve, then synthesize a cited answer with a chat model (ADR-0019). "
            "The model may cite only retrieved spans; an out-of-set citation is an "
            "error, never repaired (ADR-0018)."
        ),
    )
    answer.add_argument("query", help="The question to answer.")
    answer.add_argument("--index-dir", default=".groundkit", help="Index directory.")
    answer.add_argument("--collection", default="default", help="Collection name.")
    answer.add_argument(
        "--top-k", type=int, default=None, help=f"Results to retrieve (1-{MAX_TOP_K})."
    )
    answer.add_argument(
        "--mode",
        choices=["bm25", "dense", "hybrid"],
        default="bm25",
        help="Retrieval mode feeding synthesis. Same semantics and tradeoffs as grk search.",
    )
    answer.add_argument(
        "--rewrite",
        action="store_true",
        help=(
            "Rewrite the query with the chat model before retrieval. Retrieval runs on "
            "the rewritten query; synthesis and the judge still answer the original. A "
            "rewrite failure is an error, never a silent fallback to the original query."
        ),
    )
    answer.add_argument(
        "--judge",
        action="store_true",
        help=(
            "Run the advisory faithfulness judge over the answer. Advisory only: the "
            "verdict is reported and gates nothing — the exit code does not depend on "
            "it (SPEC.md §6; uncalibrated against human labels)."
        ),
    )
    answer.add_argument("--json", action="store_true", help="Emit the full report as JSON.")
    _add_embedding_args(answer)
    _add_chat_args(answer)
    answer.set_defaults(func=_cmd_answer)

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
    eval_parser.add_argument(
        "--rerank",
        action="store_true",
        help=(
            "Also evaluate a cross-encoder rerank stage over the best upstream stage "
            "the run produced (fusion with --dense, otherwise the BM25 baseline), "
            "reporting its delta against the baseline AND against that input stage. "
            "Requires the optional 'rerank' extra: pip install groundkit[rerank]."
        ),
    )
    eval_parser.add_argument(
        "--rerank-model",
        default=None,
        help=f"Cross-encoder for --rerank (default: {DEFAULT_RERANK_MODEL}).",
    )
    eval_parser.add_argument(
        "--synthesis",
        action="store_true",
        help=(
            "Also run the planted-marker citation-echo check (SPEC.md §2) through the "
            "configured chat provider, writing its own artifact "
            f"({DEFAULT_ECHO_REPORT_PATH}) alongside the main report. Requires a "
            "running chat provider: there is deliberately no offline double for this "
            "check, because an echo number from one would be noise presented as a "
            "measurement (SPEC.md §2)."
        ),
    )
    _add_embedding_args(eval_parser)
    _add_chat_args(eval_parser)
    eval_parser.set_defaults(func=_cmd_eval)

    serve = sub.add_parser(
        "serve",
        help="Serve the read-only REST + MCP streamable-HTTP surface over one runtime.",
    )
    _add_serve_args(serve)
    serve.add_argument(
        "--host",
        default=DEFAULT_SERVE_HOST,
        help=(
            f"Address to bind (default: {DEFAULT_SERVE_HOST}). A non-loopback address is "
            "refused unless --allow-remote-access is also passed; a hostname such as "
            "'localhost' is refused too, because only an address literal can be "
            "classified without a resolver whose answer can change."
        ),
    )
    serve.add_argument(
        "--port",
        type=int,
        default=DEFAULT_SERVE_PORT,
        help=f"Port (default: {DEFAULT_SERVE_PORT}).",
    )
    serve.add_argument(
        "--allow-remote-access",
        action="store_true",
        help=(
            "Acknowledge binding a non-loopback address. This server has NO "
            "authentication of any kind, so anyone who can reach the port can read "
            "indexed document content and the absolute filesystem paths it was ingested "
            "from. Passing this publishes the corpus."
        ),
    )
    serve.set_defaults(func=_cmd_serve)

    serve_mcp = sub.add_parser(
        "serve-mcp",
        help="Serve the MCP stdio transport (for Claude Desktop / Claude Code).",
    )
    _add_serve_args(serve_mcp)
    serve_mcp.set_defaults(func=_cmd_serve_mcp)

    return parser


def _add_serve_args(parser: argparse.ArgumentParser) -> None:
    """Add the flags ``serve`` and ``serve-mcp`` share.

    ``--host``, ``--port`` and ``--allow-remote-access`` are deliberately
    *not* here: only the HTTP transport binds a socket, so ADR-0014 decision
    7 applies to ``grk serve`` alone, and offering the flags on ``grk
    serve-mcp`` would advertise a guard with nothing to guard.

    Everything here is resolved once, at serve time, and reaches the handlers
    only through :class:`~groundkit.service.tools.ServiceContext` — ADR-0014
    decision 6, which is why no request model carries an index directory, a
    containment root or a provider setting.
    """
    parser.add_argument(
        "--index-dir",
        default=".groundkit",
        help=(
            "Index directory. Must already exist: the service never creates an index "
            "directory or a collection (ADR-0014 decision 3)."
        ),
    )
    parser.add_argument(
        "--base-dir",
        required=True,
        help=(
            "Containment root every citation must resolve within. Required, not "
            "defaulted: it is what resolve_citation checks against, and a service that "
            "cannot verify any citation must not start, because verifiable citations are "
            "the product claim (SPEC.md section 2, ADR-0014 decision 6)."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=(
            f"Results returned for a request that omits top_k (1-{MAX_TOP_K}). Defaults "
            "to ServiceContext's own default rather than a second copy of it."
        ),
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        help=(
            "Serve dense and hybrid retrieval modes as well as bm25, reading each "
            "collection's LanceDB store at <index-dir>/<collection>.lance. Opt-in "
            "because it requires the optional 'dense' extra, which a default install "
            "does not carry: pip install groundkit[dense]."
        ),
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help=(
            "Load a cross-encoder so requests may ask for rerank=true. Without it, such "
            "a request is refused rather than served unreranked. Requires the optional "
            "'rerank' extra: pip install groundkit[rerank]."
        ),
    )
    parser.add_argument(
        "--rerank-model",
        default=None,
        help=f"Cross-encoder for --rerank (default: {DEFAULT_RERANK_MODEL}).",
    )
    _add_embedding_args(parser)


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


def _add_chat_args(parser: argparse.ArgumentParser) -> None:
    """Add the ``--chat-*`` flags shared by ``answer`` and ``eval --synthesis``.

    All four default to ``None`` and are resolved into a
    :class:`~groundkit.config.ChatConfig` (falling back to its own defaults)
    by :func:`_resolve_chat_config`, the same shape the ``--embed-*`` flags
    follow. The cloud path's egress is always redaction-wrapped by
    :func:`~groundkit.providers.llm.build_chat` — there is no flag to turn
    that off, by design (ADR-0017).
    """
    parser.add_argument(
        "--chat-provider",
        choices=["ollama", "openai_compatible"],
        default=None,
        help=(
            "Chat provider for the synthesis boundary (default: ollama, local). "
            "'openai_compatible' egress is always redaction-wrapped; there is no "
            "opt-out (ADR-0017)."
        ),
    )
    parser.add_argument(
        "--chat-model", default=None, help=f"Chat model name (default: {DEFAULT_CHAT_MODEL})."
    )
    parser.add_argument(
        "--chat-base-url",
        default=None,
        help="Chat provider endpoint (default: the local Ollama endpoint).",
    )
    parser.add_argument(
        "--chat-api-key-env",
        default=None,
        help=(
            "NAME of the environment variable holding the API key for "
            "openai_compatible (default: GROUNDKIT_OPENAI_API_KEY). The value is read "
            "at call time, never stored or logged (SPEC.md §7)."
        ),
    )


def _embed_flags_supplied(args: argparse.Namespace) -> bool:
    """True if any ``--embed-*`` flag was explicitly supplied."""
    return any(getattr(args, attr) is not None for attr in _EMBED_FLAG_ATTRS)


def _chat_flags_supplied(args: argparse.Namespace) -> bool:
    """True if any ``--chat-*`` flag was explicitly supplied."""
    return any(getattr(args, attr) is not None for attr in _CHAT_FLAG_ATTRS)


def _resolve_chat_config(args: argparse.Namespace) -> ChatConfig:
    """Unpack ``--chat-*`` flags and delegate to the promoted resolver.

    The exact peer of :func:`_resolve_embedding_config`; see
    :func:`groundkit.config.resolve_chat_config` for the defaulting rule and
    the ``ValidationError`` -> ``ConfigurationError`` translation.
    """
    return resolve_chat_config(
        provider=args.chat_provider,
        model_name=args.chat_model,
        base_url=args.chat_base_url,
        api_key_env=args.chat_api_key_env,
    )


def _resolve_embedding_config(args: argparse.Namespace) -> EmbeddingConfig:
    """Unpack ``--embed-*`` flags from ``args`` and delegate to the promoted resolver.

    See :func:`groundkit.config.resolve_embedding_config` for the defaulting
    rule, the ``ValidationError`` -> ``ConfigurationError`` translation, and
    why passing ``args.embed_*`` straight through here type-checks under
    ``mypy --strict`` with no ``cast``: ``argparse.Namespace`` attributes are
    typed ``Any``, which is assignable to that function's typed keyword
    parameters.

    Raises:
        ConfigurationError: A supplied ``--embed-*`` value violates an
            :class:`~groundkit.config.EmbeddingConfig` invariant.
    """
    return resolve_embedding_config(
        provider=args.embed_provider,
        model_name=args.embed_model,
        dimensions=args.embed_dimensions,
        base_url=args.embed_base_url,
    )


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


async def _maybe_aclose(provider: object | None) -> None:
    """Close ``provider``'s underlying resources if it exposes ``aclose``.

    ``aclose`` is part of neither :class:`EmbeddingProtocol` nor
    ``ChatProtocol`` — the in-memory/scripted doubles have nothing to release
    and do not define it, while the HTTP-backed providers do (and
    ``RedactingChat`` delegates it to whatever it wraps). Duck-typed rather
    than an isinstance check against a concrete class, so a future provider
    with its own ``aclose`` is closed too without this function needing to
    know about it.
    """
    if provider is None:
        return
    aclose = getattr(provider, "aclose", None)
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


async def _cmd_answer(args: argparse.Namespace) -> int:
    if args.mode == "bm25" and _embed_flags_supplied(args):
        raise ConfigurationError(
            "--embed-provider/--embed-model/--embed-dimensions/--embed-base-url require "
            "--mode dense or --mode hybrid; bm25 mode never touches the dense path"
        )

    # SQLiteMetadataStore.open validates the collection name before any
    # path (including the LanceDB directory below) is built from it.
    store = await SQLiteMetadataStore.open(Path(args.index_dir), args.collection)
    embedder: EmbeddingProtocol | None = None
    chat: ChatProtocol | None = None
    try:
        vector_store: VectorStoreProtocol | None = None
        if args.mode != "bm25":
            embedder, vector_store = await _open_dense_deps(args)
        retriever = await Retriever.open(
            store, RetrievalConfig(), embedder=embedder, vector_store=vector_store
        )
        chat = build_chat(_resolve_chat_config(args))
        pipeline = AnswerPipeline(
            retriever.search,
            Synthesizer(chat),
            rewriter=QueryRewriter(chat) if args.rewrite else None,
            judge=FaithfulnessJudge(chat) if args.judge else None,
        )
        report = await pipeline.answer(args.query, top_k=args.top_k, mode=args.mode)
    finally:
        await store.close()
        await _maybe_aclose(embedder)
        await _maybe_aclose(chat)

    if args.json:
        print(json.dumps(report.model_dump(), indent=2))
        return 0

    _print_answer_report(report)
    return 0


def _print_answer_report(report: AnswerReport) -> None:
    """Print the answer, its citations, and any advisory verdict.

    An abstention (empty ``citations``) is stated in words rather than left
    as an answer with nothing under it — the empty tuple IS the abstention
    signal (ADR-0018), and the console should say so instead of relying on a
    reader noticing an absence.

    The verdict, when present, is labeled advisory in the output itself: it
    gates nothing and the exit code never depends on it (SPEC.md §6).
    """
    if report.rewritten_query is not None:
        print(f"rewritten query: {report.rewritten_query}")
    print(report.answer)
    if not report.citations:
        print("(abstained: the answer cites no retrieved span)")
    else:
        # Labels reuse the answer's own [n] source numbers — each citation's
        # 1-based position in report.results — never a fresh 1..k renumbering
        # of the deduplicated list: an answer citing only source 3 must print
        # a "[3]" entry, or the marker in the answer text resolves to nothing
        # while "[1]" names a result the model never cited.
        source_numbers: dict[Citation, int] = {}
        for index, result in enumerate(report.results, start=1):
            source_numbers.setdefault(result.citation, index)
        print()
        for citation in report.citations:
            number = source_numbers[citation]
            print(f"[{number}] {citation.source}#{citation.start_offset}-{citation.end_offset}")
    if report.verdict is not None:
        verdict = "faithful" if report.verdict.faithful else "NOT faithful"
        print(f"judge (advisory, uncalibrated): {verdict}")
        for claim in report.verdict.unsupported_claims:
            print(f"  unsupported: {claim}")
        if report.verdict.reasoning:
            print(f"  reasoning: {report.verdict.reasoning}")


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
    if not args.rerank and args.rerank_model is not None:
        # Same fail-closed rule the --embed-* flags follow: a flag that
        # configures a path the run will not take is a mistake to name, not
        # one to ignore. Silently accepting it lets someone believe they
        # measured a model the run never loaded.
        raise ConfigurationError(
            "--rerank-model requires --rerank; without --rerank there is no rerank "
            "stage for it to configure"
        )
    if not args.synthesis and _chat_flags_supplied(args):
        # The same fail-closed rule the --embed-* flags follow.
        raise ConfigurationError(
            "--chat-provider/--chat-model/--chat-base-url/--chat-api-key-env require "
            "--synthesis; without --synthesis the eval never touches a chat provider"
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
        reranker = (
            CrossEncoderReranker(args.rerank_model or DEFAULT_RERANK_MODEL) if args.rerank else None
        )
        report = await run_eval(
            Path(args.corpus_dir),
            judgments_path,
            top_k=args.top_k,
            embedder=embedder,
            vector_store=vector_store,
            reranker=reranker,
            # Pinned to the retriever's own ceiling rather than exposed as a
            # flag (ADR-0012 decision 1a). The depth is what decides whether
            # the stage's @10 metrics are free to move at all, so a knob here
            # would let two CLI runs be silently incomparable in the one
            # dimension a reader is least likely to check. run_eval keeps the
            # parameter for library callers, and RunConfig records whatever
            # was used either way.
            rerank_candidates=MAX_TOP_K,
        )
    finally:
        await _maybe_aclose(embedder)

    output_path = Path(args.output)
    write_report(report, output_path)

    echo_report: EchoReport | None = None
    if args.synthesis:
        # The echo check builds its own synthetic marker corpus per run in an
        # OS temp directory (its markers are generated fresh, so nesting the
        # result inside the golden-corpus report would claim a provenance the
        # two runs do not share — ADR-0018). It writes its own artifact.
        chat: ChatProtocol | None = None
        try:
            chat = build_chat(_resolve_chat_config(args))
            with tempfile.TemporaryDirectory() as tmp_dir:
                echo_report = await run_echo_check(chat, corpus_dir=Path(tmp_dir))
        finally:
            await _maybe_aclose(chat)
        write_echo_report(echo_report, DEFAULT_ECHO_REPORT_PATH)

    if args.json:
        print(json.dumps(report.model_dump(), indent=2))
        return 0

    _print_eval_summary(report, output_path)
    if echo_report is not None:
        _print_echo_summary(echo_report)
    return 0


def _print_echo_summary(report: EchoReport) -> None:
    """Print the planted-marker echo check's aggregate counts and artifact path.

    Every count is printed, including the failure-side ones — the same
    no-filtering rule :func:`_print_eval_summary` follows: a check that
    exists to catch citation echo failing must never render only its
    successes.
    """
    print(
        f"echo: cases={report.case_count} correct={report.correct_count} "
        f"wrong_source={report.wrong_source_count} abstained={report.abstained_count} "
        f"rejected={report.rejected_count} leaked={report.leaked_count}"
    )
    print(f"  chat: provider={report.chat_provider} model={report.chat_model}")
    print(f"  echo report written to {DEFAULT_ECHO_REPORT_PATH}")


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
        _print_delta(delta)

    # Printed after the baseline deltas, never instead of them. On a dense
    # run the rerank row's baseline delta sums fusion's gain with the
    # reranker's; this is the reranker's own contribution, against the stage
    # it actually reordered (ADR-0012). Both are true, and only both together
    # answer "did the cross-encoder help".
    #
    # Skipped when the input stage IS the baseline — on a BM25-only run the
    # attribution is byte-identical to the delta just printed above, so this
    # suppresses a duplicate line, never a fact. `derive_rerank_attribution`
    # deliberately still returns it, so a library caller never has to
    # reimplement this check to know whether an attribution exists.
    attribution = derive_rerank_attribution(report)
    if attribution is not None and attribution.baseline_stage != report.stages[0].stage:
        _print_delta(attribution)

    _print_rerank_provenance(report)

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


def _print_delta(delta: StageDelta) -> None:
    """Print one derived delta, its latency change, and its verdict.

    Shared by the baseline deltas and the rerank attribution so the two
    render identically — they are the same type describing the same kind of
    comparison, and formatting them differently would suggest one is more
    authoritative than the other.

    Args:
        delta: The delta to render.
    """
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


def _print_rerank_provenance(report: EvalReport) -> None:
    """Print what the rerank stage reordered, with how much, if one ran.

    The rerank row is the only stage in the report whose meaning depends on
    how the run was configured (ADR-0012 decision 1), so the console states
    it rather than leaving a reader to open the JSON. The candidate depth is
    printed alongside because it is what decides whether the ``@10`` metrics
    were free to move: reranking truncates after reordering, so a depth equal
    to ``top_k`` can only permute a fixed set.

    Args:
        report: The completed report.
    """
    config = report.run.config
    if config.rerank_input is None:
        return
    print(
        f"rerank: input={config.rerank_input} model={config.rerank_model} "
        f"candidates={config.rerank_candidates} truncated_to={config.top_k}"
    )
    if config.rerank_candidates == config.top_k:
        # ASCII only, for the reason the embedding warning in
        # `_print_eval_summary` gives.
        print(
            "  WARNING: candidate depth equals top_k, so the reranker could only "
            "permute a fixed set. The @10 metrics above are pinned to the input "
            "stage's by construction, not measured."
        )


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


# --- Phase 4: the read-only service surface (ADR-0014, ADR-0015) -----------


def _build_service_context(args: argparse.Namespace) -> ServiceContext:
    """Assemble the serve-time context both transports share.

    Every setting a handler can reach is resolved *here*, once, from the
    operator's own arguments. That is what makes ADR-0014 decision 6 hold by
    construction rather than by review: there is no resolution path inside
    ``service/`` for a request to reach, so no request model needs — or is
    allowed — an ``index_dir``, a ``base_dir`` or an ``embed_*`` field.

    ``default_top_k`` is applied with :func:`dataclasses.replace` only when
    ``--top-k`` was supplied, so the unflagged default stays
    :class:`~groundkit.service.tools.ServiceContext`'s own rather than a second
    copy of it living here — the same single-source-of-truth rule
    :func:`groundkit.config.resolve_embedding_config` follows for
    ``EmbeddingConfig``.

    Raises:
        ConfigurationError: ``--rerank-model`` without ``--rerank``; any
            ``--embed-*`` flag (see below); an out-of-range ``--top-k``; a
            missing ``--index-dir``; or a missing ``--base-dir``.
    """
    if not args.rerank and args.rerank_model is not None:
        # The same fail-closed rule `grk eval` applies: a flag configuring a
        # path this process will not take is a mistake to name, not one to
        # ignore. Silently accepting it lets an operator believe requests are
        # being reranked by a model this server never loaded.
        raise ConfigurationError(
            "--rerank-model requires --rerank; without --rerank this server holds no "
            "reranker for it to configure"
        )

    if not args.dense and _embed_flags_supplied(args):
        # The same fail-closed rule `grk ingest` and `grk search` apply: a flag
        # configuring a path this process will not take is a mistake to name,
        # not one to ignore.
        raise ConfigurationError(
            "--embed-provider/--embed-model/--embed-dimensions/--embed-base-url require "
            "--dense; without --dense this server serves bm25 only and has no dense path "
            "for them to configure"
        )

    if args.top_k is not None and not 1 <= args.top_k <= MAX_TOP_K:
        # Checked at startup because this value is the default applied to
        # every request that omits top_k: out of range, it would fail those
        # requests one at a time, from a fault the operator committed once.
        raise ConfigurationError(f"--top-k must be between 1 and {MAX_TOP_K}, got {args.top_k}")

    index_dir = Path(args.index_dir)
    if not index_dir.is_dir():
        raise ConfigurationError(
            f"--index-dir {index_dir} does not exist. The service never creates an index "
            "directory or a collection (ADR-0014 decision 3): reading one into existence "
            "would make an unauthenticated surface a disk-fill primitive. Run "
            "`grk ingest` first."
        )

    base_dir = Path(args.base_dir)
    if not base_dir.is_dir():
        raise ConfigurationError(
            f"--base-dir {base_dir} does not exist. It is the containment root every "
            "citation must resolve within, so a root that is not there can verify "
            "nothing, and a service that cannot verify any citation must not start "
            "(ADR-0014 decision 6)."
        )

    # Constructed, never loaded: CrossEncoderReranker.__init__ touches no
    # model, no filesystem and no optional extra, so a server started with
    # --rerank on an install without the extra starts, and the first request
    # asking for rerank=true fails with RerankerNotConfiguredError instead of
    # being served an unreranked list that would look identical to a reranked
    # one.
    reranker = (
        CrossEncoderReranker(args.rerank_model or DEFAULT_RERANK_MODEL) if args.rerank else None
    )

    # The registry's factory takes the COLLECTION NAME, unlike a single
    # runtime's, which takes nothing. That is what makes serving more than one
    # dense collection correct: the LanceDB store is laid out per collection,
    # so a collection-agnostic factory would search one collection's vectors
    # and join the resulting chunk ids against another's SQLite — no error, just
    # a silently thin result on a healthy collection.
    async def _open_collection_store(collection: str) -> VectorStoreProtocol:
        return await LanceDBVectorStore.open(index_dir / f"{collection}.lance")

    embedder: EmbeddingProtocol | None = None
    vector_store_factory: Callable[[str], Awaitable[VectorStoreProtocol]] | None = None
    if args.dense:
        embedder = build_embedder(_resolve_embedding_config(args))
        vector_store_factory = _open_collection_store

    ctx = ServiceContext(
        registry=CollectionRegistry(
            index_dir,
            RetrievalConfig(),
            embedder=embedder,
            vector_store_factory=vector_store_factory,
        ),
        index_dir=index_dir,
        base_dir=base_dir,
        reranker=reranker,
    )
    if args.top_k is None:
        return ctx
    return replace(ctx, default_top_k=args.top_k)


def _build_mcp_mount(ctx: ServiceContext) -> McpMount:
    """Adapt the MCP session manager into the mount :func:`create_app` accepts.

    ``StreamableHTTPSessionManager`` exposes two halves that ASGI keeps apart:
    ``run()`` is an async context manager owning the transport's lifecycle,
    and ``handle_request(scope, receive, send)`` is the per-connection entry
    point. :class:`~groundkit.service.api.McpMount` carries them separately
    for exactly that reason — the app's lifespan drives the first, the router
    the second — so the method is wrapped in an ASGI callable here rather than
    the manager being handed over as though it were an app.
    """
    from groundkit.service.api import McpMount
    from groundkit.service.mcp_server import MCP_HTTP_PATH, create_session_manager

    manager = create_session_manager(ctx)

    async def mcp_app(scope: Scope, receive: Receive, send: Send) -> None:
        await manager.handle_request(scope, receive, send)

    return McpMount(path=MCP_HTTP_PATH, app=mcp_app, lifespan=manager.run)


async def _serve_http(ctx: ServiceContext, *, host: str, port: int) -> None:
    """Run uvicorn over the FastAPI app with the MCP transport mounted on it.

    One runtime, two transports, one registry (ADR-0014 decision 5): the REST
    routes and the MCP streamable-HTTP endpoint are the same operations over
    the same ``ctx``, so two processes would mean two caches over one index
    and two snapshots that drift.

    A named module-level coroutine rather than an inline block, so the CLI's
    own tests can substitute it and exercise every startup guard without
    binding a socket.
    """
    import uvicorn

    from groundkit.service.api import create_app

    app = create_app(ctx, mcp_mount=_build_mcp_mount(ctx))
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="info"))
    await server.serve()


async def _cmd_serve(args: argparse.Namespace) -> int:
    """Serve REST and MCP streamable-HTTP from one app.

    The bind guard runs *first* — before the index directory is opened, before
    a registry exists, before a reranker is constructed. A refused host must
    cost nothing and touch nothing, and ordering it first is also what lets
    its test assert the refusal without a valid index on disk.
    """
    ensure_bindable_host(args.host, allow_remote_access=args.allow_remote_access)
    ctx = _build_service_context(args)
    try:
        await _serve_http(ctx, host=args.host, port=args.port)
    finally:
        await ctx.registry.aclose()
    return 0


async def _cmd_serve_mcp(args: argparse.Namespace) -> int:
    """Serve the MCP stdio transport.

    No socket is bound, so ADR-0014 decision 7's host guard has nothing to
    apply to and the flags driving it are absent by design. The stdio-specific
    rule that *is* load-bearing — stdout carries JSON-RPC frames and nothing
    else — is why this command prints nothing at all: diagnostics go through
    :mod:`logging`, which writes to stderr.
    """
    from groundkit.service.mcp_server import run_stdio

    ctx = _build_service_context(args)
    try:
        await run_stdio(ctx)
    finally:
        await ctx.registry.aclose()
    return 0

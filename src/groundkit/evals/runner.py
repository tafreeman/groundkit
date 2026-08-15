"""Multi-stage retrieval eval runner (Phase 2 baseline; Phase 3 Wave E stages).

Ties the pieces already built independently — :mod:`groundkit.evals.corpus`
(judgment loading and gold-span resolution), :mod:`groundkit.evals.metrics`
(pure scoring functions), and :mod:`groundkit.evals.schema` (the artifact
shape) — into one entry point, :func:`run_eval`, that builds a throwaway
index over a corpus, retrieves against it, and reports an
:class:`~groundkit.evals.schema.EvalReport`.

Phase 2 shipped this as a single-stage BM25 baseline. Wave E made it
multi-stage: given a dense pair, one run indexes once, opens **one**
retriever, and replays the same judgment set through every stage
(``bm25`` → ``dense`` → ``fusion``), emitting them into one report so the
delta each stage owes SPEC.md §6 is derivable within that artifact
(:mod:`groundkit.evals.delta`). Stages share the index and the retriever
deliberately: a per-stage rebuild would let corpus or chunking differences
leak into what is supposed to be a comparison of retrieval strategies alone.

**A losing stage is reported, not dropped.** There is no filtering anywhere
in this module — every configured stage lands in ``report.stages`` with
whatever numbers it earned. SPEC.md §6's baseline discipline makes that a
requirement, and it is the reason the dense and fusion stages are appended
unconditionally once a dense pair is supplied rather than "if they help".

Two conventions this module exists to enforce, both easy to get wrong:

- Every corpus-relative path (:class:`~groundkit.evals.schema.GoldSpanResult`
  and :class:`~groundkit.evals.schema.RetrievedHit`) is derived via
  ``Path.relative_to`` against the resolved corpus root, never by comparing
  or trimming raw absolute-path strings. ``Document.source`` is always an
  absolute realpath (``ingestion/loaders.py``); a string comparison happens
  to pass on a machine where the checkout path is predictable and breaks the
  moment CI's checkout path differs.
- Ground truth (which chunks count as relevant to a judgment) is computed
  over *every* persisted chunk, not just the chunks a particular retrieval
  call returned — nDCG's IDCG needs the true relevant count, and a query
  capped by ``top_k`` cannot supply that on its own.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from groundkit import __version__
from groundkit.config import ChunkingConfig, RetrievalConfig
from groundkit.errors import ConfigurationError, EvalError, GroundkitError
from groundkit.evals.corpus import (
    chunk_overlaps_span,
    load_judgments,
    read_corpus_doc,
    resolve_gold_span,
)
from groundkit.evals.metrics import (
    NDCG_K,
    RECALL_K_VALUES,
    mean,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from groundkit.evals.schema import (
    EvalReport,
    GoldSpanResult,
    MetricSet,
    QueryMetrics,
    QueryResult,
    RetrievedHit,
    RunConfig,
    RunMetadata,
    StageName,
    StageResult,
)
from groundkit.identity import identity_of, validate_dense_pair
from groundkit.index.metadata import SQLiteMetadataStore
from groundkit.indexer import Indexer
from groundkit.ingestion.loaders import FileLoader
from groundkit.providers.embeddings import INMEMORY_PROVIDER
from groundkit.retrieval.search import MAX_TOP_K, Retriever

if TYPE_CHECKING:
    from collections.abc import Sequence

    from groundkit.contracts import Chunk
    from groundkit.evals.corpus import GoldSpan, Judgment
    from groundkit.index.protocols import MetadataStoreProtocol, VectorStoreProtocol
    from groundkit.providers.protocols import EmbeddingProtocol
    from groundkit.retrieval.search import SearchMode

logger = logging.getLogger(__name__)

#: Eval stage -> the :meth:`Retriever.search` mode that produces it, in the
#: order stages are appended to a report. ``stages[0]`` must be ``"bm25"``
#: (``EvalReport`` validates it), so the baseline's position is pinned here
#: rather than assembled by callers.
#:
#: Note ``fusion`` maps to mode ``"hybrid"``: the *stage* is named for what
#: produced the ranking (RRF fusion) and the *mode* for what the caller asks
#: for, and ``SearchResponse.metadata["stage"]`` already reports ``"fusion"``
#: for a hybrid search. Mapping them here keeps that single rename in one
#: place instead of scattered across the runner.
_DENSE_STAGE_MODES: tuple[tuple[StageName, SearchMode], ...] = (
    ("dense", "dense"),
    ("fusion", "hybrid"),
)

#: Floor on ``run_eval``'s ``top_k``, enforced here rather than only at the
#: CLI boundary. Every stage publishes ``recall@10`` and ``nDCG@10``
#: (``MetricSet``'s fields are fixed at 1/5/10), and both are sliced from
#: the single ranked list retrieval returned — so a run below this floor
#: does not measure recall@10 at a different cutoff, it computes the
#: ``@10`` metrics over a list that was never allowed to reach ten and
#: publishes the shortfall under the ``@10`` name. ``top_k=1`` scoring an
#: otherwise-perfect rank-7 hit as ``recall_at_10 = 0.0`` is not a stricter
#: measurement; it is a mislabelled one, and ``RunConfig.top_k`` recording
#: the cutoff does not undo the mislabelling because nothing reading
#: ``recall_at_10`` is obliged to cross-check it. The CLI has always
#: rejected this; a library caller reaching :func:`run_eval` directly could
#: not, which is the gap this closes.
MIN_EVAL_TOP_K: int = 10


#: Chunking config pinned for eval runs, with every value stated
#: explicitly rather than inherited from ``ChunkingConfig()``'s defaults.
#: The explicitness is the whole point: a bare ``ChunkingConfig()`` would
#: silently adopt any future change to the library-wide defaults, moving
#: every chunk boundary in the golden corpus — and therefore straddle
#: behavior (a gold quote spanning a boundary, SPEC.md §6) and the BM25-only
#: baseline every later phase reports its delta against — while
#: ``corpus_hash`` and ``judgments_hash`` stayed identical, so two
#: incomparable runs would still look comparable. Changing these values is a
#: deliberate change to the baseline; ``tests/test_runner.py`` pins them so
#: it cannot happen by accident.
EVAL_CHUNKING_CONFIG: ChunkingConfig = ChunkingConfig(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", ". ", " ", ""],
)


async def run_eval(
    corpus_dir: Path,
    judgments_path: Path,
    *,
    top_k: int = 10,
    embedder: EmbeddingProtocol | None = None,
    vector_store: VectorStoreProtocol | None = None,
) -> EvalReport:
    """Run the retrieval eval over a corpus and judgment set, one stage per strategy.

    Resolves every gold span against the corpus text first (failing closed
    on a missing or ambiguous quote before any indexing happens), then
    builds a throwaway index in an OS temp directory — never the repo tree
    or ``corpus_dir`` itself — ingests ``corpus_dir`` with
    :data:`EVAL_CHUNKING_CONFIG`, opens a
    :class:`~groundkit.retrieval.search.Retriever` snapshot over the
    freshly-ingested store (ADR-0002: the retriever is opened *after*
    ingestion and never refreshed), and scores every judgment against it
    once per stage.

    Without a dense pair this is exactly the Phase 2 baseline run: a single
    ``bm25`` stage. With one, the same index and the same retriever also
    produce ``dense`` and ``fusion`` stages, so the three differ only in
    retrieval strategy and their deltas
    (:func:`~groundkit.evals.delta.derive_stage_deltas`) measure that alone.

    Passing :class:`~groundkit.providers.embeddings.InMemoryEmbedder` is
    supported and *logged as a warning*: it exercises the dense and fusion
    code paths deterministically and offline, but its hash-derived vectors
    carry no semantic signal, so the quality numbers of those stages are
    noise (SPEC.md §2). The embedder's identity is recorded in
    ``report.run.config.embedding`` so a reader can tell which case they are
    looking at without being told.

    Args:
        corpus_dir: Root directory of the corpus documents (e.g.
            ``evals/corpus/``, or a synthetic fixture built the same way).
        judgments_path: Path to a JSONL judgments file, as read by
            :func:`~groundkit.evals.corpus.load_judgments`.
        top_k: Results requested per query, at least
            :data:`MIN_EVAL_TOP_K` and at most
            :data:`~groundkit.retrieval.search.MAX_TOP_K`. Retrieval runs
            exactly once per query per stage at this cutoff; recall@1/5/10
            are all sliced from that single ranked list rather than
            re-querying per ``k`` — which is why the floor exists, since a
            shorter list cannot produce an ``@10`` metric worth the name.
        embedder: Optional embedding provider enabling the ``dense`` and
            ``fusion`` stages (keyword-only; both or neither with
            ``vector_store``, mirroring ``Indexer`` and ``Retriever``).
        vector_store: Optional dense vector store (keyword-only). **Must be
            empty**, and is treated as disposable: this run writes the
            corpus's vectors into it and deletes them again on the way out.
            Pass a fresh store, never a live collection's — see
            :func:`_require_empty_vector_store` for what goes wrong
            otherwise.

    Returns:
        An :class:`EvalReport` whose ``stages[0]`` is always the ``bm25``
        baseline, followed by ``dense`` and ``fusion`` when a dense pair was
        supplied.

    Raises:
        ConfigurationError: Exactly one of ``embedder`` / ``vector_store``
            was supplied, or ``vector_store`` already holds vectors.
        EvalError: A judgment or gold span fails to load or resolve against
            the corpus, or a gold span's document (or a retrieved hit's
            document) has no corresponding entry among the corpus's
            actually-indexed documents.
        IngestionError: Ingesting ``corpus_dir`` fails.
        StorageError: The throwaway index fails to open or write.
        RetrievalError: A search call fails (e.g. an index inconsistency,
            or an out-of-range ``top_k``).
    """
    if not MIN_EVAL_TOP_K <= top_k <= MAX_TOP_K:
        raise EvalError(
            f"top_k must be between {MIN_EVAL_TOP_K} and {MAX_TOP_K}, got {top_k}. The "
            f"floor is not Retriever.search's (which accepts 1): every stage publishes "
            f"recall@10 and nDCG@10 sliced from this one ranked list, so below "
            f"{MIN_EVAL_TOP_K} those fields report a list that was never allowed to "
            "reach ten under the @10 name. The upper bound is checked here rather than "
            "left to Retriever.search, which enforces the same one but only inside the "
            "stage loop — by then a dense run has embedded the entire corpus, doing "
            "real and possibly billable provider work for an invocation that was "
            "always going to be rejected."
        )
    _validate_dense_pair(embedder, vector_store)
    if embedder is not None and vector_store is not None:
        await _require_empty_vector_store(vector_store, embedder.dimensions)
    if embedder is not None and embedder.provider == INMEMORY_PROVIDER:
        logger.warning(
            # Plain ASCII deliberately: this string is emitted to a live
            # console, and the repo's primary dev platform defaults to a
            # cp1252 code page that mangles a section sign into a
            # replacement character mid-warning.
            "Eval running with the %r embedder: dense and fusion stages will exercise "
            "the code paths but their quality metrics are hash-derived noise, not a "
            "retrieval-quality measurement (SPEC.md section 2). Use a real embedding "
            "provider for any number that is meant to mean something.",
            INMEMORY_PROVIDER,
        )

    started_at = datetime.now(UTC).isoformat()
    judgments = load_judgments(judgments_path)

    # Resolve every gold span against the corpus text up front, before any
    # indexing work: fails closed (EvalError propagates) on a missing or
    # ambiguous quote rather than burning an ingest run first.
    resolved_spans: dict[str, list[tuple[GoldSpan, int, int]]] = {}
    for judgment in judgments:
        spans: list[tuple[GoldSpan, int, int]] = []
        for gold_span in judgment.gold:
            document_text = read_corpus_doc(corpus_dir, gold_span.doc)
            start, end = resolve_gold_span(document_text, gold_span.quote)
            spans.append((gold_span, start, end))
        resolved_spans[judgment.query_id] = spans

    corpus_root = corpus_dir.resolve()
    # Both reads race the filesystem (a file removed or made unreadable
    # mid-traversal) and both raise bare OSError, which would escape every
    # caller that handles GroundkitError — the CLI included.
    try:
        corpus_hash = await asyncio.to_thread(_hash_corpus, corpus_root)
    except OSError as exc:
        raise EvalError(f"Cannot hash corpus at {str(corpus_root)!r}: {exc}") from exc
    try:
        judgments_hash = hashlib.sha256(judgments_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise EvalError(f"Cannot read judgments file {str(judgments_path)!r}: {exc}") from exc
    retrieval_config = RetrievalConfig()

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        store = await SQLiteMetadataStore.open(Path(tmp_dir_name), "eval")
        try:
            indexer = Indexer(
                store,
                FileLoader(allowed_base_dir=corpus_dir),
                chunking_config=EVAL_CHUNKING_CONFIG,
                embedder=embedder,
                vector_store=vector_store,
            )
            await indexer.index_directory(str(corpus_dir))

            # Opened AFTER ingestion completes: a Retriever snapshots the
            # store at open() and never refreshes it (ADR-0002). One
            # retriever serves every stage — nothing writes between stages,
            # so the snapshot stays valid, and sharing it keeps the stages
            # comparing retrieval strategy rather than index state.
            retriever = await Retriever.open(
                store, retrieval_config, embedder=embedder, vector_store=vector_store
            )

            all_chunks = await store.get_chunks()
            sources = await store.get_document_sources()
            document_count = len(sources)
            chunk_count = len(all_chunks)

            # Document.source is always an absolute realpath — never
            # corpus-relative (ingestion/loaders.py). Map every document
            # back to a corpus-relative, forward-slashed path exactly once
            # here, via Path.relative_to, so the artifact stays identical
            # across machines and checkout locations.
            document_id_to_relpath = {
                document_id: Path(source).relative_to(corpus_root).as_posix()
                for document_id, source in sources.items()
            }
            relpath_to_document_id = {
                relpath: document_id for document_id, relpath in document_id_to_relpath.items()
            }

            chunks_by_document: dict[str, list[Chunk]] = defaultdict(list)
            for chunk in all_chunks:
                chunks_by_document[chunk.document_id].append(chunk)

            # Ground truth is a property of the corpus and the judgments,
            # not of any retrieval strategy, so it is resolved once here and
            # reused by every stage. Recomputing it per stage would be
            # wasted work and, worse, a place for two stages to disagree
            # about what "relevant" meant.
            gold_by_query: dict[str, _JudgmentGold] = {
                judgment.query_id: _build_gold(
                    judgment,
                    resolved_spans[judgment.query_id],
                    chunks_by_document=chunks_by_document,
                    relpath_to_document_id=relpath_to_document_id,
                )
                for judgment in judgments
            }

            stages: list[StageResult] = []
            for stage_name, mode in _planned_stages(embedder, vector_store):
                query_results: list[QueryResult] = []
                for judgment in judgments:
                    query_results.append(
                        await _evaluate_judgment(
                            judgment,
                            gold_by_query[judgment.query_id],
                            retriever=retriever,
                            top_k=top_k,
                            mode=mode,
                            document_id_to_relpath=document_id_to_relpath,
                        )
                    )
                stages.append(
                    _build_stage_result(
                        query_results,
                        stage=stage_name,
                        is_baseline=not stages,
                    )
                )
        finally:
            # Purge BEFORE closing: the document set lives in the store, and
            # the store is about to go away with the temp directory.
            if vector_store is not None:
                await _purge_eval_vectors(vector_store, store)
            # Must close before the TemporaryDirectory context exits: on
            # Windows the sqlite file cannot be deleted while a handle to it
            # is still open.
            await store.close()

    run_metadata = RunMetadata(
        started_at=started_at,
        groundkit_version=__version__,
        corpus_hash=corpus_hash,
        judgments_hash=judgments_hash,
        document_count=document_count,
        chunk_count=chunk_count,
        judgment_count=len(judgments),
        config=RunConfig(
            chunk_size=EVAL_CHUNKING_CONFIG.chunk_size,
            chunk_overlap=EVAL_CHUNKING_CONFIG.chunk_overlap,
            top_k=top_k,
            bm25_k1=retrieval_config.bm25_k1,
            bm25_b=retrieval_config.bm25_b,
            score_threshold=retrieval_config.score_threshold,
            embedding=identity_of(embedder) if embedder is not None else None,
            # Recorded only when a fusion stage actually ran: on a BM25-only
            # run the constant was never applied to anything, and stamping
            # the configured value into the artifact anyway would describe a
            # computation that did not happen.
            rrf_k=retrieval_config.rrf_k if embedder is not None else None,
        ),
    )
    logger.info(
        "Eval run complete: %d stages, %d judgments, %d documents, %d chunks",
        len(stages),
        len(judgments),
        document_count,
        chunk_count,
    )
    return EvalReport(run=run_metadata, stages=stages)


def _planned_stages(
    embedder: EmbeddingProtocol | None, vector_store: VectorStoreProtocol | None
) -> tuple[tuple[StageName, SearchMode], ...]:
    """The stages this run will emit, baseline first.

    Args:
        embedder: The run's embedder, or ``None``.
        vector_store: The run's vector store, or ``None``. Pairing is
            already validated by the caller; both are read here so the
            dense stages cannot be planned off half a pair.

    Returns:
        ``(stage_name, search_mode)`` pairs. Always starts with the
        ``bm25`` baseline; appends :data:`_DENSE_STAGE_MODES` when a dense
        pair is present.
    """
    baseline: tuple[tuple[StageName, SearchMode], ...] = (("bm25", "bm25"),)
    if embedder is None or vector_store is None:
        return baseline
    return baseline + _DENSE_STAGE_MODES


async def _require_empty_vector_store(vector_store: VectorStoreProtocol, dimensions: int) -> None:
    """Refuse a vector store that already holds vectors.

    An eval run builds a **throwaway** index: its SQLite half lives in an OS
    temp directory that is deleted when the run ends. The vector half is
    supplied by the caller, and nothing about the protocol makes it
    throwaway too — so writing eval vectors into a store that already holds
    a real collection would leave them behind after the SQLite side they
    reference has been deleted. Those orphans are not inert: a later dense
    search through that store fails closed with ``RetrievalError`` the
    moment one ranks into the candidate window (``retrieval/search.py``'s
    orphan check), which breaks the caller's collection permanently and for
    a reason with no visible connection to having run an eval.

    Requiring the store to be empty makes contaminating a populated
    collection unrepresentable rather than merely discouraged, and
    :func:`_purge_eval_vectors` then leaves an accepted store as it was
    found.

    Emptiness is probed with ``search(top_k=1)``, the same idiom
    :func:`~groundkit.index.dense.verify_dense_side_present` uses: a dense
    search never drops zero-scored results, so an empty result means an
    empty store, and this needs no addition to
    :class:`~groundkit.index.protocols.VectorStoreProtocol`.

    Args:
        vector_store: The caller-supplied dense store.
        dimensions: Embedding width, used to shape the probe vector.

    Raises:
        ConfigurationError: The store already holds at least one vector.
    """
    # A unit vector rather than zeros: cosine similarity is undefined
    # against a zero-magnitude query.
    probe = [1.0] + [0.0] * (dimensions - 1)
    if await vector_store.search(probe, top_k=1):
        raise ConfigurationError(
            "run_eval was given a vector_store that already holds vectors. An eval "
            "builds a throwaway index whose SQLite half is deleted when the run ends, "
            "so writing into a populated store would strand those vectors with no "
            "metadata behind them and fail that collection's later dense searches "
            "closed. Pass a fresh, empty store (the CLI builds an InMemoryVectorStore "
            "per run); evaluate an existing collection by re-ingesting its corpus here "
            "instead of handing over its live store."
        )


async def _purge_eval_vectors(
    vector_store: VectorStoreProtocol, store: MetadataStoreProtocol
) -> None:
    """Delete every vector this run wrote, leaving the store as it was found.

    The store was verified empty at entry
    (:func:`_require_empty_vector_store`), so everything in it now was
    written by this run and every document SQLite knows about is one to
    remove.

    Best-effort by construction, and deliberately not fatal: this runs in a
    ``finally``, so raising here would replace whatever real failure brought
    us into it with a cleanup error. Failures are logged loudly instead.

    One residual gap, worth knowing rather than discovering: ``Indexer``
    commits SQLite *last* (vectors are written before the metadata that
    references them, so SQLite is never ahead of the dense store). A crash
    part-way through ingest can therefore leave the in-flight document's
    vectors in a store SQLite never recorded, and this purge cannot see
    them. The emptiness guard bounds the damage — that can only ever happen
    to a store the caller declared disposable — and it is recorded in
    ``KNOWN_LIMITATIONS.md``.

    Args:
        vector_store: The caller-supplied dense store to clean.
        store: The run's temporary metadata store, still open.
    """
    try:
        document_ids = list(await store.get_document_sources())
    except GroundkitError as exc:
        logger.warning(
            "Could not read the document set to purge eval vectors from the supplied "
            "vector store; it may still hold this run's vectors: %s",
            exc,
        )
        return

    failed: list[str] = []
    for document_id in document_ids:
        try:
            await vector_store.delete(document_id)
        except GroundkitError:
            failed.append(document_id)
    if failed:
        logger.warning(
            "Failed to purge eval vectors for %d of %d documents from the supplied "
            "vector store; it still holds vectors for: %s",
            len(failed),
            len(document_ids),
            ", ".join(failed),
        )


def _validate_dense_pair(
    embedder: EmbeddingProtocol | None, vector_store: VectorStoreProtocol | None
) -> None:
    """Reject half a dense pair before any corpus work starts.

    The branch structure is shared with ``Indexer`` and ``Retriever``
    (:func:`groundkit.identity.validate_dense_pair`). Checked here too, rather
    than left to whichever of them raises first, so the eval fails on the
    caller's mistake before spending a full ingest on it.

    Raises:
        ConfigurationError: Exactly one of ``embedder`` / ``vector_store``
            was supplied.
    """
    validate_dense_pair(
        embedder,
        vector_store,
        subject="run_eval",
        without_store="there would be no dense index to search.",
        without_embedder="queries could never be embedded.",
    )


def write_report(report: EvalReport, output_path: Path) -> None:
    """Write ``report`` as indented JSON, creating parent directories as needed.

    Args:
        report: The eval report to persist.
        output_path: Destination file (e.g. ``evals/results/latest.json``).

    Raises:
        EvalError: The parent directory cannot be created or the file cannot
            be written (unwritable path, a directory where a file is
            expected, and so on). Wrapped because a bare ``OSError`` escapes
            every caller that handles :class:`GroundkitError` — the CLI
            would surface a traceback after an otherwise successful run.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    except OSError as exc:
        raise EvalError(f"Cannot write eval report to {str(output_path)!r}: {exc}") from exc


class _JudgmentGold(NamedTuple):
    """One judgment's ground truth, resolved once and reused by every stage.

    Attributes:
        chunk_ids: Every persisted chunk whose offsets overlap a gold span.
        spans: The resolved gold spans, for the artifact.
    """

    chunk_ids: set[str]
    spans: list[GoldSpanResult]


def _build_gold(
    judgment: Judgment,
    resolved_gold_spans: list[tuple[GoldSpan, int, int]],
    *,
    chunks_by_document: dict[str, list[Chunk]],
    relpath_to_document_id: dict[str, str],
) -> _JudgmentGold:
    """Resolve one judgment's ground truth against the persisted chunk set.

    Ground truth is built from every persisted chunk whose document matches
    a gold span's document and whose offsets overlap that span —
    deliberately over *all* chunks in ``chunks_by_document``, not just the
    chunks some retrieval call happens to return, so nDCG's IDCG reflects
    the true count of relevant chunks rather than being capped by ``top_k``.

    Computed once per judgment and shared across stages: it depends only on
    the corpus and the judgments, so a per-stage recomputation would be both
    wasted work and an opportunity for two stages to disagree about what
    counted as relevant.

    Args:
        judgment: The judgment to resolve.
        resolved_gold_spans: ``(gold_span, start, end)`` triples, one per
            ``judgment.gold`` entry, already resolved against the corpus
            text before ingestion.
        chunks_by_document: Every persisted chunk, grouped by
            ``document_id``.
        relpath_to_document_id: Corpus-relative posix path ->
            ``document_id``, for looking up a gold span's document.

    Returns:
        The judgment's :class:`_JudgmentGold`.

    Raises:
        EvalError: A gold span's document has no corresponding entry among
            the indexed corpus documents — it was never actually persisted,
            so ground truth cannot be computed.
    """
    chunk_ids: set[str] = set()
    spans: list[GoldSpanResult] = []
    for gold_span, start, end in resolved_gold_spans:
        spans.append(
            GoldSpanResult(
                document=gold_span.doc,
                start_offset=start,
                end_offset=end,
                quote=gold_span.quote,
            )
        )
        document_id = relpath_to_document_id.get(gold_span.doc)
        if document_id is None:
            raise EvalError(
                f"judgment {judgment.query_id!r}: gold document {gold_span.doc!r} "
                "was not found among the indexed corpus documents"
            )
        for chunk in chunks_by_document.get(document_id, []):
            if chunk_overlaps_span(chunk.start_offset, chunk.end_offset, (start, end)):
                chunk_ids.add(chunk.chunk_id)
    return _JudgmentGold(chunk_ids=chunk_ids, spans=spans)


async def _evaluate_judgment(
    judgment: Judgment,
    gold: _JudgmentGold,
    *,
    retriever: Retriever,
    top_k: int,
    mode: SearchMode,
    document_id_to_relpath: dict[str, str],
) -> QueryResult:
    """Score one judgment against a live retriever in one retrieval mode.

    Args:
        judgment: The judgment to score.
        gold: The judgment's pre-resolved ground truth (:func:`_build_gold`).
        retriever: An open retriever snapshotting the freshly-ingested
            corpus.
        top_k: Results requested for this query.
        mode: The retrieval mode this stage measures.
        document_id_to_relpath: ``document_id`` -> corpus-relative path, for
            labeling retrieved hits.

    Returns:
        This judgment's fully-scored :class:`~groundkit.evals.schema.QueryResult`.

    Raises:
        EvalError: A retrieved hit's document has no corresponding entry
            among the indexed corpus documents, so the result cannot be
            labeled.
    """
    is_no_answer = judgment.category == "no_answer"
    gold_ids = gold.chunk_ids
    gold_results = gold.spans

    response = await retriever.search(judgment.query, top_k=top_k, mode=mode)
    ranked_ids = [result.chunk_id for result in response.results]

    retrieved: list[RetrievedHit] = []
    for rank, result in enumerate(response.results, start=1):
        document = document_id_to_relpath.get(result.document_id)
        if document is None:
            raise EvalError(
                f"judgment {judgment.query_id!r}: retrieved document_id "
                f"{result.document_id!r} has no known corpus-relative path"
            )
        retrieved.append(
            RetrievedHit(
                rank=rank,
                document=document,
                start_offset=result.start_offset,
                end_offset=result.end_offset,
                score=result.score,
                is_relevant=result.chunk_id in gold_ids,
            )
        )

    latency_ms = float(response.metadata["latency_ms"])

    metrics: QueryMetrics | None = None
    total_relevant_chunks = 0
    if not is_no_answer:
        recalls = {k: recall_at_k(ranked_ids, gold_ids, k=k) for k in RECALL_K_VALUES}
        metrics = QueryMetrics(
            recall_at_1=recalls[1],
            recall_at_5=recalls[5],
            recall_at_10=recalls[10],
            reciprocal_rank=reciprocal_rank(ranked_ids, gold_ids),
            ndcg_at_10=ndcg_at_k(ranked_ids, gold_ids, k=NDCG_K),
        )
        total_relevant_chunks = len(gold_ids)

    return QueryResult(
        query_id=judgment.query_id,
        query=judgment.query,
        category=judgment.category,
        is_no_answer=is_no_answer,
        gold=gold_results,
        total_relevant_chunks=total_relevant_chunks,
        retrieved=retrieved,
        metrics=metrics,
        latency_ms=latency_ms,
    )


def _aggregate_metric_set(query_results: Sequence[QueryResult]) -> MetricSet:
    """Aggregate metrics over every answerable (non-no-answer) result.

    Args:
        query_results: Results to aggregate; entries with ``metrics is None``
            (no-answer queries) are skipped.

    Returns:
        The aggregate :class:`~groundkit.evals.schema.MetricSet`.

    Raises:
        EvalError: None of ``query_results`` is answerable — an aggregate
            over zero queries would be meaningless.
    """
    answerable = [qr.metrics for qr in query_results if qr.metrics is not None]
    if not answerable:
        raise EvalError("no answerable judgments to aggregate metrics over")
    return MetricSet(
        query_count=len(answerable),
        recall_at_1=mean([m.recall_at_1 for m in answerable]),
        recall_at_5=mean([m.recall_at_5 for m in answerable]),
        recall_at_10=mean([m.recall_at_10 for m in answerable]),
        mrr=mean([m.reciprocal_rank for m in answerable]),
        ndcg_at_10=mean([m.ndcg_at_10 for m in answerable]),
    )


def _aggregate_by_category(query_results: Sequence[QueryResult]) -> dict[str, MetricSet]:
    """Aggregate metrics per judgment category, excluding no-answer queries.

    Args:
        query_results: Results to group and aggregate.

    Returns:
        ``category -> MetricSet`` for every category with at least one
        answerable result. ``"no_answer"`` never appears as a key — those
        queries have no metrics to aggregate, by construction.
    """
    grouped: dict[str, list[QueryResult]] = defaultdict(list)
    for query_result in query_results:
        if query_result.metrics is not None:
            grouped[query_result.category].append(query_result)
    return {category: _aggregate_metric_set(results) for category, results in grouped.items()}


def _build_stage_result(
    query_results: list[QueryResult], *, stage: StageName, is_baseline: bool
) -> StageResult:
    """Assemble one stage's result from its per-query results.

    Latency percentiles are computed from *this* stage's own per-query
    latencies, which is what makes SPEC.md §6's "latency percentiles per
    stage (BM25 / dense / fusion / rerank)" a real per-stage measurement:
    every stage times its own retrieval calls independently over the same
    judgment set.

    Args:
        query_results: Every judgment's scored result for this stage, in
            judgment order.
        stage: The stage name, typed as the schema's own
            :data:`~groundkit.evals.schema.StageName` so an unrecognized
            stage is a type error rather than a runtime surprise.
        is_baseline: Whether this is the run's baseline stage. The caller
            sets it for ``stages[0]`` only; ``EvalReport``'s validators
            enforce that there is exactly one and that it is BM25.

    Returns:
        The assembled stage result.
    """
    no_answer_results = [qr for qr in query_results if qr.is_no_answer]
    no_answer_abstained = sum(1 for qr in no_answer_results if not qr.retrieved)
    latencies = sorted(qr.latency_ms for qr in query_results)

    return StageResult(
        stage=stage,
        is_baseline=is_baseline,
        aggregate=_aggregate_metric_set(query_results),
        by_category=_aggregate_by_category(query_results),
        no_answer_query_count=len(no_answer_results),
        no_answer_abstained_count=no_answer_abstained,
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=_percentile(latencies, 95),
        latency_p99_ms=_percentile(latencies, 99),
        queries=query_results,
    )


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    """Nearest-rank percentile of a pre-sorted sample.

    Uses the nearest-rank method rather than ``statistics.quantiles``'
    interpolation: rank ``= ceil(percentile / 100 * n)``, clamped to
    ``[1, n]`` and taken 1-indexed into ``sorted_values``. Nearest-rank
    always returns an observed sample value rather than one interpolated
    between two samples. Documented explicitly because with an eval run of
    a few dozen judgments, the p95/p99 tails are order-of-magnitude
    estimates regardless of method — the choice doesn't materially change
    what the numbers mean, but it does change what they are, so a reader
    diffing two runs needs to know which one was used.

    Args:
        sorted_values: Latency samples in ascending order. The caller must
            have sorted them; this function trusts the ordering rather than
            re-sorting.
        percentile: Target percentile in ``[0, 100]``.

    Returns:
        The sample at the computed rank, or ``0.0`` if ``sorted_values`` is
        empty.
    """
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    rank = max(1, min(n, math.ceil(percentile / 100 * n)))
    return sorted_values[rank - 1]


def _hash_corpus(corpus_root: Path) -> str:
    """SHA-256 over every file under ``corpus_root``, order- and path-independent.

    Hashes sorted ``(relative_posix_path, file_bytes)`` pairs rather than,
    say, concatenating files in directory-walk order: walk order is not
    guaranteed identical across platforms or filesystems, and this hash must
    be identical across machines for two runs to be comparable
    (:class:`~groundkit.evals.schema.RunMetadata`'s ``corpus_hash``).

    Args:
        corpus_root: The resolved corpus documents root.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    entries: list[tuple[str, bytes]] = []
    for path in corpus_root.rglob("*"):
        if path.is_file():
            relpath = path.relative_to(corpus_root).as_posix()
            entries.append((relpath, path.read_bytes()))
    entries.sort(key=lambda entry: entry[0])

    hasher = hashlib.sha256()
    for relpath, content in entries:
        hasher.update(relpath.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(content)
        hasher.update(b"\0")
    return hasher.hexdigest()

"""BM25-baseline retrieval eval runner (Phase 2, SPEC.md §6, §8).

Ties the pieces already built independently — :mod:`groundkit.evals.corpus`
(judgment loading and gold-span resolution), :mod:`groundkit.evals.metrics`
(pure scoring functions), and :mod:`groundkit.evals.schema` (the artifact
shape) — into one entry point, :func:`run_eval`, that builds a throwaway
index over a corpus, retrieves against it, and reports an
:class:`~groundkit.evals.schema.EvalReport`.

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
from typing import TYPE_CHECKING

from groundkit import __version__
from groundkit.config import ChunkingConfig, RetrievalConfig
from groundkit.errors import EvalError
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
    StageResult,
)
from groundkit.index.metadata import SQLiteMetadataStore
from groundkit.indexer import Indexer
from groundkit.ingestion.loaders import FileLoader
from groundkit.retrieval.search import Retriever

if TYPE_CHECKING:
    from collections.abc import Sequence

    from groundkit.contracts import Chunk
    from groundkit.evals.corpus import GoldSpan, Judgment

logger = logging.getLogger(__name__)

#: Chunking config pinned for eval runs, deliberately not
#: ``ChunkingConfig()`` inlined at each call site. Pinning it here means a
#: future change to the library-wide chunking default becomes a visible,
#: intentional edit to this constant rather than a silent shift in every
#: chunk boundary of the golden corpus — which would move straddle behavior
#: (a gold quote spanning a chunk boundary, SPEC.md §6) and therefore the
#: BM25-only baseline every later phase reports its delta against.
EVAL_CHUNKING_CONFIG: ChunkingConfig = ChunkingConfig()


async def run_eval(corpus_dir: Path, judgments_path: Path, *, top_k: int = 10) -> EvalReport:
    """Run the BM25-baseline retrieval eval over a corpus and judgment set.

    Resolves every gold span against the corpus text first (failing closed
    on a missing or ambiguous quote before any indexing happens), then
    builds a throwaway index in an OS temp directory — never the repo tree
    or ``corpus_dir`` itself — ingests ``corpus_dir`` with
    :data:`EVAL_CHUNKING_CONFIG`, opens a
    :class:`~groundkit.retrieval.search.Retriever` snapshot over the
    freshly-ingested store (ADR-0002: the retriever is opened *after*
    ingestion and never refreshed), and scores every judgment against it.

    Args:
        corpus_dir: Root directory of the corpus documents (e.g.
            ``evals/corpus/``, or a synthetic fixture built the same way).
        judgments_path: Path to a JSONL judgments file, as read by
            :func:`~groundkit.evals.corpus.load_judgments`.
        top_k: Results requested per query. Retrieval runs exactly once per
            query at this cutoff; recall@1/5/10 are all sliced from that
            single ranked list rather than re-querying per ``k``.

    Returns:
        A single-stage (``bm25``, baseline) :class:`EvalReport`.

    Raises:
        EvalError: A judgment or gold span fails to load or resolve against
            the corpus, or a gold span's document (or a retrieved hit's
            document) has no corresponding entry among the corpus's
            actually-indexed documents.
        IngestionError: Ingesting ``corpus_dir`` fails.
        StorageError: The throwaway index fails to open or write.
        RetrievalError: A search call fails (e.g. an index inconsistency,
            or an out-of-range ``top_k``).
    """
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
    corpus_hash = await asyncio.to_thread(_hash_corpus, corpus_root)
    judgments_hash = hashlib.sha256(judgments_path.read_bytes()).hexdigest()
    retrieval_config = RetrievalConfig()

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        store = await SQLiteMetadataStore.open(Path(tmp_dir_name), "eval")
        try:
            indexer = Indexer(
                store,
                FileLoader(allowed_base_dir=corpus_dir),
                chunking_config=EVAL_CHUNKING_CONFIG,
            )
            await indexer.index_directory(str(corpus_dir))

            # Opened AFTER ingestion completes: a Retriever snapshots the
            # store at open() and never refreshes it (ADR-0002).
            retriever = await Retriever.open(store, retrieval_config)

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

            query_results: list[QueryResult] = []
            for judgment in judgments:
                query_results.append(
                    await _evaluate_judgment(
                        judgment,
                        resolved_spans[judgment.query_id],
                        retriever=retriever,
                        top_k=top_k,
                        chunks_by_document=chunks_by_document,
                        relpath_to_document_id=relpath_to_document_id,
                        document_id_to_relpath=document_id_to_relpath,
                    )
                )
        finally:
            # Must close before the TemporaryDirectory context exits: on
            # Windows the sqlite file cannot be deleted while a handle to it
            # is still open.
            await store.close()

    stage = _build_stage_result(query_results)
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
        ),
    )
    logger.info(
        "Eval run complete: %d judgments, %d documents, %d chunks",
        len(judgments),
        document_count,
        chunk_count,
    )
    return EvalReport(run=run_metadata, stages=[stage])


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


async def _evaluate_judgment(
    judgment: Judgment,
    resolved_gold_spans: list[tuple[GoldSpan, int, int]],
    *,
    retriever: Retriever,
    top_k: int,
    chunks_by_document: dict[str, list[Chunk]],
    relpath_to_document_id: dict[str, str],
    document_id_to_relpath: dict[str, str],
) -> QueryResult:
    """Score one judgment against a live retriever.

    Ground truth (``gold_ids``) is built from every persisted chunk whose
    document matches a gold span's document and whose offsets overlap that
    span — deliberately over *all* chunks in ``chunks_by_document``, not
    just the chunks this call's ``retriever.search`` happens to return, so
    nDCG's IDCG reflects the true count of relevant chunks rather than being
    capped by ``top_k``.

    Args:
        judgment: The judgment to score.
        resolved_gold_spans: ``(gold_span, start, end)`` triples, one per
            ``judgment.gold`` entry, already resolved against the corpus
            text before ingestion.
        retriever: An open retriever snapshotting the freshly-ingested
            corpus.
        top_k: Results requested for this query.
        chunks_by_document: Every persisted chunk, grouped by
            ``document_id``.
        relpath_to_document_id: Corpus-relative posix path ->
            ``document_id``, for looking up a gold span's document.
        document_id_to_relpath: The inverse map, for labeling retrieved
            hits with a corpus-relative path.

    Returns:
        This judgment's fully-scored :class:`~groundkit.evals.schema.QueryResult`.

    Raises:
        EvalError: A gold span's document, or a retrieved hit's document,
            has no corresponding entry among the indexed corpus documents —
            it was never actually persisted, so ground truth (or the result
            label) cannot be computed.
    """
    is_no_answer = judgment.category == "no_answer"

    gold_ids: set[str] = set()
    gold_results: list[GoldSpanResult] = []
    for gold_span, start, end in resolved_gold_spans:
        gold_results.append(
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
                gold_ids.add(chunk.chunk_id)

    response = await retriever.search(judgment.query, top_k=top_k)
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


def _build_stage_result(query_results: list[QueryResult]) -> StageResult:
    """Assemble the single ``bm25`` baseline stage from per-query results.

    Args:
        query_results: Every judgment's scored result, in judgment order.

    Returns:
        The stage result, with ``is_baseline=True``.
    """
    no_answer_results = [qr for qr in query_results if qr.is_no_answer]
    no_answer_abstained = sum(1 for qr in no_answer_results if not qr.retrieved)
    latencies = sorted(qr.latency_ms for qr in query_results)

    return StageResult(
        stage="bm25",
        is_baseline=True,
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

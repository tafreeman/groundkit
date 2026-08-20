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

Given a reranker, one more stage follows (ADR-0012). It is unlike the others
in two ways that the code below is arranged around:

- **It is a post-step, not a mode.** A reranker reorders
  ``list[RetrievalResult]``; it does not produce candidates. So there is no
  fourth ``SearchMode`` and nothing in ``retrieval/search.py`` changed — the
  runner asks the retriever for the *input* stage's results and reranks what
  comes back. ADR-0012 decision 2.
- **Its input is the best stage available** — ``fusion`` with a dense pair,
  ``bm25`` without. That makes the row's meaning configuration-dependent,
  which is why :class:`~groundkit.evals.schema.RunConfig` records
  ``rerank_input``, ``rerank_candidates`` and ``rerank_model``, and why
  :func:`~groundkit.evals.delta.derive_rerank_attribution` exists: against
  ``stages[0]`` the rerank delta measures fusion *and* rerank together, so
  the two have to be separable or neither is interpretable. How cleanly they
  separate depends on the input stage — cleanly for ``bm25``, only partly
  for ``fusion``, because RRF is not depth-invariant and the rerank stage
  fetches a wider pool than the fusion stage reported. That distinction is
  spelled out in :mod:`groundkit.evals.delta`'s module docstring and must
  not be flattened when quoting either number.

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
import time
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
from groundkit.evals.synthesis_eval import run_synthesis_eval
from groundkit.identity import identity_of, validate_dense_pair
from groundkit.index.metadata import SQLiteMetadataStore
from groundkit.indexer import Indexer
from groundkit.ingestion.loaders import FileLoader
from groundkit.providers.embeddings import INMEMORY_PROVIDER
from groundkit.providers.synthesis import DEFAULT_SYNTHESIS_PROMPT
from groundkit.retrieval.search import MAX_TOP_K, Retriever

if TYPE_CHECKING:
    from collections.abc import Sequence

    from groundkit.contracts import Chunk, RetrievalResult
    from groundkit.evals.corpus import GoldSpan, Judgment
    from groundkit.evals.schema import SynthesisReport
    from groundkit.index.protocols import MetadataStoreProtocol, VectorStoreProtocol
    from groundkit.providers.judge import FaithfulnessJudge
    from groundkit.providers.protocols import ChatProtocol, EmbeddingProtocol
    from groundkit.retrieval.protocols import RerankerProtocol
    from groundkit.retrieval.search import SearchMode

logger = logging.getLogger(__name__)

#: Nanoseconds per millisecond, matching ``retrieval/search.py``'s own
#: conversion so a rerank stage's latency is expressed in the same unit as
#: the retrieval latency it is added to.
_NS_PER_MS: float = 1_000_000.0

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


class _StagePlan(NamedTuple):
    """One stage the run will emit, and how to produce it.

    Attributes:
        stage: The stage's name in the artifact.
        mode: The :meth:`Retriever.search` mode that produces this stage's
            candidates. For a rerank stage this is the *input* stage's mode
            — a reranker consumes results rather than producing them, so it
            has no mode of its own (ADR-0012 decision 2).
        reranks: Whether the candidates are passed through the run's
            reranker before scoring.
    """

    stage: StageName
    mode: SearchMode
    reranks: bool = False


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
    reranker: RerankerProtocol | None = None,
    rerank_candidates: int = MAX_TOP_K,
    chat: ChatProtocol | None = None,
    judge: FaithfulnessJudge | None = None,
    synthesis_prompt_template: str = DEFAULT_SYNTHESIS_PROMPT,
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

    A ``reranker`` appends one more stage, over whichever upstream stage the
    run actually produced — ``fusion`` with a dense pair, ``bm25`` without
    (ADR-0012 decision 1). Two consequences are worth stating outright
    because neither is visible from the numbers alone:

    - **Its delta against ``stages[0]`` is not the reranker's contribution**
      whenever the input was ``fusion``: that delta sums what fusion gained
      over BM25 and what rerank gained over fusion.
      :func:`~groundkit.evals.delta.derive_rerank_attribution` separates
      them, and the CLI prints both.
    - **``rerank_candidates`` decides whether ``recall_at_10`` can move at
      all.** Reranking truncates to ``top_k`` after reordering, so a
      candidate depth equal to ``top_k`` hands the model a set it can only
      permute — its ``recall_at_10`` then equals the input stage's for
      arithmetic reasons, not measured ones. The default over-fetches to
      :data:`~groundkit.retrieval.search.MAX_TOP_K`, the retriever's own
      ceiling, so every metric is free to move in both directions.

    Passing :class:`~groundkit.providers.embeddings.InMemoryEmbedder` is
    supported and *logged as a warning*: it exercises the dense and fusion
    code paths deterministically and offline, but its hash-derived vectors
    carry no semantic signal, so the quality numbers of those stages are
    noise (SPEC.md §2). The embedder's identity is recorded in
    ``report.run.config.embedding`` so a reader can tell which case they are
    looking at without being told.

    A ``chat`` argument enables one more pass, after every stage above has
    scored: :func:`~groundkit.evals.synthesis_eval.run_synthesis_eval`
    synthesizes an answer for every judgment's query against whichever stage
    this run actually produced as its best available one — ``rerank`` if a
    reranker was supplied, else ``fusion`` with a dense pair, else the
    ``bm25`` baseline — and folds the result into ``report.synthesis``
    (ADR-0018 decision 6). This is independent of, and shares no code path
    with, :mod:`groundkit.evals.echo`'s planted-marker check: that check
    answers one fixed question against a synthetic corpus it builds itself,
    while this pass answers the golden corpus's own queries against results
    this same retriever already knows how to produce. Supplying ``judge``
    alongside ``chat`` additionally judges every non-rejected synthesis
    outcome (SPEC.md §6's advisory faithfulness judge) — never gating
    anything here or anywhere downstream.

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
        reranker: Optional reranker enabling the ``rerank`` stage
            (keyword-only). Independent of the dense pair: a reranker with
            no dense pair reorders the ``bm25`` baseline, which is a
            complete and self-contained measurement.
        rerank_candidates: How many candidates the upstream stage is asked
            for before the reranker truncates back to ``top_k``. Between
            ``top_k`` and :data:`~groundkit.retrieval.search.MAX_TOP_K`;
            ignored, and not recorded, when ``reranker`` is ``None``.
        chat: Optional chat provider (keyword-only) enabling the
            golden-corpus synthesis pass described above. ``report.synthesis``
            stays ``None`` when omitted — the default, and the whole of this
            run's behavior when a caller wants only the stages above.
        judge: Optional :class:`~groundkit.providers.judge.FaithfulnessJudge`
            (keyword-only), reused exactly as given — this function never
            constructs one. Requires ``chat`` (there is nothing for a judge
            to evaluate without a synthesized answer); supplying ``judge``
            without ``chat`` is rejected before any corpus work starts.
        synthesis_prompt_template: Forwarded to the internal
            :class:`~groundkit.providers.synthesis.Synthesizer` and hashed
            into ``report.synthesis.synthesis_prompt_hash``. Ignored when
            ``chat`` is ``None``.

    Returns:
        An :class:`EvalReport` whose ``stages[0]`` is always the ``bm25``
        baseline, followed by ``dense`` and ``fusion`` when a dense pair was
        supplied and ``rerank`` when a reranker was. ``synthesis`` is set
        when ``chat`` was supplied, ``None`` otherwise.

    Raises:
        ConfigurationError: Exactly one of ``embedder`` / ``vector_store``
            was supplied, ``vector_store`` already holds vectors, or
            ``judge`` was supplied without ``chat``.
        EvalError: A judgment or gold span fails to load or resolve against
            the corpus, or a gold span's document (or a retrieved hit's
            document) has no corresponding entry among the corpus's
            actually-indexed documents, or ``rerank_candidates`` is out of
            range.
        RerankerNotConfiguredError: A ``reranker`` was supplied but cannot
            load its model — for :class:`CrossEncoderReranker`, the
            ``rerank`` extra is not installed. Never a silent passthrough:
            an unreranked stage labelled ``rerank`` would be a fabricated
            measurement.
        IngestionError: Ingesting ``corpus_dir`` fails.
        StorageError: The throwaway index fails to open or write.
        ChatError: Propagated unmodified from ``chat`` (via the synthesis or
            judge call) — a provider failure is not a per-query outcome.
        JudgeError: Never escapes this function — caught per-judgment inside
            :func:`~groundkit.evals.synthesis_eval.run_synthesis_eval` and
            folded into ``judge_error_count``.
        RetrievalError: A search call fails (e.g. an index inconsistency,
            or an out-of-range ``top_k``).
    """
    await _validate_run_inputs(
        top_k=top_k,
        embedder=embedder,
        vector_store=vector_store,
        reranker=reranker,
        rerank_candidates=rerank_candidates,
        chat=chat,
        judge=judge,
    )
    settings = _plan_run(
        top_k=top_k,
        embedder=embedder,
        vector_store=vector_store,
        reranker=reranker,
        rerank_candidates=rerank_candidates,
    )

    started_at = datetime.now(UTC).isoformat()
    prepared = await _prepare_corpus(corpus_dir, judgments_path)

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        store = await SQLiteMetadataStore.open(Path(tmp_dir_name), "eval")
        try:
            indexed = await _index_corpus(
                store, corpus_dir=corpus_dir, prepared=prepared, settings=settings
            )
            scored = await _score_stages(
                prepared.judgments, indexed, settings, capture_synthesis_inputs=chat is not None
            )

            synthesis_report: SynthesisReport | None = None
            if chat is not None:
                # The best stage this run actually produced — the same
                # "best upstream" the rerank stage itself reorders
                # (_planned_stages): `settings.planned[-1]` is `rerank` if a
                # reranker was supplied, else `fusion` with a dense pair,
                # else the `bm25` baseline. It shares the one index and one
                # retriever every stage above already shares — never a second
                # ingest of the same corpus, and since the synthesis inputs
                # were captured during that stage's own scoring pass, never a
                # second *retrieval* either. Re-fetching here was the earlier
                # shape: correct, but it ran every query through the retriever
                # twice and, on a dense or hybrid stage, through the embedding
                # provider twice.
                synthesis_input = settings.planned[-1]
                synthesis_report = await run_synthesis_eval(
                    scored.synthesis_inputs,
                    chat=chat,
                    input_stage=synthesis_input.stage,
                    judge=judge,
                    synthesis_prompt_template=synthesis_prompt_template,
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

    return _build_report(
        scored.stages,
        synthesis=synthesis_report,
        started_at=started_at,
        prepared=prepared,
        document_count=indexed.document_count,
        chunk_count=indexed.chunk_count,
        settings=settings,
    )


class _RunSettings(NamedTuple):
    """How this run retrieves, resolved once before any corpus work starts.

    Grouped rather than threaded through the pipeline one argument at a
    time, because these values are not independent of each other:
    ``rerank_input`` and ``rerank_model`` describe the same rerank stage
    ``planned`` contains, and deriving all three together — once, here — is
    what keeps the artifact's account of the run identical to the run
    (:func:`_planned_stages`). Every field is read-only for the rest of the
    pipeline; nothing below mutates it.

    Attributes:
        planned: The ordered stage plan. ``planned[0]`` is always the
            ``bm25`` baseline and ``planned[-1]`` is the best stage this run
            produces — the one the synthesis pass runs over.
        top_k: Results requested per query, per stage.
        retrieval_config: The retrieval knobs every stage shares, recorded
            into :class:`~groundkit.evals.schema.RunConfig`.
        embedder: The run's embedder, or ``None`` for a BM25-only run.
        vector_store: The run's dense store, or ``None``. Paired with
            ``embedder``, already validated.
        reranker: The run's reranker, or ``None``.
        rerank_candidates: Candidate depth fetched before reranking. Carried
            unconditionally but only *recorded* when ``reranker`` is set —
            on a run without one it describes a computation that did not
            happen.
        rerank_input: The stage a rerank stage reorders, or ``None``.
        rerank_model: The reranker's identity for the artifact, or ``None``.
    """

    planned: tuple[_StagePlan, ...]
    top_k: int
    retrieval_config: RetrievalConfig
    embedder: EmbeddingProtocol | None
    vector_store: VectorStoreProtocol | None
    reranker: RerankerProtocol | None
    rerank_candidates: int
    rerank_input: StageName | None
    rerank_model: str | None


class _CorpusPreparation(NamedTuple):
    """Everything derived from the corpus and judgments before indexing.

    Attributes:
        judgments: The judgment set, in file order.
        resolved_spans: ``query_id -> [(gold_span, start, end)]``, every gold
            quote already located in its corpus document.
        corpus_root: The resolved corpus root, the base every
            corpus-relative path in the artifact is derived against.
        corpus_hash: SHA-256 over the corpus contents.
        judgments_hash: SHA-256 over the judgments file's bytes.
    """

    judgments: list[Judgment]
    resolved_spans: dict[str, list[tuple[GoldSpan, int, int]]]
    corpus_root: Path
    corpus_hash: str
    judgments_hash: str


class _IndexedCorpus(NamedTuple):
    """The throwaway index, and the ground truth resolved against it.

    Attributes:
        retriever: An open retriever snapshotting the freshly-ingested
            store. Valid only while that store is open (ADR-0002).
        document_count: Documents actually persisted.
        chunk_count: Chunks actually persisted.
        document_id_to_relpath: ``document_id`` -> corpus-relative posix
            path, for labeling retrieved hits.
        gold_by_query: ``query_id`` -> that judgment's ground truth,
            resolved once and shared by every stage.
    """

    retriever: Retriever
    document_count: int
    chunk_count: int
    document_id_to_relpath: dict[str, str]
    gold_by_query: dict[str, _JudgmentGold]


class _ScoredStages(NamedTuple):
    """The scoring loop's output: every stage, plus synthesis's input.

    Attributes:
        stages: One :class:`~groundkit.evals.schema.StageResult` per planned
            stage, in plan order, baseline first. Nothing is filtered — a
            losing stage is reported, not dropped (SPEC.md §6).
        synthesis_inputs: ``(query, results)`` pairs captured from the last
            planned stage's own scoring pass, empty when the caller did not
            ask for them.
    """

    stages: list[StageResult]
    synthesis_inputs: list[tuple[str, list[RetrievalResult]]]


async def _validate_run_inputs(
    *,
    top_k: int,
    embedder: EmbeddingProtocol | None,
    vector_store: VectorStoreProtocol | None,
    reranker: RerankerProtocol | None,
    rerank_candidates: int,
    chat: ChatProtocol | None,
    judge: FaithfulnessJudge | None,
) -> None:
    """Reject an unusable argument combination before any corpus work starts.

    Every check here is cheap and every failure it raises is a caller
    mistake that no amount of ingesting, embedding or model-loading would
    change — so they all run first, together, rather than being discovered
    one at a time by whichever component reaches them. The one non-raising
    case is the in-memory embedder, which is legal and only warned about.

    Args:
        top_k: Results requested per query, per stage.
        embedder: The run's embedder, or ``None``.
        vector_store: The run's dense store, or ``None``.
        reranker: The run's reranker, or ``None``.
        rerank_candidates: Candidate depth fetched before reranking.
        chat: The run's chat provider, or ``None``.
        judge: The run's faithfulness judge, or ``None``.

    Raises:
        ConfigurationError: Exactly one of ``embedder`` / ``vector_store``
            was supplied, ``vector_store`` already holds vectors, or
            ``judge`` was supplied without ``chat``.
        EvalError: ``top_k`` or ``rerank_candidates`` is out of range.
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
    if reranker is not None and not top_k <= rerank_candidates <= MAX_TOP_K:
        raise EvalError(
            f"rerank_candidates must be between top_k ({top_k}) and {MAX_TOP_K}, got "
            f"{rerank_candidates}. Below top_k the reranker could not return a full "
            f"result list; above {MAX_TOP_K} Retriever.search would reject the request "
            "— and both are checked here, before an ingest and a model load are spent "
            "on a run that was always going to be rejected."
        )
    if judge is not None and chat is None:
        raise ConfigurationError(
            "run_eval was given a judge without chat. The judge evaluates a synthesized "
            "answer, and nothing here can produce one without a chat provider — pass "
            "chat too, or drop judge, before any corpus work starts."
        )
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


def _plan_run(
    *,
    top_k: int,
    embedder: EmbeddingProtocol | None,
    vector_store: VectorStoreProtocol | None,
    reranker: RerankerProtocol | None,
    rerank_candidates: int,
) -> _RunSettings:
    """Fix the stage plan and the knobs every later step reads.

    Args:
        top_k: Results requested per query, per stage.
        embedder: The run's embedder, or ``None``.
        vector_store: The run's dense store, or ``None``.
        reranker: The run's reranker, or ``None``.
        rerank_candidates: Candidate depth fetched before reranking.

    Returns:
        The run's :class:`_RunSettings`.

    Raises:
        EvalError: A rerank stage was planned first, which
            :func:`_rerank_input_stage` refuses — unreachable while
            :func:`_planned_stages` seeds the baseline.
    """
    planned = _planned_stages(embedder, vector_store, reranker)
    rerank_input = _rerank_input_stage(planned)
    rerank_model = _reranker_identity(reranker) if reranker is not None else None
    if reranker is not None:
        # Named at INFO rather than buried in the artifact alone: which stage
        # was reranked is the single fact that decides what this run's rerank
        # row means, and a reader watching a console should not have to open
        # the JSON to learn it.
        logger.info(
            "Eval will rerank the %r stage with %r, over %d candidates truncated to %d",
            rerank_input,
            rerank_model,
            rerank_candidates,
            top_k,
        )
    return _RunSettings(
        planned=planned,
        top_k=top_k,
        retrieval_config=RetrievalConfig(),
        embedder=embedder,
        vector_store=vector_store,
        reranker=reranker,
        rerank_candidates=rerank_candidates,
        rerank_input=rerank_input,
        rerank_model=rerank_model,
    )


async def _prepare_corpus(corpus_dir: Path, judgments_path: Path) -> _CorpusPreparation:
    """Load the judgments, resolve every gold span, and hash both inputs.

    All of it happens *before* any indexing: a missing or ambiguous gold
    quote fails the run closed here rather than after an ingest has been
    spent on it.

    Args:
        corpus_dir: Root directory of the corpus documents.
        judgments_path: Path to the JSONL judgments file.

    Returns:
        The run's :class:`_CorpusPreparation`.

    Raises:
        EvalError: A judgment fails to load, a gold span fails to resolve
            against the corpus text, or either input cannot be read.
    """
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
    # caller that handles GroundkitError — the CLI included. Both are also
    # content-sized, so both are dispatched off the event loop; the judgments
    # read was inline while the corpus read beside it was not, which made this
    # comment's "both" true of the error handling and false of the dispatch.
    try:
        corpus_hash = await asyncio.to_thread(_hash_corpus, corpus_root)
    except OSError as exc:
        raise EvalError(f"Cannot hash corpus at {str(corpus_root)!r}: {exc}") from exc
    try:
        judgments_hash = await asyncio.to_thread(
            lambda: hashlib.sha256(judgments_path.read_bytes()).hexdigest()
        )
    except OSError as exc:
        raise EvalError(f"Cannot read judgments file {str(judgments_path)!r}: {exc}") from exc

    return _CorpusPreparation(
        judgments=judgments,
        resolved_spans=resolved_spans,
        corpus_root=corpus_root,
        corpus_hash=corpus_hash,
        judgments_hash=judgments_hash,
    )


async def _index_corpus(
    store: MetadataStoreProtocol,
    *,
    corpus_dir: Path,
    prepared: _CorpusPreparation,
    settings: _RunSettings,
) -> _IndexedCorpus:
    """Ingest the corpus into ``store`` and resolve ground truth against it.

    Args:
        store: The run's freshly-opened throwaway metadata store.
        corpus_dir: Root directory of the corpus documents, ingested with
            :data:`EVAL_CHUNKING_CONFIG`.
        prepared: The corpus preparation, for the resolved gold spans and
            the corpus root every relative path is derived against.
        settings: The run's settings, for the dense pair and retrieval
            config the one shared retriever is opened with.

    Returns:
        The run's :class:`_IndexedCorpus`, valid while ``store`` stays open.

    Raises:
        EvalError: A gold span's document was never persisted, so its ground
            truth cannot be computed.
        IngestionError: Ingesting ``corpus_dir`` fails.
        StorageError: The throwaway index fails to write.
    """
    indexer = Indexer(
        store,
        FileLoader(allowed_base_dir=corpus_dir),
        chunking_config=EVAL_CHUNKING_CONFIG,
        embedder=settings.embedder,
        vector_store=settings.vector_store,
    )
    await indexer.index_directory(str(corpus_dir))

    # Opened AFTER ingestion completes: a Retriever snapshots the store at
    # open() and never refreshes it (ADR-0002). One retriever serves every
    # stage — nothing writes between stages, so the snapshot stays valid,
    # and sharing it keeps the stages comparing retrieval strategy rather
    # than index state.
    retriever = await Retriever.open(
        store,
        settings.retrieval_config,
        embedder=settings.embedder,
        vector_store=settings.vector_store,
    )

    all_chunks = await store.get_chunks()
    sources = await store.get_document_sources()

    # Document.source is always an absolute realpath — never corpus-relative
    # (ingestion/loaders.py). Map every document back to a corpus-relative,
    # forward-slashed path exactly once here, via Path.relative_to, so the
    # artifact stays identical across machines and checkout locations.
    document_id_to_relpath = {
        document_id: Path(source).relative_to(prepared.corpus_root).as_posix()
        for document_id, source in sources.items()
    }
    relpath_to_document_id = {
        relpath: document_id for document_id, relpath in document_id_to_relpath.items()
    }

    chunks_by_document: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in all_chunks:
        chunks_by_document[chunk.document_id].append(chunk)

    # Ground truth is a property of the corpus and the judgments, not of any
    # retrieval strategy, so it is resolved once here and reused by every
    # stage. Recomputing it per stage would be wasted work and, worse, a
    # place for two stages to disagree about what "relevant" meant.
    gold_by_query: dict[str, _JudgmentGold] = {
        judgment.query_id: _build_gold(
            judgment,
            prepared.resolved_spans[judgment.query_id],
            chunks_by_document=chunks_by_document,
            relpath_to_document_id=relpath_to_document_id,
        )
        for judgment in prepared.judgments
    }

    return _IndexedCorpus(
        retriever=retriever,
        document_count=len(sources),
        chunk_count=len(all_chunks),
        document_id_to_relpath=document_id_to_relpath,
        gold_by_query=gold_by_query,
    )


async def _score_stages(
    judgments: Sequence[Judgment],
    indexed: _IndexedCorpus,
    settings: _RunSettings,
    *,
    capture_synthesis_inputs: bool,
) -> _ScoredStages:
    """Replay the judgment set through every planned stage, in plan order.

    Every stage scores the same judgments against the same index and the
    same retriever, so the stages differ only in retrieval strategy — which
    is what makes their deltas
    (:func:`~groundkit.evals.delta.derive_stage_deltas`) measure that alone.
    Nothing is filtered here: a stage that loses to the baseline is reported
    with the numbers it earned (SPEC.md §6).

    Args:
        judgments: The judgment set, in file order.
        indexed: The open retriever and the ground truth resolved against
            the same index.
        settings: The run's settings, for the stage plan and the retrieval
            knobs each stage is scored at.
        capture_synthesis_inputs: Whether to keep the last planned stage's
            raw results for the synthesis pass. Captured *during* that
            stage's own scoring pass rather than re-retrieved afterwards —
            re-fetching would run every query through the retriever, and on
            a dense or hybrid stage through the embedding provider, twice.

    Returns:
        The run's :class:`_ScoredStages`.

    Raises:
        EvalError: A retrieved hit's document has no corpus-relative path,
            or a stage has no answerable judgment to aggregate over.
        RerankerNotConfiguredError: The reranker cannot load its model.
        RetrievalError: A search or a rerank call failed.
    """
    stages: list[StageResult] = []
    synthesis_inputs: list[tuple[str, list[RetrievalResult]]] = []
    for plan in settings.planned:
        is_synthesis_input = capture_synthesis_inputs and plan is settings.planned[-1]
        query_results: list[QueryResult] = []
        for judgment in judgments:
            query_result, raw_results = await _evaluate_judgment(
                judgment,
                indexed.gold_by_query[judgment.query_id],
                retriever=indexed.retriever,
                top_k=settings.top_k,
                mode=plan.mode,
                document_id_to_relpath=indexed.document_id_to_relpath,
                reranker=settings.reranker if plan.reranks else None,
                rerank_candidates=settings.rerank_candidates,
            )
            query_results.append(query_result)
            if is_synthesis_input:
                synthesis_inputs.append((judgment.query, raw_results))
        stages.append(
            _build_stage_result(
                query_results,
                stage=plan.stage,
                is_baseline=not stages,
            )
        )
    return _ScoredStages(stages=stages, synthesis_inputs=synthesis_inputs)


def _build_report(
    stages: list[StageResult],
    *,
    synthesis: SynthesisReport | None,
    started_at: str,
    prepared: _CorpusPreparation,
    document_count: int,
    chunk_count: int,
    settings: _RunSettings,
) -> EvalReport:
    """Assemble the run's artifact from what the stages actually produced.

    Args:
        stages: The scored stages, baseline first.
        synthesis: The synthesis pass's report, or ``None`` when no chat
            provider was supplied.
        started_at: ISO-8601 timestamp taken before any corpus work.
        prepared: The corpus preparation, for the two input hashes and the
            judgment count.
        document_count: Documents actually persisted by this run.
        chunk_count: Chunks actually persisted by this run.
        settings: The run's settings, recorded into
            :class:`~groundkit.evals.schema.RunConfig`.

    Returns:
        The assembled :class:`~groundkit.evals.schema.EvalReport`.
    """
    embedding = identity_of(settings.embedder) if settings.embedder is not None else None
    # Recorded only when a fusion stage actually ran: on a BM25-only run the
    # constant was never applied to anything, and stamping the configured
    # value into the artifact anyway would describe a computation that did
    # not happen.
    rrf_k = settings.retrieval_config.rrf_k if settings.embedder is not None else None
    # This and the other two rerank fields key off the same condition, which
    # RunConfig's validator then re-checks: a rerank record that is half
    # present is the silently-incomparable artifact ADR-0012 forbids.
    rerank_candidates = settings.rerank_candidates if settings.reranker is not None else None

    run_metadata = RunMetadata(
        started_at=started_at,
        groundkit_version=__version__,
        corpus_hash=prepared.corpus_hash,
        judgments_hash=prepared.judgments_hash,
        document_count=document_count,
        chunk_count=chunk_count,
        judgment_count=len(prepared.judgments),
        config=RunConfig(
            chunk_size=EVAL_CHUNKING_CONFIG.chunk_size,
            chunk_overlap=EVAL_CHUNKING_CONFIG.chunk_overlap,
            top_k=settings.top_k,
            bm25_k1=settings.retrieval_config.bm25_k1,
            bm25_b=settings.retrieval_config.bm25_b,
            score_threshold=settings.retrieval_config.score_threshold,
            embedding=embedding,
            rrf_k=rrf_k,
            rerank_input=settings.rerank_input,
            rerank_candidates=rerank_candidates,
            rerank_model=settings.rerank_model,
        ),
    )
    logger.info(
        "Eval run complete: %d stages, %d judgments, %d documents, %d chunks",
        len(stages),
        len(prepared.judgments),
        document_count,
        chunk_count,
    )
    return EvalReport(run=run_metadata, stages=stages, synthesis=synthesis)


def _planned_stages(
    embedder: EmbeddingProtocol | None,
    vector_store: VectorStoreProtocol | None,
    reranker: RerankerProtocol | None = None,
) -> tuple[_StagePlan, ...]:
    """The stages this run will emit, baseline first.

    Args:
        embedder: The run's embedder, or ``None``.
        vector_store: The run's vector store, or ``None``. Pairing is
            already validated by the caller; both are read here so the
            dense stages cannot be planned off half a pair.
        reranker: The run's reranker, or ``None``. When supplied, a
            ``rerank`` stage is appended over the best upstream stage
            available (ADR-0012 decision 1).

    Returns:
        The ordered stage plan. Always starts with the ``bm25`` baseline;
        appends :data:`_DENSE_STAGE_MODES` when a dense pair is present, and
        a rerank stage last when a reranker is present.
    """
    plans: tuple[_StagePlan, ...] = (_StagePlan("bm25", "bm25"),)
    if embedder is not None and vector_store is not None:
        plans += tuple(_StagePlan(stage, mode) for stage, mode in _DENSE_STAGE_MODES)
    if reranker is not None:
        # "Best available" is read off the plan already built rather than
        # re-deriving it from the dense pair: the input stage named in the
        # artifact is then the same stage the report actually contains, by
        # construction, instead of two independent answers to the same
        # question that could disagree.
        upstream = plans[-1]
        plans += (_StagePlan("rerank", upstream.mode, reranks=True),)
    return plans


def _rerank_input_stage(plans: tuple[_StagePlan, ...]) -> StageName | None:
    """The stage a ``rerank`` plan reorders, or ``None`` if there is none.

    Derived from the plan rather than recomputed from the dense pair, for
    the reason :func:`_planned_stages` gives: ``RunConfig.rerank_input`` must
    name a stage that is genuinely in this report.

    Args:
        plans: The run's stage plan.

    Returns:
        The upstream stage's name, or ``None`` when no rerank stage is
        planned.
    """
    for position, plan in enumerate(plans):
        if plan.reranks:
            if position == 0:
                # `plans[-1]` would silently return the LAST stage's name
                # rather than raising IndexError, so a rerank plan that ever
                # reached index 0 would stamp a confident, wrong
                # `rerank_input` into the artifact — contract-legal and
                # corrupt, the combination this repo treats as worse than a
                # crash. Unreachable today, since `_planned_stages` always
                # seeds the bm25 baseline first, which is exactly why it is
                # asserted rather than trusted: the guarantee lives in
                # another function that a later stage insertion could change
                # without anyone touching this one.
                raise EvalError(
                    "a rerank stage cannot be first in the plan: it reorders an upstream "
                    "stage's results, and at index 0 there is no stage before it"
                )
            return plans[position - 1].stage
    return None


def _reranker_identity(reranker: RerankerProtocol) -> str:
    """Best available identity string for ``reranker``, for the artifact.

    Read with ``getattr`` rather than through the protocol, deliberately.
    :class:`~groundkit.retrieval.protocols.RerankerProtocol` encodes
    ADR-0001 hazard 4 and is held to exact signature parity by
    ``tests/test_protocol_conformance.py``; widening it with a ``model_name``
    member to satisfy a reporting need would change a seam for a reason that
    has nothing to do with the seam.

    A test double has no model name, so it falls back to its class name —
    and that is the useful outcome, not a degraded one. It puts
    ``"_StubReranker"`` in ``RunConfig.rerank_model`` where a real run puts
    ``"cross-encoder/ms-marco-MiniLM-L-6-v2"``, so the artifact self-labels
    which of the two it is, exactly as ``embedding.provider == "inmemory"``
    does for the dense stages (SPEC.md §2).

    Args:
        reranker: The run's reranker.

    Returns:
        The reranker's ``model_name`` if it exposes a non-empty string one,
        otherwise its class name.
    """
    model_name = getattr(reranker, "model_name", None)
    if isinstance(model_name, str) and model_name:
        return model_name
    return type(reranker).__name__


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


async def _fetch_stage_results(
    retriever: Retriever,
    query: str,
    *,
    top_k: int,
    mode: SearchMode,
    reranker: RerankerProtocol | None,
    rerank_candidates: int,
) -> tuple[list[RetrievalResult], float]:
    """Retrieve one query's candidates for a stage, reranking if configured.

    The one place "what would this stage return for this query" is answered
    — shared by :func:`_evaluate_judgment` (scoring a stage against the
    golden corpus judgments) and :func:`run_eval`'s golden-corpus synthesis
    pass (the ``chat`` argument), so the two cannot independently drift on
    what a rerank stage's candidates actually are.

    Args:
        retriever: An open retriever snapshotting the freshly-ingested
            corpus.
        query: The query to retrieve for.
        top_k: Results requested. With a ``reranker``, this is the count
            reranking truncates back down to after reordering.
        mode: The retrieval mode. For a rerank stage this is the *input*
            stage's mode — a reranker consumes results rather than
            producing them (ADR-0012 decision 2).
        reranker: Reranker to apply, or ``None`` for a plain retrieval call.
        rerank_candidates: Candidates fetched before reranking. Unused when
            ``reranker`` is ``None``.

    Returns:
        The stage's results (reranked, when ``reranker`` is given) and the
        latency, in milliseconds, retrieval (plus reranking, if applied)
        took.

    Raises:
        RerankerNotConfiguredError: The reranker cannot load its model.
        RetrievalError: The search or the reranker failed.
    """
    fetch_k = rerank_candidates if reranker is not None else top_k
    response = await retriever.search(query, top_k=fetch_k, mode=mode)
    latency_ms = float(response.metadata["latency_ms"])
    results = list(response.results)

    if reranker is not None:
        # Timed here and ADDED to the retrieval latency, rather than
        # replacing it or being dropped. The cross-encoder is by a wide
        # margin the most expensive thing in this stage — it is the cost a
        # reader is weighing the quality gain against — so a rerank stage
        # reporting only its retrieval time would show the model as very
        # nearly free, which is the opposite of true and is the number
        # `latency_p50_delta_ms` would then publish.
        rerank_started = time.perf_counter_ns()
        results = await reranker.rerank(query, results, top_k=top_k)
        latency_ms += (time.perf_counter_ns() - rerank_started) / _NS_PER_MS

    return results, latency_ms


async def _evaluate_judgment(
    judgment: Judgment,
    gold: _JudgmentGold,
    *,
    retriever: Retriever,
    top_k: int,
    mode: SearchMode,
    document_id_to_relpath: dict[str, str],
    reranker: RerankerProtocol | None = None,
    rerank_candidates: int = MAX_TOP_K,
) -> tuple[QueryResult, list[RetrievalResult]]:
    """Score one judgment against a live retriever in one retrieval mode.

    With a ``reranker``, this asks the retriever for ``rerank_candidates``
    results rather than ``top_k`` and reranks them back down. The wider
    fetch is the whole point: reranking truncates *after* reordering, so
    handing the model exactly ``top_k`` candidates leaves it able only to
    permute a fixed set, and every ``@k`` metric at the full cutoff is then
    pinned to the input stage's by arithmetic (ADR-0012 decision 1a).

    Args:
        judgment: The judgment to score.
        gold: The judgment's pre-resolved ground truth (:func:`_build_gold`).
        retriever: An open retriever snapshotting the freshly-ingested
            corpus.
        top_k: Results requested for this query.
        mode: The retrieval mode this stage measures. For a rerank stage
            this is the *input* stage's mode.
        document_id_to_relpath: ``document_id`` -> corpus-relative path, for
            labeling retrieved hits.
        reranker: Reranker to apply to this stage's candidates, or ``None``
            for a plain retrieval stage.
        rerank_candidates: Candidates fetched before reranking. Unused when
            ``reranker`` is ``None``.

    Returns:
        This judgment's fully-scored
        :class:`~groundkit.evals.schema.QueryResult`, and the raw
        :class:`~groundkit.contracts.RetrievalResult` list it was scored
        from. The raw list is returned rather than discarded so the synthesis
        pass can reuse the retrieval this call already performed: it used to
        re-fetch the winning stage for every judgment, doubling retrieval work
        on the ``--judge`` path and, with a dense or hybrid stage, doubling
        the embedding-provider calls too.

    Raises:
        EvalError: A retrieved hit's document has no corresponding entry
            among the indexed corpus documents, so the result cannot be
            labeled.
        RerankerNotConfiguredError: The reranker cannot load its model.
        RetrievalError: The reranker failed while scoring.
    """
    is_no_answer = judgment.category == "no_answer"
    gold_ids = gold.chunk_ids
    gold_results = gold.spans

    results, latency_ms = await _fetch_stage_results(
        retriever,
        judgment.query,
        top_k=top_k,
        mode=mode,
        reranker=reranker,
        rerank_candidates=rerank_candidates,
    )

    ranked_ids = [result.chunk_id for result in results]

    retrieved: list[RetrievedHit] = []
    for rank, result in enumerate(results, start=1):
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

    return (
        QueryResult(
            query_id=judgment.query_id,
            query=judgment.query,
            category=judgment.category,
            is_no_answer=is_no_answer,
            gold=gold_results,
            total_relevant_chunks=total_relevant_chunks,
            retrieved=retrieved,
            metrics=metrics,
            latency_ms=latency_ms,
        ),
        results,
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

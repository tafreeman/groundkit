"""Tests for the rerank eval stage (SPEC.md §6, ADR-0012).

Synthetic fixtures only — the real ``evals/corpus/`` and ``evals/judgments.jsonl``
are authored separately and exercised by ``tests/test_corpus_integrity.py``, not
here. Async methods are driven with ``asyncio.run()`` inside sync test functions
(pytest-asyncio is not configured in this repo), mirroring ``tests/test_runner.py``.

Every reranker double here is pure and stub-only: no torch, no model, no real
cross-encoder. A stub reranker's quality *numbers* are never asserted — SPEC.md
§2 treats them as noise with a sign — only structure, provenance, and delta
*direction* are.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from groundkit.config import RetrievalConfig
from groundkit.contracts import Chunk, RetrievalResult
from groundkit.errors import EvalError
from groundkit.evals.delta import derive_stage_deltas
from groundkit.evals.runner import (
    MIN_EVAL_TOP_K,
    _rerank_input_stage,
    _StagePlan,
    run_eval,
)
from groundkit.evals.schema import EvalReport
from groundkit.index.bm25 import BM25Index
from groundkit.index.dense import InMemoryVectorStore
from groundkit.index.protocols import VectorStoreProtocol
from groundkit.providers.embeddings import InMemoryEmbedder
from groundkit.providers.protocols import EmbeddingProtocol
from groundkit.retrieval.fusion import reciprocal_rank_fusion
from groundkit.retrieval.protocols import RerankerProtocol
from groundkit.retrieval.search import MAX_TOP_K


def _write_judgments(path: Path, judgments: list[dict[str, Any]]) -> None:
    """Write ``judgments`` as JSONL, one compact JSON object per line."""
    lines = [json.dumps(judgment) for judgment in judgments]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run(
    corpus: Path,
    tmp_path: Path,
    judgments: list[dict[str, Any]],
    *,
    top_k: int = MIN_EVAL_TOP_K,
    embedder: EmbeddingProtocol | None = None,
    vector_store: VectorStoreProtocol | None = None,
    reranker: RerankerProtocol | None = None,
    rerank_candidates: int = MAX_TOP_K,
) -> EvalReport:
    """Write ``judgments`` and run ``run_eval`` with the given rerank configuration."""

    async def run() -> EvalReport:
        judgments_path = tmp_path / "judgments.jsonl"
        _write_judgments(judgments_path, judgments)
        return await run_eval(
            corpus,
            judgments_path,
            top_k=top_k,
            embedder=embedder,
            vector_store=vector_store,
            reranker=reranker,
            rerank_candidates=rerank_candidates,
        )

    return asyncio.run(run())


def _dense_pair() -> tuple[InMemoryEmbedder, InMemoryVectorStore]:
    """A working offline dense pair, for structural rerank-provenance assertions only.

    Never for a quality assertion: ``InMemoryEmbedder``'s vectors are
    hash-derived and carry no semantic signal (SPEC.md §2).
    """
    return InMemoryEmbedder(dimensions=32), InMemoryVectorStore()


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """Three short single-chunk docs a shared query matches in all three."""
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "alpha.md").write_text(
        "The quokka survey documented burrow counts across the alpha reserve.",
        encoding="utf-8",
    )
    (root / "beta.md").write_text(
        "The quokka survey recorded burrow counts across the beta reserve.",
        encoding="utf-8",
    )
    (root / "gamma.md").write_text(
        "The quokka survey tracked burrow counts across the gamma reserve.",
        encoding="utf-8",
    )
    return root


_TRIPLE_MATCH_JUDGMENT: dict[str, Any] = {
    "query_id": "quokka-survey",
    "query": "quokka survey burrow counts",
    "category": "normal",
    "gold": [{"doc": "alpha.md", "quote": "quokka survey"}],
}


@pytest.fixture
def wide_corpus(tmp_path: Path) -> Path:
    """Fifteen short single-chunk docs sharing one query term, wider than top_k=10."""
    root = tmp_path / "wide_corpus"
    root.mkdir()
    for index in range(15):
        (root / f"doc{index:02d}.md").write_text(
            f"wideterm entry number {index:02d} appears in this fixture document.",
            encoding="utf-8",
        )
    return root


_WIDE_JUDGMENT: dict[str, Any] = {
    "query_id": "wideterm-query",
    "query": "wideterm entry",
    "category": "normal",
    "gold": [{"doc": "doc07.md", "quote": "wideterm entry number 07"}],
}


def _build_loss_corpus(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    """A two-doc corpus where bm25 ranks the gold document first, by a wide margin.

    ``alpha`` shares all four query terms; ``beta`` shares only one. The wide
    margin is what a reversing reranker needs to visibly demote ``alpha`` off
    rank 1 — with a closer race the reversal might not move recall@1 at all.
    """
    corpus = tmp_path / "loss_corpus"
    corpus.mkdir()
    (corpus / "alpha.md").write_text(
        "quokka research burrow telemetry documented across the vast southern reserve",
        encoding="utf-8",
    )
    (corpus / "beta.md").write_text(
        "quokka mentioned briefly in passing today",
        encoding="utf-8",
    )
    judgment: dict[str, Any] = {
        "query_id": "loss-probe",
        "query": "quokka research burrow telemetry",
        "category": "normal",
        "gold": [{"doc": "alpha.md", "quote": "quokka research burrow telemetry"}],
    }
    return corpus, judgment


class _RecordingReranker:
    """Records how many candidates each call received; returns them unchanged.

    Satisfies :class:`~groundkit.retrieval.protocols.RerankerProtocol` structurally.
    """

    def __init__(self) -> None:
        self.candidate_counts: list[int] = []

    async def rerank(
        self, query: str, results: list[RetrievalResult], *, top_k: int = 5
    ) -> list[RetrievalResult]:
        """Record the candidate count, then return the input order truncated to top_k."""
        self.candidate_counts.append(len(results))
        return results[:top_k]


class _ReversingReranker:
    """Reverses candidate order, forcing a different, worse rank for a leading match."""

    async def rerank(
        self, query: str, results: list[RetrievalResult], *, top_k: int = 5
    ) -> list[RetrievalResult]:
        """Return the candidates in reverse order, truncated to top_k."""
        return list(reversed(results))[:top_k]


class _PromoteLastReranker:
    """Moves the final candidate to the front, leaving the rest in place."""

    async def rerank(
        self, query: str, results: list[RetrievalResult], *, top_k: int = 5
    ) -> list[RetrievalResult]:
        """Promote the last candidate to rank 1, then truncate to top_k."""
        if not results:
            return []
        promoted = [results[-1], *results[:-1]]
        return promoted[:top_k]


class _NamedReranker:
    """Exposes a non-empty ``model_name``, as a real reranker would."""

    model_name = "stub-cross-encoder-v1"

    async def rerank(
        self, query: str, results: list[RetrievalResult], *, top_k: int = 5
    ) -> list[RetrievalResult]:
        """Return the candidates unchanged, truncated to top_k."""
        return results[:top_k]


class _UnnamedReranker:
    """Deliberately exposes no ``model_name``, to exercise the class-name fallback."""

    async def rerank(
        self, query: str, results: list[RetrievalResult], *, top_k: int = 5
    ) -> list[RetrievalResult]:
        """Return the candidates unchanged, truncated to top_k."""
        return results[:top_k]


class _SlowReranker:
    """Sleeps 50ms before returning, so its cost is not free in the stage's latency."""

    async def rerank(
        self, query: str, results: list[RetrievalResult], *, top_k: int = 5
    ) -> list[RetrievalResult]:
        """Sleep, then return the candidates unchanged, truncated to top_k."""
        await asyncio.sleep(0.05)
        return results[:top_k]


class TestStageStructure:
    """The rerank stage's presence and position follow the run's configuration."""

    def test_bm25_only_run_with_reranker_appends_rerank_after_bm25(
        self, corpus: Path, tmp_path: Path
    ) -> None:
        """A BM25-only run with a reranker produces exactly ["bm25", "rerank"]."""
        report = _run(corpus, tmp_path, [_TRIPLE_MATCH_JUDGMENT], reranker=_RecordingReranker())

        assert [stage.stage for stage in report.stages] == ["bm25", "rerank"]

    def test_dense_pair_with_reranker_appends_rerank_after_fusion(
        self, corpus: Path, tmp_path: Path
    ) -> None:
        """A dense run with a reranker produces exactly ["bm25", "dense", "fusion", "rerank"]."""
        embedder, vector_store = _dense_pair()
        report = _run(
            corpus,
            tmp_path,
            [_TRIPLE_MATCH_JUDGMENT],
            embedder=embedder,
            vector_store=vector_store,
            reranker=_RecordingReranker(),
        )

        assert [stage.stage for stage in report.stages] == ["bm25", "dense", "fusion", "rerank"]

    def test_no_reranker_produces_no_rerank_stage_and_null_rerank_fields(
        self, corpus: Path, tmp_path: Path
    ) -> None:
        """Without a reranker, there is no rerank stage and every rerank_* field is None."""
        report = _run(corpus, tmp_path, [_TRIPLE_MATCH_JUDGMENT], reranker=None)

        assert "rerank" not in [stage.stage for stage in report.stages]
        assert report.run.config.rerank_input is None
        assert report.run.config.rerank_candidates is None
        assert report.run.config.rerank_model is None

    def test_exactly_one_baseline_at_index_zero_even_with_a_rerank_stage(
        self, corpus: Path, tmp_path: Path
    ) -> None:
        """Appending a rerank stage must not disturb the single-baseline invariant."""
        report = _run(corpus, tmp_path, [_TRIPLE_MATCH_JUDGMENT], reranker=_RecordingReranker())

        assert [stage.is_baseline for stage in report.stages] == [True, False]
        assert report.stages[0].stage == "bm25"


class TestRerankProvenance:
    """RunConfig records what the rerank stage measured, not just that it ran."""

    def test_rerank_input_is_bm25_on_a_bm25_only_run(self, corpus: Path, tmp_path: Path) -> None:
        """Without a dense pair, the reranker reorders the bm25 stage."""
        report = _run(corpus, tmp_path, [_TRIPLE_MATCH_JUDGMENT], reranker=_RecordingReranker())

        assert report.run.config.rerank_input == "bm25"

    def test_rerank_input_is_fusion_on_a_dense_run(self, corpus: Path, tmp_path: Path) -> None:
        """With a dense pair, the reranker reorders fusion, the best upstream stage."""
        embedder, vector_store = _dense_pair()
        report = _run(
            corpus,
            tmp_path,
            [_TRIPLE_MATCH_JUDGMENT],
            embedder=embedder,
            vector_store=vector_store,
            reranker=_RecordingReranker(),
        )

        assert report.run.config.rerank_input == "fusion"

    def test_rerank_candidates_is_recorded_as_passed(self, corpus: Path, tmp_path: Path) -> None:
        """The artifact records the depth the caller actually configured."""
        report = _run(
            corpus,
            tmp_path,
            [_TRIPLE_MATCH_JUDGMENT],
            reranker=_RecordingReranker(),
            rerank_candidates=MIN_EVAL_TOP_K,
        )

        assert report.run.config.rerank_candidates == MIN_EVAL_TOP_K

    def test_rerank_model_uses_model_name_when_present(self, corpus: Path, tmp_path: Path) -> None:
        """A reranker exposing a non-empty model_name is identified by it."""
        report = _run(corpus, tmp_path, [_TRIPLE_MATCH_JUDGMENT], reranker=_NamedReranker())

        assert report.run.config.rerank_model == _NamedReranker.model_name

    def test_rerank_model_falls_back_to_the_class_name_when_absent(
        self, corpus: Path, tmp_path: Path
    ) -> None:
        """A stub with no model_name self-labels via its class name, never silently.

        This is the property that makes a stub reranker's run self-describing:
        a reader sees ``"_UnnamedReranker"`` in the artifact where a real run
        would show a real cross-encoder identifier, rather than an invented or
        missing model identity.
        """
        report = _run(corpus, tmp_path, [_TRIPLE_MATCH_JUDGMENT], reranker=_UnnamedReranker())

        assert report.run.config.rerank_model == _UnnamedReranker.__name__


class TestCandidateDepth:
    """Candidate fetch depth is observable and, at the extremes, load-bearing."""

    def test_max_candidates_hands_the_reranker_more_than_top_k(
        self, wide_corpus: Path, tmp_path: Path
    ) -> None:
        """Over-fetching to MAX_TOP_K must actually widen what the reranker is given."""
        reranker = _RecordingReranker()
        _run(
            wide_corpus,
            tmp_path,
            [_WIDE_JUDGMENT],
            reranker=reranker,
            rerank_candidates=MAX_TOP_K,
            top_k=MIN_EVAL_TOP_K,
        )

        assert reranker.candidate_counts[0] > MIN_EVAL_TOP_K

    def test_matched_candidates_hands_the_reranker_exactly_top_k(
        self, wide_corpus: Path, tmp_path: Path
    ) -> None:
        """A depth equal to top_k leaves the reranker nothing beyond what it can permute."""
        reranker = _RecordingReranker()
        _run(
            wide_corpus,
            tmp_path,
            [_WIDE_JUDGMENT],
            reranker=reranker,
            rerank_candidates=MIN_EVAL_TOP_K,
            top_k=MIN_EVAL_TOP_K,
        )

        assert reranker.candidate_counts[0] == MIN_EVAL_TOP_K

    def test_matched_depth_pins_rerank_retrieved_set_to_the_input_stage(
        self, wide_corpus: Path, tmp_path: Path
    ) -> None:
        """At rerank_candidates == top_k, the reranker can only permute, never widen.

        The recording reranker returns candidates unchanged, so with nothing
        to truncate the rerank stage's retrieved set must exactly equal the
        input stage's — proving its recall@10 is pinned by arithmetic here,
        not measured.
        """
        report = _run(
            wide_corpus,
            tmp_path,
            [_WIDE_JUDGMENT],
            reranker=_RecordingReranker(),
            rerank_candidates=MIN_EVAL_TOP_K,
            top_k=MIN_EVAL_TOP_K,
        )
        bm25_stage, rerank_stage = report.stages[0], report.stages[1]

        bm25_hits = [
            (hit.document, hit.start_offset, hit.end_offset)
            for hit in bm25_stage.queries[0].retrieved
        ]
        rerank_hits = [
            (hit.document, hit.start_offset, hit.end_offset)
            for hit in rerank_stage.queries[0].retrieved
        ]
        assert rerank_hits == bm25_hits

        bm25_metrics = bm25_stage.queries[0].metrics
        rerank_metrics = rerank_stage.queries[0].metrics
        assert bm25_metrics is not None
        assert rerank_metrics is not None
        assert rerank_metrics.recall_at_10 == bm25_metrics.recall_at_10

    def test_over_fetch_with_promote_last_proves_deeper_candidates_change_the_winner(
        self, wide_corpus: Path, tmp_path: Path
    ) -> None:
        """The falsifiable proof that over-fetching matters.

        With rerank_candidates == MAX_TOP_K the reranker sees more than the
        input stage's top 10 (proved above). The promote-last reranker moves
        the final candidate — by construction outside that top 10 — to rank
        1, so the rerank stage's winner is provably one the input stage never
        surfaced at all.
        """
        report = _run(
            wide_corpus,
            tmp_path,
            [_WIDE_JUDGMENT],
            reranker=_PromoteLastReranker(),
            rerank_candidates=MAX_TOP_K,
            top_k=MIN_EVAL_TOP_K,
        )
        bm25_stage, rerank_stage = report.stages[0], report.stages[1]

        bm25_top10 = {
            (hit.document, hit.start_offset, hit.end_offset)
            for hit in bm25_stage.queries[0].retrieved
        }
        winner = rerank_stage.queries[0].retrieved[0]
        winner_key = (winner.document, winner.start_offset, winner.end_offset)

        assert winner_key not in bm25_top10


class TestRerankLatencyIncludesTheModelCall:
    """SPEC.md §6: the rerank stage's dominant cost must not be omitted from its latency."""

    def test_slow_reranker_latency_is_at_least_the_sleep_duration(
        self, corpus: Path, tmp_path: Path
    ) -> None:
        """asyncio.sleep(0.05) guarantees >=50ms; 45ms leaves slack for Windows timers."""
        report = _run(corpus, tmp_path, [_TRIPLE_MATCH_JUDGMENT], reranker=_SlowReranker())
        rerank_stage = report.stages[1]

        assert rerank_stage.queries[0].latency_ms >= 45.0

    def test_slow_reranker_latency_exceeds_the_input_stage_latency_for_the_same_query(
        self, corpus: Path, tmp_path: Path
    ) -> None:
        """Regression test: a rerank stage timing only retrieval would hide the model's cost.

        Without adding the rerank call's own timing, this stage's latency
        would look roughly equal to bm25's instead of at least 50ms slower —
        reporting the cross-encoder as very nearly free, which is the
        opposite of true.
        """
        report = _run(corpus, tmp_path, [_TRIPLE_MATCH_JUDGMENT], reranker=_SlowReranker())
        bm25_stage, rerank_stage = report.stages[0], report.stages[1]

        assert rerank_stage.queries[0].latency_ms > bm25_stage.queries[0].latency_ms


class TestHonestLossReporting:
    """SPEC.md §6: a stage that loses to its input is reported, never suppressed."""

    def test_reversing_reranker_stage_is_present_with_its_real_worse_numbers(
        self, tmp_path: Path
    ) -> None:
        """The reversing reranker demotes the gold hit; the loss must show up honestly."""
        loss_corpus, judgment = _build_loss_corpus(tmp_path)

        report = _run(loss_corpus, tmp_path, [judgment], reranker=_ReversingReranker())
        bm25_stage, rerank_stage = report.stages[0], report.stages[1]

        assert [stage.stage for stage in report.stages] == ["bm25", "rerank"]
        assert bm25_stage.aggregate.recall_at_1 == 1.0
        assert rerank_stage.aggregate.recall_at_1 == 0.0
        assert rerank_stage.aggregate.mrr < bm25_stage.aggregate.mrr

    def test_the_derived_delta_flags_the_reversal_as_a_regression(self, tmp_path: Path) -> None:
        """derive_stage_deltas must never filter a losing stage out of the report."""
        loss_corpus, judgment = _build_loss_corpus(tmp_path)

        report = _run(loss_corpus, tmp_path, [judgment], reranker=_ReversingReranker())
        rerank_delta = next(
            delta for delta in derive_stage_deltas(report) if delta.stage == "rerank"
        )

        assert rerank_delta.is_regression is True
        assert rerank_delta.quality["recall_at_1"] < 0.0


class TestRerankCandidatesValidation:
    """rerank_candidates is bound-checked before any ingest or reranker call."""

    def test_below_top_k_raises_eval_error(self, corpus: Path, tmp_path: Path) -> None:
        """The lower bound: a reranker cannot truncate a set it was never given."""
        with pytest.raises(EvalError, match="rerank_candidates"):
            _run(
                corpus,
                tmp_path,
                [_TRIPLE_MATCH_JUDGMENT],
                reranker=_RecordingReranker(),
                rerank_candidates=MIN_EVAL_TOP_K - 1,
                top_k=MIN_EVAL_TOP_K,
            )

    def test_above_max_top_k_raises_eval_error(self, corpus: Path, tmp_path: Path) -> None:
        """The upper bound: Retriever.search would reject this depth anyway."""
        with pytest.raises(EvalError, match="rerank_candidates"):
            _run(
                corpus,
                tmp_path,
                [_TRIPLE_MATCH_JUDGMENT],
                reranker=_RecordingReranker(),
                rerank_candidates=MAX_TOP_K + 1,
                top_k=MIN_EVAL_TOP_K,
            )

    def test_rerank_candidates_is_ignored_without_a_reranker(
        self, corpus: Path, tmp_path: Path
    ) -> None:
        """An out-of-range value is harmless when there is no reranker to bound."""
        report = _run(
            corpus,
            tmp_path,
            [_TRIPLE_MATCH_JUDGMENT],
            reranker=None,
            rerank_candidates=MIN_EVAL_TOP_K - 1,
            top_k=MIN_EVAL_TOP_K,
        )

        assert [stage.stage for stage in report.stages] == ["bm25"]
        assert report.run.config.rerank_candidates is None

    def test_bound_violation_is_rejected_before_the_reranker_is_ever_called(
        self, corpus: Path, tmp_path: Path
    ) -> None:
        """No ingest or model call is spent on a run that was always going to be rejected."""
        reranker = _RecordingReranker()

        with pytest.raises(EvalError):
            _run(
                corpus,
                tmp_path,
                [_TRIPLE_MATCH_JUDGMENT],
                reranker=reranker,
                rerank_candidates=MIN_EVAL_TOP_K - 1,
                top_k=MIN_EVAL_TOP_K,
            )

        assert reranker.candidate_counts == []


class TestStageIndependenceWithRerank:
    """Every stage, rerank included, scores the same judgment set and agrees on gold spans."""

    def test_every_stage_scores_the_same_judgment_set(self, corpus: Path, tmp_path: Path) -> None:
        """Stages must differ by strategy alone, never by which queries they answered."""
        embedder, vector_store = _dense_pair()
        report = _run(
            corpus,
            tmp_path,
            [_TRIPLE_MATCH_JUDGMENT],
            embedder=embedder,
            vector_store=vector_store,
            reranker=_RecordingReranker(),
        )

        query_ids = [[qr.query_id for qr in stage.queries] for stage in report.stages]
        assert query_ids[0] == query_ids[1] == query_ids[2] == query_ids[3]

    def test_gold_spans_agree_across_every_stage_including_rerank(
        self, corpus: Path, tmp_path: Path
    ) -> None:
        """Ground truth is corpus-derived, so no stage — rerank included — may disagree."""
        embedder, vector_store = _dense_pair()
        report = _run(
            corpus,
            tmp_path,
            [_TRIPLE_MATCH_JUDGMENT],
            embedder=embedder,
            vector_store=vector_store,
            reranker=_RecordingReranker(),
        )

        gold_per_stage = [stage.queries[0].gold for stage in report.stages]
        assert gold_per_stage[0] == gold_per_stage[1] == gold_per_stage[2] == gold_per_stage[3]


def _chunk(document_id: str, content: str, *, chunk_index: int = 0) -> Chunk:
    """A minimal valid Chunk; only ``document_id`` and ``content`` vary per call."""
    return Chunk(
        document_id=document_id,
        chunk_index=chunk_index,
        content=content,
        start_offset=0,
        end_offset=len(content),
    )


class TestRerankInputStageWraparound:
    """Fix 3: a rerank plan at index 0 must raise, never silently read ``plans[-1]``.

    ``_rerank_input_stage`` looks up the stage before a rerank plan via
    ``plans[position - 1]``. At ``position == 0`` that expression is
    ``plans[-1]`` — Python silently wraps negative indices rather than
    raising ``IndexError``, so an unguarded lookup would return the LAST
    plan's name instead of failing. That is worse than a crash: it stamps a
    confident, wrong ``rerank_input`` into the artifact rather than refusing
    to report one.
    """

    def test_rerank_plan_at_index_zero_raises_instead_of_wrapping_to_the_last_plan(
        self,
    ) -> None:
        """Two plans, not one, so a silent ``plans[-1]`` wraparound would return
        a real (and wrong) stage name — ``"fusion"`` — rather than trivially
        returning the rerank plan's own name back to itself. That distinction
        is what makes this a meaningful regression test rather than one that
        would pass by accident against either version of the source.
        """
        plans = (
            _StagePlan("rerank", "bm25", reranks=True),
            _StagePlan("fusion", "hybrid"),
        )

        with pytest.raises(EvalError, match="cannot be first"):
            _rerank_input_stage(plans)


class TestBM25IsDepthInvariant:
    """Fix 4 (property, not a defect fix): BM25 scores every chunk independently
    of ``top_k`` (``index/bm25.py``) — ``top_k`` only truncates an
    already-fully-sorted list, so a wider fetch is a strict prefix of a
    narrower one. This is what lets ``derive_rerank_attribution``
    (``evals/delta.py``) call a BM25-input rerank delta a clean isolation of
    the cross-encoder's own contribution: the reranker sees a superset of
    exactly what the baseline stage scored, and nothing about widening the
    fetch could have rescored or reordered any of it.
    """

    def test_top_10_of_top_50_equals_a_direct_top_10_search(self) -> None:
        """``search(top_k=50)[:10] == search(top_k=10)``, chunk-for-chunk and
        score-for-score — the exact prefix property the module docstring in
        ``evals/delta.py`` relies on."""
        index = BM25Index()
        chunks = [
            _chunk(
                f"doc-{i:03d}",
                f"quokka burrow telemetry entry {i:03d} across the reserve",
            )
            for i in range(20)
        ]
        index.index_chunks(chunks)

        deep = index.search("quokka burrow telemetry reserve", top_k=50)
        shallow = index.search("quokka burrow telemetry reserve", top_k=10)

        assert len(shallow) == 10
        assert deep[:10] == shallow


class TestRRFIsNotDepthInvariant:
    """Fix 4 (property, not a defect fix): RRF's score for a chunk sums
    ``1 / (rrf_k + rank)`` over every input ranking it is *visible in at the
    fetched depth* (``retrieval/fusion.py``). Widening the fetch therefore
    adds contributions that did not exist at the narrower one —
    ``fusion@50`` is a genuinely different ranking function, not a deeper
    slice of ``fusion@10``. Unlike BM25 (``TestBM25IsDepthInvariant``), this
    is exactly why ``derive_rerank_attribution`` can only call a
    fusion-input rerank delta a real production comparison, never the
    cross-encoder's isolated contribution: part of the delta may belong to
    the wider candidate pool the reranker was handed, not to the reranker.

    The "hidden" chunk below sits at rank 11 in BOTH input rankings —
    deliberately not rank 1 of either. A chunk visible at rank 1 of any
    ranking is visible at every depth >= 1, so nothing about widening the
    fetch could ever change whether it appears in the fused output; the two
    depth-10 and depth-50 top-10s would then be identical and this test
    would pass without demonstrating anything.
    """

    def test_depth_50_top_10_differs_from_depth_10_top_10(self) -> None:
        rrf_k = RetrievalConfig().rrf_k  # 60 by default (ADR-0005 decision 2)

        # Ranks 1..9 in BOTH rankings: always in every depth-10 window.
        shared = [_chunk(f"shared-{i}", f"shared chunk number {i}") for i in range(1, 10)]
        # Rank 10, each in ONE ranking only: fills out both depth-10 windows.
        unique_to_a = _chunk("unique-a", "unique to ranking a")
        unique_to_b = _chunk("unique-b", "unique to ranking b")
        # Rank 11 in BOTH rankings: outside every depth-10 window, inside
        # both once the fetch widens past it.
        hidden = _chunk("hidden", "hidden chunk outside both depth-10 windows")

        ranking_a = [(chunk, 1.0) for chunk in [*shared, unique_to_a, hidden]]
        ranking_b = [(chunk, 1.0) for chunk in [*shared, unique_to_b, hidden]]

        depth_10_fused = reciprocal_rank_fusion(
            [ranking_a[:10], ranking_b[:10]], rrf_k=rrf_k, top_k=10
        )
        depth_50_fused = reciprocal_rank_fusion(
            [ranking_a[:50], ranking_b[:50]], rrf_k=rrf_k, top_k=10
        )

        depth_10_ids = {chunk.chunk_id for chunk, _ in depth_10_fused}
        depth_50_ids = {chunk.chunk_id for chunk, _ in depth_50_fused}

        # Fixture sanity: if this fails, the fixture itself is wrong, not the
        # property under test.
        assert hidden.chunk_id not in depth_10_ids
        assert hidden.chunk_id in depth_50_ids

        assert depth_10_ids != depth_50_ids

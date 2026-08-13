"""Tests for the BM25-baseline eval runner (SPEC.md §6, §8).

Synthetic fixtures only — the real ``evals/corpus/`` and ``evals/judgments.jsonl``
are authored separately and exercised by ``tests/test_corpus_integrity.py``, not
here. Async methods are driven with ``asyncio.run()`` inside sync test functions
(pytest-asyncio is not configured in this repo), mirroring ``tests/test_indexer.py``.
"""

from __future__ import annotations

import asyncio
import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from groundkit.contracts import Chunk, Document
from groundkit.errors import EvalError
from groundkit.evals.runner import EVAL_CHUNKING_CONFIG, run_eval, write_report
from groundkit.evals.schema import EvalReport, MetricSet, RunConfig, RunMetadata, StageResult
from groundkit.ingestion.chunking import RecursiveChunker


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A tiny two-document corpus; each document is short enough to be one chunk."""
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "alpha.md").write_text(
        "The quokka expedition documented burrow temperatures across the reserve.",
        encoding="utf-8",
    )
    (root / "beta.md").write_text(
        "Citations resolve back to character offsets in the original source file.",
        encoding="utf-8",
    )
    return root


def _write_judgments(path: Path, judgments: list[dict[str, Any]]) -> None:
    """Write ``judgments`` as JSONL, one compact JSON object per line."""
    lines = [json.dumps(judgment) for judgment in judgments]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


_QUOKKA_JUDGMENT: dict[str, Any] = {
    "query_id": "quokka-burrow",
    "query": "quokka expedition burrow temperatures",
    "category": "normal",
    "gold": [{"doc": "alpha.md", "quote": "quokka expedition"}],
}


class TestBaselineStage:
    """A run always reports exactly one stage: the bm25 baseline."""

    def test_runner_produces_single_baseline_bm25_stage(self, corpus: Path, tmp_path: Path) -> None:
        """Pins ``stages == [StageResult(stage="bm25", is_baseline=True)]``."""

        async def run() -> EvalReport:
            judgments_path = tmp_path / "judgments.jsonl"
            _write_judgments(judgments_path, [_QUOKKA_JUDGMENT])
            return await run_eval(corpus, judgments_path)

        report = asyncio.run(run())

        assert len(report.stages) == 1
        assert report.stages[0].stage == "bm25"
        assert report.stages[0].is_baseline is True


class TestPathMapping:
    """Every document field in the artifact is corpus-relative, never absolute."""

    def test_runner_maps_absolute_source_to_relative_corpus_path(
        self, corpus: Path, tmp_path: Path
    ) -> None:
        """A retrieved hit's ``document`` must be a bare relative posix path.

        Constructed to fail if the runner ever compares or trims raw
        absolute-path strings instead of using ``Path.relative_to``:
        ``Document.source`` is always an absolute realpath under ``tmp_path``
        (``ingestion/loaders.py``), so a raw-string implementation would leak
        the tmp_path prefix — and, on Windows, a drive letter and
        backslashes — straight into ``document`` instead of ``"alpha.md"``.
        """

        async def run() -> EvalReport:
            judgments_path = tmp_path / "judgments.jsonl"
            _write_judgments(judgments_path, [_QUOKKA_JUDGMENT])
            return await run_eval(corpus, judgments_path)

        report = asyncio.run(run())
        query_result = report.stages[0].queries[0]

        assert query_result.gold[0].document == "alpha.md"
        assert query_result.retrieved, "the query must have matched something to test mapping"

        hit = query_result.retrieved[0]
        assert hit.document == "alpha.md"
        assert ":" not in hit.document
        assert "\\" not in hit.document
        assert str(tmp_path) not in hit.document
        assert str(corpus) not in hit.document


class TestGroundTruth:
    """Ground truth counts every overlapping persisted chunk, not just retrieved ones."""

    def test_runner_ground_truth_counts_all_overlapping_chunks_not_only_retrieved(
        self, tmp_path: Path
    ) -> None:
        """``total_relevant_chunks`` must include a chunk ``top_k`` excluded.

        Builds one long document of unique tokens so ``EVAL_CHUNKING_CONFIG``
        (chunk_size=512, overlap=64) splits it into several chunks with
        adjacent chunks sharing an overlap region. The real chunker (not a
        hand-computed offset) locates that overlap and a unique token fully
        inside it, so the resulting gold span provably overlaps exactly two
        persisted chunks. Retrieving with ``top_k=1`` then returns only one
        of those two chunks — proving ``total_relevant_chunks`` was computed
        from every persisted chunk, not from what came back on the wire.
        """
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        tokens = [f"zz{i:04d}" for i in range(200)]
        text = " ".join(tokens)
        (corpus / "gamma.md").write_text(text, encoding="utf-8")

        probe_document = Document(document_id="probe", source="probe", content=text)
        probe_chunks = RecursiveChunker().chunk(probe_document, config=EVAL_CHUNKING_CONFIG)

        overlap_pair: tuple[Chunk, Chunk] | None = None
        for earlier, later in pairwise(probe_chunks):
            if earlier.end_offset > later.start_offset:
                overlap_pair = (earlier, later)
                break
        assert overlap_pair is not None, "expected the chunker to produce an overlap"
        overlap_start, overlap_end = overlap_pair[1].start_offset, overlap_pair[0].end_offset

        quote: str | None = None
        cursor = 0
        for token in tokens:
            idx = text.index(token, cursor)
            if overlap_start <= idx and idx + len(token) <= overlap_end:
                quote = token
                break
            cursor = idx + 1
        assert quote is not None, "expected a whole token to fit inside the overlap region"

        async def run() -> EvalReport:
            judgments_path = tmp_path / "judgments.jsonl"
            _write_judgments(
                judgments_path,
                [
                    {
                        "query_id": "overlap-probe",
                        "query": quote,
                        "category": "normal",
                        "gold": [{"doc": "gamma.md", "quote": quote}],
                    }
                ],
            )
            return await run_eval(corpus, judgments_path, top_k=1)

        report = asyncio.run(run())
        query_result = report.stages[0].queries[0]

        assert query_result.total_relevant_chunks == 2
        assert len(query_result.retrieved) == 1


class TestNoAnswerHandling:
    """no_answer judgments are excluded from aggregates and tracked separately."""

    def test_runner_no_answer_query_excluded_from_aggregate_and_has_null_metrics(
        self, corpus: Path, tmp_path: Path
    ) -> None:
        """A no_answer query must have ``metrics=None`` and not enter any aggregate."""

        async def run() -> EvalReport:
            judgments_path = tmp_path / "judgments.jsonl"
            _write_judgments(
                judgments_path,
                [
                    {
                        "query_id": "no-answer-unicorns",
                        "query": "unrelated nonsense about invisible unicorns",
                        "category": "no_answer",
                        "gold": [],
                    },
                    _QUOKKA_JUDGMENT,
                ],
            )
            return await run_eval(corpus, judgments_path)

        report = asyncio.run(run())
        stage = report.stages[0]

        no_answer_result = next(qr for qr in stage.queries if qr.query_id == "no-answer-unicorns")
        assert no_answer_result.is_no_answer is True
        assert no_answer_result.metrics is None
        assert no_answer_result.gold == []
        assert no_answer_result.total_relevant_chunks == 0

        # Only the answerable "quokka-burrow" judgment feeds the aggregate.
        assert stage.aggregate.query_count == 1
        assert "no_answer" not in stage.by_category

    def test_runner_no_answer_returning_zero_results_counts_as_abstained(
        self, corpus: Path, tmp_path: Path
    ) -> None:
        """A no_answer query that retrieves nothing counts as abstained."""

        async def run() -> EvalReport:
            judgments_path = tmp_path / "judgments.jsonl"
            _write_judgments(
                judgments_path,
                [
                    {
                        "query_id": "no-answer-zzz",
                        "query": "zzznonexistentqueryterm000",
                        "category": "no_answer",
                        "gold": [],
                    },
                    _QUOKKA_JUDGMENT,
                ],
            )
            return await run_eval(corpus, judgments_path)

        report = asyncio.run(run())
        stage = report.stages[0]

        assert stage.no_answer_query_count == 1
        assert stage.no_answer_abstained_count == 1

        no_answer_result = next(qr for qr in stage.queries if qr.query_id == "no-answer-zzz")
        assert no_answer_result.retrieved == []


class TestIsolation:
    """The runner never writes its throwaway index anywhere durable."""

    def test_runner_leaves_no_files_in_repo_tree(self, corpus: Path, tmp_path: Path) -> None:
        """No ``.sqlite3`` (or any other) file appears under the corpus dir or repo tree."""

        async def run() -> EvalReport:
            judgments_path = tmp_path / "judgments.jsonl"
            _write_judgments(judgments_path, [_QUOKKA_JUDGMENT])
            return await run_eval(corpus, judgments_path)

        before = sorted(p.name for p in corpus.rglob("*"))
        asyncio.run(run())
        after = sorted(p.name for p in corpus.rglob("*"))

        assert after == before
        assert not any(corpus.rglob("*.sqlite3"))

        repo_root = Path(__file__).resolve().parents[1]
        assert not any(repo_root.rglob("*.sqlite3"))


class TestCorpusHash:
    """``corpus_hash`` changes iff corpus content changes."""

    def test_runner_corpus_hash_changes_when_a_document_changes(
        self, corpus: Path, tmp_path: Path
    ) -> None:
        """Editing a corpus document (unrelated to any judgment's gold) changes the hash."""

        async def run() -> EvalReport:
            judgments_path = tmp_path / "judgments.jsonl"
            # References only beta.md, so mutating alpha.md below cannot
            # invalidate gold-span resolution for this judgment.
            _write_judgments(
                judgments_path,
                [
                    {
                        "query_id": "citation-offsets",
                        "query": "citations resolve character offsets",
                        "category": "normal",
                        "gold": [{"doc": "beta.md", "quote": "character offsets"}],
                    }
                ],
            )
            return await run_eval(corpus, judgments_path)

        first = asyncio.run(run())
        (corpus / "alpha.md").write_text("A completely different alpha document.", encoding="utf-8")
        second = asyncio.run(run())

        assert first.run.corpus_hash != second.run.corpus_hash

    def test_corpus_hash_stable_across_identical_content(
        self, corpus: Path, tmp_path: Path
    ) -> None:
        """Two runs over unchanged corpus content produce the same hash."""

        async def run() -> EvalReport:
            judgments_path = tmp_path / "judgments.jsonl"
            _write_judgments(judgments_path, [_QUOKKA_JUDGMENT])
            return await run_eval(corpus, judgments_path)

        first = asyncio.run(run())
        second = asyncio.run(run())

        assert first.run.corpus_hash == second.run.corpus_hash


def _minimal_report() -> EvalReport:
    """A schema-valid, all-zeros report — enough to exercise the writer."""
    return EvalReport(
        run=RunMetadata(
            started_at="2026-01-01T00:00:00+00:00",
            groundkit_version="0.0.0",
            corpus_hash="a" * 64,
            judgments_hash="b" * 64,
            document_count=0,
            chunk_count=0,
            judgment_count=0,
            config=RunConfig(
                chunk_size=512,
                chunk_overlap=64,
                top_k=10,
                bm25_k1=1.5,
                bm25_b=0.75,
                score_threshold=None,
            ),
        ),
        stages=[
            StageResult(
                stage="bm25",
                is_baseline=True,
                aggregate=MetricSet(
                    query_count=0,
                    recall_at_1=0.0,
                    recall_at_5=0.0,
                    recall_at_10=0.0,
                    mrr=0.0,
                    ndcg_at_10=0.0,
                ),
                by_category={},
                no_answer_query_count=0,
                no_answer_abstained_count=0,
                latency_p50_ms=0.0,
                latency_p95_ms=0.0,
                latency_p99_ms=0.0,
                queries=[],
            )
        ],
    )


class TestWriteReport:
    """``write_report`` is a plain, parent-creating JSON writer."""

    def test_unwritable_output_raises_eval_error(self, tmp_path: Path) -> None:
        """A filesystem failure surfaces as EvalError, not a raw OSError.

        ``main()`` catches only GroundkitError, so a bare OSError here would
        print a traceback after an otherwise successful — and potentially
        expensive — eval run. Pointing --output at an existing directory is
        the portable way to force the failure on both POSIX and Windows.
        """
        report = _minimal_report()
        target = tmp_path / "a-directory"
        target.mkdir()

        with pytest.raises(EvalError, match="Cannot write eval report"):
            write_report(report, target)

    def test_write_report_creates_parent_directories(self, tmp_path: Path) -> None:
        """Writing to a nested, not-yet-existing path creates every parent dir."""
        report = _minimal_report()

        output_path = tmp_path / "nested" / "dir" / "latest.json"
        write_report(report, output_path)

        assert output_path.is_file()
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == 1


class TestEndToEndScoring:
    """A trivially-perfect retrieval scores 1.0 — proves the wiring end to end."""

    def test_runner_perfect_retrieval_scores_one(self, tmp_path: Path) -> None:
        """A query identical to its gold quote, with no lexical competition, is a perfect hit."""
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "target.md").write_text("xylophone quokka expedition telemetry", encoding="utf-8")
        (corpus / "distractor.md").write_text(
            "unrelated botanical notes about garden hedges", encoding="utf-8"
        )

        async def run() -> EvalReport:
            judgments_path = tmp_path / "judgments.jsonl"
            _write_judgments(
                judgments_path,
                [
                    {
                        "query_id": "perfect-hit",
                        "query": "xylophone quokka expedition telemetry",
                        "category": "normal",
                        "gold": [
                            {
                                "doc": "target.md",
                                "quote": "xylophone quokka expedition telemetry",
                            }
                        ],
                    }
                ],
            )
            return await run_eval(corpus, judgments_path)

        report = asyncio.run(run())
        query_result = report.stages[0].queries[0]

        assert query_result.metrics is not None
        assert query_result.metrics.recall_at_1 == 1.0
        assert query_result.metrics.reciprocal_rank == 1.0
        assert report.stages[0].aggregate.mrr == 1.0

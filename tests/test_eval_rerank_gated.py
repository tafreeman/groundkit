"""Real-model rerank *eval* run behind ``RERANK_GATED=1`` (Phase 3 Wave D/E).

Companion to ``tests/test_rerank_gated.py`` (which proves the cross-encoder
backend in isolation) and ``tests/test_eval_gated.py`` (which proves the
dense/fusion stages against a real embedder). This module proves the third
leg: that ``groundkit.evals.runner.run_eval`` can drive a real
:class:`~groundkit.retrieval.rerank.CrossEncoderReranker` end to end and
still produce a schema-valid, honestly-reported artifact. Everything here is
skipped unless the gate is explicitly opened, and the skip is the *normal*
outcome — matching both companion modules, the default suite must stay
offline, torch-free, and green with no ``RERANK_GATED`` set.

**Why a gate, not a mock, for this specific claim.** ``tests/test_rerank.py``
already covers the pure rerank path (sigmoid, ordering, contract
re-validation) with no model at all, and ``tests/test_runner.py`` covers the
runner's rerank *wiring* with a stub reranker. Neither can show that a real
model, loaded through ``run_eval``'s own code path, produces a stage whose
latency reflects genuine inference cost and whose delta is derivable exactly
as the CLI and schema promise. A stub reranker returns instantly and would
pass the latency assertion below for the wrong reason.

**Never the real golden corpus.** ``tests/test_eval_gated.py`` gates on the
committed ``evals/corpus/`` because it needs that corpus's real size and
content to be a meaningful dense/fusion measurement. This module needs
neither: reranking is a stage over whatever the retriever already returned,
so a small synthetic, disposable corpus is enough to prove the plumbing
without inheriting a dependency on golden-corpus size or content drifting
out from under this test.

**The honesty rule, restated.** Exactly as ``tests/test_eval_gated.py``
documents for dense/fusion: this module does **not** assert that reranking
*beats* the BM25 baseline on any metric. A loss or a wash from a real
cross-encoder on a three-document synthetic corpus is not evidence the
reranker is broken — it is what "small corpus, real model" can honestly
produce. Asserting a direction here would be the same trap: a test that
reddens on a legitimate measured loss pressures the next person to grow the
corpus until the number moves, which is fitting the benchmark to the result
rather than reporting it. What is asserted is that the delta *exists* and is
*derivable*, with both directions representable — never which way it went.

Opening the gate::

    uv sync --extra rerank
    RERANK_GATED=1 uv run pytest tests/test_eval_rerank_gated.py -v

The first run downloads :data:`GATED_RERANK_MODEL` (overridable by
environment variable, matching ``tests/test_rerank_gated.py``) into the
sentence-transformers cache.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING

import pytest

from groundkit.errors import GroundkitError
from groundkit.evals.delta import QUALITY_METRIC_FIELDS, derive_rerank_attribution
from groundkit.evals.runner import run_eval
from groundkit.evals.schema import EvalReport
from groundkit.retrieval.rerank import DEFAULT_RERANK_MODEL, CrossEncoderReranker
from groundkit.retrieval.search import MAX_TOP_K

if TYPE_CHECKING:
    from pathlib import Path

#: Cross-encoder used by the gated run. Overridable so the gate can be
#: pointed at a smaller or already-cached model without editing this file —
#: the same variable ``tests/test_rerank_gated.py`` reads, so one override
#: affects both gated modules together.
GATED_RERANK_MODEL: str = os.environ.get("GROUNDKIT_GATED_RERANK_MODEL", DEFAULT_RERANK_MODEL)

pytestmark = pytest.mark.skipif(
    os.environ.get("RERANK_GATED") != "1",
    reason="real cross-encoder eval run; set RERANK_GATED=1 and install the 'rerank' extra",
)


def _corpus_files() -> dict[str, str]:
    """Three short, thematically distinct passages — enough to rank, not to measure.

    Deliberately not the committed ``evals/corpus/``: see the module
    docstring. Reused across the module's fixtures rather than the golden
    corpus's size or content mattering to what this module proves.
    """
    return {
        "sky.md": (
            "Sunlight reaching Earth's atmosphere is scattered by gas molecules. Shorter "
            "blue wavelengths scatter far more than longer red ones, a process called "
            "Rayleigh scattering, which is why the daytime sky looks blue."
        ),
        "recipe.md": (
            "Preheat the oven to 200C. Combine the flour, butter and sugar, then rest "
            "the dough in the refrigerator for at least one hour before rolling it out."
        ),
        "citations.md": (
            "Citations resolve back to character offsets in the original source file, "
            "which is what makes a retrieved passage independently verifiable."
        ),
    }


def _judgment_lines() -> list[str]:
    """Two judgments against :func:`_corpus_files`, one gold span each."""
    return [
        json.dumps(
            {
                "query_id": "sky-color",
                "query": "why does the sky appear blue",
                "category": "normal",
                "gold": [{"doc": "sky.md", "quote": "Rayleigh scattering"}],
            }
        ),
        json.dumps(
            {
                "query_id": "citation-offsets",
                "query": "how are citations verified against a source",
                "category": "normal",
                "gold": [{"doc": "citations.md", "quote": "character offsets"}],
            }
        ),
    ]


@pytest.fixture(scope="module")
def synthetic_corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny, throwaway corpus directory, never the real ``evals/corpus/``."""
    root = tmp_path_factory.mktemp("rerank-gated-corpus")
    for name, text in _corpus_files().items():
        (root / name).write_text(text, encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def synthetic_judgments(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A minimal judgments JSONL matching :func:`synthetic_corpus`."""
    path = tmp_path_factory.mktemp("rerank-gated-judgments") / "judgments.jsonl"
    path.write_text("\n".join(_judgment_lines()) + "\n", encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def gated_report(synthetic_corpus: Path, synthetic_judgments: Path) -> EvalReport:
    """One real cross-encoder eval run over the synthetic corpus, no dense pair.

    Module-scoped: loading a real model and running inference is the
    expensive part, so every assertion below reads the one run rather than
    paying for a fresh model load per test. With no ``embedder``/
    ``vector_store`` supplied, ``run_eval`` plans exactly ``["bm25",
    "rerank"]`` (ADR-0012 decision 1: a reranker with no dense pair reorders
    the BM25 baseline), which is the case this module exists to prove.
    """
    reranker = CrossEncoderReranker(GATED_RERANK_MODEL)

    async def run() -> EvalReport:
        return await run_eval(
            synthetic_corpus,
            synthetic_judgments,
            reranker=reranker,
            # Pinned rather than left to the default so the value under test
            # is stated here, not merely inherited from run_eval's signature.
            rerank_candidates=MAX_TOP_K,
        )

    try:
        return asyncio.run(run())
    except GroundkitError as exc:  # pragma: no cover - only on a gated run
        pytest.fail(
            f"Gated rerank eval failed loading or running {GATED_RERANK_MODEL!r}: {exc}. "
            "The gate was opened, so the 'rerank' extra was expected to be installed "
            "and the model loadable."
        )


class TestGatedRerankStageShape:
    """A real cross-encoder run produces the rerank stage over the BM25 baseline."""

    def test_report_has_bm25_then_rerank_stages(self, gated_report: EvalReport) -> None:
        assert [stage.stage for stage in gated_report.stages] == ["bm25", "rerank"]

    def test_rerank_input_is_bm25(self, gated_report: EvalReport) -> None:
        """No dense pair was supplied, so rerank reordered the baseline itself."""
        assert gated_report.run.config.rerank_input == "bm25"

    def test_rerank_model_is_the_real_model_identifier(self, gated_report: EvalReport) -> None:
        """Names an actual cross-encoder, not a class name.

        A test double falls back to ``type(reranker).__name__`` (see
        ``groundkit.evals.runner._reranker_identity``); a real model is
        expected to name itself, which is the fact this checks.
        """
        assert gated_report.run.config.rerank_model == GATED_RERANK_MODEL
        assert gated_report.run.config.rerank_model != CrossEncoderReranker.__name__

    def test_rerank_candidates_equals_the_pinned_ceiling(self, gated_report: EvalReport) -> None:
        assert gated_report.run.config.rerank_candidates == MAX_TOP_K


class TestGatedRerankAttributionIsDerivable:
    """The point of the gate: a real, derivable delta against the input stage.

    CRITICAL — the honesty rule this file exists to enforce: nothing below
    asserts that reranking *beats* the baseline on any metric. A loss or a
    wash from a real model on a three-document synthetic corpus is a
    legitimate result, not a defect; only that a delta is reported, with
    both directions representable, is asserted (see the module docstring
    and ``tests/test_eval_gated.py``, which states the same stance for
    dense/fusion).
    """

    def test_attribution_delta_is_reported_against_bm25(self, gated_report: EvalReport) -> None:
        attribution = derive_rerank_attribution(gated_report)

        assert attribution is not None
        assert attribution.stage == "rerank"
        assert attribution.baseline_stage == "bm25"

    def test_attribution_covers_every_quality_metric(self, gated_report: EvalReport) -> None:
        """A partially-reported delta would let one metric's regression hide."""
        attribution = derive_rerank_attribution(gated_report)
        assert attribution is not None
        assert set(attribution.quality) == set(QUALITY_METRIC_FIELDS)


class TestGatedRerankLatency:
    """A real cross-encoder inference pass is not free."""

    def test_rerank_stage_latency_exceeds_the_baseline_it_reordered(
        self, gated_report: EvalReport
    ) -> None:
        stages = {stage.stage: stage for stage in gated_report.stages}
        assert stages["rerank"].latency_p50_ms > stages["bm25"].latency_p50_ms

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

**The committed golden corpus, not a fixture.** ``tests/test_eval_gated.py``
pins the real ``evals/corpus/`` and ``evals/judgments.jsonl`` — "the real
ones, not a fixture. A gated run measures the artifact this repo actually
ships" — and carries a test named
``test_run_covers_the_committed_golden_corpus`` specifically to guard
against a gated run quietly measuring something else instead. This module
now gates on that same committed corpus and carries the same guard, so
rerank's evidence is not weaker than dense/fusion's for no reason connected
to what a reranker actually needs.

**Two gated configurations.**

1. **Rerank over BM25** — gated on ``RERANK_GATED=1`` alone. No embedder, no
   vector store, no Ollama; only the ``rerank`` extra. This is the primary
   configuration, and the one whose delta is a clean isolation of the
   reranker: BM25 scores each document independently of ``top_k``
   (``index/bm25.py``), so ``bm25@50`` is a strict prefix of ``bm25@10`` and
   the cross-encoder sees a superset of exactly what the baseline stage
   scored.
2. **Rerank over fusion** — gated on ``RERANK_GATED=1`` *and* ``EVAL_GATED=1``
   together, since it additionally needs a real embedding provider (Ollama +
   ``nomic-embed-text``, built the same way ``tests/test_eval_gated.py``
   builds it). RRF is **not** depth-invariant — it sums
   ``1/(rrf_k + rank)`` over the rankings a chunk is visible in at the
   fetched depth, so a wider fetch adds contributions that did not exist at
   the narrower one, and a chunk absent from ``fusion@10`` can rank first in
   ``fusion@50`` (see :mod:`groundkit.evals.delta`'s module docstring). So
   this configuration's rerank delta measures the rerank pipeline at
   candidate depth versus the fusion stage as reported, **not** the
   cross-encoder's isolated contribution, and nothing below claims
   otherwise.

**The honesty rule, restated.** Exactly as ``tests/test_eval_gated.py``
documents for dense/fusion: this module does **not** assert that reranking
*beats* either baseline on any metric. A loss or a wash from a real
cross-encoder on the golden corpus is not evidence the reranker is broken —
it is what "real corpus, real model" can honestly produce. Asserting a
direction here would be the same trap: a test that reddens on a legitimate
measured loss pressures the next person to grow the corpus until the number
moves, which is fitting the benchmark to the result rather than reporting
it. What is asserted is that the delta *exists* and is *derivable*, with
both directions representable — never which way it went, and never, for the
fusion configuration, that it isolates the reranker.

Opening the gate::

    uv sync --extra rerank
    RERANK_GATED=1 uv run pytest tests/test_eval_rerank_gated.py -v

    # rerank-over-fusion also needs an embedding provider:
    RERANK_GATED=1 EVAL_GATED=1 uv run pytest tests/test_eval_rerank_gated.py -v

The first run downloads :data:`GATED_RERANK_MODEL` (overridable by
environment variable, matching ``tests/test_rerank_gated.py``) into the
sentence-transformers cache.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from groundkit.config import EmbeddingConfig
from groundkit.errors import GroundkitError
from groundkit.evals.delta import QUALITY_METRIC_FIELDS, derive_rerank_attribution
from groundkit.evals.runner import run_eval
from groundkit.evals.schema import EvalReport
from groundkit.index.dense import InMemoryVectorStore
from groundkit.providers.embeddings import INMEMORY_PROVIDER, build_embedder
from groundkit.retrieval.rerank import DEFAULT_RERANK_MODEL, CrossEncoderReranker
from groundkit.retrieval.search import MAX_TOP_K

#: Environment variable that opens the primary gate (rerank over BM25).
RERANK_GATED_ENV: str = "RERANK_GATED"

#: Environment variable that, opened alongside :data:`RERANK_GATED_ENV`,
#: additionally exercises the rerank-over-fusion configuration below — it
#: needs a real embedding provider, exactly what this variable already gates
#: in ``tests/test_eval_gated.py``.
EVAL_GATED_ENV: str = "EVAL_GATED"

#: Cross-encoder used by the gated run. Overridable so the gate can be
#: pointed at a smaller or already-cached model without editing this file —
#: the same variable ``tests/test_rerank_gated.py`` reads, so one override
#: affects both gated modules together.
GATED_RERANK_MODEL: str = os.environ.get("GROUNDKIT_GATED_RERANK_MODEL", DEFAULT_RERANK_MODEL)

#: Embedding model the fusion-input configuration uses. Matches
#: ``tests/test_eval_gated.py`` exactly, including its override variable
#: names, so one Ollama setup serves both gated modules.
GATED_EMBED_MODEL: str = os.environ.get("GROUNDKIT_EVAL_EMBED_MODEL", "nomic-embed-text")

#: Vector width of :data:`GATED_EMBED_MODEL`. Must match the model actually
#: served: the embedder validates the width it receives, so a wrong value
#: fails loudly rather than producing a mis-shaped index.
GATED_EMBED_DIMENSIONS: int = int(os.environ.get("GROUNDKIT_EVAL_EMBED_DIMENSIONS", "768"))

#: Endpoint the fusion-input run embeds against. Defaults to local Ollama.
GATED_EMBED_BASE_URL: str = os.environ.get(
    "GROUNDKIT_EVAL_EMBED_BASE_URL", "http://localhost:11434"
)

#: The committed golden corpus and judgments — the real ones, not a
#: fixture, matching ``tests/test_eval_gated.py``. A gated run measures the
#: artifact this repo actually ships.
_REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR: Path = _REPO_ROOT / "evals" / "corpus"
JUDGMENTS_PATH: Path = _REPO_ROOT / "evals" / "judgments.jsonl"

pytestmark = pytest.mark.skipif(
    os.environ.get(RERANK_GATED_ENV) != "1",
    reason="real cross-encoder eval run; set RERANK_GATED=1 and install the 'rerank' extra",
)


@pytest.fixture(scope="module")
def gated_bm25_report() -> EvalReport:
    """One real cross-encoder run reranking the BM25 baseline over the golden corpus.

    Module-scoped: the model load and the corpus run are the expensive
    part, so every assertion below reads the one run rather than paying for
    it again. With no ``embedder``/``vector_store`` supplied, ``run_eval``
    plans exactly ``["bm25", "rerank"]`` (ADR-0012 decision 1: a reranker
    with no dense pair reorders the BM25 baseline), which makes the delta
    below a clean isolation of the cross-encoder (see the module
    docstring).
    """
    reranker = CrossEncoderReranker(GATED_RERANK_MODEL)

    async def run() -> EvalReport:
        return await run_eval(
            CORPUS_DIR,
            JUDGMENTS_PATH,
            reranker=reranker,
            # Pinned rather than left to the default so the value under
            # test is stated here, not merely inherited from run_eval's
            # signature.
            rerank_candidates=MAX_TOP_K,
        )

    try:
        return asyncio.run(run())
    except GroundkitError as exc:  # pragma: no cover - only on a gated run
        pytest.fail(
            f"Gated rerank eval failed loading or running {GATED_RERANK_MODEL!r} over "
            f"the committed golden corpus: {exc}. The gate was opened, so the 'rerank' "
            "extra was expected to be installed and the model loadable."
        )


@pytest.fixture(scope="module")
def gated_fusion_report() -> EvalReport:
    """One real cross-encoder run reranking the fusion stage over the golden corpus.

    Needs both gates open: a real embedder (built exactly as
    ``tests/test_eval_gated.py`` builds its own) plus a real cross-encoder.
    The attribution this run's rerank stage yields is **not** an isolated
    cross-encoder measurement — see the module docstring and
    :mod:`groundkit.evals.delta` for why RRF's depth-dependence rules that
    out.
    """
    embedder = build_embedder(
        EmbeddingConfig(
            provider="ollama",
            model_name=GATED_EMBED_MODEL,
            dimensions=GATED_EMBED_DIMENSIONS,
            base_url=GATED_EMBED_BASE_URL,
        )
    )
    reranker = CrossEncoderReranker(GATED_RERANK_MODEL)

    async def run() -> EvalReport:
        try:
            return await run_eval(
                CORPUS_DIR,
                JUDGMENTS_PATH,
                embedder=embedder,
                vector_store=InMemoryVectorStore(),
                reranker=reranker,
                rerank_candidates=MAX_TOP_K,
            )
        finally:
            aclose = getattr(embedder, "aclose", None)
            if callable(aclose):
                await aclose()

    try:
        return asyncio.run(run())
    except GroundkitError as exc:  # pragma: no cover - only on a gated run
        pytest.fail(
            f"Gated rerank-over-fusion eval failed against {GATED_EMBED_BASE_URL} using "
            f"embedding model {GATED_EMBED_MODEL!r} and cross-encoder "
            f"{GATED_RERANK_MODEL!r}: {exc}. Both gates were opened, so both a real "
            "embedding provider and the 'rerank' extra were expected to be available."
        )


class TestGatedRerankOverBM25Shape:
    """A real cross-encoder run over the real golden corpus, reordering BM25."""

    def test_run_covers_the_committed_golden_corpus(self, gated_bm25_report: EvalReport) -> None:
        """Guards against this gate quietly measuring a fixture instead.

        Floors rather than exact counts, matching
        ``tests/test_eval_gated.py``: SPEC.md §6 fixes the minimums and
        ``tests/test_corpus_integrity.py`` owns them, so restating exact
        totals here would be a second copy to keep in sync.
        """
        assert gated_bm25_report.run.document_count >= 8
        assert gated_bm25_report.run.judgment_count >= 40

    def test_report_has_bm25_then_rerank_stages(self, gated_bm25_report: EvalReport) -> None:
        assert [stage.stage for stage in gated_bm25_report.stages] == ["bm25", "rerank"]

    def test_rerank_input_is_bm25(self, gated_bm25_report: EvalReport) -> None:
        """No dense pair was supplied, so rerank reordered the baseline itself."""
        assert gated_bm25_report.run.config.rerank_input == "bm25"

    def test_rerank_model_is_the_real_model_identifier(self, gated_bm25_report: EvalReport) -> None:
        """Names an actual cross-encoder, not a class name.

        A test double falls back to ``type(reranker).__name__`` (see
        ``groundkit.evals.runner._reranker_identity``); a real model is
        expected to name itself, which is the fact this checks.
        """
        assert gated_bm25_report.run.config.rerank_model == GATED_RERANK_MODEL
        assert gated_bm25_report.run.config.rerank_model != CrossEncoderReranker.__name__

    def test_rerank_candidates_equals_the_pinned_ceiling(
        self, gated_bm25_report: EvalReport
    ) -> None:
        assert gated_bm25_report.run.config.rerank_candidates == MAX_TOP_K


class TestGatedRerankOverBM25AttributionIsDerivable:
    """The point of the primary gate: a real, cleanly-isolated delta.

    CRITICAL — the honesty rule this file exists to enforce: nothing below
    asserts that reranking *beats* the baseline on any metric. A loss or a
    wash from a real model on the golden corpus is a legitimate result, not
    a defect; only that a delta is reported, with both directions
    representable, is asserted (see the module docstring and
    ``tests/test_eval_gated.py``, which states the same stance for
    dense/fusion).
    """

    def test_attribution_delta_is_reported_against_bm25(
        self, gated_bm25_report: EvalReport
    ) -> None:
        attribution = derive_rerank_attribution(gated_bm25_report)

        assert attribution is not None
        assert attribution.stage == "rerank"
        assert attribution.baseline_stage == "bm25"

    def test_attribution_covers_every_quality_metric(self, gated_bm25_report: EvalReport) -> None:
        """A partially-reported delta would let one metric's regression hide."""
        attribution = derive_rerank_attribution(gated_bm25_report)
        assert attribution is not None
        assert set(attribution.quality) == set(QUALITY_METRIC_FIELDS)


class TestGatedRerankOverBM25Latency:
    """A real cross-encoder inference pass is not free."""

    def test_rerank_stage_latency_exceeds_the_baseline_it_reordered(
        self, gated_bm25_report: EvalReport
    ) -> None:
        stages = {stage.stage: stage for stage in gated_bm25_report.stages}
        assert stages["rerank"].latency_p50_ms > stages["bm25"].latency_p50_ms


@pytest.mark.skipif(
    os.environ.get(EVAL_GATED_ENV) != "1",
    reason=(
        f"{EVAL_GATED_ENV}=1 not set: rerank-over-fusion additionally needs a real "
        "embedding provider, matching the gate tests/test_eval_gated.py itself opens on."
    ),
)
class TestGatedRerankOverFusion:
    """Rerank over the fusion stage: derivable, but NOT an isolated cross-encoder measurement.

    RRF is not depth-invariant (see the module docstring and
    :mod:`groundkit.evals.delta`'s), so this class does not — and must not
    — assert a clean-isolation property the way the BM25-input
    configuration can. It asserts only that the run produces the expected
    four-stage shape, that ``rerank_input`` names ``fusion`` honestly, that
    a real embedding provider was used, and that an attribution is
    derivable and correctly labelled.
    """

    def test_run_produces_four_stages_in_order(self, gated_fusion_report: EvalReport) -> None:
        assert [stage.stage for stage in gated_fusion_report.stages] == [
            "bm25",
            "dense",
            "fusion",
            "rerank",
        ]

    def test_rerank_input_is_fusion(self, gated_fusion_report: EvalReport) -> None:
        """A dense pair was supplied, so rerank reordered the fusion stage."""
        assert gated_fusion_report.run.config.rerank_input == "fusion"

    def test_fusion_uses_a_real_embedding_provider(self, gated_fusion_report: EvalReport) -> None:
        """The artifact must name the real semantic space, not the hash-derived one."""
        embedding = gated_fusion_report.run.config.embedding

        assert embedding is not None
        assert embedding.provider != INMEMORY_PROVIDER
        assert embedding.model_name == GATED_EMBED_MODEL

    def test_attribution_is_derivable_and_labelled_fusion(
        self, gated_fusion_report: EvalReport
    ) -> None:
        """Derivable and correctly labelled — NOT a claim of isolation.

        See the module docstring: RRF's depth-dependence means this delta
        measures the rerank pipeline at candidate depth versus fusion as
        reported, not the cross-encoder alone. Only that a delta exists,
        covers every quality metric, and is labelled against the right
        baseline is asserted.
        """
        attribution = derive_rerank_attribution(gated_fusion_report)

        assert attribution is not None
        assert attribution.stage == "rerank"
        assert attribution.baseline_stage == "fusion"
        assert set(attribution.quality) == set(QUALITY_METRIC_FIELDS)

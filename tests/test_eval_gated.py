"""Real-model eval run behind ``EVAL_GATED=1`` (SPEC.md §6).

Everything here is skipped unless the gate is explicitly opened, and the skip
is the *normal* outcome: default CI, pre-commit, and a plain ``uv run pytest``
must stay offline and credential-free, so these tests never run by accident.

**Why a gate exists at all.** ``InMemoryEmbedder`` produces hash-derived
vectors with no semantic signal, so a dense or fusion quality number computed
with it is noise presented as a number — which SPEC.md §2 forbids. Every
other test in this repo therefore asserts *structure* on the dense path and
never *quality*. A real retrieval-quality delta needs a real embedding model,
which needs a running Ollama, which cannot be a precondition of the default
test suite. That is the whole content of R1 in
``docs/specs/phase-3-hybrid-retrieval.md``, and this module is its resolution.

**What is asserted here, and what deliberately is not.** These tests assert
that a real-model multi-stage run *completes and reports honestly*: three
stages, derivable deltas, recorded embedding identity, metrics inside their
declared ranges. They do **not** assert that dense or fusion beats BM25.
Pinning a direction would make the suite fail on a legitimate measured
outcome — R2 states plainly that on a corpus this size a loss or a wash is a
real result, and a test that goes red on a loss is a test that pressures the
next person to grow the corpus until the number moves, which is fitting the
benchmark to the result. The number is an output of this run, not an
assertion of it.

Opening the gate::

    EVAL_GATED=1 uv run pytest tests/test_eval_gated.py -v

with an Ollama serving :data:`GATED_EMBED_MODEL` at
:data:`GATED_EMBED_BASE_URL` (both overridable by environment variable).
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from groundkit.config import EmbeddingConfig
from groundkit.errors import GroundkitError
from groundkit.evals.delta import QUALITY_METRIC_FIELDS, derive_stage_deltas
from groundkit.evals.runner import run_eval
from groundkit.evals.schema import EvalReport
from groundkit.index.dense import InMemoryVectorStore
from groundkit.providers.embeddings import INMEMORY_PROVIDER, build_embedder

#: Environment variable that opens the gate. Any value other than exactly
#: ``"1"`` leaves these tests skipped — including ``"0"``, ``"true"``, and
#: the unset case. A strict comparison rather than a truthiness check,
#: because "the network tests silently turned themselves on" is a worse
#: failure than "the gate needed the documented value".
EVAL_GATED_ENV: str = "EVAL_GATED"

#: Embedding model the gated run uses. Overridable so a machine with a
#: different local model can still open the gate.
GATED_EMBED_MODEL: str = os.environ.get("GROUNDKIT_EVAL_EMBED_MODEL", "nomic-embed-text")

#: Vector width of :data:`GATED_EMBED_MODEL`. Must match the model actually
#: served: the embedder validates the width it receives, so a wrong value
#: fails loudly rather than producing a mis-shaped index.
GATED_EMBED_DIMENSIONS: int = int(os.environ.get("GROUNDKIT_EVAL_EMBED_DIMENSIONS", "768"))

#: Endpoint the gated run embeds against. Defaults to local Ollama.
GATED_EMBED_BASE_URL: str = os.environ.get(
    "GROUNDKIT_EVAL_EMBED_BASE_URL", "http://localhost:11434"
)

#: The committed golden corpus and judgments — the real ones, not a fixture.
#: A gated run measures the artifact this repo actually ships.
_REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR: Path = _REPO_ROOT / "evals" / "corpus"
JUDGMENTS_PATH: Path = _REPO_ROOT / "evals" / "judgments.jsonl"

pytestmark = pytest.mark.skipif(
    os.environ.get(EVAL_GATED_ENV) != "1",
    reason=(
        f"{EVAL_GATED_ENV}=1 not set: real-model eval is opt-in and requires a running "
        "embedding provider. Skipping cleanly is the expected default (SPEC.md §6)."
    ),
)


@pytest.fixture(scope="module")
def gated_report() -> EvalReport:
    """One real-model multi-stage run over the committed golden corpus.

    Module-scoped: the run embeds the whole corpus against a live model, so
    every assertion below reads the same report rather than paying for it
    again. A provider failure fails the gated job loudly — the gate having
    been opened is a statement that a provider *is* available, so a
    connection error here is a real failure, not a reason to skip.
    """
    embedder = build_embedder(
        EmbeddingConfig(
            provider="ollama",
            model_name=GATED_EMBED_MODEL,
            dimensions=GATED_EMBED_DIMENSIONS,
            base_url=GATED_EMBED_BASE_URL,
        )
    )

    async def run() -> EvalReport:
        try:
            return await run_eval(
                CORPUS_DIR,
                JUDGMENTS_PATH,
                embedder=embedder,
                vector_store=InMemoryVectorStore(),
            )
        finally:
            aclose = getattr(embedder, "aclose", None)
            if callable(aclose):
                await aclose()

    try:
        return asyncio.run(run())
    except GroundkitError as exc:  # pragma: no cover - only on a gated run
        pytest.fail(
            f"Gated eval failed against {GATED_EMBED_BASE_URL} using model "
            f"{GATED_EMBED_MODEL!r}: {exc}. The gate was opened, so a provider was "
            "expected to be reachable."
        )


class TestGatedRunShape:
    """A real-model run produces the full multi-stage artifact."""

    def test_run_produces_bm25_dense_and_fusion_stages(self, gated_report: EvalReport) -> None:
        """The three stages Wave E is responsible for emitting."""
        assert [stage.stage for stage in gated_report.stages] == ["bm25", "dense", "fusion"]

    def test_run_records_the_real_embedding_identity(self, gated_report: EvalReport) -> None:
        """The artifact must name the semantic space it measured."""
        embedding = gated_report.run.config.embedding

        assert embedding is not None
        assert embedding.provider != INMEMORY_PROVIDER
        assert embedding.model_name == GATED_EMBED_MODEL
        assert embedding.dimensions == GATED_EMBED_DIMENSIONS

    def test_run_covers_the_committed_golden_corpus(self, gated_report: EvalReport) -> None:
        """Guards against a gated run quietly measuring a fixture instead.

        Floors rather than exact counts: SPEC.md §6 fixes the minimums and
        ``tests/test_corpus_integrity.py`` owns them, so restating exact
        totals here would be a second copy to keep in sync.
        """
        assert gated_report.run.document_count >= 8
        assert gated_report.run.judgment_count >= 40


class TestGatedDeltasAreDerivable:
    """The point of the gate: a real, signed delta against the BM25 baseline."""

    def test_every_non_baseline_stage_yields_a_delta(self, gated_report: EvalReport) -> None:
        """Both dense and fusion get one, regardless of which way they went."""
        deltas = derive_stage_deltas(gated_report)

        assert [delta.stage for delta in deltas] == ["dense", "fusion"]

    def test_each_delta_covers_every_quality_metric(self, gated_report: EvalReport) -> None:
        """A partially-reported delta would let one metric's regression hide."""
        for delta in derive_stage_deltas(gated_report):
            assert set(delta.quality) == set(QUALITY_METRIC_FIELDS)

    def test_delta_direction_flags_agree_with_the_signed_numbers(
        self, gated_report: EvalReport
    ) -> None:
        """Whatever the outcome, the flags must describe the numbers honestly.

        Deliberately not "dense beats bm25": R2 makes a loss or a wash a
        legitimate measured result on a corpus this size, and a test that
        reddens on a loss would push the next person to grow the corpus
        until it passes.
        """
        for delta in derive_stage_deltas(gated_report):
            assert delta.is_regression == any(value < 0.0 for value in delta.quality.values())
            assert delta.is_improvement == any(value > 0.0 for value in delta.quality.values())

    def test_metrics_stay_inside_their_declared_ranges(self, gated_report: EvalReport) -> None:
        """A real model must not push any aggregate outside ``[0, 1]``."""
        for stage in gated_report.stages:
            aggregate = stage.aggregate
            for name in QUALITY_METRIC_FIELDS:
                value = float(getattr(aggregate, name))
                assert 0.0 <= value <= 1.0, f"{stage.stage}.{name} out of range: {value}"


class TestGatedLatency:
    """Per-stage latency percentiles are real measurements on a real model."""

    def test_every_stage_reports_nonzero_percentiles(self, gated_report: EvalReport) -> None:
        """A live embedding call cannot take zero milliseconds."""
        for stage in gated_report.stages:
            assert stage.latency_p50_ms > 0.0
            assert stage.latency_p95_ms >= stage.latency_p50_ms
            assert stage.latency_p99_ms >= stage.latency_p95_ms

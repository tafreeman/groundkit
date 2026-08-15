"""Real cross-encoder run behind ``RERANK_GATED=1`` (Wave D's backend proof).

Skipped unless the gate is explicitly opened, and the skip is the *normal*
outcome. Wave D's plan requires that CI's default job never pull torch, so the
optional ``rerank`` extra is not in the dev group and these tests must stay
inert in a default install — exactly as ``tests/test_eval_gated.py`` stays inert
without Ollama.

**Why a gate rather than a mock.** Everything about reranking that groundkit
*controls* — the sigmoid, the ordering, the contract re-validation, every
fail-closed path — is pure and is tested in ``tests/test_rerank.py`` with no
model at all. What a mock cannot prove is the part groundkit does not control:
that a real MS MARCO cross-encoder emits raw, unbounded, frequently negative
logits, which is the premise ADR-0001 hazard 2 rests on and the reason the
sigmoid exists. A stub returning ``[-9.0, -1.5]`` asserts that premise rather
than testing it. This module is where the premise is checked against reality,
which makes it the sole proof of this backend and therefore never
``continue-on-error`` (SPEC.md §3).

**What is asserted here, and what deliberately is not.** That a real model
loads, produces finite scores, and that those scores survive the contract and
come back ordered. It does **not** assert that reranking *improves* retrieval
quality — that is an eval-harness question, measured as a stage delta and
reported honestly including when it loses (SPEC.md §6), not a unit-test
assertion. Pinning a direction here would make the suite fail on a legitimate
measured outcome, which is the same trap ``test_eval_gated.py`` documents.

Opening the gate::

    uv sync --extra rerank
    RERANK_GATED=1 uv run pytest tests/test_rerank_gated.py -v

The first run downloads :data:`GATED_RERANK_MODEL` (overridable by environment
variable) into the sentence-transformers cache.
"""

from __future__ import annotations

import asyncio
import math
import os

import pytest

from groundkit.contracts import RetrievalResult
from groundkit.retrieval.rerank import DEFAULT_RERANK_MODEL, CrossEncoderReranker

#: Cross-encoder used by the gated run. Overridable so the gate can be pointed
#: at a smaller or already-cached model without editing this file.
GATED_RERANK_MODEL: str = os.environ.get("GROUNDKIT_GATED_RERANK_MODEL", DEFAULT_RERANK_MODEL)

pytestmark = pytest.mark.skipif(
    os.environ.get("RERANK_GATED") != "1",
    reason="real cross-encoder run; set RERANK_GATED=1 and install the 'rerank' extra",
)


_QUERY = "What causes the sky to appear blue?"

_PASSAGES = {
    "rayleigh": (
        "Sunlight reaching Earth's atmosphere is scattered by gas molecules. Shorter "
        "blue wavelengths scatter far more than longer red ones, a process called "
        "Rayleigh scattering, which is why the daytime sky looks blue."
    ),
    "recipe": (
        "Preheat the oven to 200C. Combine the flour, butter and sugar, then rest the "
        "dough in the refrigerator for at least one hour before rolling it out."
    ),
    "ocean": (
        "The ocean appears blue largely because water absorbs longer wavelengths of "
        "light more strongly than shorter ones, leaving blue light to be scattered."
    ),
}


def _candidates() -> list[RetrievalResult]:
    """One result per passage, all with identical placeholder scores.

    Equal input scores on purpose: any ordering in the output is then the
    reranker's doing and not a leftover of the input order.
    """
    return [
        RetrievalResult(
            content=text,
            score=0.5,
            document_id=f"doc-{name}",
            chunk_id=f"chunk-{name}",
            source=f"{name}.md",
            start_offset=0,
            end_offset=len(text),
        )
        for name, text in _PASSAGES.items()
    ]


class TestRealCrossEncoder:
    def test_real_model_scores_survive_the_contract(self) -> None:
        """The hazard-2 premise, checked against a real model rather than assumed."""

        async def run() -> None:
            reranker = CrossEncoderReranker(GATED_RERANK_MODEL)
            reranked = await reranker.rerank(_QUERY, _candidates(), top_k=3)

            assert len(reranked) == 3
            for result in reranked:
                assert result.score >= 0.0
                assert math.isfinite(result.score)
                assert result.score <= 1.0

        asyncio.run(run())

    def test_real_model_returns_results_in_descending_score_order(self) -> None:
        async def run() -> None:
            reranker = CrossEncoderReranker(GATED_RERANK_MODEL)
            reranked = await reranker.rerank(_QUERY, _candidates(), top_k=3)

            scores = [r.score for r in reranked]
            assert scores == sorted(scores, reverse=True)

        asyncio.run(run())

    def test_real_model_ranks_the_on_topic_passage_above_the_unrelated_one(self) -> None:
        """The one quality assertion, kept deliberately coarse.

        A baking recipe against "why is the sky blue" is not a close call for
        any working reranker, so this catches a model loaded wrong, a
        query/passage pair assembled in the wrong order, or scores attached to
        the wrong passages. It does not encode a ranking between the two
        *plausible* passages, which is a genuine judgement call a model is
        allowed to make either way.
        """

        async def run() -> None:
            reranker = CrossEncoderReranker(GATED_RERANK_MODEL)
            reranked = await reranker.rerank(_QUERY, _candidates(), top_k=3)

            order = [r.chunk_id for r in reranked]
            assert order.index("chunk-rayleigh") < order.index("chunk-recipe")

        asyncio.run(run())

    def test_real_model_truncates_to_top_k(self) -> None:
        async def run() -> None:
            reranker = CrossEncoderReranker(GATED_RERANK_MODEL)
            reranked = await reranker.rerank(_QUERY, _candidates(), top_k=1)

            assert len(reranked) == 1

        asyncio.run(run())

    def test_real_model_emits_unbounded_logits(self) -> None:
        """Documents *why* the sigmoid exists, by observing the raw output.

        Reaches past the public surface to the loaded model on purpose: the
        claim under test is about what the backend produces, which
        :meth:`CrossEncoderReranker.rerank` has by then already normalized
        away. If a future model family returned pre-squashed 0-1 scores, this
        is the test that would notice.
        """

        async def run() -> None:
            reranker = CrossEncoderReranker(GATED_RERANK_MODEL)
            model = reranker._load_model()
            pairs = [[_QUERY, text] for text in _PASSAGES.values()]
            raw = [float(score) for score in await asyncio.to_thread(model.predict, pairs)]

            assert all(math.isfinite(score) for score in raw)
            assert any(score < 0.0 or score > 1.0 for score in raw), (
                f"expected unbounded logits from {GATED_RERANK_MODEL}, got {raw} — if this "
                "model now returns pre-activated scores, the identity activation in "
                "_import_cross_encoder is no longer taking effect"
            )

        asyncio.run(run())

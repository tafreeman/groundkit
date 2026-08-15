"""Cross-encoder rerank: sigmoid normalization, ordering, and fail-closed paths.

Everything here runs in a base install. The optional ``rerank`` extra is never
imported, and no model is ever loaded — :mod:`groundkit.retrieval.rerank` keeps
the heavy import behind a lazy function precisely so its whole pure surface is
reachable offline. The real model is exercised separately in
``tests/test_rerank_gated.py`` behind ``RERANK_GATED=1``.

The centrepiece is :class:`TestHazard2NegativeLogits`. ADR-0001 hazard 2 is the
ported ARP defect where raw cross-encoder logits were fed into a ``score >= 0.0``
field; MS MARCO rerankers emit negative logits routinely, so this was not an edge
case but the ordinary path.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time

import pytest

from groundkit.contracts import RetrievalResult
from groundkit.errors import RerankerNotConfiguredError, RetrievalError
from groundkit.retrieval.rerank import (
    DEFAULT_RERANK_MODEL,
    SIGMOID_SATURATION_LOGIT,
    CrossEncoderReranker,
    rerank_by_logits,
    sigmoid,
)


def _result(chunk_id: str, *, source: str = "doc.md", start: int = 0) -> RetrievalResult:
    """A minimal valid result; only the fields rerank actually reads vary."""
    return RetrievalResult(
        content=f"content of {chunk_id}",
        score=0.5,
        document_id="doc-1",
        chunk_id=chunk_id,
        source=source,
        start_offset=start,
        end_offset=start + 10,
    )


class TestSigmoid:
    def test_maps_zero_to_one_half(self) -> None:
        assert sigmoid(0.0) == pytest.approx(0.5)

    def test_is_monotonic_across_sign_change(self) -> None:
        values = [sigmoid(x) for x in (-10.0, -3.0, -0.5, 0.0, 0.5, 3.0, 10.0)]
        assert values == sorted(values)
        assert len(set(values)) == len(values)

    @pytest.mark.parametrize("logit", [-1000.0, -745.0, -50.0, -1.0, 0.0, 1.0, 50.0, 1000.0])
    def test_output_always_satisfies_the_contract_bound(self, logit: float) -> None:
        assert sigmoid(logit) >= 0.0

    def test_extreme_negative_logit_does_not_overflow(self) -> None:
        """The naive ``1/(1+exp(-x))`` spelling raises OverflowError here."""
        assert sigmoid(-100_000.0) == 0.0

    def test_extreme_positive_logit_saturates_without_error(self) -> None:
        assert sigmoid(100_000.0) == 1.0

    def test_saturation_constant_is_accurate(self) -> None:
        """The documented saturation point is a measured value, not a guess."""
        assert sigmoid(-SIGMOID_SATURATION_LOGIT) > 0.0
        assert sigmoid(-(SIGMOID_SATURATION_LOGIT + 10.0)) == 0.0

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_non_finite_logit_is_refused(self, bad: float) -> None:
        with pytest.raises(RetrievalError, match="non-finite"):
            sigmoid(bad)


class TestHazard2NegativeLogits:
    """ADR-0001 hazard 2 — raw negative logits into a ``ge=0.0`` contract.

    Wave D's plan states the assertion exactly: feed negative logits, assert no
    ``ValidationError`` and that ordering is preserved. Both halves matter, and
    the second is why the fix is a sigmoid rather than a clamp — clamping would
    also produce no ``ValidationError`` while collapsing every negative score
    into a single tie, satisfying the contract and destroying the ranking.
    """

    def test_all_negative_logits_produce_valid_results(self) -> None:
        results = [_result("c1", start=0), _result("c2", start=10), _result("c3", start=20)]
        logits = [-8.5, -2.1, -11.0]

        reranked = rerank_by_logits(results, logits, top_k=3)

        assert len(reranked) == 3
        assert all(r.score >= 0.0 for r in reranked)

    def test_ordering_is_preserved_through_normalization(self) -> None:
        """Sigmoid is monotonic, so the reranked order must equal the logit order."""
        results = [_result("c1", start=0), _result("c2", start=10), _result("c3", start=20)]
        logits = [-8.5, -2.1, -11.0]

        reranked = rerank_by_logits(results, logits, top_k=3)

        # -2.1 > -8.5 > -11.0
        assert [r.chunk_id for r in reranked] == ["c2", "c1", "c3"]

    def test_negative_logits_are_not_clamped_to_a_tie(self) -> None:
        """The distinguishing assertion: a clamp would make these three equal."""
        results = [_result("c1", start=0), _result("c2", start=10), _result("c3", start=20)]

        reranked = rerank_by_logits(results, [-8.5, -2.1, -11.0], top_k=3)

        scores = [r.score for r in reranked]
        assert len(set(scores)) == 3, "negative logits collapsed into a tie — clamped, not squashed"
        assert all(s > 0.0 for s in scores)

    def test_mixed_sign_logits_rank_positives_above_negatives(self) -> None:
        results = [_result("neg", start=0), _result("pos", start=10), _result("zero", start=20)]

        reranked = rerank_by_logits(results, [-4.0, 4.0, 0.0], top_k=3)

        assert [r.chunk_id for r in reranked] == ["pos", "zero", "neg"]

    def test_contract_is_genuinely_revalidated(self) -> None:
        """Results are rebuilt through the constructor, not ``model_copy``.

        ``model_copy(update=...)`` skips validation, which would make every
        assertion above vacuous — a negative score would pass straight through.
        Asserting the constructor still rejects one proves validation is live on
        the type these results are built as.
        """
        with pytest.raises(Exception, match="greater than or equal to 0"):
            RetrievalResult(
                content="x",
                score=-0.5,
                document_id="d",
                chunk_id="c",
                source="s.md",
                start_offset=0,
                end_offset=1,
            )


class TestRerankByLogits:
    def test_truncates_to_top_k(self) -> None:
        results = [_result(f"c{i}", start=i * 10) for i in range(5)]

        reranked = rerank_by_logits(results, [1.0, 2.0, 3.0, 4.0, 5.0], top_k=2)

        assert [r.chunk_id for r in reranked] == ["c4", "c3"]

    def test_top_k_larger_than_candidates_returns_all(self) -> None:
        results = [_result("c1", start=0), _result("c2", start=10)]

        assert len(rerank_by_logits(results, [1.0, 2.0], top_k=50)) == 2

    def test_does_not_mutate_input(self) -> None:
        results = [_result("c1", start=0), _result("c2", start=10)]
        before = [r.score for r in results]

        rerank_by_logits(results, [3.0, -3.0], top_k=2)

        assert [r.score for r in results] == before
        assert [r.chunk_id for r in results] == ["c1", "c2"]

    def test_preserves_every_citation_field(self) -> None:
        original = RetrievalResult(
            content="hello world",
            score=0.1,
            document_id="doc-9",
            chunk_id="chunk-9",
            source="notes/a.md",
            start_offset=12,
            end_offset=23,
            metadata={"heading": "Intro"},
        )

        (reranked,) = rerank_by_logits([original], [2.0], top_k=1)

        assert reranked.content == original.content
        assert reranked.document_id == original.document_id
        assert reranked.chunk_id == original.chunk_id
        assert reranked.source == original.source
        assert reranked.start_offset == original.start_offset
        assert reranked.end_offset == original.end_offset
        assert reranked.metadata == original.metadata
        assert reranked.score != original.score

    def test_ties_break_on_source_then_offset_deterministically(self) -> None:
        results = [
            _result("c3", source="b.md", start=5),
            _result("c1", source="a.md", start=30),
            _result("c2", source="a.md", start=10),
        ]

        reranked = rerank_by_logits(results, [1.0, 1.0, 1.0], top_k=3)

        assert [r.chunk_id for r in reranked] == ["c2", "c1", "c3"]

    def test_empty_candidates_returns_empty(self) -> None:
        assert rerank_by_logits([], [], top_k=5) == []

    def test_misaligned_batch_is_refused(self) -> None:
        results = [_result("c1", start=0), _result("c2", start=10)]

        with pytest.raises(RetrievalError, match="3 scores for 2 results"):
            rerank_by_logits(results, [1.0, 2.0, 3.0], top_k=2)

    @pytest.mark.parametrize("bad_top_k", [0, -1])
    def test_non_positive_top_k_is_refused(self, bad_top_k: int) -> None:
        with pytest.raises(RetrievalError, match="top_k must be > 0"):
            rerank_by_logits([_result("c1")], [1.0], top_k=bad_top_k)

    def test_non_finite_model_output_is_refused(self) -> None:
        with pytest.raises(RetrievalError, match="non-finite"):
            rerank_by_logits([_result("c1")], [math.nan], top_k=1)


class TestCrossEncoderRerankerConstruction:
    """Construction is total: no import, no filesystem, no network."""

    def test_construction_loads_nothing(self) -> None:
        reranker = CrossEncoderReranker()
        assert reranker.model_name == DEFAULT_RERANK_MODEL

    def test_model_name_is_configurable(self) -> None:
        assert CrossEncoderReranker("some/other-model").model_name == "some/other-model"

    def test_default_model_is_an_ms_marco_family_model(self) -> None:
        """The default is deliberately a model that emits unbounded logits."""
        assert "ms-marco" in DEFAULT_RERANK_MODEL


class TestCrossEncoderRerankerFailsClosed:
    """An unconfigured reranker raises; it never returns the input unchanged."""

    def test_empty_results_short_circuit_before_model_load(self) -> None:
        """Must not raise even with the extra absent — nothing to rerank."""

        async def run() -> None:
            reranker = CrossEncoderReranker("definitely/not-a-real-model")
            assert await reranker.rerank("q", [], top_k=5) == []

        asyncio.run(run())

    def test_non_positive_top_k_refused_before_model_load(self) -> None:
        async def run() -> None:
            reranker = CrossEncoderReranker("definitely/not-a-real-model")
            with pytest.raises(RetrievalError, match="top_k must be > 0"):
                await reranker.rerank("q", [_result("c1")], top_k=0)

        asyncio.run(run())

    def test_missing_backend_raises_rather_than_passing_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The fail-closed guarantee Wave D's plan names explicitly.

        A reranker that returned its input on a missing backend would be
        indistinguishable from one that worked, so a rerank eval stage would
        silently report the upstream stage's numbers under its own name.
        """
        import groundkit.retrieval.rerank as rerank_module

        def _no_backend() -> tuple[object, object]:
            raise RerankerNotConfiguredError("simulated missing extra")

        monkeypatch.setattr(rerank_module, "_import_cross_encoder", _no_backend)

        async def run() -> None:
            reranker = CrossEncoderReranker()
            candidates = [_result("c1", start=0), _result("c2", start=10)]
            with pytest.raises(RerankerNotConfiguredError):
                await reranker.rerank("q", candidates, top_k=2)

        asyncio.run(run())

    def test_model_load_failure_is_typed_and_names_the_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import groundkit.retrieval.rerank as rerank_module

        class _ExplodingCrossEncoder:
            def __init__(self, *args: object, **kwargs: object) -> None:
                raise RuntimeError("no local copy and no network")

        monkeypatch.setattr(
            rerank_module,
            "_import_cross_encoder",
            lambda: (_ExplodingCrossEncoder, object()),
        )

        async def run() -> None:
            reranker = CrossEncoderReranker("some/unreachable-model")
            with pytest.raises(RerankerNotConfiguredError, match="some/unreachable-model"):
                await reranker.rerank("q", [_result("c1")], top_k=1)

        asyncio.run(run())

    def test_model_is_loaded_once_and_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import groundkit.retrieval.rerank as rerank_module

        loads = {"count": 0}

        class _CountingCrossEncoder:
            def __init__(self, *args: object, **kwargs: object) -> None:
                loads["count"] += 1

            def predict(self, pairs: list[list[str]]) -> list[float]:
                return [float(len(pairs) - i) for i in range(len(pairs))]

        monkeypatch.setattr(
            rerank_module,
            "_import_cross_encoder",
            lambda: (_CountingCrossEncoder, object()),
        )

        async def run() -> None:
            reranker = CrossEncoderReranker()
            candidates = [_result("c1", start=0), _result("c2", start=10)]
            await reranker.rerank("q", candidates, top_k=2)
            await reranker.rerank("q", candidates, top_k=2)

        asyncio.run(run())

        assert loads["count"] == 1

    def test_model_is_loaded_off_the_event_loop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Loading must not block the loop, exactly as ``predict`` must not.

        The load is the *more* expensive of the two: a torch import plus, on a
        cold cache, a weight download. Left on the event loop it stalls every
        other coroutine for its whole duration and prevents any cancellation or
        timeout from firing.
        """
        import groundkit.retrieval.rerank as rerank_module

        threads: dict[str, int] = {}

        class _ThreadRecordingCrossEncoder:
            def __init__(self, *args: object, **kwargs: object) -> None:
                threads["load"] = threading.get_ident()

            def predict(self, pairs: list[list[str]]) -> list[float]:
                return [0.0] * len(pairs)

        monkeypatch.setattr(
            rerank_module,
            "_import_cross_encoder",
            lambda: (_ThreadRecordingCrossEncoder, "IDENTITY"),
        )

        async def run() -> None:
            threads["loop"] = threading.get_ident()
            reranker = CrossEncoderReranker()
            await reranker.rerank("q", [_result("c1")], top_k=1)

        asyncio.run(run())

        assert threads["load"] != threads["loop"], "model was constructed on the event loop thread"

    def test_concurrent_first_calls_build_only_one_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Initialization is serialized, not merely cached.

        Two coroutines reaching a cold reranker together both see ``_model is
        None``. Without a lock held across the load, both proceed and build a
        model — duplicating a multi-gigabyte allocation and, on a cold cache,
        the download with it. The stub sleeps inside ``__init__`` to make the
        window the lock closes reliably reachable rather than a race the test
        would usually lose.
        """
        import groundkit.retrieval.rerank as rerank_module

        loads = {"count": 0}

        class _SlowCrossEncoder:
            def __init__(self, *args: object, **kwargs: object) -> None:
                time.sleep(0.05)
                loads["count"] += 1

            def predict(self, pairs: list[list[str]]) -> list[float]:
                return [0.0] * len(pairs)

        monkeypatch.setattr(
            rerank_module,
            "_import_cross_encoder",
            lambda: (_SlowCrossEncoder, "IDENTITY"),
        )

        async def run() -> None:
            reranker = CrossEncoderReranker()
            candidates = [_result("c1")]
            await asyncio.gather(
                reranker.rerank("q", candidates, top_k=1),
                reranker.rerank("q", candidates, top_k=1),
                reranker.rerank("q", candidates, top_k=1),
            )

        asyncio.run(run())

        assert loads["count"] == 1

    def test_max_length_is_forwarded_to_the_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import groundkit.retrieval.rerank as rerank_module

        seen: dict[str, object] = {}

        class _RecordingCrossEncoder:
            def __init__(self, name: str, **kwargs: object) -> None:
                seen.update(kwargs)

            def predict(self, pairs: list[list[str]]) -> list[float]:
                return [0.0] * len(pairs)

        monkeypatch.setattr(
            rerank_module,
            "_import_cross_encoder",
            lambda: (_RecordingCrossEncoder, "IDENTITY"),
        )

        async def run() -> None:
            reranker = CrossEncoderReranker(max_length=128)
            await reranker.rerank("q", [_result("c1")], top_k=1)

        asyncio.run(run())

        assert seen["max_length"] == 128
        assert seen["activation_fn"] == "IDENTITY"

    def test_falls_back_to_the_legacy_activation_parameter_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """sentence-transformers renamed this across major versions.

        The fallback must still pass an *explicit* identity activation under
        the old spelling. Omitting it would inherit the library default, which
        for ``num_labels=1`` is a sigmoid — double-squashing every score
        against our own, which ADR-0005 decision 4 forbids.
        """
        import groundkit.retrieval.rerank as rerank_module

        seen: dict[str, object] = {}

        class _LegacyCrossEncoder:
            def __init__(self, name: str, **kwargs: object) -> None:
                if "activation_fn" in kwargs:
                    raise TypeError("unexpected keyword argument 'activation_fn'")
                seen.update(kwargs)

            def predict(self, pairs: list[list[str]]) -> list[float]:
                return [-2.0] * len(pairs)

        monkeypatch.setattr(
            rerank_module,
            "_import_cross_encoder",
            lambda: (_LegacyCrossEncoder, "IDENTITY"),
        )

        async def run() -> None:
            reranker = CrossEncoderReranker()
            reranked = await reranker.rerank("q", [_result("c1")], top_k=1)
            assert reranked[0].score >= 0.0

        asyncio.run(run())

        assert seen["default_activation_function"] == "IDENTITY"
        assert "activation_fn" not in seen

    def test_failure_under_both_activation_spellings_is_typed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import groundkit.retrieval.rerank as rerank_module

        class _AlwaysRejectingCrossEncoder:
            def __init__(self, name: str, **kwargs: object) -> None:
                raise TypeError("neither spelling accepted")

        monkeypatch.setattr(
            rerank_module,
            "_import_cross_encoder",
            lambda: (_AlwaysRejectingCrossEncoder, "IDENTITY"),
        )

        async def run() -> None:
            reranker = CrossEncoderReranker()
            with pytest.raises(RerankerNotConfiguredError, match="either"):
                await reranker.rerank("q", [_result("c1")], top_k=1)

        asyncio.run(run())

    def test_scores_from_a_stub_model_are_normalized_and_ordered(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end through ``rerank`` with a stub emitting negative logits."""
        import groundkit.retrieval.rerank as rerank_module

        class _NegativeLogitCrossEncoder:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def predict(self, pairs: list[list[str]]) -> list[float]:
                assert all(pair[0] == "why is the sky blue" for pair in pairs)
                return [-9.0, -1.5]

        monkeypatch.setattr(
            rerank_module,
            "_import_cross_encoder",
            lambda: (_NegativeLogitCrossEncoder, object()),
        )

        async def run() -> None:
            reranker = CrossEncoderReranker()
            candidates = [_result("c1", start=0), _result("c2", start=10)]
            reranked = await reranker.rerank("why is the sky blue", candidates, top_k=2)
            assert [r.chunk_id for r in reranked] == ["c2", "c1"]
            assert all(r.score >= 0.0 for r in reranked)

        asyncio.run(run())

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

import ast
import asyncio
import math
import threading
import time
from pathlib import Path

import pytest

from groundkit.contracts import RetrievalResult, SourceClass
from groundkit.errors import RerankerNotConfiguredError, RetrievalError
from groundkit.retrieval.rerank import (
    DEFAULT_RERANK_MODEL,
    SIGMOID_SATURATION_LOGIT,
    CrossEncoderReranker,
    rerank_by_logits,
    sigmoid,
)


def _result(
    chunk_id: str,
    *,
    source: str = "doc.md",
    start: int = 0,
    source_class: SourceClass = "text",
    extractor: str | None = None,
) -> RetrievalResult:
    """A minimal valid result; only the fields rerank actually reads vary."""
    return RetrievalResult(
        content=f"content of {chunk_id}",
        score=0.5,
        document_id="doc-1",
        chunk_id=chunk_id,
        source=source,
        source_class=source_class,
        extractor=extractor,
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

    def test_preserves_source_class_and_extractor(self) -> None:
        """GK-003: a reranked result must keep the document's real provenance.

        ``rerank_by_logits`` rebuilds every result through the
        :class:`RetrievalResult` constructor (see the module docstring, and
        ``test_contract_is_genuinely_revalidated`` above). Omitting
        ``source_class``/``extractor`` from that rebuild does not raise —
        both fields default — so a reranked ``snapshot`` or ``extracted``
        result silently reverts to ``("text", None)``. Against a collection
        holding URL-ingested documents that routes every reranked citation
        into the plain read-and-slice resolver path ADR-0016 exists to keep
        snapshots out of, without ever raising an error.
        """
        snapshot_result = _result("snap-1", source_class="snapshot", extractor=None, start=0)
        extracted_result = _result(
            "ext-1", source_class="extracted", extractor="pdfminer-six==20240706", start=10
        )

        reranked = rerank_by_logits([snapshot_result, extracted_result], [1.0, 2.0], top_k=2)
        by_id = {r.chunk_id: r for r in reranked}

        assert by_id["snap-1"].source_class == "snapshot"
        assert by_id["ext-1"].source_class == "extracted"
        assert by_id["ext-1"].extractor == "pdfminer-six==20240706"

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

    def test_cancelled_cold_call_does_not_orphan_a_second_load(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cancelled load must not free the lock while its worker still runs.

        A worker thread cannot be cancelled, but the ``await`` in front of it
        can. Holding the lock across the awaited ``to_thread`` therefore only
        looks safe: cancellation unwinds ``async with``, frees the lock, and
        leaves the worker building — so the next caller still sees ``_model is
        None``, takes the free lock, and starts a second multi-gigabyte load.
        Under a Phase 4 request timeout that is an ordinary event, not an
        exotic one.
        """
        import groundkit.retrieval.rerank as rerank_module

        loads = {"count": 0}

        class _SlowCrossEncoder:
            def __init__(self, *args: object, **kwargs: object) -> None:
                time.sleep(0.2)
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

            cold = asyncio.create_task(reranker.rerank("q", candidates, top_k=1))
            await asyncio.sleep(0.05)  # let it reach the load, not past it
            cold.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cold

            # The abandoned worker is still inside its 0.2s __init__ here.
            reranked = await reranker.rerank("q", candidates, top_k=1)
            assert len(reranked) == 1

        asyncio.run(run())

        assert loads["count"] == 1

    def test_a_failed_load_can_be_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A shared task that failed is replaced, not reused forever.

        Reusing it would make one transient fault mid-download poison the
        instance permanently, with every later call re-raising a stale
        exception from a task that will never be retried.
        """
        import groundkit.retrieval.rerank as rerank_module

        attempts = {"count": 0}

        class _FailsOnceCrossEncoder:
            def __init__(self, *args: object, **kwargs: object) -> None:
                attempts["count"] += 1
                if attempts["count"] == 1:
                    raise RuntimeError("transient network fault")

            def predict(self, pairs: list[list[str]]) -> list[float]:
                return [0.0] * len(pairs)

        monkeypatch.setattr(
            rerank_module,
            "_import_cross_encoder",
            lambda: (_FailsOnceCrossEncoder, "IDENTITY"),
        )

        async def run() -> None:
            reranker = CrossEncoderReranker()
            candidates = [_result("c1")]

            with pytest.raises(RerankerNotConfiguredError):
                await reranker.rerank("q", candidates, top_k=1)

            reranked = await reranker.rerank("q", candidates, top_k=1)
            assert len(reranked) == 1

        asyncio.run(run())

        assert attempts["count"] == 2

    def test_inference_failure_is_translated_to_a_retrieval_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A backend exception during scoring must not cross the seam raw.

        Inference fails in ways this repo does not model — CUDA OOM, a
        tokenizer rejecting an input, a device that went away. Every other
        failure in this module already arrives typed, so leaving this one
        untranslated would leave a caller handling ``RetrievalError`` (in
        Phase 4, the request boundary) unprotected on the path most likely to
        fail under load.
        """
        import groundkit.retrieval.rerank as rerank_module

        class _ExplodingAtInferenceCrossEncoder:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def predict(self, pairs: list[list[str]]) -> list[float]:
                raise RuntimeError("CUDA out of memory")

        monkeypatch.setattr(
            rerank_module,
            "_import_cross_encoder",
            lambda: (_ExplodingAtInferenceCrossEncoder, "IDENTITY"),
        )

        async def run() -> None:
            reranker = CrossEncoderReranker()
            with pytest.raises(RetrievalError) as excinfo:
                await reranker.rerank("q", [_result("c1")], top_k=1)

            # Chained, so the backend's own message survives in the traceback.
            assert isinstance(excinfo.value.__cause__, RuntimeError)
            assert "CUDA out of memory" in str(excinfo.value.__cause__)

        asyncio.run(run())

    def test_inference_failure_message_does_not_leak_query_or_content(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The raised message must not carry corpus text or the query.

        Unlike a load failure, an inference failure routinely quotes the input
        it choked on — here the query and the passages. The backend's message
        is therefore chained rather than interpolated, so it reaches a
        traceback without the raised error itself becoming a disclosure.
        """
        import groundkit.retrieval.rerank as rerank_module

        private_query = "what is the patient's diagnosis"
        private_content = "content of c1"

        class _LeakyCrossEncoder:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def predict(self, pairs: list[list[str]]) -> list[float]:
                raise ValueError(f"tokenizer failed on {pairs!r}")

        monkeypatch.setattr(
            rerank_module,
            "_import_cross_encoder",
            lambda: (_LeakyCrossEncoder, "IDENTITY"),
        )

        async def run() -> None:
            reranker = CrossEncoderReranker()
            with pytest.raises(RetrievalError) as excinfo:
                await reranker.rerank(private_query, [_result("c1")], top_k=1)

            message = str(excinfo.value)
            assert private_query not in message
            assert private_content not in message
            assert "ValueError" in message  # the type is named; the text is not

        asyncio.run(run())

    def test_backend_import_failure_that_is_not_an_import_error_is_typed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ "Installed" and "usable" are different states, and only one is ImportError.

        torch raises ``OSError`` when a native library is missing — the
        WinError 126 / missing-CUDA-``.so`` family — which is an environment
        fault, not a missing package. Catching only ``ImportError`` let those
        escape untyped from a function documented to raise
        ``RerankerNotConfiguredError``.
        """
        import builtins

        from groundkit.retrieval.rerank import _import_cross_encoder

        real_import = builtins.__import__

        def _import_raising_oserror(name: str, *args: object, **kwargs: object) -> object:
            if name == "torch":
                raise OSError("[WinError 126] The specified module could not be found")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", _import_raising_oserror)

        with pytest.raises(RerankerNotConfiguredError, match="failed to initialize"):
            _import_cross_encoder()

    def test_non_scalar_prediction_output_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A multi-label model emits a row per label, not a relevance score.

        ``model_name`` is caller-supplied, so nothing stops one being pointed
        at a multi-label cross-encoder. ``float()`` on a row raises
        ``TypeError``, which escaped the prediction handler because the
        conversion sat outside it.
        """
        import groundkit.retrieval.rerank as rerank_module

        class _MultiLabelCrossEncoder:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def predict(self, pairs: list[list[str]]) -> list[list[float]]:
                return [[0.1, 0.9] for _ in pairs]

        monkeypatch.setattr(
            rerank_module,
            "_import_cross_encoder",
            lambda: (_MultiLabelCrossEncoder, "IDENTITY"),
        )

        async def run() -> None:
            reranker = CrossEncoderReranker("some/multi-label-model")
            with pytest.raises(RetrievalError, match="non-scalar score at position 0"):
                await reranker.rerank("q", [_result("c1")], top_k=1)

        asyncio.run(run())

    def test_non_iterable_prediction_output_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import groundkit.retrieval.rerank as rerank_module

        class _ScalarReturningCrossEncoder:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def predict(self, pairs: list[list[str]]) -> float:
                return 0.5

        monkeypatch.setattr(
            rerank_module,
            "_import_cross_encoder",
            lambda: (_ScalarReturningCrossEncoder, "IDENTITY"),
        )

        async def run() -> None:
            reranker = CrossEncoderReranker()
            with pytest.raises(RetrievalError, match="non-iterable prediction"):
                await reranker.rerank("q", [_result("c1")], top_k=1)

        asyncio.run(run())

    def test_overflow_error_from_float_conversion_is_translated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``float(10**400)`` raises ``OverflowError``, which the old
        ``except (TypeError, ValueError)`` guard on the ``float(score)``
        boundary did not catch. An int too large to represent as a float is an
        ordinary outcome from a backend this repo does not control — nothing
        stops a multi-label or misconfigured model from emitting one — so it
        must translate to ``RetrievalError`` like every other coercion
        failure rather than escape as a raw ``OverflowError``.
        """
        import groundkit.retrieval.rerank as rerank_module

        class _HugeIntCrossEncoder:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def predict(self, pairs: list[list[str]]) -> list[int]:
                return [10**400 for _ in pairs]

        monkeypatch.setattr(
            rerank_module,
            "_import_cross_encoder",
            lambda: (_HugeIntCrossEncoder, "IDENTITY"),
        )

        async def run() -> None:
            reranker = CrossEncoderReranker()
            with pytest.raises(RetrievalError, match="non-scalar score at position 0"):
                await reranker.rerank("q", [_result("c1")], top_k=1)

        asyncio.run(run())

    def test_iteration_failure_other_than_type_error_is_translated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ``list(raw)`` boundary was guarded for ``TypeError`` only.

        A prediction object whose ``__iter__`` raises something else entirely
        — here a plain ``RuntimeError``, standing in for whatever a
        third-party backend this repo does not control might do — escaped
        untyped. Any single named exception is a guess at another library's
        behavior, which is why the fix catches ``Exception`` generally rather
        than enumerating a tuple.
        """
        import groundkit.retrieval.rerank as rerank_module

        class _ExplodingIterable:
            def __iter__(self) -> None:
                raise RuntimeError("iteration exploded")

        class _ExplodingIterableCrossEncoder:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def predict(self, pairs: list[list[str]]) -> _ExplodingIterable:
                return _ExplodingIterable()

        monkeypatch.setattr(
            rerank_module,
            "_import_cross_encoder",
            lambda: (_ExplodingIterableCrossEncoder, "IDENTITY"),
        )

        async def run() -> None:
            reranker = CrossEncoderReranker()
            with pytest.raises(RetrievalError, match="non-iterable prediction"):
                await reranker.rerank("q", [_result("c1")], top_k=1)

        asyncio.run(run())

    def test_numpy_style_scalars_are_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The conversion must coerce, never ``isinstance``-check.

        ``numpy.float32`` is not a subclass of Python's ``float``, so a type
        test would reject exactly what a real cross-encoder returns while every
        stub here — returning Python floats — kept passing. This stands in for
        that value: a scalar that converts but fails an ``isinstance(x, float)``
        test, so the check can never quietly be tightened into one.
        """
        import groundkit.retrieval.rerank as rerank_module

        class _Float32Like:
            def __init__(self, value: float) -> None:
                self._value = value

            def __float__(self) -> float:
                return self._value

        class _NumpyLikeCrossEncoder:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def predict(self, pairs: list[list[str]]) -> list[_Float32Like]:
                return [_Float32Like(-2.0), _Float32Like(3.0)][: len(pairs)]

        monkeypatch.setattr(
            rerank_module,
            "_import_cross_encoder",
            lambda: (_NumpyLikeCrossEncoder, "IDENTITY"),
        )

        async def run() -> None:
            reranker = CrossEncoderReranker()
            candidates = [_result("c1", start=0), _result("c2", start=10)]
            reranked = await reranker.rerank("q", candidates, top_k=2)
            assert [r.chunk_id for r in reranked] == ["c2", "c1"]
            assert all(r.score >= 0.0 for r in reranked)

        asyncio.run(run())

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


#: Source root the fourth-rebuild-site scan walks. Derived, not hardcoded, so a
#: package move does not silently make the scan walk nothing and pass vacuously.
_SRC_DIR = Path(__file__).resolve().parents[1] / "src" / "groundkit"

#: ``evals/`` is excluded, deliberately. Its two ``RetrievalResult(...)`` calls
#: (``evals/echo.py``) build synthetic fixtures for the eval harness's own
#: golden-corpus stages, not results that ever crossed a real ``source_class``
#: boundary — there is no ingested document, extractor, or snapshot upstream of
#: them to drop in the first place. Provenance in that module is invented for
#: the test, not carried through from a store record, so the failure mode this
#: guard exists to catch (GK-003: a real field silently reverting to a
#: contract default) cannot occur there. Folding it into the guard would only
#: force every future synthetic fixture to carry two dead keyword arguments.
_EXCLUDED_DIRS = {"evals"}


class TestNoFourthRebuildSite:
    """GK-003: guard against a *fifth* silent ``RetrievalResult`` rebuild site.

    ``rerank.py:170`` omitted ``source_class``/``extractor`` for over a phase
    without a type error, a test failure, or a runtime exception — both fields
    default, so the omission was invisible until ADR-0016 grew the contract
    underneath already-shipped code that rebuilds it. Two call sites already
    pass both fields explicitly (``retrieval/search.py``,
    ``providers/context_assembly.py``); this asserts every current and future
    provenance-bearing site does too, walking source rather than trusting a
    hand-maintained list that the next rebuild site would not update itself.

    Guard, demonstrated by injection (matching
    ``test_service_package_imports_no_write_path``'s idiom in
    ``tests/test_service_tools.py``): add a bare
    ``RetrievalResult(content=..., score=..., document_id=..., chunk_id=...,
    source=..., start_offset=..., end_offset=...)`` anywhere under
    ``src/groundkit/`` outside ``evals/`` and this fails.
    """

    def test_every_retrieval_result_construction_passes_source_class_and_extractor(
        self,
    ) -> None:
        offenders: list[str] = []
        py_files = [
            path
            for path in _SRC_DIR.rglob("*.py")
            if not (set(path.relative_to(_SRC_DIR).parts[:-1]) & _EXCLUDED_DIRS)
        ]
        assert py_files, f"scan found no source files under {_SRC_DIR}; path is wrong"

        for path in py_files:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and _is_retrieval_result_call(node)):
                    continue
                passed_kwargs = {kw.arg for kw in node.keywords if kw.arg is not None}
                missing = {"source_class", "extractor"} - passed_kwargs
                if missing:
                    offenders.append(
                        f"{path.relative_to(_SRC_DIR)}:{node.lineno} missing {sorted(missing)}"
                    )

        assert not offenders, (
            "RetrievalResult constructed without explicit source_class/extractor "
            f"(GK-003 fourth-rebuild-site guard): {offenders}"
        )


def _is_retrieval_result_call(node: ast.Call) -> bool:
    """True for ``RetrievalResult(...)`` and ``contracts.RetrievalResult(...)`` alike."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "RetrievalResult"
    if isinstance(func, ast.Attribute):
        return func.attr == "RetrievalResult"
    return False

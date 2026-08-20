"""Answer composition tests (Phase 5, ``AnswerPipeline`` / ``AnswerReport``).

Every collaborator ``AnswerPipeline`` composes is faked or scripted:
``_AnswerFakeRetriever`` stands in for the retrieval collaborator (a
``SearchCallable``) so no real store or BM25 index is ever built, and
``_AnswerScriptedChat`` stands in for the ``ChatProtocol`` underlying real
``Synthesizer`` / ``QueryRewriter`` / ``FaithfulnessJudge`` instances, so
those three collaborators run their real (cheap, offline) logic against
scripted completions rather than a fake of their own. pytest-asyncio is not
a dependency of this repo; async code under test is driven with
``asyncio.run()`` inside plain ``def`` test functions, matching
``tests/test_synthesis.py`` and ``tests/test_query_rewrite.py``.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from groundkit.answer import AnswerPipeline, AnswerReport
from groundkit.contracts import RetrievalResult, SearchResponse
from groundkit.errors import (
    ChatError,
    JudgeError,
    QueryRewriteError,
    RetrievalError,
    SynthesisError,
)
from groundkit.providers.judge import FaithfulnessJudge
from groundkit.providers.protocols import ChatProtocol
from groundkit.providers.query_rewrite import QueryRewriter
from groundkit.providers.synthesis import Synthesizer
from groundkit.retrieval.search import SearchMode

_FAITHFUL_VERDICT_JSON = (
    '{"faithful": true, "unsupported_claims": [], "reasoning": "fully supported"}'
)


class _AnswerScriptedChat:
    """A ``ChatProtocol`` test double returning a fixed completion, or raising a
    scripted error, with no network or real model involved.

    Records every ``(prompt, system)`` pair it receives so tests can assert
    on what a real ``Synthesizer`` / ``QueryRewriter`` / ``FaithfulnessJudge``
    actually sent, without inspecting that collaborator's internals.
    """

    def __init__(
        self,
        completion: str = "",
        *,
        error: Exception | None = None,
        provider: str = "answer-scripted",
        model_name: str = "answer-scripted-v1",
    ) -> None:
        self._completion = completion
        self._error = error
        self._provider = provider
        self._model_name = model_name
        self.prompts: list[str] = []
        self.systems: list[str | None] = []

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model_name(self) -> str:
        return self._model_name

    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        self.prompts.append(prompt)
        self.systems.append(system)
        if self._error is not None:
            raise self._error
        return self._completion


class _AnswerFakeRetriever:
    """Fake search collaborator, structurally a ``SearchCallable`` via its bound
    ``search`` method — no real ``MetadataStoreProtocol``, ``BM25Index``, or
    vector store is ever constructed.

    Records every ``(query, top_k, mode)`` call it receives so tests can
    assert what the pipeline actually searched with — in particular, that a
    rewriter's output (not the original query) reaches this fake when a
    rewriter is configured.
    """

    def __init__(
        self, response: SearchResponse | None = None, *, error: Exception | None = None
    ) -> None:
        self._response = response
        self._error = error
        self.calls: list[tuple[str, int | None, str]] = []

    async def search(
        self, query: str, top_k: int | None = None, *, mode: SearchMode = "bm25"
    ) -> SearchResponse:
        self.calls.append((query, top_k, mode))
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


async def _direct_search_fn(
    query: str, top_k: int | None = None, *, mode: SearchMode = "bm25"
) -> SearchResponse:
    """A plain ``async def`` function satisfying ``SearchCallable`` directly —
    the "callable" half of "a Retriever-like search callable or instance",
    as opposed to ``_AnswerFakeRetriever.search``'s "instance" half. Echoes
    its parameters into the response metadata rather than discarding them,
    which is both realistic (a real search function would use them) and
    keeps every parameter genuinely read."""
    return SearchResponse(
        query=query,
        results=[_result()],
        total_results=1,
        metadata={"top_k": top_k, "mode": mode},
    )


def _result(
    *,
    content: str = "the sky is blue",
    score: float = 1.0,
    document_id: str = "doc-1",
    chunk_id: str = "chunk-1",
    source: str = "doc.txt",
    start_offset: int = 0,
) -> RetrievalResult:
    """Build a valid RetrievalResult fixture with consistent offset arithmetic."""
    return RetrievalResult(
        content=content,
        score=score,
        document_id=document_id,
        chunk_id=chunk_id,
        source=source,
        start_offset=start_offset,
        end_offset=start_offset + len(content),
    )


def _search_response(results: list[RetrievalResult], *, query: str = "a query") -> SearchResponse:
    return SearchResponse(query=query, results=results, total_results=len(results), metadata={})


# ── Empty query: fail closed before any collaborator runs ──────────────────


class TestEmptyQueryPreCondition:
    def test_empty_string_rejected_before_any_collaborator_runs(self) -> None:
        retriever = _AnswerFakeRetriever(_search_response([_result()]))
        synth_chat = _AnswerScriptedChat("[1] an answer")
        rewrite_chat = _AnswerScriptedChat("rewritten")
        judge_chat = _AnswerScriptedChat(_FAITHFUL_VERDICT_JSON)
        pipeline = AnswerPipeline(
            retriever.search,
            Synthesizer(synth_chat),
            rewriter=QueryRewriter(rewrite_chat),
            judge=FaithfulnessJudge(judge_chat),
        )

        with pytest.raises(RetrievalError, match="empty"):
            asyncio.run(pipeline.answer(""))

        assert retriever.calls == []
        assert synth_chat.prompts == []
        assert rewrite_chat.prompts == []
        assert judge_chat.prompts == []

    def test_whitespace_only_rejected_before_any_collaborator_runs(self) -> None:
        retriever = _AnswerFakeRetriever(_search_response([_result()]))
        synth_chat = _AnswerScriptedChat("[1] an answer")
        pipeline = AnswerPipeline(retriever.search, Synthesizer(synth_chat))

        with pytest.raises(RetrievalError, match="empty"):
            asyncio.run(pipeline.answer("   \t\n  "))

        assert retriever.calls == []
        assert synth_chat.prompts == []


# ── Rewrite path: two strings, both recorded ────────────────────────────────


class TestRewritePath:
    def test_rewriter_present_searches_with_rewritten_query_and_records_both(self) -> None:
        retriever = _AnswerFakeRetriever(_search_response([_result()]))
        rewrite_chat = _AnswerScriptedChat("rewritten form of the query")
        synth_chat = _AnswerScriptedChat("[1] an answer")
        pipeline = AnswerPipeline(
            retriever.search, Synthesizer(synth_chat), rewriter=QueryRewriter(rewrite_chat)
        )

        report = asyncio.run(pipeline.answer("original user question"))

        assert report.query == "original user question"
        assert report.rewritten_query == "rewritten form of the query"
        assert retriever.calls == [("rewritten form of the query", None, "bm25")]

    def test_synthesis_uses_the_original_query_not_the_rewritten_one(self) -> None:
        """The rewrite is for retrieval matching, not for the question the
        answer must address (module docstring)."""
        retriever = _AnswerFakeRetriever(_search_response([_result()]))
        rewrite_chat = _AnswerScriptedChat("totally different retrieval phrasing")
        synth_chat = _AnswerScriptedChat("[1] an answer")
        pipeline = AnswerPipeline(
            retriever.search, Synthesizer(synth_chat), rewriter=QueryRewriter(rewrite_chat)
        )

        asyncio.run(pipeline.answer("what color is the sky"))

        assert "what color is the sky" in synth_chat.prompts[0]
        assert "totally different retrieval phrasing" not in synth_chat.prompts[0]

    def test_no_rewriter_records_none_and_searches_with_original_query(self) -> None:
        retriever = _AnswerFakeRetriever(_search_response([_result()]))
        synth_chat = _AnswerScriptedChat("[1] an answer")
        pipeline = AnswerPipeline(retriever.search, Synthesizer(synth_chat))

        report = asyncio.run(pipeline.answer("a plain query"))

        assert report.rewritten_query is None
        assert retriever.calls == [("a plain query", None, "bm25")]

    def test_rewrite_error_propagates_and_search_never_runs(self) -> None:
        retriever = _AnswerFakeRetriever(_search_response([_result()]))
        rewrite_chat = _AnswerScriptedChat(error=QueryRewriteError("blank rewrite"))
        synth_chat = _AnswerScriptedChat("[1] an answer")
        pipeline = AnswerPipeline(
            retriever.search, Synthesizer(synth_chat), rewriter=QueryRewriter(rewrite_chat)
        )

        with pytest.raises(QueryRewriteError):
            asyncio.run(pipeline.answer("a query"))

        assert retriever.calls == []
        assert synth_chat.prompts == []


# ── top_k / mode forwarding ──────────────────────────────────────────────────


class TestSearchParamsForwarded:
    def test_top_k_and_mode_forwarded_unchanged(self) -> None:
        retriever = _AnswerFakeRetriever(_search_response([_result()]))
        synth_chat = _AnswerScriptedChat("[1] an answer")
        pipeline = AnswerPipeline(retriever.search, Synthesizer(synth_chat))

        asyncio.run(pipeline.answer("a query", top_k=7, mode="hybrid"))

        assert retriever.calls == [("a query", 7, "hybrid")]


# ── Zero retrieved results: single source of truth is Synthesizer's own check ─


class TestZeroResultsIsSynthesizersOwnRejection:
    def test_empty_results_raises_synthesis_error_through_the_pipeline(self) -> None:
        retriever = _AnswerFakeRetriever(_search_response([]))
        synth_chat = _AnswerScriptedChat("[1] an answer")
        judge_chat = _AnswerScriptedChat(_FAITHFUL_VERDICT_JSON)
        pipeline = AnswerPipeline(
            retriever.search, Synthesizer(synth_chat), judge=FaithfulnessJudge(judge_chat)
        )

        with pytest.raises(SynthesisError, match="at least one retrieved result"):
            asyncio.run(pipeline.answer("a query"))

        # Synthesizer's own precondition rejects before it ever calls its chat,
        # and the judge — downstream of a synthesized answer that never
        # existed — never runs either.
        assert synth_chat.prompts == []
        assert judge_chat.prompts == []


# ── Provider / synthesis error propagation ──────────────────────────────────


class TestErrorPropagation:
    def test_search_error_propagates_unmodified(self) -> None:
        boom = RetrievalError("index inconsistency")
        retriever = _AnswerFakeRetriever(error=boom)
        synth_chat = _AnswerScriptedChat("[1] an answer")
        pipeline = AnswerPipeline(retriever.search, Synthesizer(synth_chat))

        with pytest.raises(RetrievalError) as excinfo:
            asyncio.run(pipeline.answer("a query"))
        assert excinfo.value is boom
        assert synth_chat.prompts == []

    def test_synthesis_chat_error_propagates_unmodified(self) -> None:
        boom = ChatError("provider exploded")
        retriever = _AnswerFakeRetriever(_search_response([_result()]))
        synth_chat = _AnswerScriptedChat(error=boom)
        pipeline = AnswerPipeline(retriever.search, Synthesizer(synth_chat))

        with pytest.raises(ChatError) as excinfo:
            asyncio.run(pipeline.answer("a query"))
        assert excinfo.value is boom

    def test_out_of_range_citation_raises_synthesis_error(self) -> None:
        retriever = _AnswerFakeRetriever(_search_response([_result()]))
        synth_chat = _AnswerScriptedChat("[7] a claim citing nothing retrieved")
        pipeline = AnswerPipeline(retriever.search, Synthesizer(synth_chat))

        with pytest.raises(SynthesisError, match=r"\[7\]"):
            asyncio.run(pipeline.answer("a query"))

    def test_judge_error_propagates_and_report_is_never_built(self) -> None:
        retriever = _AnswerFakeRetriever(_search_response([_result()]))
        synth_chat = _AnswerScriptedChat("[1] an answer")
        judge_chat = _AnswerScriptedChat("not valid json at all")
        pipeline = AnswerPipeline(
            retriever.search, Synthesizer(synth_chat), judge=FaithfulnessJudge(judge_chat)
        )

        with pytest.raises(JudgeError):
            asyncio.run(pipeline.answer("a query"))


# ── Judge: advisory, absent-by-default, and unconditional when present ──────


class TestJudge:
    def test_judge_absent_leaves_verdict_none(self) -> None:
        retriever = _AnswerFakeRetriever(_search_response([_result()]))
        synth_chat = _AnswerScriptedChat("[1] an answer")
        pipeline = AnswerPipeline(retriever.search, Synthesizer(synth_chat))

        report = asyncio.run(pipeline.answer("a query"))

        assert report.verdict is None

    def test_judge_present_and_successful_populates_verdict(self) -> None:
        retriever = _AnswerFakeRetriever(_search_response([_result()]))
        synth_chat = _AnswerScriptedChat("[1] an answer")
        judge_chat = _AnswerScriptedChat(_FAITHFUL_VERDICT_JSON)
        pipeline = AnswerPipeline(
            retriever.search, Synthesizer(synth_chat), judge=FaithfulnessJudge(judge_chat)
        )

        report = asyncio.run(pipeline.answer("a query"))

        assert report.verdict is not None
        assert report.verdict.faithful is True

    def test_judge_never_narrows_sources_to_only_cited_results(self) -> None:
        """The judge is shown every retrieved result's content, not only the
        subset the answer cited (module docstring's documented decision)."""
        r1 = _result(chunk_id="c1", content="UNCITED FIRST FACT")
        r2 = _result(chunk_id="c2", content="CITED SECOND FACT")
        retriever = _AnswerFakeRetriever(_search_response([r1, r2]))
        # Cites only [2] (r2); r1 is retrieved but never cited.
        synth_chat = _AnswerScriptedChat("[2] a claim citing only the second source")
        judge_chat = _AnswerScriptedChat(_FAITHFUL_VERDICT_JSON)
        pipeline = AnswerPipeline(
            retriever.search, Synthesizer(synth_chat), judge=FaithfulnessJudge(judge_chat)
        )

        report = asyncio.run(pipeline.answer("a query"))

        assert report.citations == (r2.citation,)
        judge_prompt = judge_chat.prompts[0]
        assert "UNCITED FIRST FACT" in judge_prompt
        assert "CITED SECOND FACT" in judge_prompt

    def test_judge_runs_even_when_the_answer_is_an_abstention(self) -> None:
        """Judging is unconditional on the presence of a judge, never
        special-cased on an empty-citations abstention."""
        retriever = _AnswerFakeRetriever(_search_response([_result()]))
        synth_chat = _AnswerScriptedChat("The sources do not answer this question.")
        judge_chat = _AnswerScriptedChat(_FAITHFUL_VERDICT_JSON)
        pipeline = AnswerPipeline(
            retriever.search, Synthesizer(synth_chat), judge=FaithfulnessJudge(judge_chat)
        )

        report = asyncio.run(pipeline.answer("a query"))

        assert report.citations == ()
        assert judge_chat.prompts != []
        assert report.verdict is not None


# ── Abstention flows through untouched ──────────────────────────────────────


class TestAbstention:
    def test_zero_marker_abstention_yields_empty_citations_no_error(self) -> None:
        retriever = _AnswerFakeRetriever(_search_response([_result()]))
        synth_chat = _AnswerScriptedChat("The sources do not answer this question.")
        pipeline = AnswerPipeline(retriever.search, Synthesizer(synth_chat))

        report = asyncio.run(pipeline.answer("a query"))

        assert report.citations == ()
        assert report.answer == "The sources do not answer this question."


# ── Result ordering preserved ────────────────────────────────────────────────


class TestResultOrderingPreserved:
    def test_results_land_in_report_in_search_order(self) -> None:
        r1 = _result(chunk_id="c1", content="first fact")
        r2 = _result(chunk_id="c2", content="second fact")
        r3 = _result(chunk_id="c3", content="third fact")
        retriever = _AnswerFakeRetriever(_search_response([r3, r1, r2]))
        synth_chat = _AnswerScriptedChat("[1] a claim")
        pipeline = AnswerPipeline(retriever.search, Synthesizer(synth_chat))

        report = asyncio.run(pipeline.answer("a query"))

        assert report.results == (r3, r1, r2)


# ── SearchCallable accepts either an instance's bound method or a plain fn ──


class TestSearchCallableAcceptsInstanceOrPlainFunction:
    def test_bound_method_of_a_fake_instance_works(self) -> None:
        retriever = _AnswerFakeRetriever(_search_response([_result()]))
        synth_chat = _AnswerScriptedChat("[1] an answer")
        pipeline = AnswerPipeline(retriever.search, Synthesizer(synth_chat))

        report = asyncio.run(pipeline.answer("a query"))

        assert report.answer == "[1] an answer"

    def test_plain_async_function_works(self) -> None:
        synth_chat = _AnswerScriptedChat("[1] an answer")
        pipeline = AnswerPipeline(_direct_search_fn, Synthesizer(synth_chat))

        report = asyncio.run(pipeline.answer("a query"))

        assert report.answer == "[1] an answer"


# ── AnswerReport contract ────────────────────────────────────────────────────


class TestAnswerReportContract:
    def _report(self) -> AnswerReport:
        return AnswerReport(
            query="q",
            rewritten_query=None,
            answer="a",
            citations=(),
            results=(),
            verdict=None,
        )

    def test_frozen(self) -> None:
        report = self._report()
        with pytest.raises(ValidationError):
            report.answer = "changed"

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AnswerReport(
                query="q",
                rewritten_query=None,
                answer="a",
                citations=(),
                results=(),
                verdict=None,
                surprise=True,  # type: ignore[call-arg]
            )


# ── ChatProtocol conformance of the test double itself ───────────────────────


class TestChatProtocolConformance:
    def test_scripted_chat_satisfies_chat_protocol(self) -> None:
        assert isinstance(_AnswerScriptedChat(), ChatProtocol)

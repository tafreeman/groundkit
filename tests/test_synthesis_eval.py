"""Golden-corpus synthesis + advisory-judge pass tests (``evals/synthesis_eval.py``).

This is the producer ``KNOWN_LIMITATIONS.md`` named as missing: before this
module existed, ``EvalReport.synthesis`` (``SynthesisReport``) was
"structure without a producer". These tests exercise
``run_synthesis_eval`` directly, bypassing ``run_eval``/the CLI entirely —
``tests/test_runner.py`` covers the ``run_eval`` wiring (stage selection,
``ConfigurationError`` when ``judge`` is given without ``chat``) and
``tests/test_cli.py`` covers ``--judge``'s argument validation. Splitting it
this way keeps each layer's tests from needing to drive the two below it.

``_SynthesisEvalScriptedChat`` is a module-local scripted ``ChatProtocol``
double, distinct from every other test module's own fake and from
``groundkit.providers.llm.ScriptedChatProvider`` (a production-facing type
this test suite never touches) — matching ``tests/test_echo.py`` and
``tests/test_synthesis.py``'s stated convention. Its script is a plain FIFO
queue: for a run with a judge, both the internal ``Synthesizer`` and the
injected ``FaithfulnessJudge`` are driven off the SAME chat instance,
matching exactly how ``groundkit.cli._cmd_eval`` constructs them (one
``chat``, reused). Async code under test is driven with ``asyncio.run()``
inside plain ``def`` test functions — pytest-asyncio is not a dependency of
this repo.
"""

from __future__ import annotations

import asyncio
import hashlib
import json

import pytest

from groundkit.contracts import RetrievalResult
from groundkit.errors import ChatError
from groundkit.evals.judge import DEFAULT_JUDGE_PROMPT, FaithfulnessJudge
from groundkit.evals.synthesis_eval import hash_prompt_template, run_synthesis_eval
from groundkit.providers.llm import RedactingChat
from groundkit.providers.protocols import ChatProtocol
from groundkit.providers.redaction import RedactionConfig
from groundkit.providers.synthesis import DEFAULT_SYNTHESIS_PROMPT


class _SynthesisEvalScriptedChat:
    """FIFO-scripted ``ChatProtocol`` double: each call returns the next
    entry in ``script``, regardless of ``prompt``/``system``.

    Fails closed once exhausted (raises :class:`~groundkit.errors.ChatError`)
    rather than cycling, mirroring
    :class:`~groundkit.providers.llm.ScriptedChatProvider`'s own rule.
    """

    def __init__(
        self,
        script: list[str],
        *,
        provider: str = "synthesis-eval-scripted",
        model_name: str = "synthesis-eval-scripted-v1",
    ) -> None:
        self._script = list(script)
        self._index = 0
        self._provider = provider
        self._model_name = model_name
        self.calls: list[str] = []

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model_name(self) -> str:
        return self._model_name

    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        self.calls.append(prompt)
        if self._index >= len(self._script):
            raise ChatError(
                f"_SynthesisEvalScriptedChat's script exhausted after {len(self._script)} "
                "completion(s)"
            )
        text = self._script[self._index]
        self._index += 1
        return text


def _result(
    *,
    content: str = "quokkas are marsupials native to Western Australia",
    document_id: str = "doc-1",
    chunk_id: str = "chunk-1",
    source: str = "doc.txt",
) -> RetrievalResult:
    """A structurally valid RetrievalResult fixture."""
    return RetrievalResult(
        content=content,
        score=1.0,
        document_id=document_id,
        chunk_id=chunk_id,
        source=source,
        start_offset=0,
        end_offset=len(content),
    )


def _verdict_json(*, faithful: bool, unsupported_claims: list[str] | None = None) -> str:
    return json.dumps(
        {
            "faithful": faithful,
            "unsupported_claims": [] if unsupported_claims is None else unsupported_claims,
            "reasoning": "scripted verdict",
        }
    )


class TestOutcomeSplitWithoutJudge:
    """answered / abstained / rejected, no judge configured."""

    def test_three_way_split_counted_correctly(self) -> None:
        chat = _SynthesisEvalScriptedChat(
            [
                "the answer is here [1]",  # answered
                "I cannot answer this from the given sources",  # abstained
                "see [99] for details",  # rejected: marker out of range
            ]
        )
        query_results = [
            ("query-answered", [_result()]),
            ("query-abstained", [_result()]),
            ("query-rejected", [_result()]),
        ]

        report = asyncio.run(run_synthesis_eval(query_results, chat=chat, input_stage="bm25"))

        assert report.answered_count == 1
        assert report.abstained_count == 1
        assert report.rejected_count == 1

    def test_judge_fields_all_none_when_no_judge_given(self) -> None:
        chat = _SynthesisEvalScriptedChat(["the answer is here [1]"])
        report = asyncio.run(
            run_synthesis_eval([("q", [_result()])], chat=chat, input_stage="bm25")
        )

        assert report.judge_provider is None
        assert report.judge_model is None
        assert report.judge_prompt_hash is None
        assert report.judged_count is None
        assert report.faithful_count is None
        assert report.unfaithful_count is None
        assert report.judge_error_count is None

    def test_input_stage_and_identities_recorded_verbatim(self) -> None:
        chat = _SynthesisEvalScriptedChat(
            ["the answer is here [1]"], provider="my-provider", model_name="my-model"
        )
        report = asyncio.run(
            run_synthesis_eval([("q", [_result()])], chat=chat, input_stage="fusion")
        )

        assert report.input_stage == "fusion"
        assert report.synthesis_provider == "my-provider"
        assert report.synthesis_model == "my-model"

    def test_redacted_false_for_a_bare_chat(self) -> None:
        chat = _SynthesisEvalScriptedChat(["the answer is here [1]"])
        report = asyncio.run(
            run_synthesis_eval([("q", [_result()])], chat=chat, input_stage="bm25")
        )
        assert report.redacted is False

    def test_redacted_true_when_chat_is_wrapped_in_redacting_chat(self) -> None:
        inner = _SynthesisEvalScriptedChat(["the answer is here [1]"])
        chat: ChatProtocol = RedactingChat(inner, RedactionConfig())
        report = asyncio.run(
            run_synthesis_eval([("q", [_result()])], chat=chat, input_stage="bm25")
        )
        assert report.redacted is True


class TestPromptHashes:
    def test_synthesis_prompt_hash_matches_the_template_actually_used(self) -> None:
        chat = _SynthesisEvalScriptedChat(["the answer is here [1]"])
        template = DEFAULT_SYNTHESIS_PROMPT
        report = asyncio.run(
            run_synthesis_eval(
                [("q", [_result()])],
                chat=chat,
                input_stage="bm25",
                synthesis_prompt_template=template,
            )
        )
        assert report.synthesis_prompt_hash == hashlib.sha256(template.encode("utf-8")).hexdigest()

    def test_judge_prompt_hash_matches_the_template_the_judge_was_built_with(self) -> None:
        chat = _SynthesisEvalScriptedChat(["the answer is here [1]", _verdict_json(faithful=True)])
        judge = FaithfulnessJudge(chat)
        template = DEFAULT_JUDGE_PROMPT
        report = asyncio.run(
            run_synthesis_eval(
                [("q", [_result()])],
                chat=chat,
                input_stage="bm25",
                judge=judge,
                judge_prompt_template=template,
            )
        )
        assert report.judge_prompt_hash == hashlib.sha256(template.encode("utf-8")).hexdigest()

    def test_hash_prompt_template_is_plain_sha256_hex(self) -> None:
        assert hash_prompt_template("hello") == hashlib.sha256(b"hello").hexdigest()


class TestJudgeTallies:
    """Judge verdicts and errors, driven through one shared scripted chat."""

    def test_judge_runs_over_answered_and_abstained_but_not_rejected(self) -> None:
        # Call order: query A's synth, then A's judge; query B's synth, then
        # B's judge; query C's synth only — C is rejected, so its judge call
        # never happens and the script must not provision one.
        chat = _SynthesisEvalScriptedChat(
            [
                "the answer is here [1]",  # A: answered
                _verdict_json(faithful=True),  # A's verdict
                "I cannot answer this from the given sources",  # B: abstained
                _verdict_json(faithful=False, unsupported_claims=["a claim"]),  # B's verdict
                "see [99] for details",  # C: rejected, never judged
            ]
        )
        judge = FaithfulnessJudge(chat)
        query_results = [
            ("query-a", [_result()]),
            ("query-b", [_result()]),
            ("query-c", [_result()]),
        ]

        report = asyncio.run(
            run_synthesis_eval(query_results, chat=chat, input_stage="bm25", judge=judge)
        )

        assert report.answered_count == 1
        assert report.abstained_count == 1
        assert report.rejected_count == 1
        assert report.judged_count == 2
        assert report.faithful_count == 1
        assert report.unfaithful_count == 1
        assert report.judge_error_count == 0
        # Exactly 5 calls made — proves C's judge call never fired.
        assert len(chat.calls) == 5

    def test_all_unfaithful_tallied_correctly(self) -> None:
        """Guards the direction the mixed-verdict cases cannot distinguish.

        A tally that counted judge *calls* rather than reading each verdict
        would pass every mixed case and fail only here, where the correct
        answer is that nothing is faithful.
        """
        chat = _SynthesisEvalScriptedChat(
            [
                "answer A",
                _verdict_json(faithful=False),
                "answer B",
                _verdict_json(faithful=False),
            ]
        )
        judge = FaithfulnessJudge(chat)
        query_results = [
            ("q1", [_result()]),
            ("q2", [_result()]),
        ]
        report = asyncio.run(
            run_synthesis_eval(query_results, chat=chat, input_stage="bm25", judge=judge)
        )
        assert report.faithful_count == 0
        assert report.unfaithful_count == 2
        assert report.judged_count == 2

    def test_judge_provider_and_model_identity_recorded(self) -> None:
        chat = _SynthesisEvalScriptedChat(
            ["the answer is here [1]", _verdict_json(faithful=True)],
            provider="judge-provider",
            model_name="judge-model",
        )
        judge = FaithfulnessJudge(chat)
        report = asyncio.run(
            run_synthesis_eval([("q", [_result()])], chat=chat, input_stage="bm25", judge=judge)
        )
        assert report.judge_provider == "judge-provider"
        assert report.judge_model == "judge-model"

    def test_malformed_verdict_counted_as_judge_error_not_faithful_or_unfaithful(self) -> None:
        chat = _SynthesisEvalScriptedChat(["the answer is here [1]", "not valid json at all"])
        judge = FaithfulnessJudge(chat)
        report = asyncio.run(
            run_synthesis_eval([("q", [_result()])], chat=chat, input_stage="bm25", judge=judge)
        )
        assert report.judged_count == 1
        assert report.faithful_count == 0
        assert report.unfaithful_count == 0
        assert report.judge_error_count == 1

    def test_tallies_sum_to_judged_count(self) -> None:
        """The invariant SynthesisReport._validate_judge_fields itself enforces —
        exercised end to end here rather than only unit-tested on the schema.
        """
        chat = _SynthesisEvalScriptedChat(
            [
                "the answer is here [1]",
                _verdict_json(faithful=True),
                "the answer is here [1]",
                "not valid json",
            ]
        )
        judge = FaithfulnessJudge(chat)
        report = asyncio.run(
            run_synthesis_eval(
                [("q1", [_result()]), ("q2", [_result()])],
                chat=chat,
                input_stage="bm25",
                judge=judge,
            )
        )
        assert (
            report.faithful_count is not None
            and report.unfaithful_count is not None
            and report.judge_error_count is not None
        )
        total = report.faithful_count + report.unfaithful_count + report.judge_error_count
        assert total == report.judged_count


class TestChatErrorPropagation:
    """A genuine provider failure is never mistaken for a per-query outcome."""

    def test_chat_error_from_synthesis_propagates_unmodified(self) -> None:
        chat = _SynthesisEvalScriptedChat([])  # exhausted immediately
        with pytest.raises(ChatError):
            asyncio.run(run_synthesis_eval([("q", [_result()])], chat=chat, input_stage="bm25"))

    def test_chat_error_from_judge_propagates_unmodified(self) -> None:
        chat = _SynthesisEvalScriptedChat(["the answer is here [1]"])  # no scripted verdict
        judge = FaithfulnessJudge(chat)
        with pytest.raises(ChatError):
            asyncio.run(
                run_synthesis_eval([("q", [_result()])], chat=chat, input_stage="bm25", judge=judge)
            )

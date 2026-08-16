"""Faithfulness judge tests (Phase 5 synthesis-mode advisory judge, SPEC.md §6).

Exercises :class:`~groundkit.evals.judge.FaithfulnessJudge` against a
module-local scripted double, ``_JudgeScriptedChat``, that conforms
structurally to ``ChatProtocol`` — mirroring
``tests/test_query_rewrite.py``'s ``_RewriteScriptedChat``. This module never
imports ``groundkit.providers.llm`` or ``groundkit.providers.synthesis``, so
these tests stay coupled only to the pinned protocol contract and to plain
strings, never to a particular chat backend or the synthesis type. No test
touches the network. pytest-asyncio is not part of this repo's dependency
set, so async code under test is driven with ``asyncio.run()`` inside plain
``def`` test functions (see ``tests/test_embeddings.py`` for the same
pattern).
"""

from __future__ import annotations

import asyncio
import json

import pytest
from pydantic import ValidationError

from groundkit.errors import ChatError, ChatProviderNotConfiguredError, JudgeError
from groundkit.evals.judge import FaithfulnessJudge, FaithfulnessVerdict
from groundkit.providers.protocols import ChatProtocol


class _JudgeScriptedChat:
    """Minimal scripted double conforming to ``ChatProtocol`` structurally.

    Deliberately not built from ``groundkit.providers.llm`` and does not
    subclass ``ChatProtocol`` — this repo's provider seams are duck-typed
    Protocols, so a plain class with matching members is exactly what
    ``FaithfulnessJudge`` is required to accept. Records every ``(prompt,
    system)`` pair it is called with, so tests can assert on what the judge
    actually sent.
    """

    def __init__(
        self, completions: list[str] | None = None, *, error: Exception | None = None
    ) -> None:
        self._completions = list(completions) if completions is not None else []
        self._error = error
        self.calls: list[tuple[str, str | None]] = []

    @property
    def provider(self) -> str:
        return "scripted-test-double"

    @property
    def model_name(self) -> str:
        return "scripted-judge-v1"

    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        self.calls.append((prompt, system))
        if self._error is not None:
            raise self._error
        return self._completions.pop(0)


def _verdict_json(
    *,
    faithful: bool = True,
    unsupported_claims: list[str] | None = None,
    reasoning: str = "ok",
) -> str:
    """Build a schema-valid verdict JSON payload as a compact string."""
    return json.dumps(
        {
            "faithful": faithful,
            "unsupported_claims": [] if unsupported_claims is None else unsupported_claims,
            "reasoning": reasoning,
        }
    )


# ── protocol conformance of the test double itself ─────────────────────────


class TestScriptedChatConformance:
    def test_scripted_double_satisfies_chat_protocol(self) -> None:
        assert isinstance(_JudgeScriptedChat(["x"]), ChatProtocol)


# ── valid verdict round-trip ─────────────────────────────────────────────


class TestValidVerdictRoundTrip:
    def test_faithful_verdict_round_trips(self) -> None:
        chat = _JudgeScriptedChat(
            [_verdict_json(faithful=True, unsupported_claims=[], reasoning="fully supported")]
        )
        judge = FaithfulnessJudge(chat)

        verdict = asyncio.run(
            judge.judge(query="q", answer="the sky is blue", sources=["the sky is blue today"])
        )

        assert verdict == FaithfulnessVerdict(
            faithful=True, unsupported_claims=(), reasoning="fully supported"
        )

    def test_unfaithful_verdict_with_named_claims_round_trips(self) -> None:
        chat = _JudgeScriptedChat(
            [
                _verdict_json(
                    faithful=False,
                    unsupported_claims=["the moon is made of cheese"],
                    reasoning="one claim unsupported",
                )
            ]
        )
        judge = FaithfulnessJudge(chat)

        verdict = asyncio.run(judge.judge(query="q", answer="a", sources=["s"]))

        assert verdict.faithful is False
        assert verdict.unsupported_claims == ("the moon is made of cheese",)
        assert verdict.reasoning == "one claim unsupported"

    def test_unfaithful_verdict_with_no_named_claims_is_valid(self) -> None:
        """faithful=False with an empty unsupported_claims is not incoherent."""
        chat = _JudgeScriptedChat(
            [_verdict_json(faithful=False, unsupported_claims=[], reasoning="vague, no pinpoint")]
        )
        judge = FaithfulnessJudge(chat)

        verdict = asyncio.run(judge.judge(query="q", answer="a", sources=["s"]))

        assert verdict.faithful is False
        assert verdict.unsupported_claims == ()


# ── fenced-JSON extraction ──────────────────────────────────────────────


class TestFencedJsonExtraction:
    def test_json_tagged_fence_is_stripped(self) -> None:
        payload = _verdict_json()
        chat = _JudgeScriptedChat([f"```json\n{payload}\n```"])
        judge = FaithfulnessJudge(chat)

        verdict = asyncio.run(judge.judge(query="q", answer="a", sources=["s"]))

        assert verdict.faithful is True

    def test_bare_fence_without_language_tag_is_stripped(self) -> None:
        payload = _verdict_json()
        chat = _JudgeScriptedChat([f"```\n{payload}\n```"])
        judge = FaithfulnessJudge(chat)

        verdict = asyncio.run(judge.judge(query="q", answer="a", sources=["s"]))

        assert verdict.faithful is True

    def test_fence_with_surrounding_whitespace_is_stripped(self) -> None:
        payload = _verdict_json()
        chat = _JudgeScriptedChat([f"  \n```json\n{payload}\n```\n  "])
        judge = FaithfulnessJudge(chat)

        verdict = asyncio.run(judge.judge(query="q", answer="a", sources=["s"]))

        assert verdict.faithful is True


# ── trailing-prose-around-JSON rejection ────────────────────────────────


class TestProseAroundJsonRejected:
    def test_leading_prose_before_json_rejected(self) -> None:
        chat = _JudgeScriptedChat([f"Sure, here is the verdict: {_verdict_json()}"])
        judge = FaithfulnessJudge(chat)

        with pytest.raises(JudgeError):
            asyncio.run(judge.judge(query="q", answer="a", sources=["s"]))

    def test_trailing_prose_after_json_rejected(self) -> None:
        chat = _JudgeScriptedChat([f"{_verdict_json()}\nHope that helps!"])
        judge = FaithfulnessJudge(chat)

        with pytest.raises(JudgeError):
            asyncio.run(judge.judge(query="q", answer="a", sources=["s"]))

    def test_trailing_prose_after_fenced_json_rejected(self) -> None:
        """Fence-stripping only fires when the *whole* completion is one fence."""
        chat = _JudgeScriptedChat([f"```json\n{_verdict_json()}\n```\nHope that helps!"])
        judge = FaithfulnessJudge(chat)

        with pytest.raises(JudgeError):
            asyncio.run(judge.judge(query="q", answer="a", sources=["s"]))

    def test_plain_prose_completion_rejected(self) -> None:
        chat = _JudgeScriptedChat(["The answer looks faithful to me."])
        judge = FaithfulnessJudge(chat)

        with pytest.raises(JudgeError):
            asyncio.run(judge.judge(query="q", answer="a", sources=["s"]))


# ── missing / extra / wrong-type field ──────────────────────────────────


class TestMissingField:
    def test_missing_reasoning_field_rejected(self) -> None:
        chat = _JudgeScriptedChat([json.dumps({"faithful": True, "unsupported_claims": []})])
        judge = FaithfulnessJudge(chat)

        with pytest.raises(JudgeError):
            asyncio.run(judge.judge(query="q", answer="a", sources=["s"]))

    def test_missing_faithful_field_rejected(self) -> None:
        chat = _JudgeScriptedChat([json.dumps({"unsupported_claims": [], "reasoning": "ok"})])
        judge = FaithfulnessJudge(chat)

        with pytest.raises(JudgeError):
            asyncio.run(judge.judge(query="q", answer="a", sources=["s"]))


class TestExtraField:
    def test_extra_field_rejected(self) -> None:
        chat = _JudgeScriptedChat(
            [
                json.dumps(
                    {
                        "faithful": True,
                        "unsupported_claims": [],
                        "reasoning": "ok",
                        "confidence": 0.9,
                    }
                )
            ]
        )
        judge = FaithfulnessJudge(chat)

        with pytest.raises(JudgeError):
            asyncio.run(judge.judge(query="q", answer="a", sources=["s"]))


class TestWrongTypeField:
    def test_faithful_as_string_rejected_not_coerced(self) -> None:
        """strict=True must block the lax-mode 'true' string -> bool coercion."""
        chat = _JudgeScriptedChat(
            [json.dumps({"faithful": "true", "unsupported_claims": [], "reasoning": "ok"})]
        )
        judge = FaithfulnessJudge(chat)

        with pytest.raises(JudgeError):
            asyncio.run(judge.judge(query="q", answer="a", sources=["s"]))

    def test_unsupported_claims_as_string_rejected(self) -> None:
        chat = _JudgeScriptedChat(
            [json.dumps({"faithful": False, "unsupported_claims": "not a list", "reasoning": "ok"})]
        )
        judge = FaithfulnessJudge(chat)

        with pytest.raises(JudgeError):
            asyncio.run(judge.judge(query="q", answer="a", sources=["s"]))

    def test_reasoning_as_number_rejected(self) -> None:
        chat = _JudgeScriptedChat(
            [json.dumps({"faithful": True, "unsupported_claims": [], "reasoning": 42})]
        )
        judge = FaithfulnessJudge(chat)

        with pytest.raises(JudgeError):
            asyncio.run(judge.judge(query="q", answer="a", sources=["s"]))


# ── incoherent verdict rejection ────────────────────────────────────────


class TestIncoherentVerdictRejection:
    def test_faithful_true_with_unsupported_claims_rejected_via_judge(self) -> None:
        chat = _JudgeScriptedChat(
            [json.dumps({"faithful": True, "unsupported_claims": ["oops"], "reasoning": "ok"})]
        )
        judge = FaithfulnessJudge(chat)

        with pytest.raises(JudgeError):
            asyncio.run(judge.judge(query="q", answer="a", sources=["s"]))

    def test_faithful_true_with_unsupported_claims_rejected_by_model_directly(self) -> None:
        """The invariant lives on the schema itself, not only on the judge's parse path."""
        with pytest.raises(ValidationError):
            FaithfulnessVerdict(faithful=True, unsupported_claims=("oops",), reasoning="ok")


# ── empty answer / sources ──────────────────────────────────────────────


class TestEmptyAnswerOrSourcesRejectedBeforeProviderCall:
    def test_empty_answer_rejected(self) -> None:
        chat = _JudgeScriptedChat(["irrelevant"])
        judge = FaithfulnessJudge(chat)

        with pytest.raises(JudgeError):
            asyncio.run(judge.judge(query="q", answer="", sources=["s"]))

        assert chat.calls == []

    def test_whitespace_only_answer_rejected(self) -> None:
        chat = _JudgeScriptedChat(["irrelevant"])
        judge = FaithfulnessJudge(chat)

        with pytest.raises(JudgeError):
            asyncio.run(judge.judge(query="q", answer="   \n\t ", sources=["s"]))

        assert chat.calls == []

    def test_empty_sources_rejected(self) -> None:
        chat = _JudgeScriptedChat(["irrelevant"])
        judge = FaithfulnessJudge(chat)

        with pytest.raises(JudgeError):
            asyncio.run(judge.judge(query="q", answer="a", sources=[]))

        assert chat.calls == []


# ── provider error propagation ───────────────────────────────────────────


class TestProviderErrorPropagation:
    def test_chat_error_propagates_untouched(self) -> None:
        original = ChatError("boom")
        chat = _JudgeScriptedChat(error=original)
        judge = FaithfulnessJudge(chat)

        with pytest.raises(ChatError) as excinfo:
            asyncio.run(judge.judge(query="q", answer="a", sources=["s"]))

        assert excinfo.value is original

    def test_chat_provider_not_configured_error_propagates_untouched(self) -> None:
        original = ChatProviderNotConfiguredError("no api key configured")
        chat = _JudgeScriptedChat(error=original)
        judge = FaithfulnessJudge(chat)

        with pytest.raises(ChatProviderNotConfiguredError) as excinfo:
            asyncio.run(judge.judge(query="q", answer="a", sources=["s"]))

        assert excinfo.value is original


# ── unicode ───────────────────────────────────────────────────────────────


class TestUnicode:
    def test_unicode_round_trips_through_verdict_and_prompt(self) -> None:
        payload = _verdict_json(
            faithful=False,
            unsupported_claims=["猫は宇宙飛行士だ 🚀"],
            reasoning="日本語のクレームは出典にない — café ☕ résumé",
        )
        chat = _JudgeScriptedChat([payload])
        judge = FaithfulnessJudge(chat)

        verdict = asyncio.run(
            judge.judge(
                query="猫について教えて",
                answer="猫は宇宙飛行士だ 🚀。café でくつろぐ。",
                sources=["猫はペットだ。café は飲み物の店だ。"],
            )
        )

        assert verdict.faithful is False
        assert verdict.unsupported_claims == ("猫は宇宙飛行士だ 🚀",)
        assert "café" in verdict.reasoning

        prompt, _system = chat.calls[0]
        assert "猫について教えて" in prompt
        assert "猫は宇宙飛行士だ 🚀。café でくつろぐ。" in prompt
        assert "猫はペットだ。café は飲み物の店だ。" in prompt


# ── prompt receives all sources numbered ────────────────────────────────


class TestPromptReceivesNumberedSources:
    def test_sources_are_numbered_and_query_and_answer_are_present(self) -> None:
        chat = _JudgeScriptedChat([_verdict_json()])
        judge = FaithfulnessJudge(chat)

        asyncio.run(
            judge.judge(
                query="what is the capital of France",
                answer="Paris is the capital of France",
                sources=[
                    "Paris is the capital and largest city of France.",
                    "France is a country in Europe.",
                    "The Eiffel Tower is in Paris.",
                ],
            )
        )

        assert len(chat.calls) == 1
        prompt, system = chat.calls[0]
        assert system is None
        assert "what is the capital of France" in prompt
        assert "Paris is the capital of France" in prompt
        assert "[1] Paris is the capital and largest city of France." in prompt
        assert "[2] France is a country in Europe." in prompt
        assert "[3] The Eiffel Tower is in Paris." in prompt

    def test_custom_prompt_template_receives_numbered_sources(self) -> None:
        chat = _JudgeScriptedChat([_verdict_json()])
        judge = FaithfulnessJudge(chat, prompt_template="Q={query} A={answer} S={numbered_sources}")

        asyncio.run(judge.judge(query="q", answer="a", sources=["one", "two"]))

        prompt, _system = chat.calls[0]
        assert prompt == "Q=q A=a S=[1] one\n[2] two"


# ── exception messages never embed answer/source/query text ────────────


class TestNoContentLeakageInExceptionMessages:
    def test_schema_validation_error_does_not_echo_answer_or_source_text(self) -> None:
        secret_answer = "TOP-SECRET-ANSWER-MARKER-12345"  # noqa: S105 — leakage-test marker, not a credential
        secret_source = "TOP-SECRET-SOURCE-MARKER-67890"  # noqa: S105 — leakage-test marker, not a credential
        chat = _JudgeScriptedChat(
            [json.dumps({"faithful": "true", "unsupported_claims": [], "reasoning": "ok"})]
        )
        judge = FaithfulnessJudge(chat)

        with pytest.raises(JudgeError) as excinfo:
            asyncio.run(judge.judge(query="q", answer=secret_answer, sources=[secret_source]))

        message = str(excinfo.value)
        assert secret_answer not in message
        assert secret_source not in message

    def test_empty_answer_error_message_has_no_query_text(self) -> None:
        chat = _JudgeScriptedChat(["irrelevant"])
        judge = FaithfulnessJudge(chat)

        with pytest.raises(JudgeError) as excinfo:
            asyncio.run(judge.judge(query="secret query text", answer="", sources=["s"]))

        assert "secret query text" not in str(excinfo.value)

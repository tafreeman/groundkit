"""Query-rewrite provider tests (Phase 5 LLM boundary).

Exercises :class:`~groundkit.providers.query_rewrite.QueryRewriter` against a
module-local scripted double, ``_RewriteScriptedChat``, that conforms
structurally to ``ChatProtocol`` — this module never imports
``groundkit.providers.llm`` (a different Phase 5 seam's implementation) so
these tests stay coupled only to the pinned protocol contract, not to any
particular chat backend. pytest-asyncio is not part of this repo's dependency
set, so async code under test is driven with ``asyncio.run()`` inside plain
``def`` test functions (see ``tests/test_embeddings.py`` for the same
pattern).
"""

from __future__ import annotations

import asyncio

import pytest

from groundkit.errors import ChatError, ChatProviderNotConfiguredError, QueryRewriteError
from groundkit.providers.protocols import ChatProtocol
from groundkit.providers.query_rewrite import DEFAULT_REWRITE_PROMPT, QueryRewriter


class _RewriteScriptedChat:
    """Minimal scripted double conforming to ``ChatProtocol`` structurally.

    Deliberately not built from ``groundkit.providers.llm`` and does not
    subclass ``ChatProtocol`` — this repo's provider seams are duck-typed
    Protocols, so a plain class with matching members is exactly what
    ``QueryRewriter`` is required to accept. Records every ``(prompt,
    system)`` pair it is called with, so tests can assert on what the
    rewriter actually sent.
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
        return "scripted-rewrite-v1"

    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        self.calls.append((prompt, system))
        if self._error is not None:
            raise self._error
        return self._completions.pop(0)


# ── protocol conformance of the test double itself ─────────────────────────


class TestScriptedChatConformance:
    def test_scripted_double_satisfies_chat_protocol(self) -> None:
        assert isinstance(_RewriteScriptedChat(["x"]), ChatProtocol)


# ── empty / whitespace input query ─────────────────────────────────────────


class TestEmptyInputQuery:
    def test_empty_string_rejected_before_any_provider_call(self) -> None:
        chat = _RewriteScriptedChat(["irrelevant"])
        rewriter = QueryRewriter(chat)

        with pytest.raises(QueryRewriteError):
            asyncio.run(rewriter.rewrite(""))

        assert chat.calls == []

    def test_whitespace_only_rejected_before_any_provider_call(self) -> None:
        chat = _RewriteScriptedChat(["irrelevant"])
        rewriter = QueryRewriter(chat)

        with pytest.raises(QueryRewriteError):
            asyncio.run(rewriter.rewrite("   \t\n  "))

        assert chat.calls == []


# ── blank completion ────────────────────────────────────────────────────────


class TestBlankCompletion:
    def test_empty_completion_raises(self) -> None:
        chat = _RewriteScriptedChat([""])
        rewriter = QueryRewriter(chat)

        with pytest.raises(QueryRewriteError):
            asyncio.run(rewriter.rewrite("original query"))

    def test_whitespace_only_completion_raises(self) -> None:
        chat = _RewriteScriptedChat(["   \n\t  "])
        rewriter = QueryRewriter(chat)

        with pytest.raises(QueryRewriteError):
            asyncio.run(rewriter.rewrite("original query"))

    def test_quoted_empty_string_completion_raises(self) -> None:
        """A completion of just a quote pair reduces to "" after quote
        stripping and must still be treated as blank, not as a valid
        two-character rewrite."""
        chat = _RewriteScriptedChat(['""'])
        rewriter = QueryRewriter(chat)

        with pytest.raises(QueryRewriteError):
            asyncio.run(rewriter.rewrite("original query"))


# ── multi-line completion handling ─────────────────────────────────────────


class TestMultiLineCompletion:
    def test_multiple_non_blank_lines_raises(self) -> None:
        chat = _RewriteScriptedChat(["first candidate\nsecond candidate"])
        rewriter = QueryRewriter(chat)

        with pytest.raises(QueryRewriteError, match="non-blank lines"):
            asyncio.run(rewriter.rewrite("original query"))

    def test_trailing_blank_lines_are_tolerated(self) -> None:
        """Only more than one NON-BLANK line is a rejection — a single real
        line followed by blank trailing newlines is fine."""
        chat = _RewriteScriptedChat(["rewritten query\n\n"])
        rewriter = QueryRewriter(chat)

        result = asyncio.run(rewriter.rewrite("original query"))

        assert result == "rewritten query"

    def test_leading_blank_lines_are_tolerated(self) -> None:
        chat = _RewriteScriptedChat(["\n\nrewritten query"])
        rewriter = QueryRewriter(chat)

        result = asyncio.run(rewriter.rewrite("original query"))

        assert result == "rewritten query"

    def test_single_line_completion_is_unaffected(self) -> None:
        chat = _RewriteScriptedChat(["plain rewritten query"])
        rewriter = QueryRewriter(chat)

        result = asyncio.run(rewriter.rewrite("original query"))

        assert result == "plain rewritten query"


# ── provider error propagation ──────────────────────────────────────────────


class TestProviderErrorPropagation:
    def test_chat_error_propagates_untouched(self) -> None:
        chat = _RewriteScriptedChat(error=ChatError("upstream failure"))
        rewriter = QueryRewriter(chat)

        with pytest.raises(ChatError):
            asyncio.run(rewriter.rewrite("original query"))

    def test_chat_provider_not_configured_error_propagates_untouched(self) -> None:
        chat = _RewriteScriptedChat(error=ChatProviderNotConfiguredError("no credential"))
        rewriter = QueryRewriter(chat)

        with pytest.raises(ChatProviderNotConfiguredError):
            asyncio.run(rewriter.rewrite("original query"))

    def test_chat_provider_not_configured_error_is_also_a_chat_error(self) -> None:
        """Documents the subclass relationship this module's error handling
        relies on: propagating unmodified requires no special-casing of
        either exception type, since both are catchable as ChatError."""
        assert issubclass(ChatProviderNotConfiguredError, ChatError)


# ── prompt template receives the query ──────────────────────────────────────


class TestPromptTemplateReceivesQuery:
    def test_default_template_embeds_the_query(self) -> None:
        chat = _RewriteScriptedChat(["rewritten"])
        rewriter = QueryRewriter(chat)

        asyncio.run(rewriter.rewrite("unique marker query xyz"))

        assert len(chat.calls) == 1
        sent_prompt, sent_system = chat.calls[0]
        assert "unique marker query xyz" in sent_prompt
        assert sent_system is None

    def test_custom_template_is_honored(self) -> None:
        chat = _RewriteScriptedChat(["rewritten"])
        rewriter = QueryRewriter(
            chat, prompt_template="CUSTOM PREFIX >>> {query} <<< CUSTOM SUFFIX"
        )

        asyncio.run(rewriter.rewrite("marker"))

        sent_prompt, _sent_system = chat.calls[0]
        assert sent_prompt == "CUSTOM PREFIX >>> marker <<< CUSTOM SUFFIX"

    def test_default_prompt_template_constant_has_a_query_placeholder(self) -> None:
        assert "{query}" in DEFAULT_REWRITE_PROMPT


# ── no-fallback property ────────────────────────────────────────────────────


class TestNoFallbackProperty:
    """A rewriter that quietly returns its input on failure is
    indistinguishable from one that actually rewrote (SPEC.md §2, module
    docstring). Every failure mode here must end in an exception, never a
    successful return of the original query text.
    """

    def test_blank_completion_failure_never_returns_the_input_query(self) -> None:
        """A blank completion must raise rather than return anything at all —
        in particular, it must never return *original* unchanged."""
        original = "original unique query text 12345"
        chat = _RewriteScriptedChat([""])
        rewriter = QueryRewriter(chat)

        with pytest.raises(QueryRewriteError):
            asyncio.run(rewriter.rewrite(original))

    def test_multiline_completion_failure_never_returns_the_input_query(self) -> None:
        original = "original unique query text 12345"
        chat = _RewriteScriptedChat([f"{original}\nsecond line"])
        rewriter = QueryRewriter(chat)

        with pytest.raises(QueryRewriteError):
            asyncio.run(rewriter.rewrite(original))

    def test_provider_error_never_returns_the_input_query(self) -> None:
        original = "original unique query text 12345"
        chat = _RewriteScriptedChat(error=ChatError("boom"))
        rewriter = QueryRewriter(chat)

        with pytest.raises(ChatError):
            asyncio.run(rewriter.rewrite(original))


# ── unicode query ────────────────────────────────────────────────────────────


class TestUnicodeQuery:
    def test_unicode_query_flows_through_to_the_prompt(self) -> None:
        chat = _RewriteScriptedChat(["rewritten"])
        rewriter = QueryRewriter(chat)
        query = "qu’est-ce que l'énergie renouvelable ? 中文查询 \U0001f30d"  # noqa: RUF001 — deliberate unicode-input test

        asyncio.run(rewriter.rewrite(query))

        sent_prompt, _sent_system = chat.calls[0]
        assert query in sent_prompt

    def test_unicode_completion_is_returned_correctly(self) -> None:
        completion = "renewable énergie mesures 中文"
        chat = _RewriteScriptedChat([completion])
        rewriter = QueryRewriter(chat)

        result = asyncio.run(rewriter.rewrite("original"))

        assert result == completion


# ── output normalization (whitespace / quote stripping) ────────────────────


class TestOutputNormalization:
    def test_surrounding_whitespace_is_stripped(self) -> None:
        chat = _RewriteScriptedChat(["   rewritten query   "])
        rewriter = QueryRewriter(chat)

        assert asyncio.run(rewriter.rewrite("q")) == "rewritten query"

    def test_surrounding_double_quotes_are_stripped(self) -> None:
        chat = _RewriteScriptedChat(['"rewritten query"'])
        rewriter = QueryRewriter(chat)

        assert asyncio.run(rewriter.rewrite("q")) == "rewritten query"

    def test_surrounding_single_quotes_are_stripped(self) -> None:
        chat = _RewriteScriptedChat(["'rewritten query'"])
        rewriter = QueryRewriter(chat)

        assert asyncio.run(rewriter.rewrite("q")) == "rewritten query"

    def test_quotes_with_interior_whitespace_are_fully_normalized(self) -> None:
        chat = _RewriteScriptedChat(['"  rewritten query  "'])
        rewriter = QueryRewriter(chat)

        assert asyncio.run(rewriter.rewrite("q")) == "rewritten query"

    def test_mismatched_quotes_are_left_alone(self) -> None:
        """Only a MATCHING enclosing pair is stripped — this is a
        normalization of a clean answer, not a general-purpose parser."""
        chat = _RewriteScriptedChat(["\"rewritten query'"])
        rewriter = QueryRewriter(chat)

        assert asyncio.run(rewriter.rewrite("q")) == "\"rewritten query'"

    def test_deterministic_across_repeated_calls(self) -> None:
        chat = _RewriteScriptedChat(["same result", "same result"])
        rewriter = QueryRewriter(chat)

        first = asyncio.run(rewriter.rewrite("q"))
        second = asyncio.run(rewriter.rewrite("q"))

        assert first == second == "same result"


# ── query/completion text never leaks into exception messages ──────────────


class TestNoQueryTextInExceptionMessages:
    def test_multiline_completion_error_omits_query_and_completion_text(self) -> None:
        query = "the confidential search phrase"
        completion = "line one unique marker\nline two other marker"
        chat = _RewriteScriptedChat([completion])
        rewriter = QueryRewriter(chat)

        with pytest.raises(QueryRewriteError) as excinfo:
            asyncio.run(rewriter.rewrite(query))

        message = str(excinfo.value)
        assert query not in message
        assert "unique marker" not in message
        assert "other marker" not in message

    def test_blank_completion_error_omits_query_text(self) -> None:
        query = "another confidential phrase"
        chat = _RewriteScriptedChat([""])
        rewriter = QueryRewriter(chat)

        with pytest.raises(QueryRewriteError) as excinfo:
            asyncio.run(rewriter.rewrite(query))

        assert query not in str(excinfo.value)

    def test_empty_input_error_omits_query_text(self) -> None:
        # Nothing to leak for a purely-whitespace query, but assert the
        # message still doesn't echo the raw (whitespace) input verbatim.
        query = "   "
        chat = _RewriteScriptedChat([])
        rewriter = QueryRewriter(chat)

        with pytest.raises(QueryRewriteError) as excinfo:
            asyncio.run(rewriter.rewrite(query))

        assert query not in str(excinfo.value)


class TestQueryLengthBoundary:
    """A rewrite is an untrusted producer inside the trust boundary.

    ``MAX_QUERY_LEN``'s own docstring says callers at a trust boundary bound
    the input before it reaches retrieval, and ``Retriever.search``
    deliberately does not police length. A model completion is exactly such
    an input (PR #14 review finding, shown to fail first).
    """

    def test_overlong_rewrite_rejected_before_search(self) -> None:
        from groundkit.retrieval.search import MAX_QUERY_LEN

        chat = _RewriteScriptedChat(["q" * (MAX_QUERY_LEN + 1)])
        rewriter = QueryRewriter(chat)
        with pytest.raises(QueryRewriteError) as excinfo:
            asyncio.run(rewriter.rewrite("a query"))
        message = str(excinfo.value)
        assert "q" * 32 not in message

    def test_rewrite_at_exactly_the_boundary_is_accepted(self) -> None:
        from groundkit.retrieval.search import MAX_QUERY_LEN

        chat = _RewriteScriptedChat(["q" * MAX_QUERY_LEN])
        rewriter = QueryRewriter(chat)
        assert asyncio.run(rewriter.rewrite("a query")) == "q" * MAX_QUERY_LEN

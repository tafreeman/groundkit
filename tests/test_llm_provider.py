"""Chat provider tests — network-free via injected httpx clients.

Mirrors ``tests/test_embeddings.py``: HTTP providers are always constructed
with a client wired to ``httpx.MockTransport``; no test ever touches the
network. pytest-asyncio is not part of this repo's dependency set, so async
code under test is driven with ``asyncio.run()`` inside plain ``def`` test
functions.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from typing import Final

import httpx
import pytest

from groundkit.config import DEFAULT_CHAT_MODEL, ChatConfig
from groundkit.errors import ChatError, ChatProviderNotConfiguredError, ConfigurationError
from groundkit.providers.llm import (
    OllamaChat,
    OpenAICompatChat,
    RedactingChat,
    ScriptedChatProvider,
    build_chat,
)
from groundkit.providers.protocols import ChatProtocol
from groundkit.providers.redaction import (
    DEFAULT_PATTERNS,
    RedactionConfig,
    UnknownRedactionTokenError,
)
from groundkit.utils import url_safety

_ChatHandler = Callable[[httpx.Request], httpx.Response]


def _chat_client(handler: _ChatHandler) -> httpx.AsyncClient:
    """Build an httpx.AsyncClient wired to a MockTransport — no network."""
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


#: A host that is not shaped like an address literal, so ``ensure_safe_endpoint``
#: (ADR-0014 decision 10) sends it through DNS resolution rather than
#: classifying it directly — mirrors ``test_embeddings.py``'s
#: ``_PUBLIC_HOST``/``_PUBLIC_BASE_URL``/``_PUBLIC_ADDRESS``. OpenAICompatChat
#: carries no loopback/private allowance, so every test that needs to
#: actually send a request is pointed here instead of at a private address.
_CHAT_PUBLIC_HOST: Final[str] = "chat-proxy.example.com"
_CHAT_PUBLIC_BASE_URL: Final[str] = f"https://{_CHAT_PUBLIC_HOST}"

#: A real, globally-routable address (example.com's long-standing IP) — used
#: only as a stand-in DNS answer, never actually contacted.
_CHAT_PUBLIC_ADDRESS: Final[str] = "93.184.216.34"


async def _resolve_chat_host_to_public_address(host: str) -> Sequence[str]:
    """Fake resolver: answers every lookup with a public address.

    Injected by replacing ``url_safety._default_resolver`` wholesale, the
    same seam ``test_embeddings.py`` uses: ``OpenAICompatChat``'s
    per-request call to ``ensure_safe_endpoint`` exposes no resolver
    parameter of its own.
    """
    assert host == _CHAT_PUBLIC_HOST
    return [_CHAT_PUBLIC_ADDRESS]


def _public_openai_chat(
    monkeypatch: pytest.MonkeyPatch, handler: _ChatHandler, **kwargs: object
) -> OpenAICompatChat:
    """Build an ``OpenAICompatChat`` pointed at a public-looking host, with
    the resolver faked so the endpoint-safety guard can run without
    touching the network. See ``_CHAT_PUBLIC_BASE_URL``.
    """
    monkeypatch.setattr(url_safety, "_default_resolver", _resolve_chat_host_to_public_address)
    return OpenAICompatChat(
        base_url=_CHAT_PUBLIC_BASE_URL,
        model_name="gpt-test",
        client=_chat_client(handler),
        **kwargs,  # type: ignore[arg-type]
    )


class _ChatRecordingFake:
    """Minimal ``ChatProtocol`` conformer that records every ``(prompt,
    system)`` pair it receives and replies from a fixed script, in order.

    Distinct from ``ScriptedChatProvider``: that class is production code
    under test elsewhere in this file (its own exhaustion/rejection rules
    matter), while this fake exists only so ``RedactingChat`` tests can
    inspect exactly what text reached the wrapped provider — the one thing
    ``ScriptedChatProvider`` does not expose.
    """

    def __init__(self, completions: Sequence[str]) -> None:
        self._completions = list(completions)
        self.calls: list[tuple[str, str | None]] = []

    @property
    def provider(self) -> str:
        return "chat-recording-fake"

    @property
    def model_name(self) -> str:
        return "chat-recording-fake-model"

    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        self.calls.append((prompt, system))
        return self._completions[len(self.calls) - 1]


# ── OllamaChat ──────────────────────────────────────────────────────────


class TestOllamaChat:
    def test_request_shape_and_response_parsing(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            body = json.loads(request.content)
            assert body == {"model": DEFAULT_CHAT_MODEL, "prompt": "hello", "stream": False}
            return httpx.Response(200, json={"response": "hi there", "done": True})

        chat = OllamaChat(client=_chat_client(handler))
        result = asyncio.run(chat.complete("hello"))

        assert result == "hi there"
        assert len(requests) == 1
        assert requests[0].url.path == "/api/generate"

    def test_system_field_is_included_only_when_given(self) -> None:
        """``/api/generate`` takes ``system`` as its own top-level field
        (ADR-0017 decision 2) -- omitted entirely, not sent as null, when
        no system instruction is given."""

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["system"] == "be terse"
            assert body["prompt"] == "hello"
            return httpx.Response(200, json={"response": "ok"})

        chat = OllamaChat(client=_chat_client(handler))
        asyncio.run(chat.complete("hello", system="be terse"))

    def test_system_field_is_omitted_when_not_given(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert "system" not in body
            return httpx.Response(200, json={"response": "ok"})

        chat = OllamaChat(client=_chat_client(handler))
        asyncio.run(chat.complete("hello"))

    def test_no_authorization_header_is_sent(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert "authorization" not in request.headers
            return httpx.Response(200, json={"response": "ok"})

        chat = OllamaChat(client=_chat_client(handler))
        asyncio.run(chat.complete("x"))

    def test_malformed_response_raises(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"unexpected": "shape"})

        chat = OllamaChat(client=_chat_client(handler))
        with pytest.raises(ChatError, match="response"):
            asyncio.run(chat.complete("x"))

    @pytest.mark.parametrize(
        "response_body",
        [
            pytest.param("not-a-dict-at-all", id="non_dict_response"),
            pytest.param({"unexpected": "shape"}, id="missing_response_field"),
            pytest.param({"response": 5}, id="response_not_a_string"),
            pytest.param({"response": None}, id="response_is_null"),
        ],
    )
    def test_malformed_shapes_raise_chat_error(self, response_body: object) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=response_body)

        chat = OllamaChat(client=_chat_client(handler))
        with pytest.raises(ChatError):
            asyncio.run(chat.complete("x"))

    @pytest.mark.parametrize(
        "content",
        [pytest.param("", id="empty_string"), pytest.param("   \n\t", id="whitespace_only")],
    )
    def test_empty_or_whitespace_completion_is_rejected(self, content: str) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"response": content})

        chat = OllamaChat(client=_chat_client(handler))
        with pytest.raises(ChatError, match="empty"):
            asyncio.run(chat.complete("x"))

    def test_transport_failure_is_a_scrubbed_chat_error(self) -> None:
        """No credential exists for Ollama, but a transport-level failure
        must still surface as a plain ChatError via the same
        wrap-and-sever path the credentialed provider uses."""

        def handler(_request: httpx.Request) -> httpx.Response:
            raise RuntimeError("connection refused")

        chat = OllamaChat(client=_chat_client(handler))
        with pytest.raises(ChatError, match="connection refused"):
            asyncio.run(chat.complete("x"))

    def test_non_2xx_status_raises_chat_error(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "boom"})

        chat = OllamaChat(client=_chat_client(handler))
        with pytest.raises(ChatError):
            asyncio.run(chat.complete("x"))

    def test_custom_base_url_and_model_are_honored(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["model"] == "custom-model"
            return httpx.Response(200, json={"response": "ok"})

        chat = OllamaChat(model_name="custom-model", client=_chat_client(handler))
        asyncio.run(chat.complete("x"))
        assert chat.model_name == "custom-model"
        assert chat.provider == "ollama"

    def test_empty_input_still_sends_a_request(self) -> None:
        """Unlike embedding batches, a chat completion has no natural
        'empty input' short-circuit — an empty prompt is still a real
        request, and this pins that as deliberate rather than an oversight."""

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["prompt"] == ""
            return httpx.Response(200, json={"response": "ok"})

        chat = OllamaChat(client=_chat_client(handler))
        assert asyncio.run(chat.complete("")) == "ok"


# ── OpenAICompatChat ────────────────────────────────────────────────────


class TestOpenAICompatChat:
    def test_auth_header_uses_env_var_read_at_call_time(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GROUNDKIT_OPENAI_API_KEY", "sk-test-value")
        seen_headers: list[str | None] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_headers.append(request.headers.get("authorization"))
            return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

        chat = _public_openai_chat(monkeypatch, handler)
        asyncio.run(chat.complete("x"))

        assert seen_headers == ["Bearer sk-test-value"]

    def test_missing_key_raises_provider_not_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GROUNDKIT_OPENAI_API_KEY", raising=False)

        def handler(_request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("no request should be sent without a configured key")

        chat = OpenAICompatChat(
            base_url="https://api.example.com", model_name="gpt-test", client=_chat_client(handler)
        )

        with pytest.raises(ChatProviderNotConfiguredError, match="GROUNDKIT_OPENAI_API_KEY"):
            asyncio.run(chat.complete("x"))

    def test_empty_key_raises_provider_not_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GROUNDKIT_OPENAI_API_KEY", "")

        def handler(_request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("no request should be sent with an empty key")

        chat = OpenAICompatChat(
            base_url="https://api.example.com", model_name="gpt-test", client=_chat_client(handler)
        )

        with pytest.raises(ChatProviderNotConfiguredError, match="GROUNDKIT_OPENAI_API_KEY"):
            asyncio.run(chat.complete("x"))

    def test_custom_api_key_env_var_is_honored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CUSTOM_CHAT_KEY_VAR", "sk-custom")

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers.get("authorization") == "Bearer sk-custom"
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        chat = _public_openai_chat(monkeypatch, handler, api_key_env="CUSTOM_CHAT_KEY_VAR")
        asyncio.run(chat.complete("x"))

    def test_malformed_response_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GROUNDKIT_OPENAI_API_KEY", "sk-test")

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"unexpected": "shape"})

        chat = _public_openai_chat(monkeypatch, handler)
        with pytest.raises(ChatError, match="choices"):
            asyncio.run(chat.complete("x"))

    @pytest.mark.parametrize(
        "response_body",
        [
            pytest.param("not-a-dict", id="non_dict_response"),
            pytest.param({"choices": []}, id="empty_choices"),
            pytest.param({"choices": "not-a-list"}, id="choices_not_a_list"),
            pytest.param({"choices": ["not-a-dict"]}, id="choice_not_a_dict"),
            pytest.param({"choices": [{"message": "not-a-dict"}]}, id="message_not_a_dict"),
            pytest.param({"choices": [{"message": {}}]}, id="missing_content"),
            pytest.param({"choices": [{"message": {"content": 5}}]}, id="content_not_a_string"),
        ],
    )
    def test_malformed_shapes_raise_chat_error(
        self, monkeypatch: pytest.MonkeyPatch, response_body: object
    ) -> None:
        monkeypatch.setenv("GROUNDKIT_OPENAI_API_KEY", "sk-test")

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=response_body)

        chat = _public_openai_chat(monkeypatch, handler)
        with pytest.raises(ChatError):
            asyncio.run(chat.complete("x"))

    @pytest.mark.parametrize(
        "content",
        [pytest.param("", id="empty_string"), pytest.param("  ", id="whitespace_only")],
    )
    def test_empty_or_whitespace_completion_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch, content: str
    ) -> None:
        monkeypatch.setenv("GROUNDKIT_OPENAI_API_KEY", "sk-test")

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

        chat = _public_openai_chat(monkeypatch, handler)
        with pytest.raises(ChatError, match="empty"):
            asyncio.run(chat.complete("x"))

    def test_cause_and_context_chain_never_carries_the_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ADR-0001 hazard 6 / ARP KNOWN_LIMITATIONS §4.10: a scrubbed message
        is not enough if the raw exception still rides along on __cause__ or
        __context__. Assert the whole chain is clean, not just the top frame.
        """
        api_key = "sk-super-secret-chat-value"
        monkeypatch.setenv("GROUNDKIT_OPENAI_API_KEY", api_key)

        def handler(_request: httpx.Request) -> httpx.Response:
            raise RuntimeError(f"upstream rejected credential {api_key}")

        chat = _public_openai_chat(monkeypatch, handler)

        with pytest.raises(ChatError) as excinfo:
            asyncio.run(chat.complete("x"))

        exc: BaseException = excinfo.value
        assert api_key not in str(exc)
        assert api_key not in repr(exc)
        assert exc.__cause__ is None

        node = exc.__context__
        while node is not None:
            assert api_key not in str(node)
            assert api_key not in repr(node)
            node = node.__context__


# ── ScriptedChatProvider ────────────────────────────────────────────────


class TestScriptedChatProvider:
    def test_satisfies_chat_protocol(self) -> None:
        assert isinstance(ScriptedChatProvider(["hi"]), ChatProtocol)

    def test_provider_and_model_identity(self) -> None:
        chat = ScriptedChatProvider(["hi"])
        assert chat.provider == "scripted"
        assert chat.model_name == "scripted-v1"

    def test_returns_script_entries_in_order(self) -> None:
        chat = ScriptedChatProvider(["first", "second", "third"])
        assert asyncio.run(chat.complete("a")) == "first"
        assert asyncio.run(chat.complete("b")) == "second"
        assert asyncio.run(chat.complete("c")) == "third"

    def test_ignores_prompt_and_system(self) -> None:
        chat = ScriptedChatProvider(["only"])
        assert asyncio.run(chat.complete("anything", system="anything else")) == "only"

    def test_exhausted_script_raises_and_never_cycles(self) -> None:
        chat = ScriptedChatProvider(["only"])
        asyncio.run(chat.complete("a"))

        with pytest.raises(ChatError, match="exhausted"):
            asyncio.run(chat.complete("b"))
        # Still exhausted on a second attempt -- never silently cycles back
        # to the start of the script.
        with pytest.raises(ChatError, match="exhausted"):
            asyncio.run(chat.complete("c"))

    def test_empty_script_raises_immediately(self) -> None:
        chat = ScriptedChatProvider([])
        with pytest.raises(ChatError, match="exhausted"):
            asyncio.run(chat.complete("a"))

    @pytest.mark.parametrize(
        "content",
        [pytest.param("", id="empty_string"), pytest.param("   ", id="whitespace_only")],
    )
    def test_empty_or_whitespace_script_entry_is_rejected(self, content: str) -> None:
        chat = ScriptedChatProvider([content])
        with pytest.raises(ChatError, match="empty"):
            asyncio.run(chat.complete("a"))

    def test_mutating_the_caller_sequence_after_construction_has_no_effect(self) -> None:
        source = ["first"]
        chat = ScriptedChatProvider(source)
        source.append("second")

        assert asyncio.run(chat.complete("a")) == "first"
        with pytest.raises(ChatError, match="exhausted"):
            asyncio.run(chat.complete("b"))


# ── RedactingChat ───────────────────────────────────────────────────────


class TestRedactingChat:
    def test_prompt_and_system_are_redacted_before_reaching_inner(self) -> None:
        inner = _ChatRecordingFake(["ok"])
        chat = RedactingChat(inner, RedactionConfig())

        asyncio.run(
            chat.complete(
                "contact prompt-user@example.com",
                system="contact system-user@example.com",
            )
        )

        assert len(inner.calls) == 1
        seen_prompt, seen_system = inner.calls[0]
        assert seen_system is not None
        assert "prompt-user@example.com" not in seen_prompt
        assert "system-user@example.com" not in seen_system
        # Same Redactor instance for both calls within one complete(): the
        # prompt is redacted first, so its email claims the first counter.
        assert "[EMAIL_1]" in seen_prompt
        assert "[EMAIL_2]" in seen_system

    def test_system_none_is_not_redacted_into_a_string(self) -> None:
        inner = _ChatRecordingFake(["ok"])
        chat = RedactingChat(inner, RedactionConfig())

        asyncio.run(chat.complete("hello, no system given"))

        _seen_prompt, seen_system = inner.calls[0]
        assert seen_system is None

    def test_completion_containing_a_token_restores_to_the_original_value(self) -> None:
        original_email = "user@example.com"
        inner = _ChatRecordingFake(["the address on file is [EMAIL_1]"])
        chat = RedactingChat(inner, RedactionConfig())

        result = asyncio.run(chat.complete(f"what is {original_email}?"))

        assert result == f"the address on file is {original_email}"
        # The inner provider itself must never have seen the raw email.
        assert original_email not in inner.calls[0][0]

    def test_fresh_redactor_per_call_blocks_cross_call_token_restoration(self) -> None:
        """The cross-request-disclosure regression ADR-0017 decision 4 exists
        to prevent: if ``RedactingChat`` reused one ``Redactor`` across calls,
        a token minted while redacting call 1's prompt would be silently
        restorable out of call 2's completion, expanding call 1's secret
        into a response for a completely unrelated request.

        A FRESH ``Redactor`` per call makes that structurally impossible:
        call 2's redactor shares call 1's ``RedactionConfig`` (so it still
        recognizes the ``EMAIL`` category) but is a different instance that
        never itself issued counter 1 -- so a literal ``[EMAIL_1]`` showing
        up in call 2's completion (simulating a coincidental or leaked
        token shape) fails closed via ``UnknownRedactionTokenError`` instead
        of being silently expanded into call 1's email address.
        """
        inner = _ChatRecordingFake(["call one reply", "[EMAIL_1]"])
        chat = RedactingChat(inner, RedactionConfig())

        # Call 1: redacts a real email, minting [EMAIL_1] on a Redactor
        # instance that is discarded the moment this call returns.
        asyncio.run(chat.complete("my email is call1-secret@example.com"))

        # Call 2: no email in this prompt at all, so its own (also fresh)
        # Redactor never mints [EMAIL_1] itself.
        with pytest.raises(UnknownRedactionTokenError):
            asyncio.run(chat.complete("no email here"))

    def test_provider_and_model_name_delegate_to_inner(self) -> None:
        inner = _ChatRecordingFake([])
        chat = RedactingChat(inner, RedactionConfig())

        assert chat.provider == inner.provider
        assert chat.model_name == inner.model_name

    def test_satisfies_chat_protocol(self) -> None:
        inner = _ChatRecordingFake([])
        assert isinstance(RedactingChat(inner, RedactionConfig()), ChatProtocol)

    def test_errors_from_inner_propagate_unmodified(self) -> None:
        class _FailingInner:
            @property
            def provider(self) -> str:
                return "failing-inner"

            @property
            def model_name(self) -> str:
                return "failing-inner-model"

            async def complete(self, prompt: str, *, system: str | None = None) -> str:
                raise ChatError("inner provider failed")

        chat = RedactingChat(_FailingInner(), RedactionConfig())

        with pytest.raises(ChatError, match="inner provider failed"):
            asyncio.run(chat.complete("hello"))


# ── build_chat ──────────────────────────────────────────────────────────


class TestBuildChat:
    def test_ollama_returns_a_bare_ollama_chat(self) -> None:
        chat = build_chat(ChatConfig(provider="ollama"))

        assert isinstance(chat, OllamaChat)
        assert not isinstance(chat, RedactingChat)

    def test_ollama_config_fields_are_threaded_through(self) -> None:
        config = ChatConfig(
            provider="ollama", model_name="custom-model", base_url="http://172.17.0.5:11434"
        )
        chat = build_chat(config)

        assert isinstance(chat, OllamaChat)
        assert chat.model_name == "custom-model"
        assert chat._base_url == "http://172.17.0.5:11434"

    def test_openai_compatible_is_wrapped_in_redacting_chat(self) -> None:
        config = ChatConfig(provider="openai_compatible", base_url="https://api.example.com")
        chat = build_chat(config)

        assert isinstance(chat, RedactingChat)
        assert isinstance(chat._inner, OpenAICompatChat)

    def test_openai_compatible_never_returned_bare(self) -> None:
        """ADR-0017 decision 4: there is no operator-facing redaction
        opt-out for cloud egress -- build_chat decides, not a flag."""
        config = ChatConfig(provider="openai_compatible", base_url="https://api.example.com")
        chat = build_chat(config)

        assert not isinstance(chat, OpenAICompatChat)

    def test_openai_compatible_default_redaction_uses_default_patterns(self) -> None:
        config = ChatConfig(provider="openai_compatible", base_url="https://api.example.com")
        chat = build_chat(config)

        assert isinstance(chat, RedactingChat)
        assert chat._redaction.patterns == DEFAULT_PATTERNS

    def test_openai_compatible_honors_an_explicit_redaction_config(self) -> None:
        config = ChatConfig(provider="openai_compatible", base_url="https://api.example.com")
        custom = RedactionConfig(patterns=(DEFAULT_PATTERNS[0],))

        chat = build_chat(config, redaction=custom)

        assert isinstance(chat, RedactingChat)
        assert chat._redaction is custom

    def test_openai_compatible_with_empty_pattern_tuple_raises_configuration_error(self) -> None:
        config = ChatConfig(provider="openai_compatible", base_url="https://api.example.com")
        empty_redaction = RedactionConfig(patterns=())

        with pytest.raises(ConfigurationError, match="redaction"):
            build_chat(config, redaction=empty_redaction)

    def test_openai_compatible_config_fields_reach_the_inner_provider(self) -> None:
        config = ChatConfig(
            provider="openai_compatible",
            model_name="gpt-test-model",
            base_url="https://api.example.com",
            api_key_env="CUSTOM_ENV_VAR",
        )
        chat = build_chat(config)

        assert isinstance(chat, RedactingChat)
        inner = chat._inner
        assert isinstance(inner, OpenAICompatChat)
        assert inner.model_name == "gpt-test-model"
        assert inner._base_url == "https://api.example.com"
        assert inner._api_key_env == "CUSTOM_ENV_VAR"


# ── outbound endpoint safety (ADR-0014 decision 10) ────────────────────────


class TestEndpointShapeValidationAtConstruction:
    def test_userinfo_in_base_url_is_rejected_at_construction(self) -> None:
        with pytest.raises(ConfigurationError, match="userinfo"):
            OllamaChat(base_url="http://user:hunter2@127.0.0.1:11434")

    def test_query_string_in_base_url_is_rejected_at_construction(self) -> None:
        with pytest.raises(ConfigurationError, match="query"):
            OpenAICompatChat(base_url="https://api.example.com/v1?x=1", model_name="gpt-test")

    def test_bad_scheme_in_base_url_is_rejected_at_construction(self) -> None:
        with pytest.raises(ConfigurationError, match="scheme"):
            OllamaChat(base_url="ftp://127.0.0.1:11434")

    def test_shape_rejection_applies_identically_to_ollama_and_openai_compat(self) -> None:
        """The shape check is not part of the Ollama private-endpoint
        allowance — same bad base_url, same rejection, regardless of which
        provider it is constructed for."""
        bad_url = "http://user:pw@127.0.0.1:11434"
        with pytest.raises(ConfigurationError, match="userinfo"):
            OllamaChat(base_url=bad_url)
        with pytest.raises(ConfigurationError, match="userinfo"):
            OpenAICompatChat(base_url=bad_url, model_name="gpt-test")

    def test_valid_shape_construction_still_succeeds(self) -> None:
        OllamaChat()


class TestEnsureSafeEndpointCallSite:
    """Mirrors ``test_embeddings.py``'s ``TestEnsureSafeEndpointCallSite``:
    each test asserts both that the call raises *and* that zero requests
    reached the mock transport, proving the refusal happens before the
    socket, not merely that the response was later discarded.
    """

    def test_openai_compat_refuses_the_default_loopback_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OpenAICompatChat carries no private-endpoint allowance, so even
        the library's own default loopback endpoint (meant for Ollama)
        must be refused for this provider once a request is attempted."""
        monkeypatch.setenv("GROUNDKIT_OPENAI_API_KEY", "sk-test")
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        chat = OpenAICompatChat(
            base_url="http://127.0.0.1:11434", model_name="gpt-test", client=_chat_client(handler)
        )

        with pytest.raises(ConfigurationError, match="loopback"):
            asyncio.run(chat.complete("x"))

        assert requests == []

    def test_openai_compat_refuses_a_private_cloud_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A private RFC1918 address masquerading as a cloud endpoint must
        be refused — no allowance exists for OpenAICompatChat, unlike
        OllamaChat."""
        monkeypatch.setenv("GROUNDKIT_OPENAI_API_KEY", "sk-test")
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        chat = OpenAICompatChat(
            base_url="http://10.0.0.5:8080", model_name="gpt-test", client=_chat_client(handler)
        )

        with pytest.raises(ConfigurationError, match="private"):
            asyncio.run(chat.complete("x"))

        assert requests == []

    def test_openai_compat_refuses_link_local_cloud_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The common cloud-metadata address must be refused even though
        it is shaped like an ordinary reachable host."""
        monkeypatch.setenv("GROUNDKIT_OPENAI_API_KEY", "sk-test")
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        chat = OpenAICompatChat(
            base_url="http://169.254.169.254", model_name="gpt-test", client=_chat_client(handler)
        )

        with pytest.raises(ConfigurationError, match="link_local"):
            asyncio.run(chat.complete("x"))

        assert requests == []

    def test_ollama_still_refuses_link_local_despite_its_allowance(self) -> None:
        """OllamaChat's allowance is scoped to loopback/private only — a
        link-local literal such as the common cloud metadata address must
        still be refused for it."""
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={"response": "ok"})

        chat = OllamaChat(base_url="http://169.254.169.254:11434", client=_chat_client(handler))

        with pytest.raises(ConfigurationError, match="link_local"):
            asyncio.run(chat.complete("x"))

        assert requests == []


class TestOllamaPrivateEndpointAllowance:
    """Proves the ``_allow_private_endpoint`` wiring actually reaches
    ``ensure_safe_endpoint`` end-to-end through a real chat provider, not
    just at the ``url_safety`` unit-test level."""

    def test_default_loopback_endpoint_still_succeeds(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"response": "ok"})

        chat = OllamaChat(client=_chat_client(handler))
        assert asyncio.run(chat.complete("x")) == "ok"

    def test_rfc1918_bridge_network_endpoint_succeeds(self) -> None:
        """SPEC.md §9's Phase 6 compose topology reaches Ollama at a
        bridge-network address, not just at loopback."""

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"response": "ok"})

        chat = OllamaChat(base_url="http://172.17.0.5:11434", client=_chat_client(handler))
        assert asyncio.run(chat.complete("x")) == "ok"


# ── ChatProtocol conformance (isinstance sanity check) ─────────────────────


class TestChatProtocolConformance:
    def test_all_three_providers_satisfy_the_protocol(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("conformance check must not send a request")

        assert isinstance(OllamaChat(client=_chat_client(handler)), ChatProtocol)
        assert isinstance(
            OpenAICompatChat(
                base_url="https://api.example.com",
                model_name="gpt-test",
                client=_chat_client(handler),
            ),
            ChatProtocol,
        )
        assert isinstance(ScriptedChatProvider(["ok"]), ChatProtocol)


class TestHttpChatLifecycle:
    def test_aclose_releases_the_injected_client(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("aclose must not send a request")

        chat = OllamaChat(client=_chat_client(handler))
        asyncio.run(chat.aclose())
        assert chat._client.is_closed

    def test_shared_properties_reflect_constructor_kwargs(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("property access must not send a request")

        chat = OllamaChat(model_name="custom-model", client=_chat_client(handler))
        assert chat.model_name == "custom-model"
        assert chat.provider == "ollama"

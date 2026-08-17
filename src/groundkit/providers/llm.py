"""Chat/completion provider interface and implementations (Phase 5 boundary).

Mirrors ``providers/embeddings.py``'s shape and hardening: no SDK, no
LiteLLM — :class:`OllamaChat` and :class:`OpenAICompatChat` call their HTTP
chat endpoints directly (SPEC.md §3, fewer deps, local-first). The same
outbound-endpoint SSRF guard applies here as there (ADR-0014 decision 10):
every ``_HttpChat`` validates its configured endpoint's *shape* once at
construction (:func:`~groundkit.utils.url_safety.validate_endpoint_shape`)
and its resolved *address* once per request, immediately before the POST
(:func:`~groundkit.utils.url_safety.ensure_safe_endpoint`). :class:`OllamaChat`
alone is exempted from the address check's loopback/private refusal — the
same named exception :class:`~groundkit.providers.embeddings.OllamaEmbedder`
carries (SPEC.md §7: local Ollama loopback is the one deliberate exception
to the guard; ADR-0014 decision 10 spells out that the exemption does not
relax the shape check, which runs identically for every conformer here).

Credential handling *reuses* ``providers/embeddings.py``'s scrubbing
helpers (:func:`~groundkit.providers.embeddings._scrub`,
:func:`~groundkit.providers.embeddings._sanitize_url`) and its
generic POST-and-catch-the-exception-as-a-value helper
(:func:`~groundkit.providers.embeddings._post_json`) rather than
reimplementing them: the scrubbing contract — an API key never in an
exception message, and the exception chain severed (``__cause__`` *and*
``__context__``) so the raw provider exception can never ride along
unscrubbed — is identical between the two provider families, so it stays
one piece of code, not a parallel copy that could drift out of sync
(ADR-0001 hazard 6 / ARP KNOWN_LIMITATIONS §4.10).

Every conformer's ``__init__`` still takes explicit constructor keyword
arguments, not a config object (mirroring how ``OllamaEmbedder`` and
``OpenAICompatibleEmbedder`` are constructed) — :func:`build_chat` is the
one place ``groundkit.config.ChatConfig`` is translated into one of them,
exactly the role :func:`~groundkit.providers.embeddings.build_embedder`
plays for ``EmbeddingConfig``.

Four conformers of :class:`~groundkit.providers.protocols.ChatProtocol`:

- :class:`OllamaChat` — direct HTTP to Ollama's non-streaming ``/api/generate``
  endpoint (ADR-0017 decision 2; ``docs/specs/phase-5-boundary-features.md``
  §3) — not ``/api/chat``: ``/api/generate``'s own ``prompt``/``system``
  fields map directly onto this seam's ``complete(prompt, *, system=None)``
  parameters, with no messages array to build or parse in between. No
  credential is used, sent, or read.
- :class:`OpenAICompatChat` — direct HTTP to an OpenAI-compatible
  ``/v1/chat/completions`` endpoint. The API key is read from
  ``os.environ[api_key_env]`` at call time — never stored, logged, or
  accepted as the key value itself (SPEC.md §7: config carries the env
  var's *name*, never a value).
- :class:`ScriptedChatProvider` — the
  :class:`~groundkit.providers.embeddings.InMemoryEmbedder` analog for
  chat: a labeled TEST DOUBLE that plays back a fixed script of
  completions. It calls no model and has zero semantic behavior; it exists
  solely so code that calls a
  :class:`~groundkit.providers.protocols.ChatProtocol` (query rewrite,
  synthesis) can be exercised deterministically and offline. Any quality
  number derived from its output is noise presented as a number (SPEC.md
  §2) — the same warning :class:`~groundkit.providers.embeddings.InMemoryEmbedder`
  carries for dense/fusion eval numbers. There is deliberately no
  ``"scripted"`` entry on ``ChatConfig.provider`` (see its docstring), so
  :func:`build_chat` never constructs this one — only direct construction
  does.
- :class:`RedactingChat` — a decorator satisfying ``ChatProtocol`` that
  wraps another ``ChatProtocol``, redacting ``prompt``/``system`` on the
  way in and restoring the completion on the way out (ADR-0017 decision
  4). A fresh :class:`~groundkit.providers.redaction.Redactor` is built on
  every :meth:`~RedactingChat.complete` call — see that class's docstring
  for why a shared, long-lived one would be a disclosure the mitigation
  itself manufactures.

:func:`build_chat` is the ``ChatConfig`` → conformer translation layer
(the ``build_embedder`` role): ``"ollama"`` returns a bare, unwrapped
:class:`OllamaChat` (local mode sends nothing anywhere, so there is
nothing to redact); ``"openai_compatible"`` always returns an
:class:`OpenAICompatChat` wrapped in :class:`RedactingChat` — never bare.
There is no operator-facing redaction opt-out for cloud egress (ADR-0017
decision 4): :func:`build_chat` decides, not a flag.

No cross-provider fallback, ever: an unconfigured or failing chat provider
is always a typed error (:class:`~groundkit.errors.ChatError` /
:class:`~groundkit.errors.ChatProviderNotConfiguredError`), never a silent
substitution (SPEC.md §2). Malformed provider output — a non-JSON-object
body, a missing or non-string completion field, or an empty/whitespace-only
completion — is a rejection, never a coercion, for the same reason.
"""

from __future__ import annotations

import abc
import logging
import os
from collections.abc import Sequence
from typing import ClassVar, Final, NoReturn

import httpx

from groundkit.config import DEFAULT_CHAT_MODEL, DEFAULT_OLLAMA_BASE_URL, ChatConfig
from groundkit.errors import ChatError, ChatProviderNotConfiguredError, ConfigurationError
from groundkit.providers.embeddings import _error_detail, _post_json, _sanitize_url
from groundkit.providers.protocols import ChatProtocol
from groundkit.providers.redaction import RedactionConfig, Redactor
from groundkit.utils.url_safety import ensure_safe_endpoint, validate_endpoint_shape

logger = logging.getLogger(__name__)

#: Default per-request timeout, seconds. This module's own default for
#: direct construction; ``ChatConfig.timeout_seconds`` (double the value,
#: 60.0) is the canonical one :func:`build_chat` actually uses — chat
#: completions legitimately run longer than an embedding batch, so the two
#: defaults are allowed to differ.
DEFAULT_TIMEOUT_SECONDS: Final[float] = 30.0

#: Default env var name holding the OpenAI-compatible API key. Same
#: default as ``EmbeddingConfig.api_key_env``: the common case is one
#: provider account serving both embeddings and chat.
DEFAULT_OPENAI_CHAT_API_KEY_ENV: Final[str] = "GROUNDKIT_OPENAI_API_KEY"

#: Provider identity of the labeled offline test double below. Public and
#: referenced by name for the same reason
#: :data:`~groundkit.providers.embeddings.INMEMORY_PROVIDER` is: a bare
#: ``"scripted"`` string literal repeated at each recognition call site is
#: a check that silently stops matching the day the identity changes.
SCRIPTED_PROVIDER: Final[str] = "scripted"


def _build_messages(prompt: str, *, system: str | None) -> list[dict[str, str]]:
    """Build a chat ``messages`` array for :class:`OpenAICompatChat`.

    :class:`OllamaChat` needs no equivalent: ``/api/generate`` takes
    ``prompt``/``system`` as top-level fields directly (ADR-0017 decision
    2), so this helper exists only because the OpenAI-compatible wire
    format has no such shortcut — a messages array is unavoidable there.

    Args:
        prompt: The user prompt.
        system: An optional system instruction, prepended as its own
            message when given.

    Returns:
        ``[{"role": "system", ...}, {"role": "user", ...}]`` when
        ``system`` is given, else just the user message.
    """
    messages: list[dict[str, str]] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def _reject_if_empty(text: str, *, provider: str, model_name: str) -> str:
    """Return ``text`` unchanged, or raise if it is empty or whitespace-only.

    Shared by every conformer in this module (SPEC.md §2: rejection, never
    coercion) so "what counts as an empty completion" cannot drift between
    the two HTTP providers and the scripted test double.

    Raises:
        ChatError: ``text`` is empty or contains only whitespace.
    """
    if not text.strip():
        raise ChatError(
            f"{provider} chat completion was empty or whitespace-only "
            f"(model {model_name!r}); refusing to return it"
        )
    return text


def _raise_chat_error(url: str, exc: Exception, *, secret: str | None) -> NoReturn:
    """Raise a scrubbed, chain-severed :class:`~groundkit.errors.ChatError` for *exc*.

    The :class:`~groundkit.errors.ChatError` analog of
    ``groundkit.providers.embeddings._raise_embedding_error`` — same
    contract, and it reuses that module's own
    :func:`~groundkit.providers.embeddings._sanitize_url` and
    :func:`~groundkit.providers.embeddings._scrub` rather than
    reimplementing them (ADR-0001 hazard 6). Must be called from a frame
    with no exception actively being handled (i.e. after ``exc`` has
    already been caught and returned as a value by
    :func:`~groundkit.providers.embeddings._post_json`), which is what
    keeps the raised error's ``__context__`` empty; the explicit
    ``from None`` keeps ``__cause__`` empty too.

    Args:
        url: The request URL, for the error message. Sanitized before it
            reaches any log line or the raised message.
        exc: The exception raised while sending or decoding the request.
        secret: The credential to scrub out of the message, or ``None``.

    Raises:
        ChatError: Always.
    """
    safe_url = _sanitize_url(url, secret)
    detail = _error_detail(exc, secret)
    logger.debug("Chat request to %s failed: %s", safe_url, detail)
    raise ChatError(f"Chat request to {safe_url} failed: {detail}") from None


def _parse_ollama_generate(data: object) -> str:
    """Parse Ollama's ``POST /api/generate`` (non-streaming) response body.

    Args:
        data: Decoded JSON response body.

    Returns:
        The generated text (``data["response"]``).

    Raises:
        ChatError: The payload is not a JSON object, or has no string
            ``response`` field.
    """
    if not isinstance(data, dict):
        raise ChatError(
            f"Ollama generate response was not a JSON object (got {type(data).__name__})"
        )
    response = data.get("response")
    if not isinstance(response, str):
        raise ChatError("Ollama generate response has no string 'response' field")
    return response


def _parse_openai_chat(data: object) -> str:
    """Parse an OpenAI-compatible ``POST /v1/chat/completions`` response body.

    Args:
        data: Decoded JSON response body.

    Returns:
        The first choice's message ``content`` string.

    Raises:
        ChatError: The payload is not a JSON object, ``choices`` is
            missing, not a list, or empty, the first choice is not a JSON
            object, it has no ``message`` object, or that message has no
            string ``content``.
    """
    if not isinstance(data, dict):
        raise ChatError(
            f"OpenAI-compatible chat response was not a JSON object (got {type(data).__name__})"
        )
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ChatError("OpenAI-compatible chat response has no non-empty 'choices' list")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ChatError("OpenAI-compatible chat response's first choice was not a JSON object")
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ChatError("OpenAI-compatible chat response's first choice has no 'message' object")
    content = message.get("content")
    if not isinstance(content, str):
        raise ChatError(
            "OpenAI-compatible chat response's first choice message has no string 'content'"
        )
    return content


class _HttpChat(abc.ABC):
    """Shared endpoint-safety and lifecycle machinery for HTTP-backed chat providers.

    Subclasses implement :attr:`provider` and :meth:`_request_completion`;
    this class owns validating the endpoint shape at construction, holding
    (or building) the HTTP client, and rejecting an empty/whitespace-only
    completion uniformly across every HTTP conformer
    (:func:`_reject_if_empty`).

    Args:
        base_url: Provider endpoint (no trailing path — request paths are
            concatenated onto it).
        model_name: Model identifier for the provider.
        timeout: Per-request timeout, seconds.
        client: Optional pre-built ``httpx.AsyncClient``. This is the seam
            tests replace: inject a client wired to ``httpx.MockTransport``
            to exercise the full request/parse/error path with zero
            network access. When omitted, a real client is built with the
            configured timeout.

    Raises:
        ConfigurationError: ``base_url``'s shape is invalid — wrong
            scheme, no host, userinfo, or a query/fragment
            (:func:`~groundkit.utils.url_safety.validate_endpoint_shape`,
            ADR-0014 decision 10).
    """

    #: Whether this provider is exempted from
    #: :func:`~groundkit.utils.url_safety.ensure_safe_endpoint`'s
    #: loopback/private refusal. A ``ClassVar``, not a constructor
    #: keyword — see ``_HttpEmbedder``'s identical attribute in
    #: ``providers/embeddings.py`` for why that shape is deliberate. Does
    #: not relax :func:`~groundkit.utils.url_safety.validate_endpoint_shape`,
    #: which runs identically for every subclass.
    _allow_private_endpoint: ClassVar[bool] = False

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        validate_endpoint_shape(base_url)
        self._base_url = base_url
        self._model_name = model_name
        self._timeout = timeout
        if client is not None:
            self._client = client
        else:
            # follow_redirects=False mirrors _HttpEmbedder's hardening: an
            # unfollowed redirect already raises via raise_for_status().
            self._client = httpx.AsyncClient(timeout=timeout, follow_redirects=False)

    @property
    @abc.abstractmethod
    def provider(self) -> str:
        """Provider identity."""

    @property
    def model_name(self) -> str:
        """Configured model identifier."""
        return self._model_name

    async def aclose(self) -> None:
        """Release the underlying HTTP client's resources."""
        await self._client.aclose()

    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        """Complete ``prompt``, optionally under a ``system`` instruction.

        Args:
            prompt: The user prompt.
            system: An optional system instruction.

        Returns:
            The provider's completion text.

        Raises:
            ChatError: On request failure, an unparseable response, or an
                empty/whitespace-only completion.
            ChatProviderNotConfiguredError: A required credential is unset.
            ConfigurationError: The endpoint's resolved address is not
                permitted — loopback, private, link-local, multicast,
                reserved, unspecified, or RFC6598 shared address space,
                unless this provider is exempted
                (:func:`~groundkit.utils.url_safety.ensure_safe_endpoint`).
                Checked per request, immediately before the POST.
        """
        text = await self._request_completion(prompt, system=system)
        return _reject_if_empty(text, provider=self.provider, model_name=self._model_name)

    @abc.abstractmethod
    async def _request_completion(self, prompt: str, *, system: str | None) -> str:
        """Send one completion request and return the raw completion text."""


class OllamaChat(_HttpChat):
    """Completes via Ollama's non-streaming ``/api/generate`` endpoint.

    POSTs ``{base_url}/api/generate`` with ``{"model": model_name, "prompt":
    prompt, "stream": False}`` — plus ``"system": system`` only when
    ``system`` is given — and parses ``{"response": "..."}``.
    ``/api/generate`` rather than ``/api/chat`` is a deliberate choice
    (ADR-0017 decision 2; ``docs/specs/phase-5-boundary-features.md`` §3):
    its own ``prompt``/``system`` fields map directly onto this class's
    ``complete`` parameters, so there is no messages array to build on the
    way in or unwrap on the way out. ``stream: False`` is load-bearing, not
    cosmetic: Ollama's default response is newline-delimited JSON chunks,
    which :func:`~groundkit.providers.embeddings._post_json`'s single
    ``response.json()`` call cannot parse. No credential is used, sent, or
    read.

    Satisfies :class:`~groundkit.providers.protocols.ChatProtocol`.
    """

    #: Ollama is the local-first default (SPEC.md §7's one named exception
    #: to the outbound SSRF guard) — see ``OllamaEmbedder``'s identical
    #: attribute in ``providers/embeddings.py``.
    _allow_private_endpoint: ClassVar[bool] = True

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        model_name: str = DEFAULT_CHAT_MODEL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(base_url=base_url, model_name=model_name, timeout=timeout, client=client)

    @property
    def provider(self) -> str:
        """Provider identity."""
        return "ollama"

    async def _request_completion(self, prompt: str, *, system: str | None) -> str:
        url = f"{self._base_url}/api/generate"
        await ensure_safe_endpoint(url, allow_private_endpoint=type(self)._allow_private_endpoint)
        payload: dict[str, object] = {
            "model": self._model_name,
            "prompt": prompt,
            "stream": False,
        }
        if system is not None:
            payload["system"] = system
        result = await _post_json(self._client, url, payload, headers=None)
        if isinstance(result, Exception):
            _raise_chat_error(url, result, secret=None)
        return _parse_ollama_generate(result)


class OpenAICompatChat(_HttpChat):
    """Completes via an OpenAI-compatible ``/v1/chat/completions`` endpoint.

    POSTs ``{base_url}/v1/chat/completions`` with ``{"model": model_name,
    "messages": [...]}`` and an ``Authorization: Bearer <key>`` header,
    where the key is read from ``os.environ[api_key_env]`` **at call
    time** — never stored, logged, or accepted as the key value itself.
    Parses ``{"choices": [{"message": {"content": ...}}]}``, reading only
    the first choice.

    Satisfies :class:`~groundkit.providers.protocols.ChatProtocol`.

    Args:
        base_url: Provider endpoint. No local-first default exists for a
            cloud provider, so this is required.
        model_name: Model identifier. Required for the same reason.
        api_key_env: Name of the environment variable holding the API key.
            Read at call time, never stored or logged.
        timeout: Per-request timeout, seconds.
        client: Optional pre-built ``httpx.AsyncClient`` (test seam).

    Raises:
        ChatProviderNotConfiguredError: ``api_key_env`` is unset or empty
            when :meth:`~_HttpChat.complete` is called. Never falls back
            to another provider (SPEC.md §2).
    """

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        api_key_env: str = DEFAULT_OPENAI_CHAT_API_KEY_ENV,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(base_url=base_url, model_name=model_name, timeout=timeout, client=client)
        self._api_key_env = api_key_env

    @property
    def provider(self) -> str:
        """Provider identity."""
        return "openai_compatible"

    async def _request_completion(self, prompt: str, *, system: str | None) -> str:
        api_key = self._resolve_api_key()
        url = f"{self._base_url}/v1/chat/completions"
        await ensure_safe_endpoint(url, allow_private_endpoint=type(self)._allow_private_endpoint)
        payload: dict[str, object] = {
            "model": self._model_name,
            "messages": _build_messages(prompt, system=system),
        }
        headers = {"Authorization": f"Bearer {api_key}"}
        result = await _post_json(self._client, url, payload, headers=headers)
        if isinstance(result, Exception):
            _raise_chat_error(url, result, secret=api_key)
        return _parse_openai_chat(result)

    def _resolve_api_key(self) -> str:
        """Read the API key from the environment at call time.

        Raises:
            ChatProviderNotConfiguredError: The configured env var is
                unset or empty. The message names the variable, never a
                value.
        """
        env_var = self._api_key_env
        api_key = os.getenv(env_var)
        if not api_key:
            raise ChatProviderNotConfiguredError(
                f"{env_var} is not set; it is required for chat provider "
                f"'openai_compatible' (model {self._model_name!r})"
            )
        return api_key


class ScriptedChatProvider:
    """Plays back a fixed script of completions — a labeled TEST DOUBLE.

    The :class:`~groundkit.providers.embeddings.InMemoryEmbedder` analog
    for chat: it calls no model and carries no semantic behavior. Each
    :meth:`complete` call returns the next entry in ``script``, in order,
    regardless of what ``prompt``/``system`` were given. It exists solely
    so retrieval-adjacent code that calls a
    :class:`~groundkit.providers.protocols.ChatProtocol` (query rewrite,
    synthesis) can be exercised deterministically and offline in tests and
    local dev — the exact role ``InMemoryEmbedder`` plays for embeddings.
    Never treat a quality number derived from its output as meaningful
    (SPEC.md §2): there is no model behind it to measure.

    Fails closed rather than cycling: once every scripted completion has
    been consumed, the next call raises :class:`~groundkit.errors.ChatError`
    instead of silently repeating the script or returning something empty.
    A caller asserting on call count would otherwise get a completion from
    a call it never provisioned a script entry for, with no signal that
    anything is wrong.

    Satisfies :class:`~groundkit.providers.protocols.ChatProtocol`.

    Args:
        script: Completions to return, in call order. Copied at
            construction, so later mutating the caller's sequence has no
            effect on this instance.
    """

    def __init__(self, script: Sequence[str]) -> None:
        self._script: list[str] = list(script)
        self._index = 0

    @property
    def provider(self) -> str:
        """Provider identity for the scripted playback test double."""
        return SCRIPTED_PROVIDER

    @property
    def model_name(self) -> str:
        """Model identity for the scripted playback test double."""
        return "scripted-v1"

    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        """Return the next scripted completion, ignoring ``prompt``/``system``.

        Args:
            prompt: Accepted for protocol conformance; playback does not
                depend on it.
            system: Accepted for protocol conformance; playback does not
                depend on it.

        Returns:
            The next entry in ``script``.

        Raises:
            ChatError: The script is already exhausted, or the next entry
                is empty or whitespace-only.
        """
        logger.debug(
            "scripted chat provider ignoring prompt (%d chars, system_set=%s) at index %d",
            len(prompt),
            system is not None,
            self._index,
        )
        if self._index >= len(self._script):
            raise ChatError(
                f"ScriptedChatProvider's script is exhausted after {len(self._script)} "
                "completion(s); refusing to cycle back to the start"
            )
        text = self._script[self._index]
        self._index += 1
        return _reject_if_empty(text, provider=self.provider, model_name=self.model_name)


class RedactingChat:
    """Wraps a :class:`~groundkit.providers.protocols.ChatProtocol`, redacting
    outbound text and restoring inbound text (ADR-0017 decision 4).

    Each :meth:`complete` call redacts ``prompt`` (and ``system``, when
    given) through a fresh :class:`~groundkit.providers.redaction.Redactor`,
    forwards the redacted text to ``inner``, and restores every token that
    *this call's* redactor issued out of the returned completion before
    handing it back.

    **A fresh ``Redactor`` per call is load-bearing, not tidiness.** Tokens
    are stable per distinct value *within one Redactor instance* — that
    stability is what :meth:`~groundkit.providers.redaction.Redactor.restore`
    depends on to invert them. A ``RedactingChat`` holding one long-lived
    redactor across many calls would let that same property work against
    it: ``restore()`` cannot tell which call a token came from, so a
    completion returned by call B that happens to contain (or echo) a token
    minted during call A's redaction would have call A's original value
    substituted back into call B's output — a cross-request disclosure the
    mitigation itself would manufacture, not a pre-existing risk it failed
    to close. Constructing one instance per call makes that outcome
    unrepresentable: there is no shared state a later call could read from
    an earlier one.

    ``provider`` and ``model_name`` delegate to ``inner`` unchanged.
    ADR-0017 does not name an override for either, and the identity that
    actually produced a completion is ``inner``'s — this wrapper adds a
    security transformation, not a different model or backend, so a report
    reading ``provider``/``model_name`` off this object should see exactly
    what it would see through ``inner`` directly (the same argument that
    puts those two properties on the seam at all, per
    ``EmbeddingProtocol.dimensions``'s reasoning one layer up).

    Satisfies :class:`~groundkit.providers.protocols.ChatProtocol`.

    Args:
        inner: The wrapped chat provider. Never constructed here — always
            supplied by the caller (:func:`build_chat` is the one caller in
            this codebase today).
        redaction: The patterns applied on every call. A fresh
            :class:`~groundkit.providers.redaction.Redactor` is built from
            this config each time :meth:`complete` runs; the config itself
            is reused across calls (it is immutable), only the ``Redactor``
            instance is not.
    """

    def __init__(self, inner: ChatProtocol, redaction: RedactionConfig) -> None:
        self._inner = inner
        self._redaction = redaction

    @property
    def provider(self) -> str:
        """Provider identity, delegated to the wrapped provider unchanged."""
        return self._inner.provider

    @property
    def model_name(self) -> str:
        """Model identity, delegated to the wrapped provider unchanged."""
        return self._inner.model_name

    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        """Redact ``prompt``/``system``, delegate to ``inner``, restore the result.

        Args:
            prompt: The user prompt. Redacted before ``inner`` ever sees it.
            system: An optional system instruction. Redacted the same way
                when given; left ``None`` (never redacted into an empty
                string) when not.

        Returns:
            ``inner``'s completion, with every token this call's fresh
            ``Redactor`` issued restored to the original value it replaced.
            A token this call never issued (from a different
            ``RedactingChat``, a different call, or literal bracketed text
            the model produced on its own) is left exactly as ``inner``
            returned it when its category is unrecognized, or raises
            :class:`~groundkit.providers.redaction.UnknownRedactionTokenError`
            when its category is recognized but its specific counter is
            not — propagated unmodified; this wrapper does not catch or
            translate it.

        Raises:
            ChatError: Propagated unmodified from ``inner``.
            ChatProviderNotConfiguredError: Propagated unmodified from
                ``inner`` (a ``ChatError`` subclass; covered above).
            UnknownRedactionTokenError: See above — a
                :class:`~groundkit.errors.RedactionError`, propagated
                unmodified.
        """
        redactor = Redactor(self._redaction)
        redacted_prompt = redactor.redact(prompt).text
        redacted_system = redactor.redact(system).text if system is not None else None

        completion = await self._inner.complete(redacted_prompt, system=redacted_system)

        return redactor.restore(completion)

    async def aclose(self) -> None:
        """Close the wrapped provider's transport, if it holds one.

        Duck-typed exactly like ``cli._maybe_aclose``: ``aclose`` is not part
        of :class:`~groundkit.providers.protocols.ChatProtocol`, but the
        HTTP-backed conformers this wrapper exists to wrap do own a client,
        and hiding it behind the wrapper must not leak it.
        """
        aclose = getattr(self._inner, "aclose", None)
        if callable(aclose):
            await aclose()


def build_chat(config: ChatConfig, *, redaction: RedactionConfig | None = None) -> ChatProtocol:
    """Construct the chat provider named by ``config.provider``, wrapping for redaction as required.

    The ``ChatConfig`` → conformer translation layer, the same role
    :func:`~groundkit.providers.embeddings.build_embedder` plays for
    ``EmbeddingConfig``. Dispatch is total over
    :attr:`~groundkit.config.ChatConfig.provider`'s closed ``Literal``
    (``"ollama"`` or ``"openai_compatible"`` only — there is no
    ``"scripted"`` entry; see :class:`~groundkit.config.ChatConfig`'s
    docstring for why :class:`ScriptedChatProvider` is never reachable
    through this factory).

    ``"ollama"`` returns a bare, unwrapped :class:`OllamaChat`: local mode
    sends nothing anywhere (SPEC.md §7 / ADR-0017 decision 4), so there is
    nothing a redaction pass would protect and wrapping it would only add a
    ``Redactor`` pass to every call for no benefit.

    ``"openai_compatible"`` always returns an :class:`OpenAICompatChat`
    wrapped in :class:`RedactingChat` — **never bare**. There is no
    operator-facing ``--no-redaction`` escape hatch (ADR-0017 decision 4):
    a lenient mode is exactly what SPEC.md §2 forbids for a cloud
    provider's egress, so this factory decides, not a flag the operator
    could set.

    Args:
        config: Chat configuration.
        redaction: Patterns to apply on the ``openai_compatible`` path.
            Defaults to a fresh
            :class:`~groundkit.providers.redaction.RedactionConfig` over
            :data:`~groundkit.providers.redaction.DEFAULT_PATTERNS` when
            omitted. Ignored entirely for ``"ollama"``.

    Returns:
        The provider matching ``config.provider`` — wrapped in
        :class:`RedactingChat` for ``"openai_compatible"``, bare for
        ``"ollama"``.

    Raises:
        ConfigurationError: ``config.provider == "openai_compatible"`` and
            the effective redaction config (*redaction*, or the default
            when omitted) carries an explicitly empty ``patterns`` tuple.
            There is no redaction opt-out for cloud egress (ADR-0017
            decision 4): a cloud provider wrapped in a ``RedactingChat``
            with nothing configured to redact is the unredacted path
            wearing a redaction label, which is precisely the lenient mode
            that decision forbids — so it is refused here rather than
            silently constructed.
    """
    if config.provider == "ollama":
        return OllamaChat(
            base_url=config.base_url,
            model_name=config.model_name,
            timeout=config.timeout_seconds,
        )
    if config.provider == "openai_compatible":
        effective_redaction = redaction if redaction is not None else RedactionConfig()
        if not effective_redaction.patterns:
            raise ConfigurationError(
                "chat provider 'openai_compatible' requires at least one configured "
                "redaction pattern; an explicitly empty pattern set is the unredacted "
                "path wearing a redaction label, and there is no redaction opt-out for "
                "cloud egress"
            )
        inner = OpenAICompatChat(
            base_url=config.base_url,
            model_name=config.model_name,
            api_key_env=config.api_key_env,
            timeout=config.timeout_seconds,
        )
        return RedactingChat(inner, effective_redaction)
    # Unreachable: config.provider is a closed Literal enforced by Pydantic at
    # construction time. Kept as a fail-loud guard, not a silent pass-through.
    raise AssertionError(f"unhandled chat provider: {config.provider!r}")  # pragma: no cover

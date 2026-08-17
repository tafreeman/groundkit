"""Embedding provider interface and implementations (Phase 1: Ollama default;
OpenAI-compatible opt-in). Unconfigured provider raises a typed error.

Ported from ARP's ``agentic_v2/rag/embeddings.py`` per ADR-0001. Kept: the
lazy-seam injectable-client pattern ("this is the seam tests replace"),
deterministic hash-based :class:`InMemoryEmbedder`, dimension checking, and
credential scrubbing. Replaced: the LiteLLM provider layer is gone —
:class:`OllamaEmbedder` and :class:`OpenAICompatibleEmbedder` call their HTTP
APIs directly (SPEC.md §3, fewer deps, local-first). Hardened: credential
scrubbing now also severs the exception chain (``__cause__`` *and*
``__context__``), closing ARP KNOWN_LIMITATIONS §4.10 / ADR-0001 hazard 6 —
see :func:`_post_json` and :func:`_raise_embedding_error`. There is no
cross-provider fallback chain: mixed semantic spaces would corrupt a
persisted index silently (SPEC.md §2), so an unconfigured or failing provider
is always a typed error, never a substitution.

Hardened further per ADR-0014 decision 10 (Phase 4 makes ``base_url``
reachable from a network-facing caller for the first time): every
``_HttpEmbedder`` validates its configured endpoint's *shape* once at
construction (:func:`~groundkit.utils.url_safety.validate_endpoint_shape`)
and its resolved *address* once per request, immediately before the POST
(:func:`~groundkit.utils.url_safety.ensure_safe_endpoint`).
:class:`OllamaEmbedder` alone is exempted from the address check's
loopback/private refusal, via a class attribute rather than a constructor
parameter — see its ``_allow_private_endpoint``.
"""

from __future__ import annotations

import abc
import asyncio
import hashlib
import logging
import math
import os
import struct
from typing import ClassVar, Final, NoReturn

import httpx

from groundkit.config import EmbeddingConfig
from groundkit.errors import EmbeddingError, ProviderNotConfiguredError
from groundkit.providers.protocols import EmbeddingProtocol
from groundkit.utils.url_safety import (
    ensure_safe_endpoint,
    sanitize_url,
    validate_endpoint_shape,
)

logger = logging.getLogger(__name__)

#: Provider identity of the labeled offline test double below. Public and
#: referenced by name wherever code has to *recognize* it rather than merely
#: build it — the eval runner warns on it, and the CLI stamps a caveat on
#: any report produced with it (SPEC.md §2: a quality number from
#: hash-derived vectors is noise presented as a number). A bare ``"inmemory"``
#: string literal repeated at each of those call sites is a check that
#: silently stops matching the day the identity changes.
INMEMORY_PROVIDER: Final[str] = "inmemory"

#: Bytes of a hash digest consumed per generated float (one big-endian uint32).
_VECTOR_BYTE_STRIDE: Final[int] = 4

#: Exclusive upper bound of the uint32 range produced by ``struct.unpack``.
_UINT32_SPACE: Final[int] = 2**32

#: Replaces a scrubbed credential wherever it appears in a log line or
#: exception message. Named to avoid tripping hardcoded-credential linters
#: (the value itself holds no secret).
_REDACTED_PLACEHOLDER: Final[str] = "***"


class InMemoryEmbedder:
    """Deterministic SHA-256 hash-expansion embedder — a labeled TEST DOUBLE.

    This embedder has **ZERO semantic signal**. It hashes each input text and
    expands the digest into a float vector; textually or semantically similar
    inputs do not produce nearby vectors. It exists solely so retrieval
    plumbing (batching, indexing, scoring, citation resolution) can be
    exercised deterministically and offline in tests and local dev — the
    exact role ARP's ``InMemoryEmbedder`` played, and the exact mistake ARP's
    own KNOWN_LIMITATIONS §4.9 warns against: never treat this as a real
    embedder, and never let a factory path return it in production
    (:func:`build_embedder` only does so when ``config.provider ==
    "inmemory"`` is set explicitly).

    Satisfies :class:`EmbeddingProtocol`.

    Args:
        dimensions: Vector width. Must be positive.

    Raises:
        ValueError: If ``dimensions`` is not positive.
    """

    def __init__(self, dimensions: int = 384) -> None:
        if dimensions <= 0:
            raise ValueError(f"dimensions must be positive, got {dimensions}")
        self._dimensions = dimensions

    @property
    def provider(self) -> str:
        """Provider identity for the deterministic in-memory embedder."""
        return INMEMORY_PROVIDER

    @property
    def model_name(self) -> str:
        """Model identity for the deterministic in-memory embedder."""
        return "inmemory-hash-v1"

    @property
    def dimensions(self) -> int:
        """Configured vector width."""
        return self._dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Hash-expand each text into a deterministic vector.

        Args:
            texts: Strings to embed. An empty list short-circuits to ``[]``
                with no computation.

        Returns:
            One vector per input text, in input order, each of length
            :attr:`dimensions` with values in ``[-1.0, 1.0]``.
        """
        if not texts:
            return []
        return [self._hash_to_vector(text) for text in texts]

    def _hash_to_vector(self, text: str) -> list[float]:
        """Expand a chained SHA-256 digest into an L2-normalized float vector."""
        vector: list[float] = []
        seed = text.encode("utf-8")

        iteration = 0
        while len(vector) < self._dimensions:
            digest = hashlib.sha256(seed + struct.pack(">I", iteration)).digest()
            for offset in range(0, len(digest), _VECTOR_BYTE_STRIDE):
                if len(vector) >= self._dimensions:
                    break
                chunk = digest[offset : offset + _VECTOR_BYTE_STRIDE]
                uint_val = struct.unpack(">I", chunk)[0]
                vector.append((uint_val / _UINT32_SPACE) * 2.0 - 1.0)
            iteration += 1

        # L2-normalize. Every component's magnitude is <= the norm by
        # construction, so this keeps values within [-1.0, 1.0] rather than
        # merely producing unit length.
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector


async def _post_json(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, object],
    *,
    headers: dict[str, str] | None,
) -> object | Exception:
    """POST *payload* to *url* and return the decoded JSON body.

    Returns the caught exception as a value instead of raising it. The caller
    then scrubs any credential out of its message and raises a fresh
    :class:`EmbeddingError` from a frame with no exception currently being
    handled — so the new error carries no ``__context__`` and (with an
    explicit ``from None``) no ``__cause__`` either. An unscrubbed provider
    exception can never ride along on either chain (ADR-0001 hazard 6; ARP
    KNOWN_LIMITATIONS §4.10).

    Args:
        client: The HTTP client to send the request on.
        url: Full request URL.
        payload: JSON request body.
        headers: Extra request headers, or ``None``.

    Returns:
        The parsed JSON response body, or the exception that prevented it.
    """
    try:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data: object = response.json()
        return data
    except Exception as exc:  # deliberately broad: see docstring
        return exc


def _raise_embedding_error(url: str, exc: Exception, *, secret: str | None) -> NoReturn:
    """Raise a scrubbed, chain-severed :class:`EmbeddingError` for *exc*.

    Must be called from a frame with no exception actively being handled
    (i.e. after ``exc`` has already been caught and returned as a value by
    :func:`_post_json`, not from inside a live ``except`` block) — that is
    what keeps ``__context__`` empty. Logs the same scrubbed text and
    sanitized URL at debug level; never logs the unscrubbed exception or the
    raw URL.

    Args:
        url: The request URL, for the error message. Sanitized via
            :func:`_sanitize_url` before it reaches any log line or the
            raised message — ``EmbeddingConfig.base_url`` is free-form
            operator-controlled ``str`` and several OpenAI-compatible/proxy
            endpoints carry a credential in the query string (ADR-0001
            hazard 6).
        exc: The exception raised while sending or decoding the request.
        secret: The credential to scrub out of the message, or ``None``.

    Raises:
        EmbeddingError: Always.
    """
    safe_url = _sanitize_url(url, secret)
    detail = _error_detail(exc, secret)
    logger.debug("Embedding request to %s failed: %s", safe_url, detail)
    raise EmbeddingError(f"Embedding request to {safe_url} failed: {detail}") from None


def _error_detail(exc: Exception, secret: str | None) -> str:
    """Scrubbed one-line description of *exc* that is never empty.

    ``str()`` on several ``httpx`` timeout types -- ``ReadTimeout``,
    ``WriteTimeout``, ``PoolTimeout``, ``ConnectTimeout`` -- is the empty
    string. Interpolated directly, that rendered a provider timeout as
    ``"... failed: "`` with nothing after the colon: the scrubbing was correct
    and the message was still undiagnosable, which is the whole failure this
    helper exists to prevent. The fallback is the exception's class name,
    which carries the one fact the empty message was hiding and cannot itself
    contain a credential.

    Shared by :func:`_raise_embedding_error` and
    :func:`groundkit.providers.llm._raise_chat_error` so the two cannot drift
    -- the chat helper already reuses this module's :func:`_scrub` and
    :func:`_sanitize_url` for the same reason (ADR-0001 hazard 6).

    Args:
        exc: The exception raised while sending or decoding the request.
        secret: The credential to scrub out of the message, or ``None``.

    Returns:
        The scrubbed message, or ``type(exc).__name__`` when it is empty or
        whitespace-only.
    """
    detail = _scrub(str(exc), secret)
    return detail if detail.strip() else type(exc).__name__


def _scrub(text: str, secret: str | None) -> str:
    """Replace every occurrence of *secret* in *text* with a placeholder."""
    if not secret:
        return text
    return text.replace(secret, _REDACTED_PLACEHOLDER)


def _sanitize_url(url: str, secret: str | None) -> str:
    """Return *url* with query values and userinfo redacted, and *secret* scrubbed.

    Thin alias over :func:`groundkit.utils.url_safety.sanitize_url`, which is
    where this logic now lives: ADR-0016's URL ingestion needs exactly the same
    redaction on exactly the same hazard, and two copies of a
    credential-redaction routine is how one of them silently stops matching the
    other. Kept as a module-local name because every call site here reads
    better for it and the tests reference it.
    """
    return sanitize_url(url, secret)


def _coerce_vector(values: list[object], *, provider_label: str) -> list[float]:
    """Convert one provider-supplied vector to floats, rejecting anything unusable.

    The single validation point both parsers share. It used to be a bare
    ``[float(value) for value in item]``, which is a *coercion* where this
    repo's contract calls for a rejection (SPEC.md §2: malformed provider
    output is refused, never coerced), and it let three distinct kinds of
    garbage into a persisted index:

    - **Non-finite numbers.** ``json.loads`` accepts JavaScript's ``NaN``,
      ``Infinity`` and ``-Infinity`` literals by default — so does
      ``httpx``'s ``response.json()`` — and ``float()`` accepts the strings
      ``"NaN"`` / ``"Infinity"`` too, as well as any overflowing literal
      such as ``1e400``. A single ``NaN`` component makes every cosine
      similarity computed against that vector ``NaN``, which is neither
      greater nor less than anything: the affected chunk's rank becomes a
      function of sort order rather than relevance, and ``_clamp_score``
      passes it straight through to a ``RetrievalResult.score`` field whose
      ``ge=0.0`` bound ``NaN`` silently satisfies. ``Infinity`` is worse in
      one direction — it dominates every ranking it appears in.
    - **Numeric-looking strings.** ``float("0.5")`` succeeds, so a provider
      returning quoted numbers was accepted rather than reported.
    - **Booleans.** ``bool`` is a subclass of ``int``, so ``float(True)``
      is ``1.0`` and a JSON ``true`` entered the index as a legitimate
      component.

    None of this survives ``_check_dimensions``, which only counts
    components. The vector's *width* is right in every case above.

    Args:
        values: The raw JSON array the provider returned for one input.
        provider_label: Provider name for the error message.

    Returns:
        The vector as floats.

    Raises:
        EmbeddingError: A component is not a finite JSON number.
    """
    vector: list[float] = []
    for position, value in enumerate(values):
        # bool first: it is a subclass of int and would pass the isinstance
        # check below.
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise EmbeddingError(
                f"{provider_label} embedding response contained a non-numeric vector "
                f"value at position {position} (got {type(value).__name__}); refusing "
                "to coerce it into an index"
            )
        number = float(value)
        if not math.isfinite(number):
            raise EmbeddingError(
                f"{provider_label} embedding response contained a non-finite vector "
                f"value at position {position} ({number}); NaN and infinities corrupt "
                "every similarity computed against the vector, so it is refused rather "
                "than stored"
            )
        vector.append(number)
    return vector


def _parse_ollama_embeddings(data: object, *, expected_count: int) -> list[list[float]]:
    """Parse Ollama's ``POST /api/embed`` response body.

    Args:
        data: Decoded JSON response body.
        expected_count: Number of texts sent in this request.

    Returns:
        One vector per input text, in the order Ollama returned them (Ollama
        preserves request order in this endpoint; there is no per-item index).

    Raises:
        EmbeddingError: If the payload is not the expected shape, does not
            contain ``expected_count`` vectors, or a vector holds a value
            that is not a finite JSON number (:func:`_coerce_vector`).
    """
    if not isinstance(data, dict):
        raise EmbeddingError(
            f"Ollama embedding response was not a JSON object (got {type(data).__name__})"
        )
    embeddings = data.get("embeddings")
    if not isinstance(embeddings, list):
        raise EmbeddingError("Ollama embedding response has no 'embeddings' list")
    if len(embeddings) != expected_count:
        raise EmbeddingError(
            f"Ollama returned {len(embeddings)} embeddings for {expected_count} input texts"
        )

    vectors: list[list[float]] = []
    for item in embeddings:
        if not isinstance(item, list):
            raise EmbeddingError("Ollama embedding response contained a non-list vector")
        vectors.append(_coerce_vector(item, provider_label="Ollama"))
    return vectors


def _parse_openai_embeddings(data: object, *, expected_count: int) -> list[list[float]]:
    """Parse an OpenAI-compatible ``POST /v1/embeddings`` response body.

    Re-sorts by each item's ``index`` field so the returned vectors match
    input order even when the provider replies out of order.

    Args:
        data: Decoded JSON response body.
        expected_count: Number of texts sent in this request.

    Returns:
        One vector per input text, in input order.

    Raises:
        EmbeddingError: If the payload is not the expected shape, does not
            contain exactly one item per input index (0..expected_count-1),
            or an item holds a vector value that is not a finite JSON
            number (:func:`_coerce_vector`).
    """
    if not isinstance(data, dict):
        raise EmbeddingError(
            f"OpenAI-compatible embedding response was not a JSON object (got "
            f"{type(data).__name__})"
        )
    items = data.get("data")
    if not isinstance(items, list):
        raise EmbeddingError("OpenAI-compatible embedding response has no 'data' list")
    if len(items) != expected_count:
        raise EmbeddingError(
            f"OpenAI-compatible provider returned {len(items)} embeddings for "
            f"{expected_count} input texts"
        )

    indexed = [_parse_openai_item(item) for item in items]
    indexed.sort(key=lambda pair: pair[0])
    indices = [index for index, _ in indexed]
    if indices != list(range(expected_count)):
        raise EmbeddingError(
            f"OpenAI-compatible embedding response indices {indices} do not cover "
            f"the expected range 0..{expected_count - 1}"
        )
    return [vector for _, vector in indexed]


def _parse_openai_item(item: object) -> tuple[int, list[float]]:
    """Parse one element of an OpenAI-compatible ``data`` list.

    Raises:
        EmbeddingError: If the item is malformed or holds a vector value
            that is not a finite JSON number.
    """
    if not isinstance(item, dict):
        raise EmbeddingError("OpenAI-compatible embedding response item was not a JSON object")
    index = item.get("index")
    # `isinstance(True, int)` is True, so a JSON `true` would otherwise be
    # accepted here and reorder the batch as index 1 — the same coercion
    # trap _coerce_vector closes for the components themselves.
    if isinstance(index, bool) or not isinstance(index, int):
        raise EmbeddingError("OpenAI-compatible embedding response item has no integer 'index'")
    vector = item.get("embedding")
    if not isinstance(vector, list):
        raise EmbeddingError("OpenAI-compatible embedding response item has no 'embedding' list")
    return index, _coerce_vector(vector, provider_label="OpenAI-compatible")


class _HttpEmbedder(abc.ABC):
    """Shared batching, concurrency, and dimension-check machinery for
    HTTP-backed embedding providers.

    Subclasses implement :attr:`provider` and :meth:`_request_batch`; this
    class owns splitting the input into ``config.batch_size`` batches,
    bounding in-flight requests with ``asyncio.Semaphore(config.max_concurrent)``,
    reassembling results in input order, and rejecting any response vector
    whose width does not match ``config.dimensions``.

    Args:
        config: Embedding configuration (model, dimensions, base_url, batch
            size, concurrency limit, timeout).
        client: Optional pre-built ``httpx.AsyncClient``. This is the seam
            tests replace: inject a client wired to ``httpx.MockTransport`` to
            exercise the full request/parse/error path with zero network
            access. When omitted, a real client is built with the configured
            timeout.

    Raises:
        ConfigurationError: If ``config.base_url``'s shape is invalid —
            wrong scheme, no host, userinfo, or a query/fragment
            (:func:`~groundkit.utils.url_safety.validate_endpoint_shape`,
            ADR-0014 decision 10).
    """

    #: Whether this provider is exempted from
    #: :func:`~groundkit.utils.url_safety.ensure_safe_endpoint`'s
    #: loopback/private refusal. A ``ClassVar``, not a constructor keyword —
    #: a keyword would make requesting the exception for *any* provider a
    #: legal thing to write, which scopes the allowance over the guard
    #: rather than around one specific, named provider. Named for permitting
    #: private endpoints generally, not loopback specifically: SPEC.md §9's
    #: Phase 6 compose topology reaches Ollama at an RFC1918 bridge-network
    #: address, not just at 127.0.0.1. Does not relax
    #: :func:`~groundkit.utils.url_safety.validate_endpoint_shape` — that
    #: check runs identically for every subclass.
    _allow_private_endpoint: ClassVar[bool] = False

    def __init__(self, config: EmbeddingConfig, client: httpx.AsyncClient | None = None) -> None:
        validate_endpoint_shape(config.base_url)
        self._config = config
        if client is not None:
            self._client = client
        else:
            # follow_redirects=False is hardening, not a fix: an unfollowed
            # redirect already raises via raise_for_status() (verified by
            # execution against unmodified source, ADR-0014 decision 11) —
            # httpx's own current default. Pinned explicitly against a
            # future httpx default change, or an injected client that sets
            # it True.
            self._client = httpx.AsyncClient(timeout=config.timeout_seconds, follow_redirects=False)

    @property
    @abc.abstractmethod
    def provider(self) -> str:
        """Provider identity."""

    @property
    def model_name(self) -> str:
        """Configured model identifier."""
        return self._config.model_name

    @property
    def dimensions(self) -> int:
        """Configured embedding vector width."""
        return self._config.dimensions

    async def aclose(self) -> None:
        """Release the underlying HTTP client's resources."""
        await self._client.aclose()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed *texts* in configured batches, preserving input order.

        Args:
            texts: Strings to embed. An empty list short-circuits with no
                request.

        Returns:
            One vector per input text, in input order.

        Raises:
            EmbeddingError: On request failure, an unparseable response, or a
                response vector whose width does not match
                ``config.dimensions``.
            ProviderNotConfiguredError: If a required credential is unset.
            ConfigurationError: If the configured endpoint's resolved address
                is not permitted — loopback, private, link-local, multicast,
                reserved, unspecified, or RFC6598 shared address space,
                unless this provider is exempted
                (:func:`~groundkit.utils.url_safety.ensure_safe_endpoint`,
                ADR-0014 decision 10). Checked per request, immediately
                before the POST.
        """
        if not texts:
            return []

        batch_size = self._config.batch_size
        batches = [texts[start : start + batch_size] for start in range(0, len(texts), batch_size)]
        semaphore = asyncio.Semaphore(self._config.max_concurrent)

        batched_vectors = await asyncio.gather(
            *(self._embed_batch_bounded(batch, semaphore) for batch in batches)
        )
        return [vector for batch_vectors in batched_vectors for vector in batch_vectors]

    async def _embed_batch_bounded(
        self, batch: list[str], semaphore: asyncio.Semaphore
    ) -> list[list[float]]:
        """Run :meth:`_request_batch` under the concurrency bound, then validate widths."""
        async with semaphore:
            vectors = await self._request_batch(batch)
        self._check_dimensions(vectors)
        return vectors

    def _check_dimensions(self, vectors: list[list[float]]) -> None:
        """Fail loud rather than index a response vector of the wrong width."""
        expected = self._config.dimensions
        for vector in vectors:
            if len(vector) != expected:
                raise EmbeddingError(
                    f"{self.provider} returned a {len(vector)}-dimensional embedding "
                    f"but {expected} dimensions are configured (model "
                    f"{self._config.model_name!r}); storing a mismatched vector "
                    "would corrupt the index"
                )

    @abc.abstractmethod
    async def _request_batch(self, batch: list[str]) -> list[list[float]]:
        """Send one request for *batch* and return vectors in input order."""


class OllamaEmbedder(_HttpEmbedder):
    """Embeds via Ollama's batch embed endpoint.

    POSTs ``{base_url}/api/embed`` with ``{"model": model_name, "input":
    [...]}`` and parses ``{"embeddings": [[...], ...]}`` — Ollama's current
    batch embedding endpoint (as opposed to the older, singular
    ``/api/embeddings`` + ``"prompt"`` endpoint, which is not implemented
    here). No credential is used, sent, or read.

    Satisfies :class:`EmbeddingProtocol`.
    """

    #: Ollama is the local-first default (SPEC.md §7's one named exception to
    #: the outbound SSRF guard) and its address must resolve to a
    #: loopback/private target to be reachable at all — see the class
    #: attribute's docstring on ``_HttpEmbedder``.
    _allow_private_endpoint: ClassVar[bool] = True

    @property
    def provider(self) -> str:
        """Provider identity."""
        return "ollama"

    async def _request_batch(self, batch: list[str]) -> list[list[float]]:
        url = f"{self._config.base_url}/api/embed"
        await ensure_safe_endpoint(url, allow_private_endpoint=type(self)._allow_private_endpoint)
        payload: dict[str, object] = {"model": self._config.model_name, "input": list(batch)}
        result = await _post_json(self._client, url, payload, headers=None)
        if isinstance(result, Exception):
            _raise_embedding_error(url, result, secret=None)
        return _parse_ollama_embeddings(result, expected_count=len(batch))


class OpenAICompatibleEmbedder(_HttpEmbedder):
    """Embeds via an OpenAI-compatible ``/v1/embeddings`` endpoint.

    POSTs ``{base_url}/v1/embeddings`` with ``{"model": model_name, "input":
    [...]}`` and an ``Authorization: Bearer <key>`` header, where the key is
    read from ``os.environ[config.api_key_env]`` **at call time** — never
    stored, logged, or accepted as a constructor argument. Parses
    ``{"data": [{"index": i, "embedding": [...]}]}``, re-sorting by ``index``
    so the returned vectors match input order even when the provider replies
    out of order.

    Satisfies :class:`EmbeddingProtocol`.

    Raises:
        ProviderNotConfiguredError: If ``config.api_key_env`` is unset or
            empty when :meth:`~_HttpEmbedder.embed` is called. Never falls
            back to another provider (SPEC.md §2).
    """

    @property
    def provider(self) -> str:
        """Provider identity."""
        return "openai_compatible"

    async def _request_batch(self, batch: list[str]) -> list[list[float]]:
        api_key = self._resolve_api_key()
        url = f"{self._config.base_url}/v1/embeddings"
        await ensure_safe_endpoint(url, allow_private_endpoint=type(self)._allow_private_endpoint)
        payload: dict[str, object] = {"model": self._config.model_name, "input": list(batch)}
        headers = {"Authorization": f"Bearer {api_key}"}
        result = await _post_json(self._client, url, payload, headers=headers)
        if isinstance(result, Exception):
            _raise_embedding_error(url, result, secret=api_key)
        return _parse_openai_embeddings(result, expected_count=len(batch))

    def _resolve_api_key(self) -> str:
        """Read the API key from the environment at call time.

        Raises:
            ProviderNotConfiguredError: If the configured env var is unset or
                empty. The message names the variable, never a value.
        """
        env_var = self._config.api_key_env
        api_key = os.getenv(env_var)
        if not api_key:
            raise ProviderNotConfiguredError(
                f"{env_var} is not set; it is required for embedding provider "
                f"'openai_compatible' (model {self._config.model_name!r})"
            )
        return api_key


def build_embedder(config: EmbeddingConfig) -> EmbeddingProtocol:
    """Construct the embedding provider named by ``config.provider``.

    Dispatch is total over :attr:`EmbeddingConfig.provider`'s closed
    ``Literal`` — every branch is a real provider; there is no cross-provider
    fallback (SPEC.md §2). ``"inmemory"`` returns a labeled test double: pass
    it only when a caller has explicitly opted in for tests or local
    development, never as a production default.

    Args:
        config: Embedding configuration.

    Returns:
        The provider matching ``config.provider``.
    """
    if config.provider == "inmemory":
        return InMemoryEmbedder(dimensions=config.dimensions)
    if config.provider == "ollama":
        return OllamaEmbedder(config)
    if config.provider == "openai_compatible":
        return OpenAICompatibleEmbedder(config)
    # Unreachable: config.provider is a closed Literal enforced by Pydantic at
    # construction time. Kept as a fail-loud guard, not a silent pass-through.
    raise AssertionError(f"unhandled embedding provider: {config.provider!r}")  # pragma: no cover

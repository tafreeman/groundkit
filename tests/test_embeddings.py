"""Embedding provider tests — network-free via injected httpx clients.

HTTP providers are always constructed with a client wired to
``httpx.MockTransport``; no test ever touches the network. pytest-asyncio is
not part of this repo's dependency set, so async code under test is driven
with ``asyncio.run()`` inside plain ``def`` test functions.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Final

import httpx
import pytest
from pydantic import ValidationError

from groundkit.config import EmbeddingConfig
from groundkit.errors import EmbeddingError, ProviderNotConfiguredError
from groundkit.providers.embeddings import (
    InMemoryEmbedder,
    OllamaEmbedder,
    OpenAICompatibleEmbedder,
    build_embedder,
)
from groundkit.providers.protocols import EmbeddingProtocol

Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: Handler) -> httpx.AsyncClient:
    """Build an httpx.AsyncClient wired to a MockTransport — no network."""
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ── InMemoryEmbedder ──────────────────────────────────────────────────────


class TestInMemoryEmbedder:
    def test_satisfies_embedding_protocol(self) -> None:
        assert isinstance(InMemoryEmbedder(), EmbeddingProtocol)

    def test_provider_and_model_identity(self) -> None:
        embedder = InMemoryEmbedder()
        assert embedder.provider == "inmemory"
        assert embedder.model_name == "inmemory-hash-v1"

    def test_rejects_non_positive_dimensions(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            InMemoryEmbedder(dimensions=0)

    def test_deterministic_across_calls(self) -> None:
        embedder = InMemoryEmbedder(dimensions=32)
        first = asyncio.run(embedder.embed(["hello world"]))
        second = asyncio.run(embedder.embed(["hello world"]))
        assert first == second

    def test_different_inputs_produce_different_vectors(self) -> None:
        embedder = InMemoryEmbedder(dimensions=32)
        result = asyncio.run(embedder.embed(["hello", "goodbye"]))
        assert result[0] != result[1]

    def test_respects_custom_dimensions(self) -> None:
        embedder = InMemoryEmbedder(dimensions=128)
        result = asyncio.run(embedder.embed(["test"]))
        assert len(result[0]) == 128
        assert embedder.dimensions == 128

    def test_values_are_bounded(self) -> None:
        embedder = InMemoryEmbedder(dimensions=16)
        result = asyncio.run(embedder.embed(["a", "bb", "ccc"]))
        for vector in result:
            assert all(-1.0 <= value <= 1.0 for value in vector)

    def test_empty_input_short_circuits(self) -> None:
        embedder = InMemoryEmbedder()
        assert asyncio.run(embedder.embed([])) == []

    def test_dimensions_not_a_multiple_of_the_digest_stride(self) -> None:
        """32-byte SHA-256 digests yield 8 floats per hash iteration; a width
        that isn't a multiple of 8 must still land exactly on ``dimensions``,
        exercising the inner-loop early break."""
        embedder = InMemoryEmbedder(dimensions=10)
        result = asyncio.run(embedder.embed(["test"]))
        assert len(result[0]) == 10


# ── OllamaEmbedder ────────────────────────────────────────────────────────


class TestOllamaEmbedder:
    def test_request_shape_and_response_parsing(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            body = json.loads(request.content)
            assert body == {"model": "nomic-embed-text", "input": ["hello", "world"]}
            return httpx.Response(200, json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]})

        config = EmbeddingConfig(provider="ollama", model_name="nomic-embed-text", dimensions=2)
        embedder = OllamaEmbedder(config, client=_client(handler))

        result = asyncio.run(embedder.embed(["hello", "world"]))

        assert result == [[0.1, 0.2], [0.3, 0.4]]
        assert len(requests) == 1
        assert requests[0].url.path == "/api/embed"

    def test_no_authorization_header_is_sent(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert "authorization" not in request.headers
            return httpx.Response(200, json={"embeddings": [[0.0, 0.0]]})

        config = EmbeddingConfig(provider="ollama", dimensions=2)
        embedder = OllamaEmbedder(config, client=_client(handler))
        asyncio.run(embedder.embed(["x"]))

    def test_dimension_mismatch_raises(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3]]})

        config = EmbeddingConfig(provider="ollama", dimensions=2)
        embedder = OllamaEmbedder(config, client=_client(handler))

        with pytest.raises(EmbeddingError, match="dimension"):
            asyncio.run(embedder.embed(["x"]))

    def test_malformed_response_raises(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"unexpected": "shape"})

        config = EmbeddingConfig(provider="ollama", dimensions=2)
        embedder = OllamaEmbedder(config, client=_client(handler))

        with pytest.raises(EmbeddingError, match="embeddings"):
            asyncio.run(embedder.embed(["x"]))

    def test_wrong_vector_count_raises(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"embeddings": [[0.1, 0.2]]})

        config = EmbeddingConfig(provider="ollama", dimensions=2)
        embedder = OllamaEmbedder(config, client=_client(handler))

        with pytest.raises(EmbeddingError, match="2 input texts"):
            asyncio.run(embedder.embed(["x", "y"]))

    def test_transport_failure_is_a_scrubbed_embedding_error(self) -> None:
        """No credential exists for Ollama, but a transport-level failure must
        still surface as a plain EmbeddingError via the same wrap-and-sever
        path the credentialed provider uses."""

        def handler(_request: httpx.Request) -> httpx.Response:
            raise RuntimeError("connection refused")

        config = EmbeddingConfig(provider="ollama", dimensions=2)
        embedder = OllamaEmbedder(config, client=_client(handler))

        with pytest.raises(EmbeddingError, match="connection refused"):
            asyncio.run(embedder.embed(["x"]))

    @pytest.mark.parametrize(
        "response_body",
        [
            pytest.param("not-a-dict-at-all", id="non_dict_response"),
            pytest.param({"embeddings": "not-a-list"}, id="embeddings_not_a_list"),
            pytest.param({"embeddings": ["not-a-list"]}, id="vector_not_a_list"),
            pytest.param({"embeddings": [["x", "y"]]}, id="non_numeric_value"),
        ],
    )
    def test_malformed_shapes_raise_embedding_error(self, response_body: object) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=response_body)

        config = EmbeddingConfig(provider="ollama", dimensions=2)
        embedder = OllamaEmbedder(config, client=_client(handler))

        with pytest.raises(EmbeddingError):
            asyncio.run(embedder.embed(["x"]))

    def test_batches_by_batch_size_and_preserves_order(self) -> None:
        seen_batches: list[list[str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            batch = body["input"]
            seen_batches.append(batch)
            vectors = [[float(len(text)), 0.0] for text in batch]
            return httpx.Response(200, json={"embeddings": vectors})

        config = EmbeddingConfig(provider="ollama", dimensions=2, batch_size=2, max_concurrent=2)
        embedder = OllamaEmbedder(config, client=_client(handler))

        texts = ["a", "bb", "ccc", "dddd", "eeeee"]
        result = asyncio.run(embedder.embed(texts))

        # 5 texts at batch_size=2 -> 3 requests. Sort by first element before
        # comparing: concurrent batches may reach the handler in any order,
        # but each batch's own contents are deterministic from the slicing.
        assert len(seen_batches) == 3
        assert sorted(seen_batches, key=lambda batch: batch[0]) == [
            ["a", "bb"],
            ["ccc", "dddd"],
            ["eeeee"],
        ]
        # Final reassembly must match input order regardless of completion order.
        assert result == [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]]

    def test_empty_input_sends_no_request(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("no request should be sent for an empty input list")

        config = EmbeddingConfig(provider="ollama", dimensions=2)
        embedder = OllamaEmbedder(config, client=_client(handler))
        assert asyncio.run(embedder.embed([])) == []


# ── OpenAICompatibleEmbedder ──────────────────────────────────────────────


class TestOpenAICompatibleEmbedder:
    def test_auth_header_uses_env_var_read_at_call_time(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GROUNDKIT_OPENAI_API_KEY", "sk-test-value")
        seen_headers: list[str | None] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_headers.append(request.headers.get("authorization"))
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1, 0.2]}]})

        config = EmbeddingConfig(provider="openai_compatible", dimensions=2)
        embedder = OpenAICompatibleEmbedder(config, client=_client(handler))

        asyncio.run(embedder.embed(["x"]))

        assert seen_headers == ["Bearer sk-test-value"]

    def test_missing_key_raises_provider_not_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GROUNDKIT_OPENAI_API_KEY", raising=False)

        def handler(_request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("no request should be sent without a configured key")

        config = EmbeddingConfig(provider="openai_compatible", dimensions=2)
        embedder = OpenAICompatibleEmbedder(config, client=_client(handler))

        with pytest.raises(ProviderNotConfiguredError, match="GROUNDKIT_OPENAI_API_KEY"):
            asyncio.run(embedder.embed(["x"]))

    def test_empty_key_raises_provider_not_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GROUNDKIT_OPENAI_API_KEY", "")

        def handler(_request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("no request should be sent with an empty key")

        config = EmbeddingConfig(provider="openai_compatible", dimensions=2)
        embedder = OpenAICompatibleEmbedder(config, client=_client(handler))

        with pytest.raises(ProviderNotConfiguredError, match="GROUNDKIT_OPENAI_API_KEY"):
            asyncio.run(embedder.embed(["x"]))

    def test_custom_api_key_env_var_is_honored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CUSTOM_KEY_VAR", "sk-custom")

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers.get("authorization") == "Bearer sk-custom"
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1, 0.2]}]})

        config = EmbeddingConfig(
            provider="openai_compatible", dimensions=2, api_key_env="CUSTOM_KEY_VAR"
        )
        embedder = OpenAICompatibleEmbedder(config, client=_client(handler))
        asyncio.run(embedder.embed(["x"]))

    def test_reorders_out_of_order_index(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GROUNDKIT_OPENAI_API_KEY", "sk-test")

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": 1, "embedding": [2.0, 2.0]},
                        {"index": 0, "embedding": [1.0, 1.0]},
                    ]
                },
            )

        config = EmbeddingConfig(provider="openai_compatible", dimensions=2)
        embedder = OpenAICompatibleEmbedder(config, client=_client(handler))

        result = asyncio.run(embedder.embed(["first", "second"]))

        assert result == [[1.0, 1.0], [2.0, 2.0]]

    def test_dimension_mismatch_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GROUNDKIT_OPENAI_API_KEY", "sk-test")

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 2.0, 3.0]}]})

        config = EmbeddingConfig(provider="openai_compatible", dimensions=2)
        embedder = OpenAICompatibleEmbedder(config, client=_client(handler))

        with pytest.raises(EmbeddingError, match="dimension"):
            asyncio.run(embedder.embed(["x"]))

    def test_wrong_vector_count_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GROUNDKIT_OPENAI_API_KEY", "sk-test")

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.1, 0.2]}]})

        config = EmbeddingConfig(provider="openai_compatible", dimensions=2)
        embedder = OpenAICompatibleEmbedder(config, client=_client(handler))

        with pytest.raises(EmbeddingError, match="2 input texts"):
            asyncio.run(embedder.embed(["x", "y"]))

    @pytest.mark.parametrize(
        "response_body",
        [
            pytest.param("not-a-dict", id="non_dict_response"),
            pytest.param({"data": "not-a-list"}, id="data_not_a_list"),
            pytest.param({"data": ["not-a-dict"]}, id="item_not_a_dict"),
            pytest.param({"data": [{"embedding": [1.0, 2.0]}]}, id="missing_index"),
            pytest.param(
                {"data": [{"index": 0, "embedding": "not-a-list"}]}, id="embedding_not_a_list"
            ),
            pytest.param({"data": [{"index": 0, "embedding": ["x", "y"]}]}, id="non_numeric_value"),
        ],
    )
    def test_malformed_shapes_raise_embedding_error(
        self, monkeypatch: pytest.MonkeyPatch, response_body: object
    ) -> None:
        monkeypatch.setenv("GROUNDKIT_OPENAI_API_KEY", "sk-test")

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=response_body)

        config = EmbeddingConfig(provider="openai_compatible", dimensions=2)
        embedder = OpenAICompatibleEmbedder(config, client=_client(handler))

        with pytest.raises(EmbeddingError):
            asyncio.run(embedder.embed(["x"]))

    def test_duplicate_index_leaves_a_gap_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GROUNDKIT_OPENAI_API_KEY", "sk-test")

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": 0, "embedding": [1.0, 2.0]},
                        {"index": 0, "embedding": [3.0, 4.0]},
                    ]
                },
            )

        config = EmbeddingConfig(provider="openai_compatible", dimensions=2)
        embedder = OpenAICompatibleEmbedder(config, client=_client(handler))

        with pytest.raises(EmbeddingError, match="indices"):
            asyncio.run(embedder.embed(["first", "second"]))

    def test_cause_and_context_chain_never_carries_the_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ADR-0001 hazard 6 / ARP KNOWN_LIMITATIONS §4.10: a scrubbed message
        is not enough if the raw exception still rides along on __cause__ or
        __context__. Assert the whole chain is clean, not just the top frame.
        """
        api_key = "sk-super-secret-value"
        monkeypatch.setenv("GROUNDKIT_OPENAI_API_KEY", api_key)

        def handler(_request: httpx.Request) -> httpx.Response:
            raise RuntimeError(f"upstream rejected credential {api_key}")

        config = EmbeddingConfig(provider="openai_compatible", dimensions=2)
        embedder = OpenAICompatibleEmbedder(config, client=_client(handler))

        with pytest.raises(EmbeddingError) as excinfo:
            asyncio.run(embedder.embed(["x"]))

        exc: BaseException = excinfo.value
        assert api_key not in str(exc)
        assert api_key not in repr(exc)
        assert exc.__cause__ is None

        node = exc.__context__
        while node is not None:
            assert api_key not in str(node)
            assert api_key not in repr(node)
            node = node.__context__

    def test_empty_input_sends_no_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GROUNDKIT_OPENAI_API_KEY", raising=False)

        def handler(_request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("no request should be sent for an empty input list")

        config = EmbeddingConfig(provider="openai_compatible", dimensions=2)
        embedder = OpenAICompatibleEmbedder(config, client=_client(handler))
        # No key is configured, but an empty batch must short-circuit before
        # the key is ever resolved.
        assert asyncio.run(embedder.embed([])) == []

    def test_configured_secret_embedded_in_base_url_query_is_scrubbed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """HIGH 1: EmbeddingConfig.base_url is free-form operator-controlled
        str; several OpenAI-compatible/proxy endpoints carry the credential
        in the query string (Azure-style ``?api-key=...``). The URL must be
        scrubbed before it reaches the raised EmbeddingError, exactly like
        the exception message already is (ADR-0001 hazard 6).
        """
        api_key = "sk-super-secret-value"
        monkeypatch.setenv("GROUNDKIT_OPENAI_API_KEY", api_key)

        def handler(_request: httpx.Request) -> httpx.Response:
            raise RuntimeError("upstream rejected the request")

        config = EmbeddingConfig(
            provider="openai_compatible",
            dimensions=2,
            base_url=f"https://embed-proxy.example.com/openai-proxy?api-key={api_key}",
        )
        embedder = OpenAICompatibleEmbedder(config, client=_client(handler))

        with pytest.raises(EmbeddingError) as excinfo:
            asyncio.run(embedder.embed(["x"]))

        assert api_key not in str(excinfo.value)

    def test_unrelated_query_string_credential_is_also_redacted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Proves generic query redaction, not only exact-secret replacement:
        a token embedded in base_url that is NOT the configured api_key must
        still be redacted, since base_url is free-form and groundkit cannot
        know every credential shape a proxy might embed there.
        """
        monkeypatch.setenv("GROUNDKIT_OPENAI_API_KEY", "sk-bearer-value")
        leaked_token = "different-leaked-token-999"  # noqa: S105 - test fixture, not a real secret

        def handler(_request: httpx.Request) -> httpx.Response:
            raise RuntimeError("upstream rejected the request")

        config = EmbeddingConfig(
            provider="openai_compatible",
            dimensions=2,
            base_url=f"https://embed-proxy.example.com/openai-proxy?key={leaked_token}",
        )
        embedder = OpenAICompatibleEmbedder(config, client=_client(handler))

        with pytest.raises(EmbeddingError) as excinfo:
            asyncio.run(embedder.embed(["x"]))

        assert leaked_token not in str(excinfo.value)

    def test_scrubbed_url_keeps_scheme_host_and_path_for_debuggability(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Redaction must not destroy the operator's ability to tell which
        endpoint failed — only the query-string VALUES are sensitive."""
        api_key = "sk-super-secret-value"
        monkeypatch.setenv("GROUNDKIT_OPENAI_API_KEY", api_key)

        def handler(_request: httpx.Request) -> httpx.Response:
            raise RuntimeError("upstream rejected the request")

        config = EmbeddingConfig(
            provider="openai_compatible",
            dimensions=2,
            base_url=f"https://embed-proxy.example.com/openai-proxy?api-key={api_key}",
        )
        embedder = OpenAICompatibleEmbedder(config, client=_client(handler))

        with pytest.raises(EmbeddingError) as excinfo:
            asyncio.run(embedder.embed(["x"]))

        message = str(excinfo.value)
        assert "https://embed-proxy.example.com" in message
        assert "/openai-proxy" in message
        assert api_key not in message


# ── vector value validation (shared by both HTTP parsers) ─────────────────


#: Response bodies whose vectors are the right *width* but hold values that
#: must never reach an index. Passed as raw JSON text, not a dict, because
#: two of them are only expressible as literals: ``json.loads`` accepts
#: JavaScript's bare ``NaN`` / ``Infinity`` by default, which is exactly how
#: a real provider's reply would carry them through ``response.json()``.
_POISONED_VECTORS: Final[list[tuple[str, str]]] = [
    ("bare NaN literal", "[NaN, 0.5]"),
    ("bare Infinity literal", "[Infinity, 0.5]"),
    ("bare -Infinity literal", "[-Infinity, 0.5]"),
    ("NaN as a string", '["NaN", 0.5]'),
    ("Infinity as a string", '["Infinity", 0.5]'),
    ("overflowing literal", "[1e400, 0.5]"),
    ("numeric-looking string", '["0.25", 0.5]'),
    ("boolean", "[true, 0.5]"),
    ("null", "[null, 0.5]"),
]


class TestVectorValuesAreValidatedNotCoerced:
    """A right-width vector full of garbage must be refused, not stored.

    Both parsers used to convert every component with a bare ``float()``,
    which accepts ``NaN``, ``Infinity``, numeric strings and booleans.
    ``_check_dimensions`` never catches any of it — the width is correct in
    every case — so the values landed in the persisted index. A single
    ``NaN`` component poisons every cosine similarity computed against that
    vector, and ``NaN`` compares false against everything, so the chunk's
    rank stops being a function of relevance; it also satisfies
    ``RetrievalResult.score``'s ``ge=0.0`` bound silently. SPEC.md §2 calls
    for rejection over coercion at exactly this boundary.
    """

    @pytest.mark.parametrize(("label", "vector_json"), _POISONED_VECTORS)
    def test_ollama_rejects(self, label: str, vector_json: str) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=f'{{"embeddings": [{vector_json}]}}',
                headers={"content-type": "application/json"},
            )

        config = EmbeddingConfig(provider="ollama", dimensions=2)
        embedder = OllamaEmbedder(config, client=_client(handler))

        with pytest.raises(EmbeddingError, match=r"non-finite|non-numeric"):
            asyncio.run(embedder.embed(["x"]))

    @pytest.mark.parametrize(("label", "vector_json"), _POISONED_VECTORS)
    def test_openai_compatible_rejects(
        self, label: str, vector_json: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GROUNDKIT_OPENAI_API_KEY", "test-key")

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=f'{{"data": [{{"index": 0, "embedding": {vector_json}}}]}}',
                headers={"content-type": "application/json"},
            )

        config = EmbeddingConfig(provider="openai_compatible", dimensions=2)
        embedder = OpenAICompatibleEmbedder(config, client=_client(handler))

        with pytest.raises(EmbeddingError, match=r"non-finite|non-numeric"):
            asyncio.run(embedder.embed(["x"]))

    def test_a_boolean_index_cannot_reorder_an_openai_batch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``isinstance(True, int)`` is True — the same coercion trap, on ``index``.

        A JSON ``true`` passed the integer check and was sorted as index 1,
        which silently misattributes a vector to the wrong input text.
        """
        monkeypatch.setenv("GROUNDKIT_OPENAI_API_KEY", "test-key")

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": 0, "embedding": [0.1, 0.2]},
                        {"index": True, "embedding": [0.3, 0.4]},
                    ]
                },
            )

        config = EmbeddingConfig(provider="openai_compatible", dimensions=2)
        embedder = OpenAICompatibleEmbedder(config, client=_client(handler))

        with pytest.raises(EmbeddingError, match="integer 'index'"):
            asyncio.run(embedder.embed(["x", "y"]))

    def test_ordinary_finite_values_still_parse(self) -> None:
        """The guard must not reject the values a real provider actually sends.

        Integers included: JSON has one number type, so a provider is free
        to serialize ``0`` rather than ``0.0``.
        """

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"embeddings": [[0, -1.5, 2.25, 1e-9]]})

        config = EmbeddingConfig(provider="ollama", dimensions=4)
        embedder = OllamaEmbedder(config, client=_client(handler))

        assert asyncio.run(embedder.embed(["x"])) == [[0.0, -1.5, 2.25, 1e-9]]


# ── build_embedder / EmbeddingProtocol conformance ─────────────────────────


class TestBuildEmbedder:
    def test_dispatches_ollama(self) -> None:
        embedder = build_embedder(EmbeddingConfig(provider="ollama"))
        assert isinstance(embedder, OllamaEmbedder)
        assert isinstance(embedder, EmbeddingProtocol)

    def test_dispatches_openai_compatible(self) -> None:
        embedder = build_embedder(EmbeddingConfig(provider="openai_compatible"))
        assert isinstance(embedder, OpenAICompatibleEmbedder)
        assert isinstance(embedder, EmbeddingProtocol)

    def test_dispatches_inmemory(self) -> None:
        embedder = build_embedder(EmbeddingConfig(provider="inmemory", dimensions=64))
        assert isinstance(embedder, InMemoryEmbedder)
        assert isinstance(embedder, EmbeddingProtocol)
        assert embedder.dimensions == 64

    def test_unknown_provider_is_impossible_to_construct(self) -> None:
        """The provider Literal fails closed at config construction — there is
        no runtime value build_embedder could receive that isn't one of the
        three handled branches."""
        with pytest.raises(ValidationError):
            EmbeddingConfig(provider="not-a-real-provider")  # type: ignore[arg-type]


class TestEmbeddingProtocolConformance:
    def test_all_three_providers_satisfy_the_protocol(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("conformance check must not send a request")

        assert isinstance(InMemoryEmbedder(), EmbeddingProtocol)
        assert isinstance(
            OllamaEmbedder(EmbeddingConfig(provider="ollama"), client=_client(handler)),
            EmbeddingProtocol,
        )
        assert isinstance(
            OpenAICompatibleEmbedder(
                EmbeddingConfig(provider="openai_compatible"), client=_client(handler)
            ),
            EmbeddingProtocol,
        )


class TestHttpEmbedderLifecycle:
    def test_aclose_releases_the_injected_client(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("aclose must not send a request")

        embedder = OllamaEmbedder(EmbeddingConfig(provider="ollama"), client=_client(handler))
        asyncio.run(embedder.aclose())
        assert embedder._client.is_closed

    def test_shared_properties_reflect_config(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError("property access must not send a request")

        config = EmbeddingConfig(provider="ollama", model_name="custom-model", dimensions=17)
        embedder = OllamaEmbedder(config, client=_client(handler))
        assert embedder.model_name == "custom-model"
        assert embedder.dimensions == 17

"""Config tests: local-first defaults and fail-closed validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from groundkit.config import (
    DEFAULT_OLLAMA_BASE_URL,
    ChunkingConfig,
    EmbeddingConfig,
    GroundkitConfig,
    IndexConfig,
    RetrievalConfig,
    resolve_embedding_config,
)
from groundkit.errors import ConfigurationError


class TestChunkingConfig:
    def test_overlap_must_be_less_than_size(self) -> None:
        with pytest.raises(ValidationError, match="chunk_overlap"):
            ChunkingConfig(chunk_size=100, chunk_overlap=100)

    def test_unknown_keys_fail_closed(self) -> None:
        with pytest.raises(ValidationError):
            ChunkingConfig(chunk_sizes=512)  # type: ignore[call-arg]


class TestEmbeddingConfig:
    def test_local_first_defaults(self) -> None:
        c = EmbeddingConfig()
        assert c.provider == "ollama"
        assert c.model_name == "nomic-embed-text"
        assert c.base_url == DEFAULT_OLLAMA_BASE_URL

    def test_unrecognized_provider_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EmbeddingConfig(provider="voyage")  # type: ignore[arg-type]

    def test_config_never_holds_a_key_value(self) -> None:
        with pytest.raises(ValidationError):
            EmbeddingConfig(api_key="sk-nope")  # type: ignore[call-arg]


class TestResolveEmbeddingConfig:
    """Tests for the promoted ``resolve_embedding_config`` (moved from ``cli.py``).

    This is a refactor, not a defect fix (SPEC.md §8): the behaviour under
    test already existed as the private ``cli._resolve_embedding_config``
    before the move, so these are new-surface tests over the promoted
    function rather than regression tests, and there is no unfixed-code
    state to show them failing against. What they prove is that the move
    preserved behaviour exactly -- the same thing the untouched existing
    suite (``TestEmbeddingConfig`` above, and ``tests/test_cli.py``'s
    embed-flag tests) already proves by continuing to pass unmodified.
    """

    def test_all_none_gives_defaults(self) -> None:
        resolved = resolve_embedding_config(
            provider=None, model_name=None, dimensions=None, base_url=None
        )
        assert resolved == EmbeddingConfig()

    def test_provider_override(self) -> None:
        resolved = resolve_embedding_config(
            provider="inmemory", model_name=None, dimensions=None, base_url=None
        )
        defaults = EmbeddingConfig()
        assert resolved.provider == "inmemory"
        assert resolved.model_name == defaults.model_name
        assert resolved.dimensions == defaults.dimensions
        assert resolved.base_url == defaults.base_url

    def test_model_name_override(self) -> None:
        resolved = resolve_embedding_config(
            provider=None, model_name="all-minilm", dimensions=None, base_url=None
        )
        defaults = EmbeddingConfig()
        assert resolved.model_name == "all-minilm"
        assert resolved.provider == defaults.provider
        assert resolved.dimensions == defaults.dimensions
        assert resolved.base_url == defaults.base_url

    def test_dimensions_override(self) -> None:
        resolved = resolve_embedding_config(
            provider=None, model_name=None, dimensions=384, base_url=None
        )
        defaults = EmbeddingConfig()
        assert resolved.dimensions == 384
        assert resolved.provider == defaults.provider
        assert resolved.model_name == defaults.model_name
        assert resolved.base_url == defaults.base_url

    def test_base_url_override(self) -> None:
        resolved = resolve_embedding_config(
            provider=None, model_name=None, dimensions=None, base_url="http://example:1234"
        )
        defaults = EmbeddingConfig()
        assert resolved.base_url == "http://example:1234"
        assert resolved.provider == defaults.provider
        assert resolved.model_name == defaults.model_name
        assert resolved.dimensions == defaults.dimensions

    def test_all_fields_override_together(self) -> None:
        resolved = resolve_embedding_config(
            provider="openai_compatible",
            model_name="text-embedding-3-small",
            dimensions=1536,
            base_url="https://api.example.com",
        )
        assert resolved == EmbeddingConfig(
            provider="openai_compatible",
            model_name="text-embedding-3-small",
            dimensions=1536,
            base_url="https://api.example.com",
        )

    def test_invalid_dimensions_translates_to_configuration_error(self) -> None:
        """A bad ``dimensions`` must raise ``ConfigurationError``, not ``ValidationError``.

        Untranslated, ``ValidationError`` is not a ``GroundkitError``, so
        ``cli.main``'s handler would never catch it and the command would
        exit on a raw pydantic traceback instead of a one-line ``error:``
        message.
        """
        with pytest.raises(ConfigurationError, match="dimensions") as excinfo:
            resolve_embedding_config(provider=None, model_name=None, dimensions=0, base_url=None)
        assert isinstance(excinfo.value.__cause__, ValidationError)


class TestRetrievalConfig:
    def test_threshold_off_is_explicit_none(self) -> None:
        assert RetrievalConfig().score_threshold is None

    def test_bm25_b_bounded(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalConfig(bm25_b=1.5)


class TestGroundkitConfig:
    def test_index_dir_is_required(self) -> None:
        with pytest.raises(ValidationError):
            GroundkitConfig()  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        c = GroundkitConfig(index=IndexConfig(index_dir="idx"))
        with pytest.raises(ValidationError):
            c.retrieval = RetrievalConfig()

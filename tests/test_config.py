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
)


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

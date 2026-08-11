"""Typed configuration — frozen Pydantic models, fail closed on unknown keys.

Adapted from ARP's ``agentic_v2/rag/config.py`` per ADR-0001: the frozen +
``extra="forbid"`` + real-invariant-validator pattern is kept; the enums and
defaults are reshaped for groundkit's local-first design (Ollama default,
configurable endpoint, hybrid retrieval parameters). Unknown keys raise at
construction time — there is no lenient mode.

API keys are never configuration values: config carries the *name* of the
environment variable to read at call time (SPEC.md §7).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: Default local Ollama endpoint — loopback by design (SPEC.md §7: the one
#: named exception to the SSRF guard).
DEFAULT_OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"


class ChunkingConfig(BaseModel):
    """Configuration for document chunking.

    Sizes are in **characters** (ARP's docs said tokens while the code split
    characters; groundkit documents what the code does).

    Attributes:
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks, in characters.
        separators: Split separators, tried in order (recursive strategy).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=64, ge=0)
    separators: list[str] = Field(default_factory=lambda: ["\n\n", "\n", ". ", " ", ""])

    @model_validator(mode="after")
    def _validate_overlap(self) -> ChunkingConfig:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        return self


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation. Local-first: Ollama by default.

    Attributes:
        provider: ``"ollama"`` (default, local), ``"openai_compatible"``
            (opt-in cloud), or ``"inmemory"`` (deterministic test double —
            never a production choice; it has no semantic signal).
        model_name: Model identifier for the provider.
        dimensions: Expected embedding vector width; responses that do not
            match raise ``EmbeddingError`` (fail loud, never corrupt an index).
        base_url: Provider endpoint. Defaults to the local Ollama endpoint.
        api_key_env: Name of the environment variable holding the API key for
            ``openai_compatible``. Read at call time, never stored or logged.
        batch_size: Max texts per embedding request.
        max_concurrent: Concurrency limit for embedding requests.
        timeout_seconds: Per-request timeout.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["ollama", "openai_compatible", "inmemory"] = "ollama"
    model_name: str = "nomic-embed-text"
    dimensions: int = Field(default=768, gt=0)
    base_url: str = DEFAULT_OLLAMA_BASE_URL
    api_key_env: str = "GROUNDKIT_OPENAI_API_KEY"
    batch_size: int = Field(default=32, gt=0)
    max_concurrent: int = Field(default=4, gt=0)
    timeout_seconds: float = Field(default=30.0, gt=0)


class IndexConfig(BaseModel):
    """Configuration for the persisted index.

    Attributes:
        index_dir: Directory holding the collection's persisted state
            (SQLite metadata store; LanceDB table arrives in Phase 3).
        collection: Collection name.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    index_dir: str
    collection: str = "default"


class RetrievalConfig(BaseModel):
    """Configuration for retrieval.

    Attributes:
        top_k: Default number of results to return.
        bm25_k1: BM25 term-frequency saturation parameter.
        bm25_b: BM25 length-normalization parameter.
        score_threshold: Minimum score filter. ``None`` disables filtering —
            explicit, instead of ARP's silent ``0.0``-means-off sentinel.
        rrf_k: Reciprocal-rank-fusion constant (used from Phase 3).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    top_k: int = Field(default=5, gt=0)
    bm25_k1: float = Field(default=1.5, gt=0)
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0)
    score_threshold: float | None = Field(default=None, ge=0.0)
    rrf_k: int = Field(default=60, gt=0)


class GroundkitConfig(BaseModel):
    """Top-level configuration composing all component configs.

    Attributes:
        chunking: Document chunking settings.
        embedding: Embedding provider settings.
        index: Persisted index settings.
        retrieval: Retrieval settings.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    index: IndexConfig
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)

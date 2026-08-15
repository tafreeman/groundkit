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

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from groundkit.errors import ConfigurationError

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


def resolve_embedding_config(
    *,
    provider: Literal["ollama", "openai_compatible", "inmemory"] | None,
    model_name: str | None,
    dimensions: int | None,
    base_url: str | None,
) -> EmbeddingConfig:
    """Build an :class:`EmbeddingConfig`, defaulting any ``None`` field.

    Each parameter is typed as the corresponding :class:`EmbeddingConfig`
    field's own type — ``provider`` is the same ``Literal["ollama",
    "openai_compatible", "inmemory"] | None`` that
    :attr:`EmbeddingConfig.provider` uses, not ``str | None`` — rather than
    a heterogeneous ``dict[str, ...]`` splat, which cannot type-check
    cleanly against :class:`EmbeddingConfig`'s typed fields under ``mypy
    --strict``.

    A caller that already holds typed values gets full checking at this
    boundary. A caller holding an *untyped* value can still pass it straight
    through with no ``cast``: ``argparse.Namespace`` attributes are typed
    ``Any``, and ``Any`` is assignable to any parameter type, so the
    permissiveness that makes passthrough type-check lives at the *call
    site*, not here. ``groundkit.cli._resolve_embedding_config`` is exactly
    that call site — it unpacks a parsed ``args.embed_*`` namespace and
    passes the attributes straight into these keyword parameters. Pydantic
    still validates the ``provider`` literal at construction below (argparse's
    own ``choices`` already constrains it on that call path, so this is
    defense in depth, not the only check).

    Defaults come from a fresh :class:`EmbeddingConfig` — one source of
    truth, never a second copy of its field defaults.

    Pydantic's own field invariants (``dimensions`` must be ``> 0``, and any
    other bound :class:`EmbeddingConfig` grows later) are enforced here and
    nowhere else on this path, so the ``ValidationError`` they raise is
    translated into a :class:`~groundkit.errors.ConfigurationError`.
    Untranslated it is not a :class:`~groundkit.errors.GroundkitError`, so
    ``cli.main``'s handler never sees it and ``grk ingest --dense
    --embed-dimensions 0`` exits on a pydantic traceback instead of the
    one-line ``error:`` message every other bad flag produces. Translating
    at this single construction site rather than re-checking each bound at
    every caller keeps :class:`EmbeddingConfig` the only place a bound is
    stated.

    Raises:
        ConfigurationError: A supplied value violates an
            :class:`EmbeddingConfig` invariant.
    """
    defaults = EmbeddingConfig()
    try:
        return EmbeddingConfig(
            provider=provider if provider is not None else defaults.provider,
            model_name=model_name if model_name is not None else defaults.model_name,
            dimensions=dimensions if dimensions is not None else defaults.dimensions,
            base_url=base_url if base_url is not None else defaults.base_url,
        )
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        raise ConfigurationError(f"invalid embedding configuration ({details})") from exc


#: Default chat model for the Phase 5 boundary features. The tag is explicit
#: (``:3b``) rather than a floating alias so the gated real-model measurement
#: means one thing: two runs under this default exercised the same weights.
DEFAULT_CHAT_MODEL: str = "llama3.2:3b"


class ChatConfig(BaseModel):
    """Configuration for the chat/completion provider (Phase 5, ADR-0017).

    The exact peer of :class:`EmbeddingConfig`: local-first, Ollama by
    default, cloud opt-in. There is no ``"scripted"`` provider literal on
    purpose — :class:`~groundkit.providers.llm.ScriptedChatProvider` is a
    test double constructed directly with its script; a config claiming to
    describe one would name a runtime artifact no config can reconstruct.

    Attributes:
        provider: ``"ollama"`` (default, local) or ``"openai_compatible"``
            (opt-in cloud; its egress is redacted — ADR-0017).
        model_name: Model identifier for the provider.
        base_url: Provider endpoint. Defaults to the local Ollama endpoint.
        api_key_env: Name of the environment variable holding the API key for
            ``openai_compatible``. Read at call time, never stored or logged.
        timeout_seconds: Per-request timeout.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["ollama", "openai_compatible"] = "ollama"
    model_name: str = DEFAULT_CHAT_MODEL
    base_url: str = DEFAULT_OLLAMA_BASE_URL
    api_key_env: str = "GROUNDKIT_OPENAI_API_KEY"
    timeout_seconds: float = Field(default=60.0, gt=0)


def resolve_chat_config(
    *,
    provider: Literal["ollama", "openai_compatible"] | None,
    model_name: str | None,
    base_url: str | None,
    api_key_env: str | None,
) -> ChatConfig:
    """Build a :class:`ChatConfig`, defaulting any ``None`` field.

    The same boundary-translation shape as :func:`resolve_embedding_config`,
    for the same reasons — see its docstring for the typing rationale and why
    the ``ValidationError`` is translated exactly here.

    Raises:
        ConfigurationError: A supplied value violates a :class:`ChatConfig`
            invariant.
    """
    defaults = ChatConfig()
    try:
        return ChatConfig(
            provider=provider if provider is not None else defaults.provider,
            model_name=model_name if model_name is not None else defaults.model_name,
            base_url=base_url if base_url is not None else defaults.base_url,
            api_key_env=api_key_env if api_key_env is not None else defaults.api_key_env,
        )
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        raise ConfigurationError(f"invalid chat configuration ({details})") from exc


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

"""Provider boundary: embedding providers (Ollama default, OpenAI-compatible
opt-in), optional query rewrite / synthesis, and the anonymization pass that
runs before any text leaves the process (Phases 1/5).
"""

from __future__ import annotations

from groundkit.providers.embeddings import (
    InMemoryEmbedder,
    OllamaEmbedder,
    OpenAICompatibleEmbedder,
    build_embedder,
)
from groundkit.providers.protocols import EmbeddingProtocol

__all__ = [
    "EmbeddingProtocol",
    "InMemoryEmbedder",
    "OllamaEmbedder",
    "OpenAICompatibleEmbedder",
    "build_embedder",
]

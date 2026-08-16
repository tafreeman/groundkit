"""Provider boundary: embedding providers (Ollama default, OpenAI-compatible
opt-in), the chat/completion seam behind query rewrite and synthesis, and the
anonymization pass that runs before any text leaves the process (Phases 1/5).
"""

from __future__ import annotations

from groundkit.providers.embeddings import (
    InMemoryEmbedder,
    OllamaEmbedder,
    OpenAICompatibleEmbedder,
    build_embedder,
)
from groundkit.providers.llm import (
    OllamaChat,
    OpenAICompatChat,
    RedactingChat,
    ScriptedChatProvider,
    build_chat,
)
from groundkit.providers.protocols import ChatProtocol, EmbeddingProtocol

__all__ = [
    "ChatProtocol",
    "EmbeddingProtocol",
    "InMemoryEmbedder",
    "OllamaChat",
    "OllamaEmbedder",
    "OpenAICompatChat",
    "OpenAICompatibleEmbedder",
    "RedactingChat",
    "ScriptedChatProvider",
    "build_chat",
    "build_embedder",
]

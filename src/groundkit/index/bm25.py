"""Persisted BM25 lexical index (Phase 1). Library choice justified in an ADR.

Ported from ARP's ``agentic_v2.rag.retrieval.BM25Index`` (ADR-0001: pure-Python
BM25, promoted near-verbatim — zero deps, already pinned by ARP's behavioral
tests). Two deliberate changes for groundkit's layering:

- :meth:`BM25Index.search` returns ``(Chunk, score)`` pairs instead of
  constructing ``RetrievalResult`` directly. ``RetrievalResult.source`` has
  no counterpart on ``Chunk`` (only ``Document`` carries the source path/URL),
  so assembling a citation-bearing result means joining against the metadata
  store — that join is the retrieval layer's job, not this module's.
- :meth:`BM25Index.from_store` rebuilds a fresh in-memory index from
  :class:`~groundkit.index.metadata.SQLiteMetadataStore` (ADR-0002): SQLite
  is the durable truth for chunks; BM25 postings are never pickled to disk.
"""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from groundkit.contracts import Chunk
    from groundkit.index.metadata import SQLiteMetadataStore

logger = logging.getLogger(__name__)

#: Okapi BM25's IDF smoothing term — keeps the ratio inside the log at or
#: above 1.0 so no query term ever contributes a negative score component,
#: matching the RetrievalResult ``score >= 0`` contract.
_IDF_SMOOTHING: float = 1.0


class BM25Index:
    """Pure-Python in-memory BM25 keyword index.

    Builds an inverted index from chunks, tokenizes on lowercased word
    characters, and scores candidates with Okapi BM25.

    Args:
        k1: Term-frequency saturation parameter.
        b: Length-normalization parameter.
    """

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._chunks: list[Chunk] = []
        self._doc_freqs: dict[str, int] = defaultdict(int)
        self._doc_term_freqs: list[dict[str, int]] = []
        self._doc_lengths: list[int] = []
        self._avg_doc_length: float = 0.0

    @property
    def size(self) -> int:
        """Number of chunks currently indexed."""
        return len(self._chunks)

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Incrementally add chunks to the BM25 index.

        Existing indexed chunks are preserved — this is an accumulation,
        not a replacement. Rebuilding from scratch means constructing a new
        :class:`BM25Index` (see :meth:`from_store`).

        Args:
            chunks: Chunks to add.
        """
        for chunk in chunks:
            self._chunks.append(chunk)
            tokens = _tokenize(chunk.content)
            self._doc_lengths.append(len(tokens))

            term_freq: dict[str, int] = defaultdict(int)
            seen_terms: set[str] = set()
            for token in tokens:
                term_freq[token] += 1
                if token not in seen_terms:
                    self._doc_freqs[token] += 1
                    seen_terms.add(token)
            self._doc_term_freqs.append(dict(term_freq))

        total_length = sum(self._doc_lengths)
        self._avg_doc_length = total_length / len(self._chunks) if self._chunks else 0.0
        logger.debug("BM25 index now holds %d chunks", len(self._chunks))

    def search(self, query: str, *, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Search the index with a BM25 query.

        Args:
            query: The search query string.
            top_k: Maximum number of results to return.

        Returns:
            ``(chunk, score)`` pairs ranked by descending BM25 score, each
            score ``>= 0.0``. An empty query or an empty index returns
            ``[]``, as does a query whose terms match nothing.
        """
        query_tokens = _tokenize(query)
        if not query_tokens or not self._chunks:
            return []

        scored: list[tuple[float, int]] = []
        for doc_idx in range(len(self._chunks)):
            score = self._score_document(query_tokens, doc_idx)
            if score > 0.0:
                scored.append((score, doc_idx))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [(self._chunks[doc_idx], score) for score, doc_idx in scored[:top_k]]

    def _score_document(self, query_tokens: list[str], doc_idx: int) -> float:
        """Compute the BM25 score for one indexed chunk against query tokens."""
        score = 0.0
        doc_length = self._doc_lengths[doc_idx]
        term_freqs = self._doc_term_freqs[doc_idx]
        n_docs = len(self._chunks)

        for token in query_tokens:
            term_frequency = term_freqs.get(token)
            if term_frequency is None:
                continue

            doc_frequency = self._doc_freqs.get(token, 0)
            idf = math.log((n_docs - doc_frequency + 0.5) / (doc_frequency + 0.5) + _IDF_SMOOTHING)
            tf_norm = (term_frequency * (self._k1 + 1.0)) / (
                term_frequency
                + self._k1 * (1.0 - self._b + self._b * doc_length / self._avg_doc_length)
            )
            score += idf * tf_norm

        return score

    @classmethod
    async def from_store(
        cls, store: SQLiteMetadataStore, *, k1: float = 1.5, b: float = 0.75
    ) -> BM25Index:
        """Rebuild a fresh in-memory BM25 index from persisted chunks.

        ADR-0002: BM25 postings are never pickled — SQLite is durable truth,
        and the in-memory index is reconstructed from it at process start.
        Cost is O(corpus size); acceptable for v1 and reconsidered if
        rebuild time is ever measured to be a problem.

        Args:
            store: The metadata store to read every persisted chunk from.
            k1: Term-frequency saturation parameter for the new index.
            b: Length-normalization parameter for the new index.

        Returns:
            A new :class:`BM25Index` populated with every chunk currently
            persisted in ``store``.
        """
        index = cls(k1=k1, b=b)
        chunks = await store.get_chunks()
        index.index_chunks(chunks)
        return index


def _tokenize(text: str) -> list[str]:
    """Tokenize text by lowercasing and extracting word characters.

    Strips punctuation so that ``"Python."`` and ``"python"`` match.

    Args:
        text: The input text to tokenize.

    Returns:
        Lowercase word tokens with punctuation removed.
    """
    return re.findall(r"\w+", text.lower())

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
    from groundkit.index.protocols import MetadataStoreProtocol

logger = logging.getLogger(__name__)

#: Okapi BM25's IDF smoothing term — keeps the ratio inside the log at or
#: above 1.0 so no query term ever contributes a negative score component,
#: matching the RetrievalResult ``score >= 0`` contract.
_IDF_SMOOTHING: float = 1.0

#: The postings list of a term the index has never seen. A module constant so
#: :meth:`BM25Index.search` can look a term up through ``dict.get`` and still
#: name what "absent" means: ``self._postings[token]`` reads identically on
#: the first call and then, the map being a ``defaultdict``, inserts an empty
#: list — an unbounded write driven by query content, on the read path of an
#: index :class:`BM25Index` documents as frozen while a search is running.
_NO_POSTINGS: tuple[int, ...] = ()


class BM25Index:
    """Pure-Python in-memory BM25 keyword index.

    Builds an inverted index from chunks — a ``term -> [chunk index]``
    postings map (``_postings``) beside the document-frequency counter
    (``_doc_freqs``) and the length statistics — tokenizes on lowercased word
    characters, and scores candidates with Okapi BM25.

    ADR-0002 decision 2 named all three of those from its first version, and
    until GK-018 only two of them existed — so :meth:`search` scored every
    indexed chunk however selective the query was, making query cost a
    function of corpus size. The postings map closes that, recorded as an
    erratum on the decision itself rather than silently. What it buys is
    **score-identical, not an approximation**: a chunk holding no query term
    skips every term inside :meth:`_score_document`, scores exactly ``0.0``,
    and was already dropped by that method's caller, so narrowing the walk to
    the union of the query terms' postings removes work and never a result —
    the low-scoring-but-nonzero tail included, which is what an approximate
    candidate scheme would have cost and why none was taken.

    **Not safe to mutate while a search is running.**
    :class:`~groundkit.retrieval.search.Retriever` dispatches :meth:`search` to
    a worker thread, where it reads the parallel per-chunk lists and the
    postings map below live; :meth:`index_chunks` extends them in separate
    statements, so an append landing mid-scan would be observed torn — a
    posting, or ``len(self._chunks)``, past the matching ``_doc_lengths`` or
    ``_doc_term_freqs`` entry, raising ``IndexError`` from
    :meth:`_score_document`. Build the index fully (normally via
    :meth:`from_store`) and treat it as frozen thereafter; rebuilding means
    constructing a new instance, which is what
    :class:`~groundkit.runtime.CollectionRuntime` does.

    Args:
        k1: Term-frequency saturation parameter.
        b: Length-normalization parameter.
    """

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._chunks: list[Chunk] = []
        self._tie_keys: list[str] = []
        self._doc_freqs: dict[str, int] = defaultdict(int)
        # term -> the indices of every chunk holding it, strictly ascending
        # and duplicate-free (see index_chunks). Not a second copy of
        # _doc_freqs: that counter answers "how many chunks hold this term"
        # for IDF, this answers "which ones" for candidate selection, and
        # ``len(self._postings[term]) == self._doc_freqs[term]`` holds by
        # construction because both are written from the same first-sighting
        # branch.
        self._postings: dict[str, list[int]] = defaultdict(list)
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

        A chunk's index joins the postings list of every term it holds, on
        that term's *first* occurrence in the chunk rather than on every
        occurrence — the same branch that increments the document-frequency
        counter, so the two cannot disagree about which chunks hold a term.
        Appending in chunk order then leaves each postings list strictly
        ascending and duplicate-free, which is the property :meth:`search`
        relies on to visit candidates in the order a whole-corpus scan
        would have.

        Args:
            chunks: Chunks to add.
        """
        for chunk in chunks:
            doc_idx = len(self._chunks)
            self._chunks.append(chunk)
            self._tie_keys.append(chunk.content_hash)
            tokens = _tokenize(chunk.content)
            self._doc_lengths.append(len(tokens))

            term_freq: dict[str, int] = defaultdict(int)
            seen_terms: set[str] = set()
            for token in tokens:
                term_freq[token] += 1
                if token not in seen_terms:
                    self._doc_freqs[token] += 1
                    self._postings[token].append(doc_idx)
                    seen_terms.add(token)
            self._doc_term_freqs.append(dict(term_freq))

        total_length = sum(self._doc_lengths)
        self._avg_doc_length = total_length / len(self._chunks) if self._chunks else 0.0
        logger.debug("BM25 index now holds %d chunks", len(self._chunks))

    def search(self, query: str, *, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Search the index with a BM25 query.

        Only chunks holding at least one query term are scored (GK-018), so
        the cost is proportional to the number of matching chunks rather than
        to the corpus. The returned list is exactly what scoring every chunk
        produced, ties included — see the class docstring for why that is an
        identity rather than an approximation.

        Args:
            query: The search query string.
            top_k: Maximum number of results to return.

        Returns:
            ``(chunk, score)`` pairs ranked by descending BM25 score, each
            score ``>= 0.0``. An empty query or an empty index returns
            ``[]``, as does a query whose terms match nothing. Ties are
            broken by ascending ``content_hash`` so that ranking is stable
            across re-ingestion of the same corpus (see the sort below).
            The one case this can't disambiguate: two byte-identical chunks
            share a ``content_hash`` and fall back to insertion order — no
            content-derived key can distinguish them.
        """
        query_tokens = _tokenize(query)
        if not query_tokens or not self._chunks:
            return []

        candidates: set[int] = set()
        for token in query_tokens:
            candidates.update(self._postings.get(token, _NO_POSTINGS))
        if not candidates:
            return []

        # Ascending, so the walk visits candidates in the order a
        # whole-corpus scan visited them. ``list.sort`` below is stable, so
        # that order is what two chunks tied on *both* score and
        # ``content_hash`` — byte-identical chunks, the one case the
        # tie-break cannot resolve — still fall back to. Iterating
        # ``candidates`` directly would hand that fallback to set layout.
        scored: list[tuple[float, int]] = []
        for doc_idx in sorted(candidates):
            score = self._score_document(query_tokens, doc_idx)
            if score > 0.0:
                scored.append((score, doc_idx))

        # Tie-break on content_hash, not document_id/chunk_id (contracts.py:38,64):
        # both are uuid4, regenerated on every ingest, so using either as a
        # secondary key would make tied ordering just as unstable as no
        # secondary key at all. Negate the score (rather than reverse=True)
        # so both sort keys are ascending — reverse=True would also flip the
        # tie-break to descending hash order, which isn't the intent.
        scored.sort(key=lambda pair: (-pair[0], self._tie_keys[pair[1]]))
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
        cls, store: MetadataStoreProtocol, *, k1: float = 1.5, b: float = 0.75
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

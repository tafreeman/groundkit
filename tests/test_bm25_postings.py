"""Postings-list candidate selection in ``BM25Index`` (GK-018).

ADR-0002 decision 2 asserted that "the postings, document frequencies, and
length statistics are pure functions of the persisted chunk set", and the
class docstring said it built an inverted index. Only the document
frequencies and the length statistics were ever built, so ``search`` scored
every indexed chunk regardless of how selective the query was. The decision
now carries an erratum recording that, and ``BM25Index`` now carries the
``term -> [chunk index]`` map the word *postings* names.

The fix is worth only as much as the proof that it changed no output, so the
central test here compares ``search`` against a transcription of the pre-fix
whole-corpus loop over the real golden corpus, byte for byte -- same length,
same chunk objects, same order, exactly equal floats -- for every query in
``evals/judgments.jsonl``. ``test_dropping_one_posting_changes_the_result_list``
exists to show that comparison has teeth, because a comparison that cannot
fail proves nothing about a change whose whole claim is that nothing changed.

Async methods are driven with ``asyncio.run()`` inside sync test functions
(pytest-asyncio is not configured in this repo).
"""

from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path
from typing import Final

import pytest

from groundkit.contracts import Chunk
from groundkit.evals.corpus import load_judgments
from groundkit.evals.runner import EVAL_CHUNKING_CONFIG
from groundkit.index.bm25 import BM25Index, _tokenize
from groundkit.index.metadata import SQLiteMetadataStore
from groundkit.indexer import Indexer
from groundkit.ingestion.loaders import FileLoader

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
CORPUS_DIR: Final[Path] = _REPO_ROOT / "evals" / "corpus"
JUDGMENTS_PATH: Final[Path] = _REPO_ROOT / "evals" / "judgments.jsonl"

#: Corpus shape for the tie test. The two twin indices are chosen so that
#: CPython iterates the resulting three-element candidate set in an order that
#: is *not* ascending (``40 & 7 == 0`` lands ahead of ``5 & 7 == 5`` in an
#: eight-slot table), which is what makes that test able to catch a walk over
#: the raw set instead of over ``sorted(...)``. That is a strengthener, not a
#: correctness requirement: the assertions state the right answer either way,
#: so a future CPython that iterates small-int sets ascending makes the test
#: weaker, never flaky.
_TIE_TERM: Final[str] = "quorum"
_TIE_CORPUS_SIZE: Final[int] = 48
_DISTINCT_IDX: Final[int] = 3
_FIRST_TWIN_IDX: Final[int] = 5
_SECOND_TWIN_IDX: Final[int] = 40


def _make_chunk(
    content: str,
    *,
    chunk_id: str,
    document_id: str = "doc-1",
    chunk_index: int = 0,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        content=content,
        start_offset=0,
        end_offset=len(content),
    )


def _full_scan(index: BM25Index, query: str, *, top_k: int) -> list[tuple[Chunk, float]]:
    """``BM25Index.search`` as it read before GK-018: score every chunk, then rank.

    A deliberate transcription of the pre-fix loop rather than a call back
    into the class, so the equivalence assertions compare the new candidate
    walk against something a postings bug cannot also narrow.

    Everything *after* the loop -- the ``score > 0.0`` filter, the
    negated-score plus ``content_hash`` sort key, the ``top_k`` slice -- is
    character-identical to the live implementation on purpose. What GK-018
    changed is candidate *selection*; re-deriving the ranking here would pin
    this file's arithmetic instead of the module's.
    """
    query_tokens = _tokenize(query)
    if not query_tokens or index.size == 0:
        return []

    scored: list[tuple[float, int]] = []
    for doc_idx in range(index.size):
        score = index._score_document(query_tokens, doc_idx)
        if score > 0.0:
            scored.append((score, doc_idx))

    scored.sort(key=lambda pair: (-pair[0], index._tie_keys[pair[1]]))
    return [(index._chunks[doc_idx], score) for score, doc_idx in scored[:top_k]]


def _assert_identical(
    actual: list[tuple[Chunk, float]],
    expected: list[tuple[Chunk, float]],
    *,
    context: str,
) -> None:
    """Assert two result lists are identical: length, chunk identity, order, exact scores.

    ``is`` on the chunk and ``==`` on the score, never ``chunk_id`` plus
    ``pytest.approx``. The two failures a postings bug actually produces are
    both quiet: dropping a chunk that scores low but nonzero shortens the
    tail, and reordering a tie swaps two entries whose scores compare equal
    to *any* tolerance. An approximate comparison sees neither.
    """
    assert len(actual) == len(expected), (
        f"{context}: postings walk returned {len(actual)} results, "
        f"whole-corpus scan returns {len(expected)}"
    )
    for rank, ((got_chunk, got_score), (want_chunk, want_score)) in enumerate(
        zip(actual, expected, strict=True)
    ):
        assert got_chunk is want_chunk, (
            f"{context}: rank {rank} is chunk {got_chunk.chunk_id!r}, "
            f"whole-corpus scan gives {want_chunk.chunk_id!r}"
        )
        assert got_score == want_score, (
            f"{context}: rank {rank} scored {got_score!r}, whole-corpus scan gives {want_score!r}"
        )


class _RecordingBM25Index(BM25Index):
    """A :class:`BM25Index` that records which chunk indices ``search`` scored.

    Subclassed rather than monkeypatched onto the class object, which
    ``tests/test_retrieval.py`` already does for an unrelated property
    (which thread the scoring runs on); two patches of one method are a
    collision waiting to happen and an instance attribute is not.
    """

    scored_indices: list[int]

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        super().__init__(k1=k1, b=b)
        self.scored_indices = []

    def _score_document(self, query_tokens: list[str], doc_idx: int) -> float:
        self.scored_indices.append(doc_idx)
        return super()._score_document(query_tokens, doc_idx)


def _selective_query(chunks: list[Chunk]) -> str:
    """Two rare terms drawn from two different corpus chunks.

    Derived from the corpus rather than hardcoded: what the selectivity test
    needs is a query whose postings union is a non-empty *strict* subset of
    the corpus and spans more than one postings list, and that has to be true
    of the corpus as it stands rather than of a term somebody picked once and
    a later authoring pass made common.
    """
    doc_freqs: Counter[str] = Counter()
    for chunk in chunks:
        doc_freqs.update(set(_tokenize(chunk.content)))

    def _rarest_in(chunk: Chunk) -> str:
        tokens = set(_tokenize(chunk.content))
        assert tokens, f"corpus chunk {chunk.chunk_id!r} tokenizes to nothing"
        return min(tokens, key=lambda term: (doc_freqs[term], term))

    return f"{_rarest_in(chunks[0])} {_rarest_in(chunks[-1])}"


@pytest.fixture(scope="module")
def corpus_chunks(tmp_path_factory: pytest.TempPathFactory) -> list[Chunk]:
    """The real ``evals/corpus/``, chunked exactly as ``grk eval`` chunks it.

    Mirrors ``tests/test_corpus_integrity.py``'s ingest fixture, pinned to
    :data:`~groundkit.evals.runner.EVAL_CHUNKING_CONFIG` so the equivalence
    check runs over the same chunk boundaries the golden baseline was
    measured against, not over whatever the library defaults are today.
    """

    async def _build() -> list[Chunk]:
        index_dir = tmp_path_factory.mktemp("bm25-postings")
        store = await SQLiteMetadataStore.open(index_dir, "default")
        try:
            indexer = Indexer(
                store,
                FileLoader(allowed_base_dir=CORPUS_DIR),
                chunking_config=EVAL_CHUNKING_CONFIG,
            )
            await indexer.index_directory(str(CORPUS_DIR))
            return await store.get_chunks()
        finally:
            await store.close()

    return asyncio.run(_build())


@pytest.fixture(scope="module")
def corpus_index(corpus_chunks: list[Chunk]) -> BM25Index:
    """A ``BM25Index`` over the golden corpus. Read-only: no test may mutate it."""
    index = BM25Index()
    index.index_chunks(corpus_chunks)
    return index


class TestPostingsEquivalence:
    """GK-018's central claim: restricting the walk changed no output."""

    def test_search_matches_a_whole_corpus_scan_for_every_golden_query(
        self, corpus_index: BM25Index, corpus_chunks: list[Chunk]
    ) -> None:
        """Every query in ``evals/judgments.jsonl``, full ranking, byte-identical.

        ``top_k`` is the whole corpus rather than the eval harness's cutoff:
        the head of the ranking is dominated by chunks holding several query
        terms, and the entry a dropped posting removes is a chunk sharing one
        term with the query, scoring low but nonzero, sitting in the tail
        nobody would have compared under a small ``top_k``.
        """
        queries = [judgment.query for judgment in load_judgments(JUDGMENTS_PATH)]
        assert queries, "the golden judgment set is empty; this test would assert nothing"

        top_k = corpus_index.size
        assert top_k, "the golden corpus indexed to zero chunks"

        ranking_lengths: list[int] = []
        for query in queries:
            actual = corpus_index.search(query, top_k=top_k)
            expected = _full_scan(corpus_index, query, top_k=top_k)
            _assert_identical(actual, expected, context=f"query {query!r}")
            ranking_lengths.append(len(actual))

        assert any(ranking_lengths), (
            "no golden query matched any chunk; the comparison above asserted nothing"
        )

        # A natural-language query carries stopwords, so most of the rankings
        # above are near-corpus-wide and none of them is evidence that the
        # walk narrowed. One deliberately selective query supplies that: its
        # ranking is a strict, non-empty subset of the corpus, and it is
        # compared the same way.
        selective = _selective_query(corpus_chunks)
        actual = corpus_index.search(selective, top_k=top_k)
        _assert_identical(
            actual,
            _full_scan(corpus_index, selective, top_k=top_k),
            context=f"selective query {selective!r}",
        )
        assert 0 < len(actual) < corpus_index.size, (
            f"{selective!r} ranked {len(actual)} of {corpus_index.size} chunks; the "
            "comparison needs a query that neither misses everything nor matches everything"
        )

    def test_search_scores_only_the_chunks_holding_a_query_term(
        self, corpus_chunks: list[Chunk]
    ) -> None:
        """The defect itself: which chunks ``search`` scores.

        Equivalence alone cannot see GK-018 -- the pre-fix code was
        score-equivalent to the fixed code by construction, that being the
        entire point. What distinguishes them is the *work*: this asserts the
        exact set of chunk indices handed to ``_score_document``, which before
        the postings map was every chunk in the corpus.
        """
        index = _RecordingBM25Index()
        index.index_chunks(corpus_chunks)

        query = _selective_query(corpus_chunks)
        query_terms = set(_tokenize(query))
        expected_candidates = [
            doc_idx
            for doc_idx, chunk in enumerate(corpus_chunks)
            if query_terms & set(_tokenize(chunk.content))
        ]
        assert expected_candidates, f"{query!r} matches no golden-corpus chunk"
        assert len(expected_candidates) < len(corpus_chunks), (
            f"{query!r} matches every chunk, so this test could not tell a postings "
            "walk from a whole-corpus scan"
        )

        index.search(query, top_k=len(corpus_chunks))

        # List equality, not set equality: it pins the ascending walk order
        # the stable sort's insertion-order fallback depends on, as well as
        # membership.
        assert index.scored_indices == expected_candidates, (
            f"{query!r} scored {len(index.scored_indices)} of {len(corpus_chunks)} chunks; "
            f"only the {len(expected_candidates)} holding a query term may be scored, "
            "in ascending index order"
        )

    def test_postings_agree_with_the_document_frequency_counter(
        self, corpus_index: BM25Index
    ) -> None:
        """``len(postings[term]) == doc_freqs[term]``, and each list is a sorted set.

        Both are written from one first-sighting branch in ``index_chunks``,
        so they cannot disagree about which chunks hold a term unless that
        branch is split -- at which point IDF and candidate selection would be
        answering the same question differently. The sorted/duplicate-free
        shape is the property ``search`` leans on for its walk order.
        """
        assert corpus_index._postings, "the golden corpus produced no postings at all"
        assert set(corpus_index._postings) == set(corpus_index._doc_freqs)

        for term, postings in corpus_index._postings.items():
            assert postings == sorted(set(postings)), (
                f"postings for {term!r} are not strictly ascending and duplicate-free: {postings}"
            )
            assert len(postings) == corpus_index._doc_freqs[term], (
                f"postings for {term!r} name {len(postings)} chunks but the document-frequency "
                f"counter says {corpus_index._doc_freqs[term]}"
            )

    def test_dropping_one_posting_changes_the_result_list(self) -> None:
        """The equivalence comparison above has teeth.

        Not a test of production behaviour -- nothing in ``src/`` mutates
        ``_postings`` -- but of the *check*. A comparison between ``search``
        and a whole-corpus scan is worth its runtime only if a wrong postings
        list makes the two disagree, and the disagreement a real bug produces
        is the quiet one: a low-scoring-but-nonzero chunk missing from the
        tail, not a crash.
        """
        chunks = [
            _make_chunk("shared alpha filler filler filler", chunk_id="c0", chunk_index=0),
            _make_chunk("shared beta", chunk_id="c1", chunk_index=1),
        ]
        index = BM25Index()
        index.index_chunks(chunks)
        top_k = len(chunks)

        intact = index.search("shared", top_k=top_k)
        _assert_identical(intact, _full_scan(index, "shared", top_k=top_k), context="intact")
        assert len(intact) == len(chunks), "both chunks must score nonzero for the drop to matter"

        positions = {chunk.chunk_id: doc_idx for doc_idx, chunk in enumerate(chunks)}
        dropped_id = intact[-1][0].chunk_id  # the tail entry, where the quiet bug hides
        index._postings["shared"].remove(positions[dropped_id])

        sabotaged = index.search("shared", top_k=top_k)

        assert [chunk.chunk_id for chunk, _ in sabotaged] == [
            chunk.chunk_id for chunk, _ in intact[:-1]
        ]
        with pytest.raises(AssertionError):
            _assert_identical(
                sabotaged,
                _full_scan(index, "shared", top_k=top_k),
                context="sabotaged",
            )


class TestPostingsWalkOrder:
    """The walk is ordered, because the tie-break's last resort is insertion order."""

    def test_tied_and_byte_identical_chunks_keep_the_whole_corpus_scan_order(self) -> None:
        """Three chunks tie exactly; two of them are byte-identical.

        ``content_hash`` separates the distinct chunk from the twins, and
        nothing can separate the twins from each other -- so they fall back
        to the order the walk visited them in, which is only insertion order
        while the walk stays ascending. This is the one way GK-018 could have
        changed output with every score-equality assertion still passing.
        """
        twin_text = f"{_TIE_TERM} twin"
        distinct_text = f"{_TIE_TERM} distinct"
        chunks: list[Chunk] = []
        for doc_idx in range(_TIE_CORPUS_SIZE):
            if doc_idx == _DISTINCT_IDX:
                content, chunk_id = distinct_text, "distinct"
            elif doc_idx == _FIRST_TWIN_IDX:
                content, chunk_id = twin_text, "twin-first"
            elif doc_idx == _SECOND_TWIN_IDX:
                content, chunk_id = twin_text, "twin-second"
            else:
                content, chunk_id = f"filler prose number {doc_idx}", f"filler-{doc_idx}"
            chunks.append(_make_chunk(content, chunk_id=chunk_id, chunk_index=doc_idx))

        index = BM25Index()
        index.index_chunks(chunks)

        results = index.search(_TIE_TERM, top_k=_TIE_CORPUS_SIZE)
        _assert_identical(
            results,
            _full_scan(index, _TIE_TERM, top_k=_TIE_CORPUS_SIZE),
            context=f"tie query {_TIE_TERM!r}",
        )

        result_ids = [chunk.chunk_id for chunk, _ in results]
        assert sorted(result_ids) == ["distinct", "twin-first", "twin-second"]

        scores = [score for _, score in results]
        assert scores[0] == scores[1] == scores[2], (
            "the three chunks must score exactly equal for this to test a tie"
        )
        assert result_ids.index("twin-first") < result_ids.index("twin-second"), (
            "byte-identical chunks share a content_hash, so their order is whatever the "
            f"candidate walk produced: {result_ids}"
        )


class TestPostingsMapIsNotWrittenOnTheReadPath:
    """``_postings`` is a ``defaultdict``, and ``search`` must not subscript it."""

    def test_an_unknown_query_term_does_not_grow_the_postings_map(self) -> None:
        """A term the index has never seen leaves the map exactly as it was.

        ``self._postings[token]`` reads identically to ``.get(token, ...)`` on
        the first call and then inserts an empty list, so every never-matched
        term a caller ever searched for would accumulate -- unbounded growth
        driven by request content on the read path of an index both this class
        and ``Retriever._bm25_search`` document as frozen while a search runs,
        and a mutation a concurrent scan of the same map would see.
        """
        index = BM25Index()
        index.index_chunks([_make_chunk("indexed content", chunk_id="c0")])
        before = {term: list(postings) for term, postings in index._postings.items()}

        assert index.search("xyzzy plugh", top_k=5) == []

        assert {term: list(postings) for term, postings in index._postings.items()} == before

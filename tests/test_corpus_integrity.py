"""Corpus-integrity tests against the REAL golden eval corpus (SPEC.md §6).

EXPECTED TO FAIL right now: ``evals/corpus/`` and ``evals/judgments.jsonl``
do not exist yet on this branch — a parallel authoring pass populates them
next, using these tests as its gate. Do not skip, guard (``skipif``), or
weaken any assertion here to make it pass early, and do not create
placeholder corpus files to satisfy it; a red run is the intended interim
state, not a bug in this file.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from groundkit.contracts import Chunk
from groundkit.errors import EvalError
from groundkit.evals.corpus import (
    INJECTION_MARKERS,
    MAX_QUERY_GOLD_TOKEN_OVERLAP,
    MIN_CORPUS_DOCS,
    MIN_CORPUS_JUDGMENTS,
    Judgment,
    chunk_overlaps_span,
    load_judgments,
    query_gold_token_overlap,
    read_corpus_doc,
    resolve_gold_span,
    tokenize,
)
from groundkit.index.metadata import SQLiteMetadataStore
from groundkit.indexer import Indexer
from groundkit.ingestion.loaders import FileLoader

_REPO_ROOT: Path = Path(__file__).resolve().parent.parent
EVALS_DIR: Path = _REPO_ROOT / "evals"
CORPUS_DIR: Path = EVALS_DIR / "corpus"
JUDGMENTS_PATH: Path = EVALS_DIR / "judgments.jsonl"


@pytest.fixture(scope="module")
def judgments() -> list[Judgment]:
    """The real, parsed ``evals/judgments.jsonl`` — module-scoped, read once."""
    return load_judgments(JUDGMENTS_PATH)


@pytest.fixture(scope="module")
def corpus_index(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[list[Chunk], dict[str, str]]:
    """Build a real index over ``evals/corpus/`` and return its chunks + sources.

    Mirrors the ``SQLiteMetadataStore.open`` -> ``Indexer(store,
    FileLoader(...))`` -> ``index_directory`` -> ``get_chunks()`` ->
    ``store.close()`` pattern used throughout ``tests/test_indexer.py``.

    Returns:
        ``(chunks, document_sources)`` where ``document_sources`` maps
        ``document_id -> source`` (an absolute, resolved path string), so a
        chunk's ``document_id`` can be joined back to the corpus-relative
        ``doc`` a :class:`~groundkit.evals.corpus.GoldSpan` names.
    """

    async def _build() -> tuple[list[Chunk], dict[str, str]]:
        index_dir = tmp_path_factory.mktemp("corpus-index")
        store = await SQLiteMetadataStore.open(index_dir, "default")
        try:
            indexer = Indexer(store, FileLoader(allowed_base_dir=CORPUS_DIR))
            await indexer.index_directory(str(CORPUS_DIR))
            chunks = await store.get_chunks()
            sources = await store.get_document_sources()
            return chunks, sources
        finally:
            await store.close()

    return asyncio.run(_build())


class TestCorpusSizeFloors:
    """SPEC.md §6: the size floor is asserted in the test, not the README."""

    def test_document_count_meets_floor(
        self, corpus_index: tuple[list[Chunk], dict[str, str]]
    ) -> None:
        _, sources = corpus_index
        assert len(sources) >= 8
        # The module constant must never drift below the literal floor above.
        assert MIN_CORPUS_DOCS >= 8

    def test_judgment_count_meets_floor(self, judgments: list[Judgment]) -> None:
        assert len(judgments) >= 40
        # The module constant must never drift below the literal floor above.
        assert MIN_CORPUS_JUDGMENTS >= 40


class TestJudgmentsFileHygiene:
    def test_unique_query_ids(self, judgments: list[Judgment]) -> None:
        ids = [j.query_id for j in judgments]
        assert len(ids) == len(set(ids))

    def test_sorted_by_query_id(self, judgments: list[Judgment]) -> None:
        ids = [j.query_id for j in judgments]
        assert ids == sorted(ids)

    def test_every_referenced_doc_exists(self, judgments: list[Judgment]) -> None:
        missing = sorted(
            {
                span.doc
                for j in judgments
                for span in j.gold
                if not (CORPUS_DIR / span.doc).is_file()
            }
        )
        assert not missing, f"judgments reference missing corpus docs: {missing}"


class TestGoldQuoteResolution:
    def test_every_gold_quote_resolves_exactly_once(self, judgments: list[Judgment]) -> None:
        """Collects every broken quote across the whole corpus in one run,
        rather than failing on the first one — a corpus author needs to see
        every break at once, not fix-and-rerun one at a time."""
        failures: list[str] = []
        doc_text_cache: dict[str, str] = {}
        for judgment in judgments:
            for span in judgment.gold:
                if span.doc not in doc_text_cache:
                    try:
                        doc_text_cache[span.doc] = read_corpus_doc(CORPUS_DIR, span.doc)
                    except EvalError as exc:
                        failures.append(f"{judgment.query_id} ({span.doc}): {exc}")
                        continue
                text = doc_text_cache[span.doc]
                try:
                    resolve_gold_span(text, span.quote)
                except EvalError as exc:
                    failures.append(f"{judgment.query_id} ({span.doc}): {exc}")
        assert not failures, "unresolved gold quotes:\n" + "\n".join(failures)

    def test_every_resolved_span_overlaps_an_indexed_chunk(
        self,
        judgments: list[Judgment],
        corpus_index: tuple[list[Chunk], dict[str, str]],
    ) -> None:
        chunks, sources = corpus_index
        chunks_by_source: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            source = sources.get(chunk.document_id)
            if source is not None:
                chunks_by_source.setdefault(source, []).append(chunk)

        failures: list[str] = []
        for judgment in judgments:
            for span in judgment.gold:
                try:
                    text = read_corpus_doc(CORPUS_DIR, span.doc)
                    resolved = resolve_gold_span(text, span.quote)
                except EvalError:
                    continue  # already reported by test_every_gold_quote_resolves_exactly_once
                expected_source = str((CORPUS_DIR / span.doc).resolve())
                doc_chunks = chunks_by_source.get(expected_source, [])
                if not any(
                    chunk_overlaps_span(c.start_offset, c.end_offset, resolved) for c in doc_chunks
                ):
                    failures.append(
                        f"{judgment.query_id} ({span.doc}): resolved span {resolved} overlaps "
                        f"none of the {len(doc_chunks)} indexed chunks for this doc"
                    )
        assert not failures, "gold spans with no overlapping chunk:\n" + "\n".join(failures)


class TestCategoryCoverage:
    def test_all_four_categories_present(self, judgments: list[Judgment]) -> None:
        categories = {j.category for j in judgments}
        assert categories == {"normal", "ambiguous", "no_answer", "adversarial"}


class TestCircularityGuard:
    def test_query_gold_overlap_below_circularity_threshold(
        self, judgments: list[Judgment]
    ) -> None:
        failures: list[str] = []
        for judgment in judgments:
            if judgment.category == "no_answer":
                continue
            for span in judgment.gold:
                overlap = query_gold_token_overlap(judgment.query, span.quote)
                if overlap > MAX_QUERY_GOLD_TOKEN_OVERLAP:
                    failures.append(
                        f"{judgment.query_id} ({span.doc}): overlap {overlap:.2f} exceeds "
                        f"MAX_QUERY_GOLD_TOKEN_OVERLAP ({MAX_QUERY_GOLD_TOKEN_OVERLAP})"
                    )
        assert not failures, "queries too lexically similar to their gold quotes:\n" + "\n".join(
            failures
        )


class TestNoAnswerVocabulary:
    def test_no_answer_queries_share_zero_corpus_vocabulary(
        self,
        judgments: list[Judgment],
        corpus_index: tuple[list[Chunk], dict[str, str]],
    ) -> None:
        chunks, _ = corpus_index
        vocabulary: set[str] = set()
        for chunk in chunks:
            vocabulary.update(tokenize(chunk.content))

        failures: list[str] = []
        for judgment in judgments:
            if judgment.category != "no_answer":
                continue
            shared = set(tokenize(judgment.query)) & vocabulary
            if shared:
                failures.append(f"{judgment.query_id}: shares tokens {sorted(shared)} with corpus")
        assert not failures, "no_answer queries share corpus vocabulary:\n" + "\n".join(failures)


class TestAdversarialInjectionMarkers:
    def test_adversarial_docs_contain_injection_marker(self, judgments: list[Judgment]) -> None:
        adversarial_docs = {
            span.doc for j in judgments if j.category == "adversarial" for span in j.gold
        }
        failures: list[str] = []
        for doc in sorted(adversarial_docs):
            text = read_corpus_doc(CORPUS_DIR, doc).lower()
            if not any(marker in text for marker in INJECTION_MARKERS):
                failures.append(doc)
        assert not failures, f"adversarial docs missing an INJECTION_MARKERS phrase: {failures}"


class TestReadmeLocation:
    """Defends the ingest-root rule: a markdown file inside evals/corpus/
    would silently become a corpus document (FileLoader ingests .md)."""

    def test_evals_readme_exists(self) -> None:
        assert (EVALS_DIR / "README.md").is_file()

    def test_corpus_readme_does_not_exist(self) -> None:
        assert not (CORPUS_DIR / "README.md").exists()

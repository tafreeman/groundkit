"""Unit tests for the golden-eval corpus schema and quote resolution.

Pure, synthetic fixtures only (``tmp_path``) — the real ``evals/`` corpus is
exercised separately by ``tests/test_corpus_integrity.py``, which is expected
to fail until a parallel authoring pass populates it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from groundkit.errors import EvalError
from groundkit.evals.corpus import (
    Category,
    GoldSpan,
    Judgment,
    chunk_overlaps_span,
    load_judgments,
    query_gold_token_overlap,
    read_corpus_doc,
    resolve_gold_span,
    tokenize,
)


def _gold(doc: str = "a.md", quote: str = "hello world") -> GoldSpan:
    return GoldSpan(doc=doc, quote=quote)


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestGoldSpanDocValidation:
    @pytest.mark.parametrize("doc", ["", "   ", "\t\n"])
    def test_empty_or_whitespace_rejected(self, doc: str) -> None:
        with pytest.raises(ValidationError, match="empty or whitespace"):
            GoldSpan(doc=doc, quote="q")

    def test_backslash_rejected(self) -> None:
        with pytest.raises(ValidationError, match="backslash"):
            GoldSpan(doc="sub\\doc.md", quote="q")

    def test_leading_slash_absolute_rejected(self) -> None:
        with pytest.raises(ValidationError, match="absolute"):
            GoldSpan(doc="/etc/passwd", quote="q")

    def test_drive_letter_absolute_rejected(self) -> None:
        with pytest.raises(ValidationError, match="absolute"):
            GoldSpan(doc="C:/secrets.md", quote="q")

    def test_dotdot_leading_segment_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"\.\."):
            GoldSpan(doc="../outside.md", quote="q")

    def test_dotdot_mid_path_segment_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"\.\."):
            GoldSpan(doc="a/../b.md", quote="q")

    def test_valid_relative_doc_accepted(self) -> None:
        span = GoldSpan(doc="guides/setup.md", quote="q")
        assert span.doc == "guides/setup.md"

    def test_empty_quote_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GoldSpan(doc="a.md", quote="")


class TestGoldSpanFrozenAndForbid:
    def test_frozen(self) -> None:
        span = _gold()
        with pytest.raises(ValidationError):
            span.doc = "other.md"

    def test_extra_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            GoldSpan(doc="a.md", quote="q", bogus="nope")  # type: ignore[call-arg]


class TestJudgmentQueryIdPattern:
    def test_kebab_case_accepted(self) -> None:
        j = Judgment(query_id="setup-timeout", query="q", category="normal", gold=[_gold()])
        assert j.query_id == "setup-timeout"

    @pytest.mark.parametrize(
        "query_id",
        [
            "Setup-Timeout",
            "setup_timeout",
            "setup timeout",
            "-setup",
            "setup-",
            "",
            "setup--timeout",
        ],
    )
    def test_invalid_query_id_rejected(self, query_id: str) -> None:
        with pytest.raises(ValidationError):
            Judgment(query_id=query_id, query="q", category="normal", gold=[_gold()])


class TestJudgmentCategoryGoldInvariants:
    def test_no_answer_requires_empty_gold(self) -> None:
        with pytest.raises(ValidationError, match="no_answer"):
            Judgment(query_id="q1", query="q", category="no_answer", gold=[_gold()])

    def test_no_answer_with_empty_gold_accepted(self) -> None:
        j = Judgment(query_id="q1", query="q", category="no_answer", gold=[])
        assert j.gold == []

    @pytest.mark.parametrize("category", ["normal", "adversarial"])
    def test_answerable_categories_require_nonempty_gold(self, category: Category) -> None:
        with pytest.raises(ValidationError, match="at least one gold span"):
            Judgment(query_id="q1", query="q", category=category, gold=[])

    def test_ambiguous_requires_at_least_two_gold(self) -> None:
        with pytest.raises(ValidationError, match="at least 2 gold spans"):
            Judgment(query_id="q1", query="q", category="ambiguous", gold=[_gold()])

    def test_ambiguous_with_two_gold_accepted(self) -> None:
        j = Judgment(
            query_id="q1",
            query="q",
            category="ambiguous",
            gold=[_gold(quote="first answer"), _gold(quote="second answer")],
        )
        assert len(j.gold) == 2

    def test_normal_with_one_gold_accepted(self) -> None:
        j = Judgment(query_id="q1", query="q", category="normal", gold=[_gold()])
        assert len(j.gold) == 1

    def test_unknown_category_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Judgment(query_id="q1", query="q", category="bogus", gold=[_gold()])  # type: ignore[arg-type]


class TestJudgmentFrozenAndForbid:
    def test_frozen(self) -> None:
        j = Judgment(query_id="q1", query="q", category="normal", gold=[_gold()])
        with pytest.raises(ValidationError):
            j.query = "different"

    def test_extra_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            Judgment(
                query_id="q1",
                query="q",
                category="normal",
                gold=[_gold()],
                bogus="nope",  # type: ignore[call-arg]
            )


class TestLoadJudgments:
    def test_loads_valid_sorted_file(self, tmp_path: Path) -> None:
        path = tmp_path / "judgments.jsonl"
        _write_lines(
            path,
            [
                json.dumps(
                    {
                        "query_id": "aaa-first",
                        "query": "q1",
                        "category": "normal",
                        "gold": [{"doc": "a.md", "quote": "hi"}],
                    }
                ),
                json.dumps(
                    {"query_id": "bbb-second", "query": "q2", "category": "no_answer", "gold": []}
                ),
            ],
        )
        judgments = load_judgments(path)
        assert [j.query_id for j in judgments] == ["aaa-first", "bbb-second"]

    def test_blank_lines_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "judgments.jsonl"
        record = json.dumps({"query_id": "aaa", "query": "q", "category": "no_answer", "gold": []})
        path.write_text(f"\n{record}\n\n   \n", encoding="utf-8")
        judgments = load_judgments(path)
        assert len(judgments) == 1

    def test_malformed_json_names_line_number(self, tmp_path: Path) -> None:
        path = tmp_path / "judgments.jsonl"
        valid = json.dumps({"query_id": "aaa", "query": "q", "category": "no_answer", "gold": []})
        _write_lines(path, [valid, "{not json"])
        with pytest.raises(EvalError, match=r":2:"):
            load_judgments(path)

    def test_schema_violation_names_line_number(self, tmp_path: Path) -> None:
        path = tmp_path / "judgments.jsonl"
        valid = json.dumps({"query_id": "aaa", "query": "q", "category": "no_answer", "gold": []})
        bad = json.dumps({"query_id": "bbb", "query": "q", "category": "bogus", "gold": []})
        _write_lines(path, [valid, bad])
        with pytest.raises(EvalError, match=r":2:"):
            load_judgments(path)

    def test_duplicate_query_id_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "judgments.jsonl"
        record = json.dumps({"query_id": "aaa", "query": "q", "category": "no_answer", "gold": []})
        _write_lines(path, [record, record])
        with pytest.raises(EvalError, match="duplicate query_id"):
            load_judgments(path)

    def test_out_of_order_query_id_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "judgments.jsonl"
        first = json.dumps({"query_id": "bbb", "query": "q", "category": "no_answer", "gold": []})
        second = json.dumps({"query_id": "aaa", "query": "q", "category": "no_answer", "gold": []})
        _write_lines(path, [first, second])
        with pytest.raises(EvalError, match="ascending order"):
            load_judgments(path)


class TestResolveGoldSpan:
    def test_found_returns_span(self) -> None:
        text = "The quick brown fox jumps."
        start, end = resolve_gold_span(text, "brown fox")
        assert text[start:end] == "brown fox"

    def test_not_found_raises(self) -> None:
        with pytest.raises(EvalError, match="not found"):
            resolve_gold_span("The quick brown fox.", "slow turtle")

    def test_ambiguous_duplicate_raises(self) -> None:
        with pytest.raises(EvalError, match="more than once"):
            resolve_gold_span("cat sat on the cat mat", "cat")


class TestReadCorpusDoc:
    def test_reads_utf8_text(self, tmp_path: Path) -> None:
        (tmp_path / "a.md").write_text("hello world", encoding="utf-8")
        assert read_corpus_doc(tmp_path, "a.md") == "hello world"

    def test_traversal_rejected(self, tmp_path: Path) -> None:
        docs_dir = tmp_path / "corpus"
        docs_dir.mkdir()
        (tmp_path / "secret.md").write_text("top secret", encoding="utf-8")
        with pytest.raises(EvalError):
            read_corpus_doc(docs_dir, "../secret.md")

    def test_invalid_utf8_raises_eval_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.md"
        bad.write_bytes(b"\xff\xfe\x00invalid")
        with pytest.raises(EvalError, match="not valid UTF-8") as exc_info:
            read_corpus_doc(tmp_path, "bad.md")
        assert isinstance(exc_info.value.__cause__, UnicodeDecodeError)

    def test_missing_file_raises_eval_error(self, tmp_path: Path) -> None:
        with pytest.raises(EvalError):
            read_corpus_doc(tmp_path, "missing.md")


class TestChunkOverlapsSpan:
    def test_full_containment_overlaps(self) -> None:
        assert chunk_overlaps_span(0, 100, (10, 20)) is True

    def test_partial_overlap_left(self) -> None:
        assert chunk_overlaps_span(5, 15, (10, 20)) is True

    def test_partial_overlap_right(self) -> None:
        assert chunk_overlaps_span(15, 25, (10, 20)) is True

    def test_chunk_ends_where_span_starts_not_overlapping(self) -> None:
        assert chunk_overlaps_span(0, 10, (10, 20)) is False

    def test_span_ends_where_chunk_starts_not_overlapping(self) -> None:
        assert chunk_overlaps_span(20, 30, (10, 20)) is False

    def test_disjoint_not_overlapping(self) -> None:
        assert chunk_overlaps_span(0, 5, (10, 20)) is False

    def test_span_contains_chunk_overlaps(self) -> None:
        assert chunk_overlaps_span(12, 15, (10, 20)) is True


class TestTokenize:
    def test_lowercases_and_strips_punctuation(self) -> None:
        assert tokenize("Hello, World!") == ["hello", "world"]

    def test_empty_string(self) -> None:
        assert tokenize("") == []


class TestQueryGoldTokenOverlap:
    def test_full_overlap_is_one(self) -> None:
        assert query_gold_token_overlap("brown fox", "the quick brown fox jumps") == 1.0

    def test_no_overlap_is_zero(self) -> None:
        assert query_gold_token_overlap("slow turtle", "the quick brown fox") == 0.0

    def test_partial_overlap_arithmetic(self) -> None:
        assert query_gold_token_overlap("alpha beta", "beta gamma") == 0.5

    def test_empty_query_is_zero(self) -> None:
        assert query_gold_token_overlap("", "beta gamma") == 0.0

    def test_duplicate_query_tokens_counted_once(self) -> None:
        assert query_gold_token_overlap("alpha beta beta", "beta") == 0.5

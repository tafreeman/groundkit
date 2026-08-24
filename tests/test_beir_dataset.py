from __future__ import annotations

import json
from pathlib import Path

import pytest

from groundkit.errors import EvalError
from groundkit.evals.beir import adapt_beir_dataset
from groundkit.evals.corpus import load_judgments


def _write_beir(root: Path, *, document_id: str = "doc-1") -> None:
    root.mkdir()
    (root / "qrels").mkdir()
    (root / "corpus.jsonl").write_text(
        json.dumps({"_id": document_id, "title": "Title", "text": "Body"}) + "\n",
        encoding="utf-8",
    )
    (root / "queries.jsonl").write_text(
        json.dumps({"_id": "q-1", "text": "What is in the body?"}) + "\n",
        encoding="utf-8",
    )
    (root / "qrels" / "test.tsv").write_text(
        f"query-id\tcorpus-id\tscore\nq-1\t{document_id}\t1\n",
        encoding="utf-8",
    )


def test_adapt_beir_dataset_preserves_document_level_qrel_as_whole_document_quote(
    tmp_path: Path,
) -> None:
    source = tmp_path / "beir"
    output = tmp_path / "adapted"
    _write_beir(source)

    report = adapt_beir_dataset(source, output)

    assert report.document_count == 1
    assert report.query_count == 1
    assert report.relevance_pair_count == 1
    assert (output / "corpus" / "doc-1.txt").read_text(encoding="utf-8") == "Title\n\nBody"
    judgments = load_judgments(output / "judgments.jsonl")
    assert judgments[0].gold[0].doc == "doc-1.txt"
    assert judgments[0].gold[0].quote == "Title\n\nBody"


def test_adapt_beir_dataset_ignores_zero_relevance_rows(tmp_path: Path) -> None:
    source = tmp_path / "beir"
    output = tmp_path / "adapted"
    _write_beir(source)
    (source / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq-1\tdoc-1\t0\n",
        encoding="utf-8",
    )

    with pytest.raises(EvalError, match="no positive relevance"):
        adapt_beir_dataset(source, output)


def test_adapt_beir_dataset_rejects_path_like_document_ids(tmp_path: Path) -> None:
    source = tmp_path / "beir"
    _write_beir(source, document_id="../escape")

    with pytest.raises(EvalError, match="unsafe BEIR document id"):
        adapt_beir_dataset(source, tmp_path / "adapted")


def test_adapt_beir_dataset_rejects_a_path_shaped_split(tmp_path: Path) -> None:
    """``split`` is interpolated into ``qrels/<split>.tsv``, so it takes the
    same character-class discipline as every other identifier here."""
    source = tmp_path / "beir"
    _write_beir(source)

    with pytest.raises(EvalError, match="unsafe BEIR split"):
        adapt_beir_dataset(source, tmp_path / "adapted", split="../../etc/passwd")


def test_query_ids_the_harness_would_reject_fail_before_anything_is_written(
    tmp_path: Path,
) -> None:
    """The adapter must not leave a half-populated corpus behind.

    ``_SAFE_IDENTIFIER`` is a path-safety class and admits ``MED-10``; the
    harness's ``Judgment.query_id`` contract is stricter (kebab-case
    lowercase) and does not. Discovering that only after writing the corpus
    left the output directory populated but unusable.
    """
    source = tmp_path / "beir"
    output = tmp_path / "adapted"
    _write_beir(source)
    (source / "queries.jsonl").write_text(
        json.dumps({"_id": "MED-10", "text": "What is in the body?"}) + "\n",
        encoding="utf-8",
    )
    (source / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nMED-10\tdoc-1\t1\n",
        encoding="utf-8",
    )

    with pytest.raises(EvalError, match="not admissible as groundkit judgment ids"):
        adapt_beir_dataset(source, output)

    # The refusal must precede every write, not follow it.
    assert not (output / "corpus").exists()
    assert not (output / "judgments.jsonl").exists()


def test_adapt_beir_dataset_refuses_to_write_into_a_populated_corpus_dir(
    tmp_path: Path,
) -> None:
    """Adapting a second dataset over the first silently unions them: the
    stale documents stay retrievable and get scored against judgments that
    never referenced them."""
    source = tmp_path / "beir"
    output = tmp_path / "adapted"
    _write_beir(source)
    adapt_beir_dataset(source, output)
    stale = output / "corpus" / "left-over-from-another-dataset.txt"
    stale.write_text("stale", encoding="utf-8")

    with pytest.raises(EvalError, match="already contains"):
        adapt_beir_dataset(source, output)

    assert stale.read_text(encoding="utf-8") == "stale"


# -- malformed source data ------------------------------------------------------
#
# A BEIR directory is an untrusted third-party download, so every parse failure
# must surface as a catchable EvalError naming the file and line rather than an
# IndexError, KeyError or ValueError escaping from the parser.


def test_missing_source_files_are_reported_as_eval_errors(tmp_path: Path) -> None:
    with pytest.raises(EvalError, match="cannot read BEIR file"):
        adapt_beir_dataset(tmp_path / "absent", tmp_path / "adapted")


def test_malformed_jsonl_line_names_the_file_and_line(tmp_path: Path) -> None:
    source = tmp_path / "beir"
    _write_beir(source)
    (source / "corpus.jsonl").write_text("{not json\n", encoding="utf-8")

    with pytest.raises(EvalError, match=r"invalid JSON in .* line 1"):
        adapt_beir_dataset(source, tmp_path / "adapted")


def test_a_jsonl_row_that_is_not_an_object_is_refused(tmp_path: Path) -> None:
    source = tmp_path / "beir"
    _write_beir(source)
    (source / "corpus.jsonl").write_text('["a list, not an object"]\n', encoding="utf-8")

    with pytest.raises(EvalError, match="must be a JSON object"):
        adapt_beir_dataset(source, tmp_path / "adapted")


def test_a_row_missing_its_id_is_refused(tmp_path: Path) -> None:
    source = tmp_path / "beir"
    _write_beir(source)
    (source / "corpus.jsonl").write_text(
        json.dumps({"title": "T", "text": "B"}) + "\n", encoding="utf-8"
    )

    with pytest.raises(EvalError, match="requires non-empty string field '_id'"):
        adapt_beir_dataset(source, tmp_path / "adapted")


def test_a_non_string_title_is_refused_rather_than_coerced(tmp_path: Path) -> None:
    source = tmp_path / "beir"
    _write_beir(source)
    (source / "corpus.jsonl").write_text(
        json.dumps({"_id": "doc-1", "title": 7, "text": "B"}) + "\n", encoding="utf-8"
    )

    with pytest.raises(EvalError, match="must be a string when present"):
        adapt_beir_dataset(source, tmp_path / "adapted")


def test_a_document_with_neither_title_nor_text_is_refused(tmp_path: Path) -> None:
    source = tmp_path / "beir"
    _write_beir(source)
    (source / "corpus.jsonl").write_text(
        json.dumps({"_id": "doc-1", "title": "  ", "text": "  "}) + "\n", encoding="utf-8"
    )

    with pytest.raises(EvalError, match="has no title or text"):
        adapt_beir_dataset(source, tmp_path / "adapted")


def test_duplicate_document_ids_are_refused(tmp_path: Path) -> None:
    source = tmp_path / "beir"
    _write_beir(source)
    row = json.dumps({"_id": "doc-1", "title": "T", "text": "B"})
    (source / "corpus.jsonl").write_text(f"{row}\n{row}\n", encoding="utf-8")

    with pytest.raises(EvalError, match="duplicate BEIR document id"):
        adapt_beir_dataset(source, tmp_path / "adapted")


def test_an_empty_query_is_refused(tmp_path: Path) -> None:
    source = tmp_path / "beir"
    _write_beir(source)
    (source / "queries.jsonl").write_text(
        json.dumps({"_id": "q-1", "text": "   "}) + "\n", encoding="utf-8"
    )

    with pytest.raises(EvalError, match="is empty"):
        adapt_beir_dataset(source, tmp_path / "adapted")


def test_qrels_naming_an_absent_document_are_refused(tmp_path: Path) -> None:
    source = tmp_path / "beir"
    _write_beir(source)
    (source / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq-1\tdoc-absent\t1\n", encoding="utf-8"
    )

    with pytest.raises(EvalError, match="reference missing documents"):
        adapt_beir_dataset(source, tmp_path / "adapted")


def test_a_wrong_qrels_header_is_refused(tmp_path: Path) -> None:
    source = tmp_path / "beir"
    _write_beir(source)
    (source / "qrels" / "test.tsv").write_text("a\tb\tc\nq-1\tdoc-1\t1\n", encoding="utf-8")

    with pytest.raises(EvalError, match="unexpected or missing BEIR qrels header"):
        adapt_beir_dataset(source, tmp_path / "adapted")


def test_a_short_qrels_row_is_refused(tmp_path: Path) -> None:
    source = tmp_path / "beir"
    _write_beir(source)
    (source / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq-1\tdoc-1\n", encoding="utf-8"
    )

    with pytest.raises(EvalError, match="invalid BEIR qrels row"):
        adapt_beir_dataset(source, tmp_path / "adapted")


def test_a_non_integer_relevance_score_is_refused(tmp_path: Path) -> None:
    source = tmp_path / "beir"
    _write_beir(source)
    (source / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq-1\tdoc-1\tvery\n", encoding="utf-8"
    )

    with pytest.raises(EvalError, match="invalid integer relevance score"):
        adapt_beir_dataset(source, tmp_path / "adapted")


@pytest.mark.parametrize("filename", ["corpus.jsonl", "queries.jsonl"])
def test_invalid_utf8_in_a_jsonl_file_is_an_eval_error_not_a_traceback(
    tmp_path: Path, filename: str
) -> None:
    """``UnicodeDecodeError`` is a ``ValueError``, not an ``OSError``, so it
    needs its own except clause -- without one it escapes this module's
    documented ``EvalError`` contract and reaches the CLI as a traceback."""
    source = tmp_path / "beir"
    _write_beir(source)
    (source / filename).write_bytes(b'{"_id": "\xff\xfe not utf-8"}\n')

    with pytest.raises(EvalError, match="not valid UTF-8"):
        adapt_beir_dataset(source, tmp_path / "adapted")


def test_invalid_utf8_in_the_qrels_is_an_eval_error_not_a_traceback(tmp_path: Path) -> None:
    source = tmp_path / "beir"
    _write_beir(source)
    (source / "qrels" / "test.tsv").write_bytes(b"query-id\tcorpus-id\tscore\n\xff\xfe\t1\t1\n")

    with pytest.raises(EvalError, match="not valid UTF-8"):
        adapt_beir_dataset(source, tmp_path / "adapted")

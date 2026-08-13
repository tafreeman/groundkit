"""End-to-end CLI tests: grk ingest -> grk search on a real temp index."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from groundkit.cli import main


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.md").write_text(
        "Reciprocal rank fusion combines lexical and dense rankings.", encoding="utf-8"
    )
    (docs / "other.txt").write_text("Unrelated text about gardening.", encoding="utf-8")
    return docs


def test_ingest_then_search_end_to_end(
    corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    idx = str(tmp_path / "idx")
    assert main(["ingest", str(corpus), "--index-dir", idx]) == 0
    out = capsys.readouterr().out
    assert "2 indexed" in out

    assert main(["search", "reciprocal rank fusion", "--index-dir", idx]) == 0
    out = capsys.readouterr().out
    assert "notes.md" in out
    assert "#0-" in out  # offsets shown

    # Re-ingest: everything unchanged.
    assert main(["ingest", str(corpus), "--index-dir", idx]) == 0
    assert "2 unchanged" in capsys.readouterr().out


def test_search_json_output(
    corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    idx = str(tmp_path / "idx")
    main(["ingest", str(corpus), "--index-dir", idx])
    capsys.readouterr()

    assert main(["search", "gardening", "--index-dir", idx, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["query"] == "gardening"
    assert payload["total_results"] == 1
    result = payload["results"][0]
    assert result["citation"]["source"].endswith("other.txt")
    assert result["start_offset"] == 0


def test_search_no_results(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    idx = str(tmp_path / "idx")
    assert main(["search", "anything", "--index-dir", idx]) == 0
    assert "no results" in capsys.readouterr().out


def test_error_paths_exit_nonzero(
    corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    idx = str(tmp_path / "idx")
    assert main(["ingest", str(corpus / "missing.md"), "--index-dir", idx]) == 1
    assert "error:" in capsys.readouterr().err

    main(["ingest", str(corpus), "--index-dir", idx])
    capsys.readouterr()
    assert main(["search", "x", "--index-dir", idx, "--top-k", "0"]) == 1
    assert "top_k" in capsys.readouterr().err


def test_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "usage: grk" in capsys.readouterr().out


@pytest.fixture
def eval_corpus(tmp_path: Path) -> Path:
    """A tiny two-document golden corpus for ``grk eval``."""
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "alpha.md").write_text(
        "The quokka expedition documented burrow temperatures across the reserve.",
        encoding="utf-8",
    )
    (root / "beta.md").write_text(
        "Citations resolve back to character offsets in the original source file.",
        encoding="utf-8",
    )
    return root


@pytest.fixture
def eval_judgments(tmp_path: Path) -> Path:
    """A minimal judgments JSONL matching ``eval_corpus``."""
    judgments_path = tmp_path / "judgments.jsonl"
    lines = [
        json.dumps(
            {
                "query_id": "citation-offsets",
                "query": "citations resolve character offsets",
                "category": "normal",
                "gold": [{"doc": "beta.md", "quote": "character offsets"}],
            }
        ),
        json.dumps(
            {
                "query_id": "quokka-burrow",
                "query": "quokka expedition burrow temperatures",
                "category": "normal",
                "gold": [{"doc": "alpha.md", "quote": "quokka expedition"}],
            }
        ),
    ]
    judgments_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return judgments_path


def test_eval_writes_artifact_and_returns_zero(
    eval_corpus: Path, eval_judgments: Path, tmp_path: Path
) -> None:
    output = tmp_path / "results" / "latest.json"
    assert (
        main(
            [
                "eval",
                "--corpus-dir",
                str(eval_corpus),
                "--judgments",
                str(eval_judgments),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert output.is_file()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["stages"][0]["stage"] == "bm25"


def test_eval_json_prints_valid_eval_report(
    eval_corpus: Path, eval_judgments: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "results" / "latest.json"
    assert (
        main(
            [
                "eval",
                "--corpus-dir",
                str(eval_corpus),
                "--judgments",
                str(eval_judgments),
                "--output",
                str(output),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["run"]["judgment_count"] == 2
    assert payload["stages"][0]["is_baseline"] is True
    assert "aggregate" in payload["stages"][0]


def test_eval_top_k_below_ten_fails_cleanly(
    eval_corpus: Path, eval_judgments: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "results" / "latest.json"
    assert (
        main(
            [
                "eval",
                "--corpus-dir",
                str(eval_corpus),
                "--judgments",
                str(eval_judgments),
                "--output",
                str(output),
                "--top-k",
                "9",
            ]
        )
        == 1
    )
    err = capsys.readouterr().err
    assert "error:" in err
    assert "--top-k" in err
    assert not output.exists()


@pytest.mark.parametrize("top_k", [1, 5])
def test_eval_top_k_other_sub_ten_values_fail_cleanly(
    eval_corpus: Path,
    eval_judgments: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    top_k: int,
) -> None:
    output = tmp_path / "results" / "latest.json"
    assert (
        main(
            [
                "eval",
                "--corpus-dir",
                str(eval_corpus),
                "--judgments",
                str(eval_judgments),
                "--output",
                str(output),
                "--top-k",
                str(top_k),
            ]
        )
        == 1
    )
    assert "error:" in capsys.readouterr().err


def test_eval_missing_judgments_file_exits_nonzero(
    eval_corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "results" / "latest.json"
    missing = tmp_path / "does-not-exist.jsonl"
    assert (
        main(
            [
                "eval",
                "--corpus-dir",
                str(eval_corpus),
                "--judgments",
                str(missing),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    err = capsys.readouterr().err
    assert "error:" in err
    assert not output.exists()


def test_eval_human_summary_reports_metrics_and_output_path(
    eval_corpus: Path, eval_judgments: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "results" / "latest.json"
    assert (
        main(
            [
                "eval",
                "--corpus-dir",
                str(eval_corpus),
                "--judgments",
                str(eval_judgments),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "recall@1=" in out
    assert "recall@5=" in out
    assert "recall@10=" in out
    assert "mrr=" in out
    assert "ndcg@10=" in out
    assert "no_answer:" in out
    assert "p50=" in out
    assert "p95=" in out
    assert str(output) in out

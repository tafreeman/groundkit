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

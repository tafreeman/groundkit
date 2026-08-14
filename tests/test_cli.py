"""End-to-end CLI tests: grk ingest -> grk search on a real temp index."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from groundkit.cli import main
from groundkit.retrieval.search import MAX_TOP_K


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


def test_ingest_dense_writes_vectors_and_lance_store(
    corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    idx = str(tmp_path / "idx")
    assert (
        main(
            [
                "ingest",
                str(corpus),
                "--index-dir",
                idx,
                "--dense",
                "--embed-provider",
                "inmemory",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert out == (
        "ingested: 2 files seen, 2 indexed, 0 unchanged, 2 chunks written, "
        "2 vectors written, 0 vectors deleted\n"
    )
    assert (Path(idx) / "default.lance").is_dir()


def test_ingest_without_dense_output_format_unchanged(
    corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    idx = str(tmp_path / "idx")
    assert main(["ingest", str(corpus), "--index-dir", idx]) == 0
    out = capsys.readouterr().out
    assert out == "ingested: 2 files seen, 2 indexed, 0 unchanged, 2 chunks written\n"


def test_search_mode_dense_over_dense_ingested_collection(
    corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    idx = str(tmp_path / "idx")
    main(["ingest", str(corpus), "--index-dir", idx, "--dense", "--embed-provider", "inmemory"])
    capsys.readouterr()

    assert (
        main(
            [
                "search",
                "reciprocal rank fusion",
                "--index-dir",
                idx,
                "--mode",
                "dense",
                "--embed-provider",
                "inmemory",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "no results" not in out
    assert "1. [" in out

    assert (
        main(
            [
                "search",
                "reciprocal rank fusion",
                "--index-dir",
                idx,
                "--mode",
                "dense",
                "--embed-provider",
                "inmemory",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["metadata"]["stage"] == "dense"


def test_search_mode_hybrid_over_dense_ingested_collection_reports_fusion_stage(
    corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    idx = str(tmp_path / "idx")
    main(["ingest", str(corpus), "--index-dir", idx, "--dense", "--embed-provider", "inmemory"])
    capsys.readouterr()

    assert (
        main(
            [
                "search",
                "reciprocal rank fusion",
                "--index-dir",
                idx,
                "--mode",
                "hybrid",
                "--embed-provider",
                "inmemory",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["metadata"]["stage"] == "fusion"


def test_search_default_mode_stays_bm25_over_dense_ingested_collection(
    corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Q1 is decided (ADR-0007): the default mode stays "bm25" even over a
    # collection that also carries a dense index. Wave E measured the delta
    # and hybrid won it on quality — the default did not move, because
    # hybrid cannot abstain and would require an embedding provider the
    # default install does not ship. If this test ever fails because the
    # default changed, that change needs an accepted ADR superseding
    # ADR-0007, not a larger delta and not a CLI-wiring tweak.
    idx = str(tmp_path / "idx")
    main(["ingest", str(corpus), "--index-dir", idx, "--dense", "--embed-provider", "inmemory"])
    capsys.readouterr()

    assert main(["search", "reciprocal rank fusion", "--index-dir", idx, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["metadata"]["stage"] == "bm25"


def test_ingest_embed_flag_without_dense_fails_closed(
    corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    idx = str(tmp_path / "idx")
    assert main(["ingest", str(corpus), "--index-dir", idx, "--embed-model", "some-model"]) == 1
    err = capsys.readouterr().err
    assert "error:" in err
    assert "--dense" in err


def test_search_embed_flag_without_dense_mode_fails_closed(
    corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    idx = str(tmp_path / "idx")
    main(["ingest", str(corpus), "--index-dir", idx])
    capsys.readouterr()

    assert (
        main(
            [
                "search",
                "reciprocal rank fusion",
                "--index-dir",
                idx,
                "--embed-model",
                "some-model",
            ]
        )
        == 1
    )
    err = capsys.readouterr().err
    assert "error:" in err
    assert "--mode" in err


def test_search_dense_identity_mismatch_fails_closed(
    corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    idx = str(tmp_path / "idx")
    main(
        [
            "ingest",
            str(corpus),
            "--index-dir",
            idx,
            "--dense",
            "--embed-provider",
            "inmemory",
            "--embed-dimensions",
            "384",
        ]
    )
    capsys.readouterr()

    assert (
        main(
            [
                "search",
                "reciprocal rank fusion",
                "--index-dir",
                idx,
                "--mode",
                "dense",
                "--embed-provider",
                "inmemory",
                "--embed-dimensions",
                "512",
            ]
        )
        == 1
    )
    err = capsys.readouterr().err
    assert "error:" in err
    assert "identity" in err


def test_search_dense_over_never_dense_ingested_collection_returns_no_results(
    corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    idx = str(tmp_path / "idx")
    main(["ingest", str(corpus), "--index-dir", idx])  # BM25-only: no --dense
    capsys.readouterr()

    assert (
        main(
            [
                "search",
                "reciprocal rank fusion",
                "--index-dir",
                idx,
                "--mode",
                "dense",
                "--embed-provider",
                "inmemory",
            ]
        )
        == 0
    )
    assert "no results" in capsys.readouterr().out


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


def test_eval_top_k_above_the_cap_fails_before_writing_a_report(
    eval_corpus: Path, eval_judgments: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An out-of-range cutoff is rejected up front, not after a full embed pass.

    Without the upper bound, ``Retriever.search`` rejects it instead — but
    only inside the stage loop, after a dense run has embedded the whole
    corpus. Against a hosted provider that is billable work for an
    invocation that could never have produced a report. Asserting the report
    file was never created pins "failed before doing the work", not merely
    "failed".
    """
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
                str(MAX_TOP_K + 1),
                "--dense",
                "--embed-provider",
                "inmemory",
                "--embed-dimensions",
                "32",
            ]
        )
        == 1
    )
    err = capsys.readouterr().err
    assert "error:" in err
    assert "--top-k" in err
    assert not output.exists()
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


def _run_dense_eval(
    eval_corpus: Path, eval_judgments: Path, output: Path, *, extra: list[str] | None = None
) -> int:
    """Run ``grk eval --dense`` with the offline embedder against a temp corpus."""
    return main(
        [
            "eval",
            "--corpus-dir",
            str(eval_corpus),
            "--judgments",
            str(eval_judgments),
            "--output",
            str(output),
            "--dense",
            "--embed-provider",
            "inmemory",
            "--embed-dimensions",
            "32",
            *(extra or []),
        ]
    )


def test_eval_dense_writes_all_three_stages(
    eval_corpus: Path, eval_judgments: Path, tmp_path: Path
) -> None:
    output = tmp_path / "results" / "latest.json"
    assert _run_dense_eval(eval_corpus, eval_judgments, output) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert [stage["stage"] for stage in payload["stages"]] == ["bm25", "dense", "fusion"]
    assert payload["run"]["config"]["embedding"]["provider"] == "inmemory"
    assert payload["run"]["config"]["rrf_k"] is not None


def test_eval_dense_summary_prints_a_signed_delta_per_stage(
    eval_corpus: Path,
    eval_judgments: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "results" / "latest.json"
    assert _run_dense_eval(eval_corpus, eval_judgments, output) == 0

    out = capsys.readouterr().out
    assert "delta[dense vs bm25]:" in out
    assert "delta[fusion vs bm25]:" in out
    # Signs are always explicit, so a reader never has to infer direction.
    assert "recall_at_1=+" in out or "recall_at_1=-" in out


def test_eval_dense_summary_warns_that_inmemory_numbers_are_not_a_measurement(
    eval_corpus: Path,
    eval_judgments: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SPEC.md §2: hash-derived vectors must never be presented as a quality number."""
    output = tmp_path / "results" / "latest.json"
    assert _run_dense_eval(eval_corpus, eval_judgments, output) == 0

    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "NOT a retrieval-quality measurement" in out
    assert "provider=inmemory" in out


def test_eval_baseline_only_prints_no_delta_and_no_embedding_line(
    eval_corpus: Path,
    eval_judgments: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A BM25-only run has nothing to diff and no semantic space to name."""
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
    assert "delta[" not in out
    assert "embedding:" not in out


def test_eval_embed_flags_without_dense_fail_cleanly(
    eval_corpus: Path,
    eval_judgments: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A flag that would be silently ignored is an error, matching ingest/search."""
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
                "--embed-provider",
                "inmemory",
            ]
        )
        == 1
    )
    assert "require --dense" in capsys.readouterr().err


def test_eval_dense_verdict_matches_the_artifact_it_describes(
    eval_corpus: Path,
    eval_judgments: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SPEC.md §6 at the CLI boundary: the printed verdict must reach the operator.

    The expected verdict is *derived from the artifact* rather than assumed.
    Asserting "REGRESSION" outright would bake in which way this embedder's
    hashes happen to fall — deterministic, but an arbitrary fact about the
    fixture rather than a property of the code. Reading the stage's own
    numbers back and requiring the printed line to agree with them tests the
    thing that actually matters: the summary cannot describe a loss as
    anything else.
    """
    output = tmp_path / "results" / "latest.json"
    assert _run_dense_eval(eval_corpus, eval_judgments, output) == 0

    out = capsys.readouterr().out
    payload = json.loads(output.read_text(encoding="utf-8"))
    stages = {stage["stage"]: stage["aggregate"] for stage in payload["stages"]}
    assert "dense" in stages, "the losing stage must stay in the artifact, not be dropped"

    metric_names = ("recall_at_1", "recall_at_5", "recall_at_10", "mrr", "ndcg_at_10")
    regressed = any(stages["dense"][name] < stages["bm25"][name] for name in metric_names)
    improved = any(stages["dense"][name] > stages["bm25"][name] for name in metric_names)

    if regressed and improved:
        assert "MIXED vs baseline" in out
    elif regressed:
        assert "REGRESSION vs baseline" in out
    elif improved:
        assert "improvement vs baseline" in out
    else:
        assert "no change vs baseline" in out

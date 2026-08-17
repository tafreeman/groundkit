"""End-to-end CLI tests: grk ingest -> grk search on a real temp index."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from groundkit.cli import _build_parser, _print_eval_summary, main
from groundkit.evals.schema import (
    EvalReport,
    MetricSet,
    RunConfig,
    RunMetadata,
    StageName,
    StageResult,
)
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


@pytest.mark.parametrize("bad_dimensions", ["0", "-1"])
def test_invalid_embed_dimensions_fails_cleanly_not_with_a_traceback(
    corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], bad_dimensions: str
) -> None:
    """A bad ``--embed-dimensions`` must exit 1 with ``error:``, not raise.

    ``EmbeddingConfig.dimensions`` is ``Field(gt=0)``, and the CLI's only
    construction site passed argparse's already-int-converted value straight
    in. Pydantic's ``ValidationError`` is not a ``GroundkitError``, so
    ``main``'s handler never saw it and the command died on a raw traceback
    — every other rejected flag on the same command prints one ``error:``
    line. Asserted through ``main`` rather than the helper, since the escape
    was the exception crossing ``main``'s boundary.
    """
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
                "--embed-dimensions",
                bad_dimensions,
            ]
        )
        == 1
    )
    err = capsys.readouterr().err
    assert "error:" in err
    assert "dimensions" in err
    assert "Traceback" not in err


def test_invalid_embed_dimensions_on_search_and_eval_fail_cleanly_too(
    corpus: Path,
    eval_corpus: Path,
    eval_judgments: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The same guard covers every command sharing the ``--embed-*`` flags.

    All three route through one ``_resolve_embedding_config``, so this pins
    that the translation lives at that shared site rather than in whichever
    command happened to be tested first.
    """
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
                "--mode",
                "dense",
                "--embed-provider",
                "inmemory",
                "--embed-dimensions",
                "0",
            ]
        )
        == 1
    )
    assert "error:" in capsys.readouterr().err

    assert (
        main(
            [
                "eval",
                "--corpus-dir",
                str(eval_corpus),
                "--judgments",
                str(eval_judgments),
                "--output",
                str(tmp_path / "out.json"),
                "--dense",
                "--embed-provider",
                "inmemory",
                "--embed-dimensions",
                "0",
            ]
        )
        == 1
    )
    assert "error:" in capsys.readouterr().err


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


@pytest.mark.parametrize("mode", ["dense", "hybrid"])
def test_search_dense_or_hybrid_over_never_dense_collection_fails_cleanly(
    mode: str, corpus: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """ADR-0008 at the CLI: the operator gets an error, not plausible results.

    Previously this exited 0 — ``dense`` printed "no results" and ``hybrid``
    printed BM25's ranking labelled as fusion. Both are the trap the README
    documents; the CLI now refuses rather than relying on the operator having
    read it.
    """
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
                mode,
                "--embed-provider",
                "inmemory",
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert "no embedding-identity manifest" in captured.err
    # The remedy that actually works, not the one that silently no-ops.
    assert "grk ingest --dense" in captured.err
    assert "no results" not in captured.out


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


# --- --rerank / --rerank-model: argument validation, mirroring --embed-* ---


def test_eval_rerank_flag_defaults_to_false_and_parses_when_supplied() -> None:
    """The parser accepts --rerank and it defaults to False when absent."""
    assert _build_parser().parse_args(["eval"]).rerank is False
    assert _build_parser().parse_args(["eval", "--rerank"]).rerank is True


def test_eval_rerank_model_without_rerank_fails_closed(
    eval_corpus: Path,
    eval_judgments: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A --rerank-model without --rerank is an error, mirroring the --embed-* rule.

    Same fail-closed shape as ``test_eval_embed_flags_without_dense_fail_cleanly``:
    a flag that would configure a path the run never takes is a mistake to
    name, not one to silently ignore.
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
                "--rerank-model",
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
            ]
        )
        == 1
    )
    err = capsys.readouterr().err
    assert "error:" in err
    assert "--rerank" in err
    assert "requires --rerank" in err


def test_eval_rerank_model_without_rerank_fails_before_touching_corpus_or_judgments(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Flag validation runs before any corpus read or judgments load.

    Both the corpus dir and the judgments path are nonexistent. If the CLI
    did any real work before validating ``--rerank-model``, this would fail
    with the missing-judgments ``EvalError`` instead — the same ordering
    ``test_eval_top_k_above_the_cap_fails_before_writing_a_report`` pins for
    ``--top-k``.
    """
    output = tmp_path / "results" / "latest.json"
    assert (
        main(
            [
                "eval",
                "--corpus-dir",
                str(tmp_path / "no-such-corpus"),
                "--judgments",
                str(tmp_path / "no-such-judgments.jsonl"),
                "--output",
                str(output),
                "--rerank-model",
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
            ]
        )
        == 1
    )
    err = capsys.readouterr().err
    assert "requires --rerank" in err
    assert "judgments file not found" not in err
    assert not output.exists()


# --- --judge: argument validation. No end-to-end run through main() here —
# --judge (like --synthesis) needs a live chat provider, and build_chat only
# ever constructs "ollama" or "openai_compatible" (ScriptedChatProvider is
# deliberately unreachable through it), so an end-to-end run would need a
# real or reachable provider. The judge/synthesis LOGIC itself is exercised
# directly, bypassing the CLI, in tests/test_synthesis_eval.py and
# tests/test_runner.py's TestSynthesisPass.


def test_eval_judge_flag_defaults_to_false_and_parses_when_supplied() -> None:
    """The parser accepts --judge and it defaults to False when absent."""
    assert _build_parser().parse_args(["eval"]).judge is False
    assert _build_parser().parse_args(["eval", "--judge"]).judge is True


def test_eval_judge_without_synthesis_fails_closed(
    eval_corpus: Path,
    eval_judgments: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--judge without --synthesis is an error, mirroring the --rerank-model rule.

    SynthesisReport's docstring (evals/schema.py) states this exact
    requirement literally: a report with half a judge record would describe
    a run that could not have happened.
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
                "--judge",
            ]
        )
        == 1
    )
    err = capsys.readouterr().err
    assert "error:" in err
    assert "--judge" in err
    assert "requires --synthesis" in err


def test_eval_judge_without_synthesis_fails_before_touching_corpus_or_judgments(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Flag validation runs before any corpus read, judgments load, or chat build.

    Both the corpus dir and the judgments path are nonexistent, and no
    --chat-* flags are supplied (so build_chat would fall back to its
    Ollama default and could hang on a real connection attempt if this
    check did not run first).
    """
    output = tmp_path / "results" / "latest.json"
    assert (
        main(
            [
                "eval",
                "--corpus-dir",
                str(tmp_path / "no-such-corpus"),
                "--judgments",
                str(tmp_path / "no-such-judgments.jsonl"),
                "--output",
                str(output),
                "--judge",
            ]
        )
        == 1
    )
    err = capsys.readouterr().err
    assert "requires --synthesis" in err
    assert "judgments file not found" not in err
    assert not output.exists()


# --- Rerank provenance and attribution printing, driven by a hand-built
# EvalReport rather than a real CrossEncoderReranker. The 'rerank' extra
# (torch, multi-gigabyte) is deliberately absent from the dev group, so these
# tests build a valid report directly instead of running --rerank end to end.


def _metric_set() -> MetricSet:
    """A structurally valid MetricSet; the specific values are not asserted on."""
    return MetricSet(
        query_count=2,
        recall_at_1=0.5,
        recall_at_5=0.5,
        recall_at_10=0.5,
        mrr=0.5,
        ndcg_at_10=0.5,
    )


def _stage_result(stage: StageName, *, is_baseline: bool) -> StageResult:
    """A structurally valid, empty-queries StageResult for the given stage name."""
    return StageResult(
        stage=stage,
        is_baseline=is_baseline,
        aggregate=_metric_set(),
        by_category={},
        no_answer_query_count=0,
        no_answer_abstained_count=0,
        latency_p50_ms=1.0,
        latency_p95_ms=2.0,
        latency_p99_ms=3.0,
        queries=[],
    )


def _rerank_report(
    *,
    stage_names: list[StageName],
    top_k: int,
    rerank_input: StageName,
    rerank_candidates: int,
    rerank_model: str,
) -> EvalReport:
    """A hand-built, schema-valid EvalReport carrying a rerank stage.

    No retrieval, no index, no model — every field is either fixed or taken
    from the caller, which is what lets this exercise ``_print_eval_summary``
    and ``_print_rerank_provenance`` without the optional 'rerank' extra.
    """
    stages = [_stage_result(name, is_baseline=(i == 0)) for i, name in enumerate(stage_names)]
    config = RunConfig(
        chunk_size=512,
        chunk_overlap=64,
        top_k=top_k,
        bm25_k1=1.5,
        bm25_b=0.75,
        score_threshold=None,
        rerank_input=rerank_input,
        rerank_candidates=rerank_candidates,
        rerank_model=rerank_model,
    )
    run = RunMetadata(
        started_at="2026-08-15T00:00:00+00:00",
        groundkit_version="0.1.0.dev0",
        corpus_hash="deadbeef",
        judgments_hash="cafebabe",
        document_count=2,
        chunk_count=2,
        judgment_count=2,
        config=config,
    )
    return EvalReport(run=run, stages=stages)


def test_eval_rerank_provenance_line_reports_input_model_candidates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = _rerank_report(
        stage_names=["bm25", "rerank"],
        top_k=10,
        rerank_input="bm25",
        rerank_candidates=MAX_TOP_K,
        rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    _print_eval_summary(report, tmp_path / "out.json")
    out = capsys.readouterr().out
    assert (
        "rerank: input=bm25 model=cross-encoder/ms-marco-MiniLM-L-6-v2 "
        f"candidates={MAX_TOP_K} truncated_to=10"
    ) in out
    assert "WARNING" not in out


def test_eval_rerank_provenance_warns_when_candidates_equal_top_k(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = _rerank_report(
        stage_names=["bm25", "rerank"],
        top_k=10,
        rerank_input="bm25",
        rerank_candidates=10,
        rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    _print_eval_summary(report, tmp_path / "out.json")
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "candidate depth equals top_k" in out


def test_eval_rerank_provenance_warning_absent_when_candidates_exceed_top_k(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = _rerank_report(
        stage_names=["bm25", "rerank"],
        top_k=10,
        rerank_input="bm25",
        rerank_candidates=MAX_TOP_K,
        rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    _print_eval_summary(report, tmp_path / "out.json")
    assert "WARNING" not in capsys.readouterr().out


def test_eval_rerank_attribution_delta_not_duplicated_on_bm25_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """On a BM25-only run, the attribution would be byte-identical to the
    baseline delta already printed for 'rerank' — so it must not repeat."""
    report = _rerank_report(
        stage_names=["bm25", "rerank"],
        top_k=10,
        rerank_input="bm25",
        rerank_candidates=MAX_TOP_K,
        rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    _print_eval_summary(report, tmp_path / "out.json")
    out = capsys.readouterr().out
    assert out.count("delta[rerank vs bm25]:") == 1


def test_eval_rerank_attribution_delta_printed_on_fusion_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """On a dense run, both the baseline delta and the reranker's own
    contribution against 'fusion' must be printed — two different facts."""
    report = _rerank_report(
        stage_names=["bm25", "dense", "fusion", "rerank"],
        top_k=10,
        rerank_input="fusion",
        rerank_candidates=MAX_TOP_K,
        rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    _print_eval_summary(report, tmp_path / "out.json")
    out = capsys.readouterr().out
    assert "delta[rerank vs bm25]:" in out
    assert "delta[rerank vs fusion]:" in out


def test_answer_citation_labels_preserve_source_numbers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The printed citation list must reuse the answer's own [n] source numbers.

    An answer citing only source 3 must print a "[3]" entry — renumbering the
    deduplicated citation list from 1 would leave the answer's "[3]" marker
    unresolved and falsely associate "[1]" with the third retrieved result
    (PR #14 review finding, shown to fail against the renumbering printer).
    """
    from groundkit.answer import AnswerReport
    from groundkit.cli import _print_answer_report
    from groundkit.contracts import RetrievalResult

    results = tuple(
        RetrievalResult(
            content=f"source text {n}",
            score=1.0,
            document_id=f"doc-{n}",
            chunk_id=f"chunk-{n}",
            source=f"doc-{n}.txt",
            start_offset=0,
            end_offset=len(f"source text {n}"),
        )
        for n in (1, 2, 3)
    )
    report = AnswerReport(
        query="which source?",
        rewritten_query=None,
        answer="Only the third source answers this [3].",
        citations=(results[2].citation,),
        results=results,
        verdict=None,
    )
    _print_answer_report(report)
    out = capsys.readouterr().out
    assert "[3] doc-3.txt#0-13" in out
    assert "[1] doc-3.txt" not in out


def test_print_synthesis_summary_without_judge(capsys: pytest.CaptureFixture[str]) -> None:
    """No live chat provider needed: this drives the printer directly with a
    hand-built ``SynthesisReport``, the same way
    ``test_answer_citation_labels_preserve_source_numbers`` drives
    ``_print_answer_report`` directly rather than through ``main()``.
    """
    from groundkit.cli import _print_synthesis_summary
    from groundkit.evals.schema import SynthesisReport

    report = SynthesisReport(
        input_stage="bm25",
        synthesis_provider="ollama",
        synthesis_model="llama3",
        synthesis_prompt_hash="a" * 64,
        redacted=False,
        answered_count=2,
        abstained_count=1,
        rejected_count=0,
    )
    _print_synthesis_summary(report)
    out = capsys.readouterr().out
    assert "synthesis (bm25): answered=2 abstained=1 rejected=0" in out
    assert "provider=ollama model=llama3" in out
    assert "judge" not in out


def test_print_synthesis_summary_with_judge_labels_it_advisory(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from groundkit.cli import _print_synthesis_summary
    from groundkit.evals.schema import SynthesisReport

    report = SynthesisReport(
        input_stage="fusion",
        synthesis_provider="ollama",
        synthesis_model="llama3",
        synthesis_prompt_hash="a" * 64,
        redacted=True,
        answered_count=1,
        abstained_count=0,
        rejected_count=0,
        judge_provider="ollama",
        judge_model="llama3",
        judge_prompt_hash="b" * 64,
        judged_count=1,
        faithful_count=1,
        unfaithful_count=0,
        judge_error_count=0,
    )
    _print_synthesis_summary(report)
    out = capsys.readouterr().out
    assert "synthesis (fusion): answered=1 abstained=0 rejected=0" in out
    assert "judge (advisory, uncalibrated): judged=1 faithful=1 unfaithful=0 errors=0" in out

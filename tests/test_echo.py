"""Planted-marker citation-echo check tests (SPEC.md §2, ADR-0018 decision 4).

Async code under test is driven with ``asyncio.run()`` inside plain ``def``
test functions, matching this repo's house style (pytest-asyncio is not a
dependency) — see ``tests/test_synthesis.py``.

``_EchoScriptedChat`` is a module-local scripted double, distinct from
``tests/test_synthesis.py``'s own ``_SynthScriptedChat``: each test module
owns its own fake rather than sharing one, and neither imports
``groundkit.providers.llm`` — that module's ``ScriptedChatProvider`` is a
different, production-facing type this test suite never touches.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from groundkit.errors import ChatError, EvalError, RetrievalError
from groundkit.evals.echo import (
    DEFAULT_ECHO_REPORT_PATH,
    EchoCaseResult,
    EchoReport,
    _evaluate_echo_case,
    build_echo_case,
    generate_marker,
    run_echo_check,
    write_echo_report,
)
from groundkit.providers.protocols import ChatProtocol
from groundkit.providers.synthesis import Synthesizer

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: Citation-marker and redaction-token formats reserved at this boundary
#: (see ``groundkit/evals/echo.py``'s module docstring). A generated marker
#: must match neither.
_CITATION_MARKER_PATTERN = re.compile(r"\[(\d+)\]")
_REDACTION_TOKEN_PATTERN = re.compile(r"\[[A-Z_]+_\d+\]")


class _EchoScriptedChat:
    """A ``ChatProtocol`` test double returning a fixed completion, or raising a
    scripted error, with no network or real model involved.

    Records every prompt it receives so a test can assert on prompt content
    without inspecting the synthesizer's internals, matching
    ``tests/test_synthesis.py``'s ``_SynthScriptedChat`` shape exactly (it is
    not reused across the two test modules on purpose — see this module's
    docstring).
    """

    def __init__(
        self,
        completion: str = "",
        *,
        error: Exception | None = None,
        provider: str = "echo-scripted",
        model_name: str = "echo-scripted-v1",
    ) -> None:
        self._completion = completion
        self._error = error
        self._provider = provider
        self._model_name = model_name
        self.prompts: list[str] = []

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model_name(self) -> str:
        return self._model_name

    async def complete(self, prompt: str, *, system: str | None = None) -> str:
        self.prompts.append(prompt)
        if self._error is not None:
            raise self._error
        return self._completion


def _make_echo_case_result(**overrides: object) -> EchoCaseResult:
    """Build a valid, coherent EchoCaseResult fixture, one field overridable at a time."""
    defaults: dict[str, object] = {
        "case_id": "case-1",
        "citation_result": "cited_positive",
        "positive_grounded": True,
        "leaked_absent_marker": False,
        "correct": True,
    }
    defaults.update(overrides)
    return EchoCaseResult(**defaults)  # type: ignore[arg-type]


def _make_echo_report(**overrides: object) -> EchoReport:
    """Build a valid, coherent single-case EchoReport fixture."""
    defaults: dict[str, object] = {
        "generated_at": "2026-08-15T00:00:00+00:00",
        "chat_provider": "echo-scripted",
        "chat_model": "echo-scripted-v1",
        "cases": [_make_echo_case_result()],
        "case_count": 1,
        "correct_count": 1,
        "wrong_source_count": 0,
        "abstained_count": 0,
        "rejected_count": 0,
        "leaked_count": 0,
    }
    defaults.update(overrides)
    return EchoReport(**defaults)  # type: ignore[arg-type]


# ── Marker generation and the collision guard ────────────────────────────


class TestMarkerGeneration:
    def test_marker_never_matches_citation_marker_pattern(self) -> None:
        assert _CITATION_MARKER_PATTERN.search(generate_marker()) is None

    def test_marker_never_matches_redaction_token_pattern(self) -> None:
        assert _REDACTION_TOKEN_PATTERN.search(generate_marker()) is None

    def test_markers_have_the_documented_prefix(self) -> None:
        assert generate_marker().startswith("GK-ECHO-")

    def test_markers_are_collision_resistant(self) -> None:
        markers = {generate_marker() for _ in range(200)}
        assert len(markers) == 200


# ── Case construction ─────────────────────────────────────────────────────


class TestBuildEchoCase:
    def test_positive_and_decoy_written_to_disk_verbatim(self, tmp_path: Path) -> None:
        fixture = build_echo_case("case-1", tmp_path)
        assert (
            Path(fixture.positive_result.source).read_text(encoding="utf-8")
            == fixture.positive_result.content
        )
        assert (
            Path(fixture.decoy_result.source).read_text(encoding="utf-8")
            == fixture.decoy_result.content
        )

    def test_positive_marker_only_in_positive_document(self, tmp_path: Path) -> None:
        fixture = build_echo_case("case-1", tmp_path)
        assert fixture.positive_marker in fixture.positive_result.content
        assert fixture.positive_marker not in fixture.decoy_result.content

    def test_absent_marker_written_nowhere(self, tmp_path: Path) -> None:
        fixture = build_echo_case("case-1", tmp_path)
        assert fixture.absent_marker not in fixture.positive_result.content
        assert fixture.absent_marker not in fixture.decoy_result.content

    def test_missing_directory_raises_eval_error(self, tmp_path: Path) -> None:
        missing_dir = tmp_path / "does-not-exist"
        with pytest.raises(EvalError, match="could not write marker documents"):
            build_echo_case("case-x", missing_dir)


# ── Cites the planted doc correctly → clean case ─────────────────────────


class TestCitesPositiveCorrectly:
    def test_clean_case(self, tmp_path: Path) -> None:
        fixture = build_echo_case("clean", tmp_path)
        chat = _EchoScriptedChat("Documented in the first source. [1]")
        result = asyncio.run(_evaluate_echo_case(fixture, Synthesizer(chat), tmp_path))
        assert result.citation_result == "cited_positive"
        assert result.positive_grounded is True
        assert result.leaked_absent_marker is False
        assert result.correct is True


# ── Cites the WRONG source → flagged ─────────────────────────────────────


class TestCitesWrongSource:
    def test_decoy_citation_is_flagged_incorrect(self, tmp_path: Path) -> None:
        fixture = build_echo_case("wrong", tmp_path)
        chat = _EchoScriptedChat("Documented in the second source. [2]")
        result = asyncio.run(_evaluate_echo_case(fixture, Synthesizer(chat), tmp_path))
        assert result.citation_result == "cited_decoy"
        assert result.correct is False

    def test_citing_both_is_still_wrong_source(self, tmp_path: Path) -> None:
        """Citing the decoy at all is wrong, even alongside a correct citation —
        precision, not just recall, is what this check measures."""
        fixture = build_echo_case("both", tmp_path)
        chat = _EchoScriptedChat("Documented in sources one and two. [1][2]")
        result = asyncio.run(_evaluate_echo_case(fixture, Synthesizer(chat), tmp_path))
        assert result.citation_result == "cited_decoy"
        assert result.correct is False


# ── Abstention handling ───────────────────────────────────────────────────


class TestAbstention:
    def test_abstained_case_is_flagged_but_not_rejected(self, tmp_path: Path) -> None:
        fixture = build_echo_case("abstain", tmp_path)
        chat = _EchoScriptedChat("The sources do not answer this question.")
        result = asyncio.run(_evaluate_echo_case(fixture, Synthesizer(chat), tmp_path))
        assert result.citation_result == "abstained"
        assert result.positive_grounded is False
        assert result.leaked_absent_marker is False
        assert result.correct is False


# ── Rejected completions (SynthesisError) are recorded, never raised out ──


class TestRejectedCases:
    def test_out_of_range_marker_is_recorded_rejected_not_raised(self, tmp_path: Path) -> None:
        fixture = build_echo_case("reject", tmp_path)
        chat = _EchoScriptedChat("[9] an unresolvable claim")
        result = asyncio.run(_evaluate_echo_case(fixture, Synthesizer(chat), tmp_path))
        assert result.citation_result == "rejected"
        assert result.positive_grounded is False
        assert result.leaked_absent_marker is False
        assert result.correct is False

    def test_chat_error_propagates_untouched(self, tmp_path: Path) -> None:
        fixture = build_echo_case("boom", tmp_path)
        boom = ChatError("provider exploded")
        chat = _EchoScriptedChat(error=boom)
        with pytest.raises(ChatError) as excinfo:
            asyncio.run(_evaluate_echo_case(fixture, Synthesizer(chat), tmp_path))
        assert excinfo.value is boom


# ── Negative-side leak detection ─────────────────────────────────────────


class TestLeakDetection:
    def test_leaked_absent_marker_flagged_even_when_citation_is_correct(
        self, tmp_path: Path
    ) -> None:
        """The negative-side check must catch a leak independently of citation
        correctness (ADR-0018 decision 4). Shown to fail first (SPEC.md §8)
        against a checker that only inspects ``citation_result`` and never
        the answer text: that checker would report this case ``correct``."""
        fixture = build_echo_case("leak", tmp_path)
        chat = _EchoScriptedChat(
            f"Documented in the first source. [1] "
            f"(internal note, not for citation: {fixture.absent_marker})"
        )
        result = asyncio.run(_evaluate_echo_case(fixture, Synthesizer(chat), tmp_path))
        assert result.citation_result == "cited_positive"
        assert result.leaked_absent_marker is True
        assert result.correct is False


# ── Grounding is verified from disk, never trusted from memory ──────────


class TestGroundingReadsFromDisk:
    def test_positive_grounded_reflects_disk_content_not_memory(self, tmp_path: Path) -> None:
        """Regression test for the ADR's "re-read from source, never trust the
        in-memory result" rule (SPEC.md §8): a checker that compared against
        ``fixture.positive_result.content`` instead of re-reading the file
        would report ``positive_grounded=True`` here, which is wrong — the
        marker has been corrupted out of the file that citation resolution
        actually reads."""
        fixture = build_echo_case("disk-check", tmp_path)
        original_content = fixture.positive_result.content
        corrupted_content = original_content.replace(
            fixture.positive_marker, "X" * len(fixture.positive_marker)
        )
        assert len(corrupted_content) == len(original_content)
        Path(fixture.positive_result.source).write_text(corrupted_content, encoding="utf-8")

        chat = _EchoScriptedChat("Documented in the first source. [1]")
        result = asyncio.run(_evaluate_echo_case(fixture, Synthesizer(chat), tmp_path))
        assert result.citation_result == "cited_positive"
        assert result.positive_grounded is False

    def test_missing_source_file_propagates_retrieval_error(self, tmp_path: Path) -> None:
        fixture = build_echo_case("vanish", tmp_path)
        Path(fixture.positive_result.source).unlink()
        chat = _EchoScriptedChat("Documented in the first source. [1]")
        with pytest.raises(RetrievalError):
            asyncio.run(_evaluate_echo_case(fixture, Synthesizer(chat), tmp_path))


# ── EchoCaseResult coherence ──────────────────────────────────────────────


class TestEchoCaseResultCoherence:
    def test_frozen(self) -> None:
        result = _make_echo_case_result()
        with pytest.raises(ValidationError):
            result.correct = False

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _make_echo_case_result(surprise=True)

    def test_positive_grounded_true_while_abstained_rejected(self) -> None:
        with pytest.raises(ValidationError, match="positive_grounded cannot be True"):
            _make_echo_case_result(
                citation_result="abstained", positive_grounded=True, correct=False
            )

    def test_positive_grounded_true_while_rejected_outcome_rejected(self) -> None:
        with pytest.raises(ValidationError, match="positive_grounded cannot be True"):
            _make_echo_case_result(
                citation_result="rejected", positive_grounded=True, correct=False
            )

    def test_leaked_marker_true_with_rejected_outcome_rejected(self) -> None:
        with pytest.raises(ValidationError, match="leaked_absent_marker cannot be True"):
            _make_echo_case_result(
                citation_result="rejected",
                positive_grounded=False,
                leaked_absent_marker=True,
                correct=False,
            )

    def test_correct_disagreeing_with_its_own_fields_rejected(self) -> None:
        with pytest.raises(ValidationError, match="disagrees with citation_result"):
            _make_echo_case_result(
                citation_result="cited_positive",
                positive_grounded=True,
                leaked_absent_marker=False,
                correct=False,
            )


# ── EchoReport coherence ──────────────────────────────────────────────────


class TestEchoReportCoherence:
    def test_empty_cases_rejected(self) -> None:
        with pytest.raises(ValidationError, match="cases must not be empty"):
            _make_echo_report(cases=[], case_count=0, correct_count=0)

    def test_mismatched_case_count_rejected(self) -> None:
        with pytest.raises(ValidationError, match="aggregate counts disagree"):
            _make_echo_report(case_count=2)

    def test_mismatched_correct_count_rejected(self) -> None:
        with pytest.raises(ValidationError, match="aggregate counts disagree"):
            _make_echo_report(correct_count=0)

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _make_echo_report(surprise=True)

    def test_frozen(self) -> None:
        report = _make_echo_report()
        with pytest.raises(ValidationError):
            report.case_count = 99


# ── run_echo_check orchestration ──────────────────────────────────────────


class TestRunEchoCheck:
    def test_clean_run_produces_a_fully_correct_report(self, tmp_path: Path) -> None:
        chat = _EchoScriptedChat("Documented in the first source. [1]")
        report = asyncio.run(run_echo_check(chat, corpus_dir=tmp_path, case_count=3))
        assert report.case_count == 3
        assert report.correct_count == 3
        assert report.wrong_source_count == 0
        assert report.abstained_count == 0
        assert report.rejected_count == 0
        assert report.leaked_count == 0
        assert report.chat_provider == "echo-scripted"
        assert report.chat_model == "echo-scripted-v1"

    def test_wrong_source_run_is_reflected_in_aggregate_counts(self, tmp_path: Path) -> None:
        chat = _EchoScriptedChat("Documented in the second source. [2]")
        report = asyncio.run(run_echo_check(chat, corpus_dir=tmp_path, case_count=2))
        assert report.wrong_source_count == 2
        assert report.correct_count == 0

    def test_case_count_below_one_rejected(self, tmp_path: Path) -> None:
        chat = _EchoScriptedChat("[1] anything")
        with pytest.raises(EvalError, match="case_count must be at least 1"):
            asyncio.run(run_echo_check(chat, corpus_dir=tmp_path, case_count=0))


# ── Serialization round-trip, always into a tmp_path ──────────────────────


class TestSerializationRoundTrip:
    def test_round_trip_into_tmp_path(self, tmp_path: Path) -> None:
        corpus_dir = tmp_path / "corpus"
        corpus_dir.mkdir()
        chat = _EchoScriptedChat("Documented in the first source. [1]")
        report = asyncio.run(run_echo_check(chat, corpus_dir=corpus_dir, case_count=2))

        output_path = tmp_path / "results" / "echo-latest.json"
        write_echo_report(report, output_path)

        assert output_path.exists()
        loaded = EchoReport.model_validate_json(output_path.read_text(encoding="utf-8"))
        assert loaded == report

    def test_unwritable_path_raises_eval_error(self, tmp_path: Path) -> None:
        chat = _EchoScriptedChat("Documented in the first source. [1]")
        report = asyncio.run(run_echo_check(chat, corpus_dir=tmp_path, case_count=1))
        blocking_file = tmp_path / "blocked"
        blocking_file.write_text("not a directory", encoding="utf-8")
        bad_output = blocking_file / "echo-latest.json"
        with pytest.raises(EvalError, match="Cannot write echo report"):
            write_echo_report(report, bad_output)


# ── evals/results/ must stay gitignored (ADR-0018 decision 6) ────────────


class TestEvalResultsGitignored:
    def test_evals_results_directory_is_gitignored(self) -> None:
        """Converts ``evals/schema.py``'s additive-with-default licence
        precondition into an assertion instead of a docstring a reader has
        to remember: the licence — and this artifact's own
        :data:`~groundkit.evals.echo.DEFAULT_ECHO_REPORT_PATH` living under
        the same directory — holds only while ``evals/results/`` stays out
        of version control."""
        gitignore_text = (_REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        assert "evals/results/" in gitignore_text.splitlines()
        assert Path("evals/results/echo-latest.json") == DEFAULT_ECHO_REPORT_PATH


# ── ChatProtocol conformance of the test double itself ───────────────────


class TestChatProtocolConformance:
    def test_scripted_chat_satisfies_chat_protocol(self) -> None:
        assert isinstance(_EchoScriptedChat(), ChatProtocol)


# ── SynthesisError is the only exception type recorded, never re-raised ──


class TestFailClosedOnlyCatchesSynthesisError:
    def test_arbitrary_exception_is_not_swallowed_into_a_rejected_case(
        self, tmp_path: Path
    ) -> None:
        fixture = build_echo_case("unexpected", tmp_path)
        boom = RuntimeError("unexpected backend failure")
        chat = _EchoScriptedChat(error=boom)
        with pytest.raises(RuntimeError) as excinfo:
            asyncio.run(_evaluate_echo_case(fixture, Synthesizer(chat), tmp_path))
        assert excinfo.value is boom

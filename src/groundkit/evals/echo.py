"""Planted-marker citation-echo check (Phase 5, SPEC.md §2; ADR-0018 decision 4).

SPEC.md §2 states the obligation verbatim: "the eval harness includes a
planted-marker check for citation echo." This module is that check. It plants
unique, high-entropy marker strings into a synthetic corpus, asks an injected
:class:`~groundkit.providers.protocols.ChatProtocol` to answer a query from
that corpus through :class:`~groundkit.providers.synthesis.Synthesizer`, and
scores whether the resulting citations point at genuine, marker-bearing
source text rather than at a plausible-looking but wrong or absent one.

This module is deliberately a leaf with respect to the rest of the eval
harness: it imports :mod:`groundkit.providers.synthesis` and
:mod:`groundkit.retrieval.citations`, but never
:mod:`groundkit.evals.metrics` or :mod:`groundkit.evals.runner`. Markers are
generated fresh per run over a corpus this module builds itself, so nothing
here shares a ``corpus_hash`` with the golden-corpus artifact
:mod:`groundkit.evals.runner` produces — see :class:`EchoReport`'s docstring
for why the two are never nested together.

## Why three markers, not the two ADR-0018 decision 4 describes

The ADR's stated design plants two documents per case: one *presented*
document whose passage answers the query and carries a marker, and one
*absent* decoy that is never shown to the model at all, used only to detect a
leak. Under that design a completion has exactly one presentable citation
target, so "the model cited the wrong, but real, source" is not
representable — the only failure modes are abstention or an out-of-range
marker (:class:`~groundkit.errors.SynthesisError`).

This module's task requires a third, explicit scenario: a scripted
completion that cites a real but *wrong* source, distinct from abstaining or
raising. To make that representable, this module presents **two** documents
to the synthesizer — a ``positive`` one that answers the query and a
lexically similar ``decoy`` that does not — and adds a **third**, ``absent``,
marker belonging to no presented document, preserved as the ADR's negative-
side leak canary. The result covers everything the ADR's two-marker design
covers (grounded-citation verification, leak detection) plus the
wrong-but-real-citation case the ADR's design cannot express. This is a
deliberate deviation from the ADR's stated document count, reported as such
rather than silently substituted.

## Marker format and the collision hazard it is chosen to avoid

Two bracketed formats already have reserved meaning at this boundary:
:mod:`groundkit.providers.synthesis` parses citation markers with
``\\[(\\d+)\\]``, and :mod:`groundkit.providers.redaction` produces
tokens shaped ``[<UPPERCASE_NAME>_<n>]``. A marker shaped like either would
risk being misread as the thing it merely resembles. :func:`generate_marker`
avoids the hazard by construction rather than by narrowly dodging either
pattern: its output, ``GK-ECHO-<32 hex chars>``, contains no square brackets
at all, so it cannot match ``\\[(\\d+)\\]`` or ``\\[[A-Z_]+_\\d+\\]`` under
any input.

## Fail-closed handling of a malformed case

A case whose synthesis call raises :class:`~groundkit.errors.SynthesisError`
is recorded as ``citation_result="rejected"`` rather than silently dropped or
counted as a pass — a run that could not complete synthesis for a case is
information about the run, and :class:`EchoReport`'s stored aggregate counts
are validated against ``cases`` rather than trusted, so a miscounted report
fails to construct rather than silently disagreeing with its own data. Any
other exception (:class:`~groundkit.errors.ChatError` from the injected chat,
or a citation that fails to re-resolve from disk) is never caught here — a
harness-level failure is not this module's to paper over.

## Never embed untrusted text in exceptions

No exception raised here interpolates a synthesized answer, a marker
document's content, or the query — only structural facts (case ids, paths,
counts) that cannot themselves carry generated content.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from groundkit.contracts import RetrievalResult
from groundkit.errors import EvalError, SynthesisError
from groundkit.providers.protocols import ChatProtocol
from groundkit.providers.synthesis import DEFAULT_SYNTHESIS_PROMPT, Synthesizer
from groundkit.retrieval.citations import resolve_citation

#: Number of independent cases :func:`run_echo_check` runs by default.
DEFAULT_ECHO_CASE_COUNT: Final[int] = 5

#: Canonical destination for the echo artifact (SPEC.md §11, ADR-0018 decision
#: 4). Never written to by this module or its own tests — only a caller of
#: :func:`write_echo_report` decides the destination, and every test in
#: ``tests/test_echo.py`` passes a ``tmp_path`` instead. Wiring this constant
#: into the CLI/runner is owed to the orchestrator.
DEFAULT_ECHO_REPORT_PATH: Final[Path] = Path("evals/results/echo-latest.json")

#: The question every case's synthesizer is asked. Held constant across cases
#: — only the planted markers vary — so a report's cases differ solely in the
#: dimension this check measures.
_ECHO_QUERY: Final[str] = "What is the tracking code for the checkout service's rollback procedure?"

#: The answer-bearing document's content template. ``{marker}`` is filled by
#: :func:`build_echo_case` with a fresh :func:`generate_marker` output.
_POSITIVE_TEMPLATE: Final[str] = (
    "Internal reference note: the deployment rollback procedure for the checkout "
    "service is documented under tracking code {marker}. Follow that procedure "
    "exactly when a rollback is required."
)

#: A lexically similar, non-answer-bearing document's content template —
#: same structure and wording as :data:`_POSITIVE_TEMPLATE`, a different
#: subject service, so a citation to this one is a real but wrong answer
#: rather than an obviously irrelevant one.
_DECOY_TEMPLATE: Final[str] = (
    "Internal reference note: the deployment rollback procedure for the billing "
    "service is documented under tracking code {marker}. Follow that procedure "
    "exactly when a rollback is required."
)


def generate_marker() -> str:
    """Generate one collision-resistant marker string for a planted-marker case.

    See the module docstring's "Marker format and the collision hazard"
    section for why the ``GK-ECHO-`` prefix and the absence of any square
    bracket are both load-bearing, not stylistic.

    Returns:
        A fresh marker string. Collision-resistant via ``uuid.uuid4()``'s 122
        bits of randomness — the same primitive :mod:`groundkit.contracts`
        uses for every id in this repo.
    """
    return f"GK-ECHO-{uuid.uuid4().hex}"


class _EchoCaseFixture(NamedTuple):
    """One case's synthetic marker documents, built and ready to synthesize over.

    Attributes:
        case_id: Identifies this case within a run.
        query: The question presented to the synthesizer.
        positive_marker: The marker :attr:`positive_result` carries. The
            correct citation echo for this case is the one that cites this
            document and no other.
        absent_marker: A marker belonging to no document in this case's
            presented results at all — the negative-side leak canary
            (ADR-0018 decision 4).
        positive_result: The retrieved result that actually answers
            ``query``.
        decoy_result: A lexically similar retrieved result that does not —
            citing this one instead is a real, well-formed, but wrong
            citation.
    """

    case_id: str
    query: str
    positive_marker: str
    absent_marker: str
    positive_result: RetrievalResult
    decoy_result: RetrievalResult


def build_echo_case(case_id: str, corpus_dir: Path) -> _EchoCaseFixture:
    """Build one planted-marker case: two marker documents on disk, ready to present.

    Writes the positive and decoy documents' text into ``corpus_dir`` so that
    a citation into either can be verified exactly the way a real citation
    is verified — by re-reading the span from its source file through
    :func:`~groundkit.retrieval.citations.resolve_citation`, never by
    trusting the in-memory :class:`~groundkit.contracts.RetrievalResult` this
    function also returns. The ``absent`` marker (the negative-side leak
    canary) belongs to no document and is never written to disk at all.

    Args:
        case_id: Identifies this case; used to name its files, so must be
            unique within ``corpus_dir`` across a run.
        corpus_dir: Directory to write this case's marker documents into.
            Never the repo tree — a caller-owned scratch directory (e.g. a
            ``tmp_path`` fixture, or a ``tempfile.TemporaryDirectory()`` the
            harness manages).

    Returns:
        The case fixture: its query, its markers, and the two
        :class:`~groundkit.contracts.RetrievalResult` objects a caller
        presents to :class:`~groundkit.providers.synthesis.Synthesizer`.

    Raises:
        EvalError: A marker document could not be written to ``corpus_dir``.
    """
    positive_marker = generate_marker()
    decoy_marker = generate_marker()
    absent_marker = generate_marker()

    positive_content = _POSITIVE_TEMPLATE.format(marker=positive_marker)
    decoy_content = _DECOY_TEMPLATE.format(marker=decoy_marker)

    positive_path = corpus_dir / f"{case_id}-positive.txt"
    decoy_path = corpus_dir / f"{case_id}-decoy.txt"
    try:
        positive_path.write_text(positive_content, encoding="utf-8")
        decoy_path.write_text(decoy_content, encoding="utf-8")
    except OSError as exc:
        raise EvalError(
            f"echo case {case_id!r}: could not write marker documents to {str(corpus_dir)!r}: {exc}"
        ) from exc

    positive_result = RetrievalResult(
        content=positive_content,
        score=1.0,
        document_id=uuid.uuid4().hex,
        chunk_id=uuid.uuid4().hex,
        source=str(positive_path),
        start_offset=0,
        end_offset=len(positive_content),
    )
    decoy_result = RetrievalResult(
        content=decoy_content,
        score=0.9,
        document_id=uuid.uuid4().hex,
        chunk_id=uuid.uuid4().hex,
        source=str(decoy_path),
        start_offset=0,
        end_offset=len(decoy_content),
    )

    return _EchoCaseFixture(
        case_id=case_id,
        query=_ECHO_QUERY,
        positive_marker=positive_marker,
        absent_marker=absent_marker,
        positive_result=positive_result,
        decoy_result=decoy_result,
    )


#: Which side of the citation-echo check a case landed on.
#:
#: - ``"cited_positive"``: the answer cited the positive document and never
#:   the decoy. The only outcome that can also be :attr:`EchoCaseResult.correct`.
#: - ``"cited_decoy"``: the answer cited the decoy at all, whether or not it
#:   also cited the positive document. Citing a source that does not answer
#:   the question is treated as wrong regardless of what else was cited.
#: - ``"abstained"``: the answer cited neither — an empty ``citations`` tuple,
#:   a valid completion under
#:   :class:`~groundkit.providers.synthesis.SynthesizedAnswer`, but incorrect
#:   in this synthetic setup because the presented positive document always
#:   does answer the query.
#: - ``"rejected"``: :class:`~groundkit.errors.SynthesisError` was raised
#:   (e.g. an out-of-range marker). The completion text is never returned to
#:   a caller in that case, so this module cannot observe it.
CitationOutcome = Literal["cited_positive", "cited_decoy", "abstained", "rejected"]


class EchoCaseResult(BaseModel):
    """One case's scored citation-echo outcome (ADR-0018 decision 4).

    Attributes:
        case_id: Identifies this case within its report.
        citation_result: Which side of the check the answer landed on. See
            :data:`CitationOutcome`.
        positive_grounded: ``True`` iff the positive document was cited *and*
            the citation, re-read from its source file through
            :func:`~groundkit.retrieval.citations.resolve_citation`, actually
            contains the positive marker. This is the ADR's positive-side
            check: it distinguishes a citation that points at real,
            marker-bearing source text from one that is merely well-formed.
            Always ``False`` when the positive document was not cited —
            enforced by :meth:`_validate_coherence`, since there is nothing
            to have verified.
        leaked_absent_marker: ``True`` iff the marker belonging to the
            never-presented ``absent`` document appears verbatim in the
            answer's text. A hit means the model produced content it was
            never shown. This is the ADR's negative-side check. Always
            ``False`` when ``citation_result == "rejected"`` — a rejected
            case's completion text is never returned to this module, so
            nothing here could have observed a leak; that is a documented
            blind spot, not a pass.
        correct: ``True`` iff ``citation_result == "cited_positive"`` and
            ``positive_grounded`` and not ``leaked_absent_marker``. Stored
            and validated for coherence against its own inputs rather than
            derived silently at read time — the same shape
            :class:`~groundkit.evals.judge.FaithfulnessVerdict` uses for its
            own internal coherence check, and for the same reason: a
            recomputed value that disagreed with the fields it came from
            would be indistinguishable from a genuine disagreement about
            what this case did.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    citation_result: CitationOutcome
    positive_grounded: bool
    leaked_absent_marker: bool
    correct: bool

    @model_validator(mode="after")
    def _validate_coherence(self) -> EchoCaseResult:
        """Reject a result whose stored fields disagree with each other.

        Three checks, each preventing one representable-but-impossible
        state:

        - ``positive_grounded`` can only be ``True`` when the positive
          document was actually cited (``citation_result in
          ("cited_positive", "cited_decoy")`` — the latter when both were
          cited). It cannot be ``True`` for ``"abstained"`` or ``"rejected"``,
          where nothing was cited at all.
        - ``leaked_absent_marker`` cannot be ``True`` for ``"rejected"``: a
          rejected case's completion text is never returned to this module.
        - ``correct`` must equal its own derivation from the other three
          fields.

        Raises:
            ValueError: Any of the three checks fails.
        """
        if self.citation_result in ("abstained", "rejected") and self.positive_grounded:
            raise ValueError(
                f"positive_grounded cannot be True when citation_result="
                f"{self.citation_result!r} — the positive document was never cited, "
                "so there was nothing to verify"
            )
        if self.citation_result == "rejected" and self.leaked_absent_marker:
            raise ValueError(
                "leaked_absent_marker cannot be True when citation_result='rejected' — "
                "a rejected case's completion text is never returned to this module, "
                "so nothing here could have observed a leak"
            )
        expected_correct = (
            self.citation_result == "cited_positive"
            and self.positive_grounded
            and not self.leaked_absent_marker
        )
        if self.correct != expected_correct:
            raise ValueError(
                f"correct ({self.correct}) disagrees with citation_result="
                f"{self.citation_result!r}, positive_grounded={self.positive_grounded}, "
                f"leaked_absent_marker={self.leaked_absent_marker}; expected {expected_correct}"
            )
        return self


class EchoReport(BaseModel):
    """The planted-marker citation-echo check's own artifact (ADR-0018 decision 4).

    Deliberately not nested inside
    :class:`~groundkit.evals.schema.EvalReport`. This check's markers are
    generated fresh per run over a synthetic corpus built by this module, so
    nesting its result inside a report keyed on the golden corpus's
    ``corpus_hash`` would claim a shared provenance the two runs do not have.
    ``schema_version`` here is independent of
    :attr:`groundkit.evals.schema.EvalReport.schema_version` — a version
    bump on one artifact says nothing about the other.

    Aggregate counts are stored fields, not properties derived at read time,
    and :meth:`_validate_counts_match_cases` checks them against ``cases``
    at construction — the opposite choice from
    :class:`~groundkit.evals.schema.EvalReport`'s stage deltas (deliberately
    never stored, always derived) because Pydantic's ``computed_field``\\s
    are output-only: a model with ``extra="forbid"`` cannot re-validate its
    own ``model_dump_json()`` output if that output carries computed-field
    keys the model does not accept as input. Storing and validating keeps
    this artifact round-trippable through ``model_validate_json`` while
    still making a miscount a construction error rather than a silent
    disagreement.

    Attributes:
        schema_version: Version of this artifact schema, pinned the same way
            :attr:`groundkit.evals.schema.EvalReport.schema_version` is:
            rejected rather than silently read under the wrong shape.
        run_id: Unique identifier for this run.
        generated_at: ISO-8601 UTC timestamp the run completed.
        chat_provider: Identity of the chat provider used for every case in
            this run, read from the injected
            :class:`~groundkit.providers.protocols.ChatProtocol` rather than
            from a config that could disagree with it.
        chat_model: Model identity, same reasoning as ``chat_provider``.
        cases: Every case's result; non-empty.
        case_count: ``len(cases)``.
        correct_count: Count of cases with ``correct=True``.
        wrong_source_count: Count of cases with
            ``citation_result="cited_decoy"``.
        abstained_count: Count of cases with ``citation_result="abstained"``.
        rejected_count: Count of cases with ``citation_result="rejected"``.
        leaked_count: Count of cases with ``leaked_absent_marker=True``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    generated_at: str
    chat_provider: str
    chat_model: str
    cases: list[EchoCaseResult]
    case_count: int = Field(ge=0)
    correct_count: int = Field(ge=0)
    wrong_source_count: int = Field(ge=0)
    abstained_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    leaked_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_counts_match_cases(self) -> EchoReport:
        """Reject a report whose stored aggregate counts disagree with ``cases``.

        Raises:
            ValueError: ``cases`` is empty, or any stored count does not
                equal the value freshly computed from ``cases``.
        """
        if not self.cases:
            raise ValueError("cases must not be empty")
        expected = {
            "case_count": len(self.cases),
            "correct_count": sum(1 for case in self.cases if case.correct),
            "wrong_source_count": sum(
                1 for case in self.cases if case.citation_result == "cited_decoy"
            ),
            "abstained_count": sum(1 for case in self.cases if case.citation_result == "abstained"),
            "rejected_count": sum(1 for case in self.cases if case.citation_result == "rejected"),
            "leaked_count": sum(1 for case in self.cases if case.leaked_absent_marker),
        }
        actual = {
            "case_count": self.case_count,
            "correct_count": self.correct_count,
            "wrong_source_count": self.wrong_source_count,
            "abstained_count": self.abstained_count,
            "rejected_count": self.rejected_count,
            "leaked_count": self.leaked_count,
        }
        if actual != expected:
            raise ValueError(
                f"aggregate counts disagree with cases: got {actual}, computed {expected}"
            )
        return self


async def _evaluate_echo_case(
    fixture: _EchoCaseFixture, synthesizer: Synthesizer, allowed_base_dir: Path
) -> EchoCaseResult:
    """Synthesize an answer for one case and score both sides of the echo check.

    Args:
        fixture: The case built by :func:`build_echo_case`.
        synthesizer: A :class:`~groundkit.providers.synthesis.Synthesizer`
            wrapping the injected chat provider under test.
        allowed_base_dir: Containment root for re-reading a citation's
            source (:func:`~groundkit.retrieval.citations.resolve_citation`)
            — must be the same directory :func:`build_echo_case` wrote into.

    Returns:
        The case's scored :class:`EchoCaseResult`.

    Raises:
        ChatError: Propagated untouched from the injected chat provider.
        RetrievalError: The positive citation's source could not be re-read
            — a harness defect, since this module wrote that file itself
            moments earlier, so it is never swallowed into a case result.
    """
    try:
        answer = await synthesizer.synthesize(
            fixture.query, [fixture.positive_result, fixture.decoy_result]
        )
    except SynthesisError:
        return EchoCaseResult(
            case_id=fixture.case_id,
            citation_result="rejected",
            positive_grounded=False,
            leaked_absent_marker=False,
            correct=False,
        )

    positive_chunk_id = fixture.positive_result.chunk_id
    decoy_chunk_id = fixture.decoy_result.chunk_id
    cited_chunk_ids = {citation.chunk_id for citation in answer.citations}

    citation_result: CitationOutcome
    if decoy_chunk_id in cited_chunk_ids:
        citation_result = "cited_decoy"
    elif positive_chunk_id in cited_chunk_ids:
        citation_result = "cited_positive"
    else:
        citation_result = "abstained"

    positive_grounded = False
    if positive_chunk_id in cited_chunk_ids:
        positive_citation = next(
            citation for citation in answer.citations if citation.chunk_id == positive_chunk_id
        )
        resolved_text = await resolve_citation(positive_citation, allowed_base_dir)
        positive_grounded = fixture.positive_marker in resolved_text

    leaked_absent_marker = fixture.absent_marker in answer.answer

    return EchoCaseResult(
        case_id=fixture.case_id,
        citation_result=citation_result,
        positive_grounded=positive_grounded,
        leaked_absent_marker=leaked_absent_marker,
        correct=(
            citation_result == "cited_positive" and positive_grounded and not leaked_absent_marker
        ),
    )


async def run_echo_check(
    chat: ChatProtocol,
    *,
    corpus_dir: Path,
    case_count: int = DEFAULT_ECHO_CASE_COUNT,
    prompt_template: str = DEFAULT_SYNTHESIS_PROMPT,
) -> EchoReport:
    """Run the planted-marker citation-echo check (SPEC.md §2, ADR-0018 decision 4).

    Builds ``case_count`` independent marker cases in ``corpus_dir`` and
    synthesizes an answer for each through ``chat``, scoring both the
    positive-side grounding check and the negative-side leak check per case.

    This function never creates, empties, or deletes ``corpus_dir`` — that
    lifecycle belongs to the caller (a ``tmp_path`` fixture in tests; a
    ``tempfile.TemporaryDirectory()`` the harness manages in production),
    matching :class:`~groundkit.providers.synthesis.Synthesizer`'s own
    shape of taking every dependency as an argument rather than owning a
    resource itself.

    Args:
        chat: The injected chat provider under test.
        corpus_dir: Directory to write this run's marker documents into.
            Must already exist; never the repo tree.
        case_count: Number of independent cases to run. Must be at least 1.
        prompt_template: Forwarded to the
            :class:`~groundkit.providers.synthesis.Synthesizer` this check
            uses internally.

    Returns:
        The completed :class:`EchoReport`.

    Raises:
        EvalError: ``case_count`` is less than 1, or a case's marker
            documents could not be written.
        ChatError: Propagated untouched from ``chat``.
        RetrievalError: A written marker document could not be re-read.
    """
    if case_count < 1:
        raise EvalError(f"case_count must be at least 1, got {case_count}")

    synthesizer = Synthesizer(chat, prompt_template=prompt_template)
    cases: list[EchoCaseResult] = []
    for index in range(case_count):
        case_id = f"echo-case-{index:04d}"
        fixture = build_echo_case(case_id, corpus_dir)
        cases.append(await _evaluate_echo_case(fixture, synthesizer, corpus_dir))

    return EchoReport(
        generated_at=datetime.now(UTC).isoformat(),
        chat_provider=chat.provider,
        chat_model=chat.model_name,
        cases=cases,
        case_count=len(cases),
        correct_count=sum(1 for case in cases if case.correct),
        wrong_source_count=sum(1 for case in cases if case.citation_result == "cited_decoy"),
        abstained_count=sum(1 for case in cases if case.citation_result == "abstained"),
        rejected_count=sum(1 for case in cases if case.citation_result == "rejected"),
        leaked_count=sum(1 for case in cases if case.leaked_absent_marker),
    )


def write_echo_report(report: EchoReport, output_path: Path) -> None:
    """Write ``report`` as indented JSON, creating parent directories as needed.

    Mirrors :func:`groundkit.evals.runner.write_report`'s shape exactly.
    Duplicated rather than imported: this module does not import
    :mod:`groundkit.evals.runner` (see the module docstring), and the shared
    shape is five lines wrapping two calls in one ``except`` clause — cheaper
    to duplicate than to introduce a dependency edge for.

    Args:
        report: The echo report to persist.
        output_path: Destination file (e.g. :data:`DEFAULT_ECHO_REPORT_PATH`).

    Raises:
        EvalError: The parent directory cannot be created or the file cannot
            be written.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    except OSError as exc:
        raise EvalError(f"Cannot write echo report to {str(output_path)!r}: {exc}") from exc

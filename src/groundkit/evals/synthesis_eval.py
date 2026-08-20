"""Golden-corpus synthesis + advisory-judge pass, folded into ``EvalReport.synthesis``.

``KNOWN_LIMITATIONS.md`` named the gap this module closes precisely:
``EvalReport.synthesis`` (:class:`~groundkit.evals.schema.SynthesisReport`) was
"structure without a producer" — ``grk eval`` did not fold judge tallies over
the golden corpus into the main artifact. This module is that producer.

It is deliberately **not** the planted-marker citation-echo check
(:mod:`groundkit.evals.echo`). The echo check builds its own synthetic
two-document corpus per run and answers one fixed question against it,
scoring citation echo specifically; its report is intentionally never nested
inside :class:`~groundkit.evals.schema.EvalReport` because it shares no
``corpus_hash`` with the golden-corpus run (ADR-0018 decision 4). This module
runs the opposite case: real synthesis over the *actual* golden-corpus
queries, against whichever retrieval stage the surrounding
:func:`~groundkit.evals.runner.run_eval` run produced as its best available
stage, so its result *does* belong inside that same report (ADR-0018
decision 6).

**Reuses, never duplicates, the judge and synthesis machinery.** This module
constructs a :class:`~groundkit.providers.synthesis.Synthesizer` around the
injected ``chat`` and calls the caller-supplied
:class:`~groundkit.providers.judge.FaithfulnessJudge` exactly the way
:class:`~groundkit.answer.AnswerPipeline` does for ``grk answer --judge`` —
unconditionally, with no attempt to special-case an abstained answer. Neither
class's internals (prompt rendering, completion parsing, verdict validation)
are reimplemented here.

## The three-way outcome split, and where the judge fits

Every judgment falls into exactly one of three synthesis outcomes:

- **answered** — synthesis produced a non-empty ``citations`` tuple.
- **abstained** — synthesis produced a valid completion with an *empty*
  ``citations`` tuple (a genuine abstention, not an error — ADR-0018
  decision 2).
- **rejected** — :class:`~groundkit.errors.SynthesisError` was raised.
  Recorded, never re-raised: a scripted or a real model producing many
  rejections is information about the run, not a harness failure.

When a judge is supplied, every *non-rejected* outcome (answered or
abstained alike) is judged — the same "no special-casing abstention" rule
:mod:`groundkit.answer` documents, because "abstained" is not a branch this
module takes, only a shape ``citations`` happens to have. A rejected
judgment produced no answer at all, so there is nothing to hand the judge; it
never contributes to ``judged_count``. Each judged case then lands in
exactly one of ``faithful`` / ``unfaithful`` / a caught
:class:`~groundkit.errors.JudgeError` (malformed verdict JSON) — mirroring
:mod:`groundkit.evals.echo`'s own "record, don't drop" handling of a
per-case failure.

## Advisory only

Nothing in this module inspects a verdict to change behavior, and nothing
here raises on an unfaithful verdict or a judge failure. Both are tallied
and returned; SPEC.md §6 and :mod:`groundkit.providers.judge`'s module docstring
own the calibration procedure that would ever be allowed to change that.

## What is, and is not, caught

Only :class:`~groundkit.errors.SynthesisError` (from ``synthesizer``) and
:class:`~groundkit.errors.JudgeError` (from ``judge``) are caught, per the
outcome buckets above. :class:`~groundkit.errors.ChatError` — a genuine
provider failure — is never caught here, the same rule
:mod:`groundkit.evals.echo` states for the identical reason: a harness-level
failure (the chat provider being unreachable or misconfigured) is not this
module's to paper over by mislabeling it as a per-query rejection.

## Prompt hashes name a template, not a per-call rendering

:attr:`~groundkit.evals.schema.SynthesisReport.synthesis_prompt_hash` and
:attr:`~groundkit.evals.schema.SynthesisReport.judge_prompt_hash` are each a
single value for the whole run, so they hash the constant *template* string
(``{query}``/``{sources}`` placeholders and all) — never a per-query
rendering, which would differ by query and could not collapse into one
run-level hash. The synthesis template is the one this function is given; the
judge template is read off the judge itself, never taken from a caller.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from groundkit.errors import JudgeError, SynthesisError
from groundkit.evals.schema import SynthesisReport
from groundkit.providers.llm import RedactingChat
from groundkit.providers.synthesis import DEFAULT_SYNTHESIS_PROMPT, Synthesizer

if TYPE_CHECKING:
    from collections.abc import Sequence

    from groundkit.contracts import RetrievalResult
    from groundkit.evals.schema import StageName
    from groundkit.providers.judge import FaithfulnessJudge
    from groundkit.providers.protocols import ChatProtocol


def hash_prompt_template(template: str) -> str:
    """SHA-256 hex digest of a prompt template string, UTF-8 encoded.

    The one place this run-level hash is computed, shared by both the
    synthesis and judge prompt hashes on
    :class:`~groundkit.evals.schema.SynthesisReport` so the two cannot
    silently diverge in hashing method.

    Args:
        template: The raw template string (e.g.
            :data:`~groundkit.providers.synthesis.DEFAULT_SYNTHESIS_PROMPT`).

    Returns:
        Hex-encoded SHA-256 digest.
    """
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


async def run_synthesis_eval(
    query_results: Sequence[tuple[str, Sequence[RetrievalResult]]],
    *,
    chat: ChatProtocol,
    input_stage: StageName,
    judge: FaithfulnessJudge | None = None,
    synthesis_prompt_template: str = DEFAULT_SYNTHESIS_PROMPT,
) -> SynthesisReport:
    """Synthesize an answer for every ``(query, results)`` pair, judging each if asked.

    One :class:`~groundkit.providers.synthesis.Synthesizer` is constructed
    here, around ``chat`` and ``synthesis_prompt_template``, and reused for
    every pair — mirroring how :func:`~groundkit.evals.runner.run_eval`
    reuses one :class:`~groundkit.retrieval.search.Retriever` across every
    judgment. ``judge``, when supplied, is used exactly as given — this
    function never constructs a judge itself, and every ``judge_*`` field on
    the report is read **off the judge**, never off ``chat``. Those are the
    same object on the ``grk eval --judge`` path, which is exactly why
    inferring them from ``chat`` was a trap: it stayed correct until someone
    judged with a different provider, and then the artifact misreported which
    model produced the verdicts while every test still passed. An earlier
    revision took a ``judge_prompt_template`` argument and asked the caller to
    keep it in agreement with the judge's real one; that is the hand-maintained
    parity ``tests/test_protocol_conformance.py`` exists to argue against, so
    the argument is gone rather than documented.

    Args:
        query_results: One entry per judgment: the query text and the
            retrieved results a caller has already fetched for the chosen
            ``input_stage`` (including any rerank step already applied).
            This function never retrieves anything itself.
        chat: The chat provider under test. Its ``provider``/``model_name``
            are recorded verbatim, and
            ``isinstance(chat, `` :class:`~groundkit.providers.llm.RedactingChat`
            ``)`` decides :attr:`~groundkit.evals.schema.SynthesisReport.redacted`
            — the same identity-reading rule
            :func:`~groundkit.evals.runner.run_eval` already follows for
            ``embedder``/``reranker`` (SPEC.md §2: an identity a caller could
            pass and disagree with is not this).
        input_stage: Which retrieval stage produced ``query_results`` —
            recorded verbatim onto the report.
        judge: Optional. When supplied, every non-rejected synthesis outcome
            is judged (see the module docstring), and its own
            ``provider``/``model_name``/``prompt_template`` become the
            report's judge provenance. When omitted, every ``judge_*`` field
            on the returned report is ``None``, honoring
            :meth:`~groundkit.evals.schema.SynthesisReport._validate_judge_fields`'s
            all-or-nothing group.
        synthesis_prompt_template: Template the internal ``Synthesizer``
            renders; also what :attr:`SynthesisReport.synthesis_prompt_hash`
            hashes.

    Returns:
        The completed :class:`~groundkit.evals.schema.SynthesisReport`.

    Raises:
        ChatError: Propagated unmodified from ``chat`` (via ``synthesizer``
            or ``judge``) — a provider failure is not a per-query outcome.
    """
    synthesizer = Synthesizer(chat, prompt_template=synthesis_prompt_template)

    answered_count = 0
    abstained_count = 0
    rejected_count = 0
    judged_count = 0
    faithful_count = 0
    unfaithful_count = 0
    judge_error_count = 0

    for query, results in query_results:
        try:
            synthesized = await synthesizer.synthesize(query, results)
        except SynthesisError:
            rejected_count += 1
            continue

        if synthesized.citations:
            answered_count += 1
        else:
            abstained_count += 1

        if judge is not None:
            judged_count += 1
            sources = [result.content for result in results]
            try:
                verdict = await judge.judge(query=query, answer=synthesized.answer, sources=sources)
            except JudgeError:
                judge_error_count += 1
                continue
            if verdict.faithful:
                faithful_count += 1
            else:
                unfaithful_count += 1

    run_judge = judge is not None
    return SynthesisReport(
        input_stage=input_stage,
        synthesis_provider=chat.provider,
        synthesis_model=chat.model_name,
        synthesis_prompt_hash=hash_prompt_template(synthesis_prompt_template),
        redacted=isinstance(chat, RedactingChat),
        answered_count=answered_count,
        abstained_count=abstained_count,
        rejected_count=rejected_count,
        # Read from the judge, never from `chat`. They are the same object on
        # the `grk eval --judge` path, which is exactly why taking the easy
        # one is a trap: it is right until someone judges with a different
        # provider, and then the artifact misreports which model produced the
        # verdicts while every test still passes.
        judge_provider=judge.provider if judge is not None else None,
        judge_model=judge.model_name if judge is not None else None,
        judge_prompt_hash=(
            hash_prompt_template(judge.prompt_template) if judge is not None else None
        ),
        judged_count=judged_count if run_judge else None,
        faithful_count=faithful_count if run_judge else None,
        unfaithful_count=unfaithful_count if run_judge else None,
        judge_error_count=judge_error_count if run_judge else None,
    )

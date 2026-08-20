"""Faithfulness judge (Phase 5, synthesis mode) — SPEC.md §6.

LLM-as-judge over a synthesized answer and the sources it was built from.
Given a query, an answer, and the source texts the answer is supposed to be
grounded in, :class:`FaithfulnessJudge` asks a chat model whether every claim
in the answer is supported, and validates the model's response against the
schema-pinned :class:`FaithfulnessVerdict` before returning it. The chat call
is injected via :class:`~groundkit.providers.protocols.ChatProtocol`, so unit
tests exercise this module with a scripted fake and never touch the network.

## Why this lives in ``providers``, not ``evals``

This is a :class:`~groundkit.providers.protocols.ChatProtocol` consumer,
exactly like :mod:`groundkit.providers.synthesis` and
:mod:`groundkit.providers.query_rewrite`, and it sits beside them. It began
in ``groundkit.evals`` because the eval harness needed it first, but
:mod:`groundkit.answer` — a runtime module on the ``grk answer`` path,
nothing to do with the harness — composes it too. A production module
importing the eval package makes ``evals`` undroppable from a runtime
install and would make this module the one special case in
``tests/test_deterministic_core.py``'s "no runtime module imports
``groundkit.evals``" scan; needing that exemption is the signal it was on
the wrong side of the line, so it moved rather than being exempted.

The import direction is now one-way and this module is at the far end of
it: it imports **nothing** from :mod:`groundkit.evals` and nothing from
:mod:`groundkit.providers.llm` or :mod:`groundkit.providers.synthesis`
either. It takes and returns plain strings and a verdict model, never an
:class:`~groundkit.evals.schema.EvalReport`, a
:class:`~groundkit.providers.synthesis.SynthesizedAnswer`, or anything the
eval runner produces. Mapping a synthesized answer and its retrieved
sources into the plain ``str`` arguments this module expects, and folding a
verdict into an eval artifact, are both jobs for a caller one layer up
(:mod:`groundkit.answer`, :mod:`groundkit.evals.synthesis_eval`) — never
for this module.

**ADVISORY ONLY.** :class:`FaithfulnessJudge` returns a
:class:`FaithfulnessVerdict` or raises :class:`~groundkit.errors.JudgeError`.
It never calls ``sys.exit``, never raises ``SystemExit``, and has no notion
of a process exit code — that policy, and any CI-gating decision built on
top of a verdict, belongs at the harness surface (the CLI command or eval
runner that calls this module), never here. Until the calibration procedure
below has been carried out and its result recorded, no caller may treat a
``faithful=False`` verdict — or a :class:`~groundkit.errors.JudgeError` — as
a build failure. A broken or disagreeing judge is signal for a human reader,
not a gate.

**Calibration procedure (required before gating may ever be proposed):**

1. **Commit a human-labeled verdict set.** Run actual synthesis over a
   sample of golden-corpus queries, then have a human independently label
   each ``(query, answer, sources)`` triple faithful or unfaithful — ideally
   also noting which claims are unsupported. Commit that label set alongside
   the golden corpus so it is versioned and reviewable like any other eval
   fixture, not held in a scratch file that can drift or disappear.
2. **Measure agreement.** Run this judge over the same sample and compare
   its verdicts to the human labels. Which statistic to use (e.g. Cohen's
   kappa, raw percent agreement, precision/recall against the human
   "unfaithful" label) and what bar counts as acceptable are decisions to be
   made and recorded *at calibration time*, not assumed here — they depend
   on the label distribution and on how costly a false negative is in this
   harness.
3. **Propose gating only via an ADR, only after that measurement exists.**
   The ADR must name the chosen statistic, the acceptance bar, the measured
   value, and the exact point in the harness where a verdict would gate.
   Until such an ADR is written and merged, every verdict this module
   produces is advisory input for a human reader — never a pass/fail signal.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from groundkit.errors import JudgeError
from groundkit.providers.protocols import ChatProtocol

#: Matches a completion that is, in its entirety (after stripping surrounding
#: whitespace), a single markdown code fence — optionally tagged (``` ```json ```)
#: — wrapping a JSON body. Anchored at both ends so leading or trailing prose
#: outside the fence prevents a match; see :func:`_extract_json_text`.
_FENCE_RE = re.compile(r"^```[^\n]*\n(?P<body>.*)\n```$", re.DOTALL)


class FaithfulnessVerdict(BaseModel):
    """Schema-validated verdict the model's chat completion is checked against.

    This is deliberately strict: ``strict=True`` disables Pydantic's default
    lax-mode scalar coercions (a JSON string ``"true"`` will not become the
    boolean ``True``; a JSON number will not become ``reasoning``'s ``str``).
    The container conversion JSON needs — a JSON array populating
    ``unsupported_claims: tuple[str, ...]`` — is unaffected, because Pydantic
    treats "JSON array becomes tuple" as part of parsing JSON's native
    shapes, not as the kind of type coercion strict mode exists to forbid.
    Malformed model output must be rejected, never repaired (SPEC.md §2), so
    there is no default-filling and no second attempt inside this class or
    :class:`FaithfulnessJudge` — a schema mismatch is exactly one
    :class:`~groundkit.errors.JudgeError`, once.

    Attributes:
        faithful: Whether every claim in the judged answer is supported by
            the judged sources.
        unsupported_claims: Claims from the answer the judge found
            unsupported. Must be empty when ``faithful`` is ``True`` — see
            :meth:`_validate_coherence`.
        reasoning: The judge's explanation for the verdict.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    faithful: bool
    unsupported_claims: tuple[str, ...]
    reasoning: str

    @model_validator(mode="after")
    def _validate_coherence(self) -> FaithfulnessVerdict:
        """Reject a verdict that claims full faithfulness but lists claims it isn't.

        ``faithful=True`` together with a non-empty ``unsupported_claims`` is
        not two independent facts the caller could reconcile — it is one
        incoherent verdict, and an incoherent verdict is malformed output
        exactly like a missing field is (SPEC.md §2, fail closed). The
        reverse is not asserted: ``faithful=False`` with an empty
        ``unsupported_claims`` is left valid, since a judge may reasonably
        find an answer unfaithful for reasons that don't reduce to a list of
        quoted claims.

        Raises:
            ValueError: ``faithful`` is ``True`` and ``unsupported_claims``
                is non-empty.
        """
        if self.faithful and self.unsupported_claims:
            raise ValueError(
                "faithful=True is inconsistent with "
                f"{len(self.unsupported_claims)} unsupported_claims entry(ies); "
                "a fully faithful verdict must report zero"
            )
        return self


#: Template presenting the query, the answer, and the numbered source texts,
#: instructing the model to return only a JSON object shaped like
#: :class:`FaithfulnessVerdict`. :meth:`FaithfulnessJudge.judge` fills in
#: ``{query}``, ``{answer}``, and ``{numbered_sources}`` (sources rendered as
#: a ``[1] ...`` / ``[2] ...`` list so every source is distinguishable in the
#: prompt).
DEFAULT_JUDGE_PROMPT = """You are a faithfulness judge for a retrieval-augmented \
answer. You are given a query, an answer produced by another system, and the \
numbered source texts the answer is supposed to be grounded in.

Query:
{query}

Answer:
{answer}

Sources:
{numbered_sources}

Judge whether every claim made in the Answer is supported by at least one of \
the numbered Sources above. A claim is unsupported if no source states it, \
implies it, or provides evidence for it.

Respond with ONLY a JSON object — no prose, no markdown, no code fence — with \
exactly these three fields and no others:

{{"faithful": true or false, "unsupported_claims": [list of exact quotes from \
the Answer that are not supported by any Source; empty if faithful is true], \
"reasoning": "one or two sentences explaining the verdict"}}
"""


class FaithfulnessJudge:
    """LLM-as-judge producing a schema-validated :class:`FaithfulnessVerdict`.

    The chat call is injected as a :class:`~groundkit.providers.protocols.ChatProtocol`
    so this class never imports a concrete provider and unit tests never
    touch the network. See the module docstring for this judge's advisory
    scope and the calibration procedure required before any caller may gate
    on its output.
    """

    def __init__(self, chat: ChatProtocol, *, prompt_template: str = DEFAULT_JUDGE_PROMPT) -> None:
        """Initialize the judge.

        Args:
            chat: The chat completion provider to use. Never constructed or
                imported by this class — always supplied by the caller, so a
                unit test can hand in a scripted fake.
            prompt_template: Template rendered before each judge call, with
                ``{query}``, ``{answer}``, and ``{numbered_sources}``
                substituted in. Defaults to :data:`DEFAULT_JUDGE_PROMPT`.
        """
        self._chat = chat
        self._prompt_template = prompt_template

    @property
    def provider(self) -> str:
        """Provider identity of the chat this judge actually calls.

        Exposed so an artifact can record the judge's *own* provenance rather
        than inferring it from whatever chat the caller happened to use for
        synthesis. Those are the same object on the `grk eval --judge` path
        and need not be in general, and a report that records the wrong one
        is worse than a report that records nothing: `judge_provider`,
        `judge_model` and `judge_prompt_hash` exist precisely so a reader can
        tell whether two runs are comparable.
        """
        return self._chat.provider

    @property
    def model_name(self) -> str:
        """Model identity of the chat this judge actually calls."""
        return self._chat.model_name

    @property
    def prompt_template(self) -> str:
        """The template this judge actually renders, for provenance hashing."""
        return self._prompt_template

    async def judge(
        self, *, query: str, answer: str, sources: Sequence[str]
    ) -> FaithfulnessVerdict:
        """Judge whether ``answer`` is faithful to ``sources``.

        Fails closed at every stage: an empty ``answer`` or empty ``sources``
        is rejected before the provider is ever called; a provider error
        propagates untouched (never caught or wrapped here); and a
        completion that is not schema-valid JSON — missing a field, carrying
        an extra one, using the wrong type, or internally incoherent — is
        rejected as :class:`~groundkit.errors.JudgeError`, never coerced or
        retried.

        A completion wrapped in a single markdown code fence (` ```json ... ``` `
        or plain ` ``` ... ``` `) has the fence stripped before parsing —
        that is a deterministic extraction of a well-known wrapper, not a
        repair. Anything looser (leading or trailing prose around the JSON,
        JSON embedded mid-sentence) is rejected rather than searched for.

        Args:
            query: The query the answer was produced for.
            answer: The synthesized answer to judge.
            sources: The source texts the answer is supposed to be grounded
                in. Rendered into the prompt numbered from 1.

        Returns:
            The schema-validated verdict.

        Raises:
            JudgeError: ``answer`` or ``sources`` is empty, or the chat
                completion could not be parsed into a valid
                :class:`FaithfulnessVerdict`.
        """
        if not answer or not answer.strip():
            raise JudgeError("answer must not be empty")
        if not sources:
            raise JudgeError("sources must not be empty")

        numbered_sources = "\n".join(
            f"[{index}] {source}" for index, source in enumerate(sources, start=1)
        )
        prompt = self._prompt_template.format(
            query=query, answer=answer, numbered_sources=numbered_sources
        )

        completion = await self._chat.complete(prompt)

        json_text = _extract_json_text(completion)
        try:
            return FaithfulnessVerdict.model_validate_json(json_text)
        except ValidationError as exc:
            field_summary = "; ".join(
                f"{_format_loc(err['loc'])} ({err['type']})" for err in exc.errors()
            )
            raise JudgeError(
                f"chat completion (length={len(completion)} chars) failed "
                f"verdict schema validation: {field_summary}"
            ) from exc


def _extract_json_text(completion: str) -> str:
    """Strip a single wrapping markdown code fence, if the whole completion is one.

    Fires only when the *entire* trimmed completion is one fenced block —
    optionally tagged (e.g. ` ```json `) — from the first ``` to the last.
    Any leading or trailing prose outside the fence prevents a match, so
    that text reaches JSON parsing completely unmodified and is rejected
    there. This is deterministic extraction of a known wrapper shape, not a
    search for JSON somewhere in the string.

    Args:
        completion: The raw chat completion text.

    Returns:
        The completion with a single wrapping fence removed, or the
        whitespace-trimmed completion unchanged if it was not one.
    """
    stripped = completion.strip()
    match = _FENCE_RE.match(stripped)
    if match:
        return match.group("body").strip()
    return stripped


def _format_loc(loc: tuple[int | str, ...]) -> str:
    """Render a Pydantic error location as a dotted path, with no field values.

    Args:
        loc: The ``loc`` tuple from one ``ValidationError.errors()`` entry.

    Returns:
        Dot-joined location, or ``"<root>"`` for a whole-document error
        (e.g. invalid JSON syntax, which has an empty ``loc``).
    """
    return ".".join(str(part) for part in loc) if loc else "<root>"

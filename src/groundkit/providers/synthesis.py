"""Cited synthesis over retrieved spans (Phase 5, SPEC.md §2 boundary).

SPEC.md §2 states the obligation verbatim: "Synthesis may cite only retrieved
spans." An answer that cites anything else is rejected outright — never
repaired, never silently dropped — because coercing an out-of-set marker into
something citable would assert a verifiable citation that verifies nothing
(see :class:`~groundkit.errors.SynthesisError`).

This module enforces exactly that set-membership contract: every ``[n]``
marker a completion emits must name one of the
:class:`~groundkit.contracts.RetrievalResult` objects the caller actually
retrieved, or synthesis fails closed. It makes **no** claim about *echo
quality* — whether the model's prose faithfully reflects what a cited source
actually says. That is a different, harder property, measured by the eval
harness's planted-marker echo check (SPEC.md §2), which is built separately
on top of this module rather than inside it. A citation set can pass every
check here and still misrepresent its sources; this module only guarantees
that the set it points at was retrieved.

## Marker parsing rule

Citation markers are recognized with a single deterministic regular
expression, ``\\[(\\d+)\\]``, applied to the entire completion. Every match
counts, in left-to-right order, with no attempt to distinguish a marker the
model placed as its own citation from a bracketed number that happens to
appear inside source text the model echoed back verbatim (e.g. a footnote
reference embedded in a quoted passage). Two properties this buys:

- **Determinism.** The same completion always yields the same citation set;
  nothing here depends on prompt phrasing or a model-specific convention for
  "real" vs. "quoted" markers, which text alone cannot reliably distinguish.
- **Honesty about the limit.** Since real citation markers and echoed source
  content look identical to this parser, a source chunk that itself contains
  bracketed numbers could inflate the detected citation count.
  ``DEFAULT_SYNTHESIS_PROMPT`` does not ask the model to quote sources
  verbatim, which limits the practical exposure, but the tradeoff is real and
  is accepted deliberately in exchange for a simple, testable rule rather
  than a heuristic that would disagree with itself across inputs.

Duplicate markers, and markers naming distinct
:class:`~groundkit.contracts.RetrievalResult` entries that are themselves
duplicates of one another (the same underlying chunk retrieved twice), never
produce duplicate citations — the result is deduplicated on the
:class:`~groundkit.contracts.Citation` value itself, preserving first-mention
order.

## Retrieved content is sanitized before it enters the prompt

Each source's content passes through
:func:`~groundkit.providers.context_assembly.sanitize_content` (SPEC.md §6)
before :func:`_format_sources` writes it into the prompt: control characters
are stripped, any literal ``<retrieved_context>``/``</retrieved_context>``
delimiter text is neutralized, and every line is quote-prefixed. This is the
same structural defense :mod:`~groundkit.providers.context_assembly`
documents for its own ``frame_content``/``TokenBudgetAssembler`` path; see
that module's docstring for exactly what it does and does not guarantee
(structural, not semantic — instruction-like phrasing survives, merely
quote-prefixed). That assessment applies here unchanged and is not restated.
Sanitization touches only the rendered prompt text; it never touches a
result's own ``citation`` property, so citation offsets are unaffected by
anything done here.

## Never embed untrusted text in exceptions

No exception raised here interpolates the answer, chunk content, or the query
into its message — only structural facts (marker numbers, valid ranges,
counts, provider/model identity) that cannot themselves carry retrieved or
user-supplied content.

## The synthesis span is the sharpest of SPEC.md §3's three (ADR-0022 decision 5)

:meth:`Synthesizer.synthesize` opens one OTel span, ``groundkit.synthesize.synthesize``,
around the whole call. It sits closer to prompt text, completion text, and
citation spans than either of SPEC.md §3's other two span sites, which is
exactly why the allowlist matters most here: **none of the query, the
retrieved content, the rendered prompt, the completion, or a citation ever
becomes a span attribute — not even truncated, not even on the error path.**
Only a result count, latency, the chat provider/model identity ADR-0022
decision 5 permits on this span specifically (configuration the operator
chose, never content a user or a document supplied — and the attribute that
makes "which model was slow" answerable without any of the above), and — on
failure — the raised exception's type name as a typed failure code reach the
span, all via
:func:`~groundkit.telemetry.span_attributes`, which has no parameter any of
the forbidden values could be passed through even by mistake.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from typing import Final

from pydantic import BaseModel, ConfigDict

from groundkit.contracts import Citation, RetrievalResult
from groundkit.errors import SynthesisError
from groundkit.providers.context_assembly import sanitize_content
from groundkit.providers.protocols import ChatProtocol
from groundkit.telemetry import get_tracer, span_attributes

#: Matches one bracketed citation marker, e.g. ``[1]``. See the module
#: docstring's "Marker parsing rule" section for what counts as a match and why.
_MARKER_PATTERN: Final[re.Pattern[str]] = re.compile(r"\[(\d+)\]")

#: Tracer for this module's one span, ``groundkit.synthesize.synthesize``
#: (ADR-0022 decision 5). See the module docstring's final section for why
#: this is the span the allowlist matters most on.
tracer = get_tracer()

#: Default prompt template. Presents each retrieved result as a numbered source
#: and instructs the model to answer only from those sources, mark every claim
#: with the source number(s) it draws from, and say so plainly when the sources
#: do not contain an answer — the three obligations SPEC.md §2 and the Phase 5
#: design place on synthesis. The instructions deliberately use the placeholder
#: letters ``[N]``/``[M]`` rather than concrete digits, so the *sources* section
#: is the only place in the rendered prompt where a real ``[n]`` marker
#: appears — this is what lets a test assert each numbered source appears
#: exactly once in the rendered prompt.
DEFAULT_SYNTHESIS_PROMPT: Final[str] = """\
Answer the question using ONLY the numbered sources below. Do not use any outside \
knowledge, and do not invent sources that are not listed.

Sources:
{sources}

Question: {query}

Instructions:
- Answer strictly from the sources above; nothing else.
- Mark every claim with the bracketed number of the source it draws from, e.g. \
[N], or [N][M] when a claim draws from more than one source.
- If the sources do not contain the answer, say so plainly instead of guessing.

Answer:"""


class SynthesizedAnswer(BaseModel):
    """A synthesized answer and the retrieved citations it actually used.

    ``citations`` is never constructed fresh — every entry is one of the input
    :class:`~groundkit.contracts.RetrievalResult` objects' own ``citation``
    property, taken verbatim (SPEC.md §2: synthesis may cite only retrieved
    spans). An empty tuple is a valid abstention, not an error: a completion
    that cites nothing because the sources did not contain an answer is
    exactly the behavior :data:`DEFAULT_SYNTHESIS_PROMPT` asks for.

    Attributes:
        answer: The model's completion text, citation markers included.
        citations: The cited results' :class:`~groundkit.contracts.Citation`
            objects, deduplicated, in first-mention order.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    answer: str
    citations: tuple[Citation, ...]


def _format_sources(results: Sequence[RetrievalResult]) -> str:
    """Render ``results`` as a one-indexed numbered source list.

    Each result's content is passed through
    :func:`~groundkit.providers.context_assembly.sanitize_content` before it
    is written into the list — see the module docstring's "Retrieved content
    is sanitized before it enters the prompt" section for what that defense
    does and does not guarantee.

    Source ``i`` in the rendered list is ``results[i - 1]`` — the only
    contract between the prompt and :func:`_resolve_citations`, which inverts
    exactly this mapping. The ``[i]`` marker prefix itself is not sanitized
    content — it is written by this function, never taken from a result — so
    sanitization cannot alter which source number resolves to which citation.

    Args:
        results: The retrieved results to present as numbered sources.

    Returns:
        The sanitized sources, one-indexed, separated by a blank line.
    """
    lines = [
        f"[{index}] {sanitize_content(result.content)}"
        for index, result in enumerate(results, start=1)
    ]
    return "\n\n".join(lines)


def _resolve_citations(completion: str, results: Sequence[RetrievalResult]) -> tuple[Citation, ...]:
    """Parse ``[n]`` markers out of ``completion`` and resolve them against ``results``.

    Args:
        completion: The model's completion text.
        results: The retrieved results the completion may cite, in the same
            order they were numbered into the prompt by :func:`_format_sources`.

    Returns:
        The cited results' ``Citation`` objects, deduplicated by value, in
        first-mention order. Empty when the completion cites nothing at all —
        a valid abstention.

    Raises:
        SynthesisError: A marker names a source number outside
            ``1..len(results)``. The message names the marker and the valid
            range only — never the completion, chunk content, or query.
    """
    valid_count = len(results)
    seen: set[Citation] = set()
    citations: list[Citation] = []
    for match in _MARKER_PATTERN.finditer(completion):
        marker = int(match.group(1))
        if marker < 1 or marker > valid_count:
            raise SynthesisError(
                f"completion cited source marker [{marker}], which is outside the "
                f"retrieved range [1]..[{valid_count}]; synthesis may cite only "
                "retrieved spans"
            )
        citation = results[marker - 1].citation
        if citation not in seen:
            seen.add(citation)
            citations.append(citation)
    return tuple(citations)


class Synthesizer:
    """Synthesizes a cited answer from a query and its retrieved results.

    Args:
        chat: A :class:`~groundkit.providers.protocols.ChatProtocol`
            implementation. This class calls only ``chat.complete``; any
            provider configuration, retries, or timeouts are the chat
            implementation's concern, not this module's.
        prompt_template: A ``str.format``-style template with ``{query}`` and
            ``{sources}`` placeholders. Defaults to
            :data:`DEFAULT_SYNTHESIS_PROMPT`.
    """

    def __init__(
        self, chat: ChatProtocol, *, prompt_template: str = DEFAULT_SYNTHESIS_PROMPT
    ) -> None:
        self._chat = chat
        self._prompt_template = prompt_template

    async def synthesize(self, query: str, results: Sequence[RetrievalResult]) -> SynthesizedAnswer:
        """Answer ``query`` from ``results``, rejecting any citation outside them.

        Args:
            query: The user's question.
            results: The retrieved spans the answer may draw from and cite.
                Must be non-empty — synthesis without retrieved spans has
                nothing it could legitimately cite.

        Returns:
            The synthesized answer with its resolved citations (SPEC.md §2).

        Raises:
            SynthesisError: ``results`` is empty, ``query`` is empty or
                whitespace-only, the provider's completion is empty or
                whitespace-only, or the completion cites a marker outside
                ``1..len(results)``. Never repaired — a synthesis call that
                fails this contract produces no answer at all.
            ChatError: Propagated untouched from ``chat.complete`` — this
                method never catches or wraps a provider failure. The span
                below labels it with a typed failure code for telemetry and
                then re-raises it exactly as received; that is not "catching"
                it in the sense this docstring means, since nothing about the
                exception is altered, inspected beyond its type, or swallowed.

        Wrapped in one OTel span, ``groundkit.synthesize.synthesize`` — see
        the module docstring's final section for what may and, above all,
        may NOT become an attribute on it.
        """
        with tracer.start_as_current_span("groundkit.synthesize.synthesize") as span:
            started = time.perf_counter()
            try:
                if not results:
                    raise SynthesisError(
                        "synthesis requires at least one retrieved result; an answer cannot "
                        "cite spans that were never retrieved"
                    )
                if not query.strip():
                    raise SynthesisError("synthesis requires a non-empty query")

                prompt = self._prompt_template.format(query=query, sources=_format_sources(results))
                completion = await self._chat.complete(prompt)
                answer = completion.strip()
                if not answer:
                    raise SynthesisError(
                        f"{self._chat.provider}/{self._chat.model_name} returned an "
                        "empty completion"
                    )

                citations = _resolve_citations(answer, results)
                synthesized = SynthesizedAnswer(answer=answer, citations=citations)
            except Exception as exc:
                # Telemetry only: this except exists to label the span, never
                # to handle the failure. The bare `raise` below re-raises the
                # exact exception, unwrapped — see the Raises section above.
                # Only the exception's TYPE NAME is read; its message (which,
                # for SynthesisError, is guaranteed free of retrieved/user
                # content per the module docstring, but is not re-verified
                # here) is never touched.
                span.set_attributes(
                    span_attributes(
                        duration_ms=(time.perf_counter() - started) * 1000,
                        failure_kind=type(exc).__name__,
                        chat_provider=self._chat.provider,
                        chat_model=self._chat.model_name,
                    )
                )
                raise
            span.set_attributes(
                span_attributes(
                    result_count=len(results),
                    duration_ms=(time.perf_counter() - started) * 1000,
                    chat_provider=self._chat.provider,
                    chat_model=self._chat.model_name,
                )
            )
            return synthesized

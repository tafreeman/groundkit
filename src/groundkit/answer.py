"""Phase 5 answer composition: rewrite, retrieve, synthesize, judge (SPEC.md §9).

This is the composition root behind the future ``grk answer`` CLI verb
(ADR-0019): the code that wires an optional query rewrite, a retrieval
search, a cited synthesis, and an optional advisory faithfulness judge into
one call and one report. Building the collaborators it composes — resolving
a chat config, constructing a concrete chat provider, opening a
``Retriever`` over a persisted collection — is deliberately not this
module's job: this module never imports ``groundkit.config`` or
``groundkit.providers.llm``, and constructs no provider itself.

## What "injected" does and does not mean here

Every collaborator *instance* is supplied by the caller — none is
constructed, discovered, or defaulted here. The *types* are not
correspondingly abstract, and nothing below claims they are: only
``search`` is a structural Protocol (:class:`SearchCallable`, and even that
exists so this module's own tests need not build a real ``Retriever``,
not as a declared extension point). ``synthesizer``, ``rewriter`` and
``judge`` are annotated as the concrete
:class:`~groundkit.providers.synthesis.Synthesizer`,
:class:`~groundkit.providers.query_rewrite.QueryRewriter` and
:class:`~groundkit.providers.judge.FaithfulnessJudge` classes, which this
module therefore imports; a substitute must be an instance of the class,
not merely something with a matching shape.

That is a deliberate trade rather than an oversight. All three are
themselves :class:`~groundkit.providers.protocols.ChatProtocol` consumers
whose own seam is the chat provider, so the substitution a caller actually
wants — a different model, a scripted fake, a redacting wrapper — is
reached by injecting a different ``ChatProtocol`` into them, and a second
Protocol layer here would only restate that seam one level out. All three
also now live together under ``groundkit.providers`` (GK-021 moved the
judge there from ``groundkit.evals``), so importing them commits this
module to no dependency on the eval harness.

## Abstention has one representation

:class:`~groundkit.providers.synthesis.SynthesizedAnswer` has no
``abstained`` field — an empty ``citations`` tuple *is* abstention
(``docs/specs/phase-5-boundary-features.md`` §6, left open there as Q2 and
settled here for this pipeline). :class:`AnswerReport` does not add a
second, redundant boolean: it exposes exactly the same rule its
``citations`` field already carries, unchanged, all the way through.

## The judge is advisory, never a gate, and never special-cases abstention

When a :class:`~groundkit.providers.judge.FaithfulnessJudge` is injected, it
runs after synthesis and its verdict is recorded — unconditionally, with no
attempt to detect or skip judging an abstained answer, because "abstained"
is not a branch this module takes, only a shape ``citations`` happens to
have. If judging is requested and the judge raises
:class:`~groundkit.errors.JudgeError` (or anything else), that error
propagates out of :meth:`AnswerPipeline.answer` uncaught: a requested
component that breaks is a typed failure, not a silently-skipped opinion
(ADR-0018 decision 5). "Advisory" describes what a *successful* verdict is
permitted to influence — nothing here inspects ``verdict.faithful`` to
change the answer or the citations — not whether a broken judge is allowed
to fail quietly.

## What the judge is shown

The judge is given every retrieved result's content, in the same rank
order the synthesizer was shown them — not only the subset of results the
answer ended up citing. ADR-0018 frames the judge's question as "whether
every claim in the Answer is supported by at least one of the numbered
Sources", and the synthesis prompt already presented the *full* retrieved
set as those sources. Narrowing the judge to only the cited subset would
judge the answer against a smaller context than the one it was actually
generated from, and would make "was this text in front of the model when
it wrote that claim" a question the verdict cannot actually answer.

## Query rewrite makes two strings, and both survive

When a :class:`~groundkit.providers.query_rewrite.QueryRewriter` is
injected, retrieval runs against its *rewritten* output, but synthesis and
judging run against the caller's *original* query text — the rewrite exists
to help retrieval match, not to replace the question the answer must
actually address. :class:`AnswerReport` records both strings explicitly
(``query`` and ``rewritten_query``) rather than picking one, which is
exactly the ambiguity ADR-0019 §2 says a single-``query``-field model like
``SearchResponse`` cannot express.

## Fail-closed, single source of truth

An empty or whitespace-only query is rejected before any collaborator
runs — no rewrite, no search, no synthesis call is ever made for it. A
search that comes back with zero results is *not* separately checked
here: :class:`~groundkit.providers.synthesis.Synthesizer` already raises
:class:`~groundkit.errors.SynthesisError` on an empty result set, and
duplicating that check here would create two sources of truth for one
rule. Every other collaborator failure — a rewrite error, a search error, a
synthesis error, a judge error — propagates unmodified: this module neither
catches nor wraps any of them.

No exception raised directly by this module interpolates query, answer, or
retrieved chunk text.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from groundkit.contracts import Citation, RetrievalResult, SearchResponse
from groundkit.errors import RetrievalError
from groundkit.providers.judge import FaithfulnessJudge, FaithfulnessVerdict
from groundkit.providers.query_rewrite import QueryRewriter
from groundkit.providers.synthesis import Synthesizer
from groundkit.retrieval.search import SearchMode


@runtime_checkable
class SearchCallable(Protocol):
    """Structural shape of the retrieval collaborator :class:`AnswerPipeline` composes.

    Matches ``Retriever.search``'s exact positional/keyword shape, so a
    bound method reference (``retriever.search``) satisfies this Protocol
    with no adapter, and a plain ``async def`` function with the same
    shape satisfies it too — "a Retriever-like search callable or
    instance" resolves to one requirement, callability with this
    signature, regardless of which of the two a caller hands in.

    This is a private, orchestration-level structural type local to this
    module — distinct from ``ChatProtocol``, the one provider-boundary
    Protocol Phase 5 formally introduces in ``providers/protocols.py``
    (``docs/specs/phase-5-boundary-features.md`` §3). It exists only so
    this module's own tests can supply a minimal scripted double instead of
    constructing a real ``Retriever`` — real stores, a real BM25 index —
    for every case; it is not a declared multi-implementation extension
    point the way ``ChatProtocol`` is.
    """

    async def __call__(
        self,
        query: str,
        top_k: int | None = None,
        *,
        mode: SearchMode = "bm25",
    ) -> SearchResponse: ...


class AnswerReport(BaseModel):
    """The full record of one ``grk answer`` composition (ADR-0019).

    Attributes:
        query: The caller's original query text, recorded verbatim
            regardless of whether a rewrite ran.
        rewritten_query: The rewritten query text, when a
            :class:`~groundkit.providers.query_rewrite.QueryRewriter` was
            injected and retrieval ran against its output. ``None`` when no
            rewriter was configured for this pipeline.
        answer: The synthesized answer text, citation markers intact.
        citations: The cited results' verifiable
            :class:`~groundkit.contracts.Citation` objects, deduplicated, in
            first-mention order — taken verbatim from
            ``SynthesizedAnswer.citations``. An empty tuple is a valid
            abstention, not an error; there is no separate boolean for it
            (see the module docstring).
        results: Every result that was retrieved and therefore available to
            synthesis (and, if judged, to the judge), in the order
            retrieval returned them — what was retrieved, so the answer is
            auditable against its actual inputs rather than only against
            the subset it happened to cite.
        verdict: The advisory faithfulness verdict, when a
            :class:`~groundkit.providers.judge.FaithfulnessJudge` was injected
            and ran successfully. ``None`` when no judge was configured.
            Never inspected by this module to alter ``answer`` or
            ``citations`` — advisory in the strict sense (ADR-0018).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str
    rewritten_query: str | None
    answer: str
    citations: tuple[Citation, ...]
    results: tuple[RetrievalResult, ...]
    verdict: FaithfulnessVerdict | None


class AnswerPipeline:
    """Composes rewrite, retrieval, synthesis, and an advisory judge into one call.

    Every collaborator *instance* is supplied by the caller — this class
    builds none of them, imports no configuration, and constructs no
    provider. Their *types* are concrete rather than abstract, deliberately;
    see the module docstring for what that does and does not buy. A class
    (rather than a bare function) because every optional collaborator it
    holds — ``rewriter``, ``judge`` — is fixed for the lifetime of a caller's
    usage (one ``grk answer`` invocation, one eval run) and this repo's
    existing Phase 5 collaborators (``Synthesizer``, ``QueryRewriter``,
    ``FaithfulnessJudge``) are themselves constructor-injected classes; a
    function would need every collaborator re-passed on each call instead of
    bound once. See the module docstring for the rewrite/synthesis query
    split, the judge's input set, and the fail-closed precondition this
    class itself enforces.

    Args:
        search: The retrieval collaborator, e.g. a
            :class:`~groundkit.retrieval.search.Retriever` instance's bound
            ``search`` method. See :class:`SearchCallable`.
        synthesizer: Produces the cited answer from the query and the
            search results.
        rewriter: Optional. When supplied, every :meth:`answer` call rewrites
            the query first and searches with the rewritten text (see the
            module docstring). When omitted,
            :attr:`AnswerReport.rewritten_query` is always ``None``.
        judge: Optional. When supplied, every :meth:`answer` call judges the
            synthesized answer against every retrieved result's content
            (see the module docstring). When omitted,
            :attr:`AnswerReport.verdict` is always ``None``.
    """

    def __init__(
        self,
        search: SearchCallable,
        synthesizer: Synthesizer,
        *,
        rewriter: QueryRewriter | None = None,
        judge: FaithfulnessJudge | None = None,
    ) -> None:
        self._search = search
        self._synthesizer = synthesizer
        self._rewriter = rewriter
        self._judge = judge

    async def answer(
        self,
        query: str,
        *,
        top_k: int | None = None,
        mode: SearchMode = "bm25",
    ) -> AnswerReport:
        """Compose one answer for ``query``.

        Args:
            query: The caller's question. Rejected before any collaborator
                runs if empty or whitespace-only.
            top_k: Result cap forwarded to ``search`` unchanged; ``None``
                defers to whatever default ``search`` itself applies.
            mode: Search mode forwarded to ``search`` unchanged.

        Returns:
            The composed :class:`AnswerReport`.

        Raises:
            RetrievalError: ``query`` is empty or whitespace-only (raised
                before ``rewriter``, ``search``, or ``synthesizer`` is ever
                called), or propagated unmodified from ``search`` (e.g. an
                out-of-range ``top_k``, or a search-time index
                inconsistency).
            QueryRewriteError: Propagated unmodified from ``rewriter``, when
                one is configured.
            ChatError: Propagated unmodified from any collaborator's
                underlying chat provider call.
            ChatProviderNotConfiguredError: Propagated unmodified, as above
                (a ``ChatError`` subclass).
            SynthesisError: Propagated unmodified from ``synthesizer`` —
                including the empty-results case, deliberately not
                pre-checked here (see the module docstring).
            JudgeError: Propagated unmodified from ``judge``, when one is
                configured.
        """
        if not query.strip():
            raise RetrievalError("Query must not be empty")

        rewritten_query: str | None = None
        search_query = query
        if self._rewriter is not None:
            rewritten_query = await self._rewriter.rewrite(query)
            search_query = rewritten_query

        response = await self._search(search_query, top_k=top_k, mode=mode)

        synthesized = await self._synthesizer.synthesize(query, response.results)

        verdict: FaithfulnessVerdict | None = None
        if self._judge is not None:
            sources = [result.content for result in response.results]
            verdict = await self._judge.judge(
                query=query, answer=synthesized.answer, sources=sources
            )

        return AnswerReport(
            query=query,
            rewritten_query=rewritten_query,
            answer=synthesized.answer,
            citations=synthesized.citations,
            results=tuple(response.results),
            verdict=verdict,
        )

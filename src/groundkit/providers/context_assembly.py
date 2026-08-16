"""RAG context assembly — token-budget-aware result assembly and
prompt-injection framing (Phase 5, SPEC.md §6 boundary).

Ported near-verbatim from ARP's ``agentic_v2/rag/context_assembly.py``
(agentic-runtime-platform, commit ``b41567ca2cf047b050ca034ce6c2966d2552de69``
— the file was deleted on that repo's ``rag_removal`` branch by commit
``9db6129f`` "refactor(rag): remove the agentic_v2.rag package, CLI, tests,
and CI job", so its content was read via ``git show <parent>:<path>`` rather
than the checked-out working tree) per ADR-0001's module table entry:
"Promote → providers/ (Phase 5). Port near-verbatim — the
``<retrieved_context>`` framing/sanitization is verified by 16 tests and
matches groundkit's trust-boundary requirement."

Provides:
- :func:`sanitize_content` / :func:`sanitize_provenance_value`: normalize
  retrieved text and provenance metadata so neither can forge the wrapper's
  delimiter structure or smuggle raw control characters into a prompt.
- :func:`frame_content`: wraps retrieved content in ``<retrieved_context>``
  delimiter tags, signaling to a completion model that the enclosed text is
  untrusted retrieved data, not an instruction.
- :class:`TokenBudgetAssembler`: greedily assembles
  :class:`~groundkit.contracts.RetrievalResult` objects into a
  :class:`~groundkit.contracts.SearchResponse` within a configurable token
  budget, framing each result's content along the way.

**Adapted from the ARP source** (every behavioral difference from the
original, itemized):

- *Data types.* ARP's RAG-local ``RetrievalResult``/``RAGResponse`` (defined
  in its own, now-deleted ``agentic_v2/rag/contracts.py``) are replaced with
  groundkit's :class:`~groundkit.contracts.RetrievalResult` and
  :class:`~groundkit.contracts.SearchResponse`. groundkit's
  ``RetrievalResult`` additionally carries ``source``, ``source_class``,
  ``extractor``, ``start_offset``, and ``end_offset`` — all citation-bearing
  fields that :meth:`TokenBudgetAssembler.assemble` carries through
  unchanged when it re-wraps a result's content with framing, so the framed
  copy's ``.citation`` property still resolves to the original, unframed
  span.
- *``frame_content`` signature.* ARP's version accepted a generic
  ``metadata: dict[str, object] | None`` parameter used for exactly one
  purpose: looking up an optional ``"source"`` key for the provenance block.
  groundkit's contract makes ``source`` a first-class, always-present field
  on ``RetrievalResult`` rather than free-form metadata, so that parameter
  is replaced here with a direct ``source: str | None`` parameter. The
  condition for whether a ``source:`` line is emitted is unchanged (present
  and non-blank after stripping) — only *where the value is read from*
  changed. One consequence worth naming: because groundkit's ``source`` is
  mandatory on every real ``RetrievalResult``, a result framed via
  :meth:`TokenBudgetAssembler.assemble` will now *always* carry a
  ``source:`` provenance line, whereas ARP's framed results carried one only
  when the caller happened to populate ``metadata["source"]``. This is not a
  weakening or strengthening of the sanitization itself — the same
  :func:`sanitize_provenance_value` scrubbing applies either way — only a
  change in how often the line appears, driven by the stricter contract.
- *No exception ancestry to sever.* The ARP module raised no exceptions of
  its own — nothing here is a fail-closed boundary in the sense
  :mod:`groundkit.errors` models. Token-budget exhaustion (a result being
  dropped because it would not fit) is normal, expected control flow, not an
  error condition.

**Sanitization honesty note.** ``sanitize_content`` and ``frame_content``
provide a *structural* defense, not a *semantic* one: they prevent retrieved
text from forging the ``<retrieved_context>`` delimiter tags (any literal
occurrence is replaced with a blocked-tag placeholder), strip control
characters, and quote-prefix every line so instruction-like phrasing is
visually and structurally marked as quoted data. They do **not** rewrite,
filter, or otherwise neutralize the semantic content of what a document
says — a chunk containing "Ignore all previous instructions" still contains
that exact phrase, quote-prefixed, inside the envelope. Labeling text as
untrusted is a mitigation that raises the bar for a naive attack (raw
delimiter/control-character smuggling), not a guarantee that a downstream
model will always treat framed content as inert. Where a stated requirement
is *no prompt-injection text ever surfaces as an instruction*, this module's
delimiter-and-quoting approach is a step toward that requirement, not a
proof of it. This gap exists in the ARP source unchanged; closing it further
(e.g. detecting and specially marking instruction-shaped phrases, or
enforcing the framing at the model-serving layer rather than only in the
prompt text) is out of scope for this near-verbatim port and is flagged here
as a follow-up rather than silently addressed.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

from groundkit.contracts import RetrievalResult, SearchResponse

logger = logging.getLogger(__name__)

# Delimiter tags for prompt injection defense. The LLM system prompt should
# instruct the model to treat content within these tags as untrusted
# retrieved data, never as instructions.
CONTEXT_DELIMITER_START = "<retrieved_context>"
CONTEXT_DELIMITER_END = "</retrieved_context>"
_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


def sanitize_content(content: str) -> str:
    """Normalize retrieved content before it reaches the model.

    The sanitization step deliberately avoids semantic rewriting. It removes
    control characters, neutralizes attempts to smuggle retrieval
    delimiters, and prefixes each line so instruction-like text is preserved
    as quoted data.

    Args:
        content: Raw retrieved chunk content.

    Returns:
        The sanitized, line-quoted content.
    """
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _CONTROL_CHAR_PATTERN.sub(" ", normalized)
    normalized = normalized.replace(
        CONTEXT_DELIMITER_START,
        "[blocked-retrieved-context-start]",
    )
    normalized = normalized.replace(
        CONTEXT_DELIMITER_END,
        "[blocked-retrieved-context-end]",
    )

    quoted_lines = [f"| {line}" if line else "|" for line in normalized.split("\n")]
    return "\n".join(quoted_lines)


def sanitize_provenance_value(value: object | None) -> str:
    """Normalize provenance metadata so it cannot forge wrapper structure.

    Args:
        value: The provenance value to normalize; may be any object (its
            ``str()`` form is used) or ``None``.

    Returns:
        A single-line, delimiter-safe string; ``"unknown"`` if ``value`` was
        ``None`` or normalized to nothing at all.
    """
    normalized = "" if value is None else str(value)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _CONTROL_CHAR_PATTERN.sub(" ", normalized)
    normalized = normalized.replace(
        CONTEXT_DELIMITER_START,
        "[blocked-retrieved-context-start]",
    )
    normalized = normalized.replace(
        CONTEXT_DELIMITER_END,
        "[blocked-retrieved-context-end]",
    )
    return " ".join(part for part in normalized.split("\n") if part).strip() or "unknown"


def frame_content(
    content: str,
    *,
    document_id: str | None = None,
    chunk_id: str | None = None,
    score: float | None = None,
    source: str | None = None,
) -> str:
    """Wrap retrieved content in a provenance-aware untrusted-data envelope.

    Args:
        content: Raw retrieved chunk content.
        document_id: Source document identifier, when available.
        chunk_id: Source chunk identifier, when available.
        score: Retrieval relevance score, when available.
        source: The document's source identifier (path/URL), when available.
            Adapted from ARP's ``metadata: dict[str, object] | None``
            parameter, which served only to look up an optional
            ``"source"`` key — see this module's docstring.

    Returns:
        Content wrapped in ``<retrieved_context>`` delimiters.
    """
    safe_content = sanitize_content(content)
    provenance_lines = [
        "trust_level: untrusted_retrieved_data",
        f"document_id: {sanitize_provenance_value(document_id)}",
        f"chunk_id: {sanitize_provenance_value(chunk_id)}",
    ]
    if score is not None:
        provenance_lines.append(f"retrieval_score: {score:.4f}")

    if source is not None and source.strip():
        provenance_lines.append(f"source: {sanitize_provenance_value(source)}")

    provenance_block = "\n".join(provenance_lines)
    return (
        f"{CONTEXT_DELIMITER_START}\n"
        "[retrieval_provenance]\n"
        f"{provenance_block}\n"
        "[/retrieval_provenance]\n"
        "[retrieved_data]\n"
        f"{safe_content}\n"
        "[/retrieved_data]\n"
        f"{CONTEXT_DELIMITER_END}"
    )


def _default_token_estimator(text: str) -> int:
    """Estimate token count as ``len(text) // 4``.

    A reasonable approximation for English text across most tokenizers when
    no real tokenizer is wired up.

    Args:
        text: The text to estimate tokens for.

    Returns:
        Estimated token count.
    """
    return len(text) // 4


class TokenBudgetAssembler:
    """Assemble retrieval results into a SearchResponse within a token budget.

    Greedily adds results in descending score order until the token budget
    is exhausted. Never exceeds ``max_tokens``.

    When ``frame_results`` is enabled (default), each result's content is
    wrapped in ``<retrieved_context>`` delimiters for prompt injection
    defense (see this module's docstring for what that framing does and does
    not guarantee). The token budget accounts for the framing overhead.

    Args:
        max_tokens: Maximum token budget for assembled context.
        token_estimator: Callable that estimates tokens for a given text.
            Defaults to ``len(text) // 4``.
        frame_results: Whether to wrap results in injection-defense
            delimiters. Defaults to ``True``.
    """

    def __init__(
        self,
        *,
        max_tokens: int = 4000,
        token_estimator: Callable[[str], int] | None = None,
        frame_results: bool = True,
    ) -> None:
        self._max_tokens = max_tokens
        self._estimate_tokens = token_estimator or _default_token_estimator
        self._frame_results = frame_results

    def assemble(
        self,
        results: list[RetrievalResult],
        *,
        query: str | None = None,
    ) -> SearchResponse:
        """Assemble retrieval results into a SearchResponse within token budget.

        Results are sorted by score (descending) and greedily added until
        the budget is exhausted.

        Args:
            results: Retrieval results to assemble.
            query: Original query string (included in the response).

        Returns:
            A :class:`~groundkit.contracts.SearchResponse` with results that
            fit within the budget.
        """
        sorted_results = sorted(results, key=lambda r: r.score, reverse=True)

        assembled: list[RetrievalResult] = []
        tokens_used = 0

        for result in sorted_results:
            framed_content = (
                frame_content(
                    result.content,
                    document_id=result.document_id,
                    chunk_id=result.chunk_id,
                    score=result.score,
                    source=result.source,
                )
                if self._frame_results
                else result.content
            )
            result_tokens = self._estimate_tokens(framed_content)
            if tokens_used + result_tokens > self._max_tokens:
                logger.debug(
                    "Token budget exhausted at %d/%d tokens, dropping remaining %d results",
                    tokens_used,
                    self._max_tokens,
                    len(sorted_results) - len(assembled),
                )
                break

            if self._frame_results:
                framed = RetrievalResult(
                    content=framed_content,
                    score=result.score,
                    document_id=result.document_id,
                    chunk_id=result.chunk_id,
                    source=result.source,
                    source_class=result.source_class,
                    extractor=result.extractor,
                    start_offset=result.start_offset,
                    end_offset=result.end_offset,
                    metadata=dict(result.metadata),
                )
                assembled.append(framed)
            else:
                assembled.append(result)
            tokens_used += result_tokens

        return SearchResponse(
            query=query or "",
            results=assembled,
            total_results=len(assembled),
            metadata={
                "max_tokens": self._max_tokens,
                "tokens_used": tokens_used,
                "results_considered": len(results),
                "results_included": len(assembled),
                "framing_enabled": self._frame_results,
                "sanitization_enabled": self._frame_results,
            },
        )

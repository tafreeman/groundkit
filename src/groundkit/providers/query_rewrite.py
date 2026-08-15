"""Optional, skippable query-rewrite feature (Phase 5 LLM boundary, SPEC.md §2).

SPEC.md §2 confines LLM use to two named boundaries, both optional and both
skippable: query rewrite (this module) and synthesis. Nothing in the
deterministic retrieval core constructs or calls :class:`QueryRewriter`
unless a caller explicitly opts in — retrieval remains fully usable with
this feature absent entirely.

Enabling query rewrite changes what is actually searched for: once a caller
substitutes :meth:`QueryRewriter.rewrite`'s output for the literal query text
it received, a downstream ``Retriever.search()`` call runs against different
input than "rewrite off" would have used. That makes any comparison between
a rewrite-on and a rewrite-off run a comparison across a *different query
set*, not a controlled delta over the same one. Accounting for that —
whether to disclose it, gate on it, or report it separately — is the eval
harness's concern (SPEC.md §8's baseline-comparability discipline), not this
module's; this module only performs the rewrite.

Fail-closed throughout, mirroring ``providers/embeddings.py``'s no-fallback
discipline: an empty input, a provider failure, or an unusable completion is
a typed error, never a silent substitution of the original query. A
rewriter that quietly falls back to its input on failure is indistinguishable
from one that is genuinely rewriting, which would make any measured
"rewrite on" quality delta really belong to the un-rewritten path (see
:class:`~groundkit.errors.QueryRewriteError`).
"""

from __future__ import annotations

from typing import Final

from groundkit.errors import QueryRewriteError
from groundkit.providers.protocols import ChatProtocol

#: Default prompt template sent verbatim as the ``prompt`` argument to
#: ``ChatProtocol.complete`` (never split into a separate ``system``
#: message — a custom ``prompt_template`` can still do that itself by
#: embedding its own preamble, but this default keeps the seam to one
#: string). Must contain the literal placeholder ``{query}``; it is filled
#: in via ``str.format(query=query)``, which never touches instruction text
#: — a ``{`` or ``}`` character inside the *query itself* is inert once it is
#: the argument, not part of the format string.
DEFAULT_REWRITE_PROMPT: Final[str] = (
    "You rewrite search queries for a hybrid retrieval system that combines "
    "lexical (BM25 keyword) scoring with dense semantic embedding similarity. "
    "Reformulate the query below so it matches well under BOTH: keep terms a "
    "keyword search would need, and add synonyms or related phrasing a "
    "semantic search would need, without changing the user's underlying "
    "intent or adding new claims.\n"
    "\n"
    "Respond with ONLY the rewritten query, as a single line of plain text. "
    "Do not add a preamble, an explanation, quotation marks, or a label such "
    'as "Rewritten query:".\n'
    "\n"
    "Query: {query}"
)

#: Characters :func:`_strip_surrounding_quotes` treats as a matching
#: enclosing pair. Straight quotes only, deliberately: a model asked for
#: "no quotation marks" that wraps its answer in one straight-quote pair
#: anyway is common and safe to normalize away; guessing at curly/smart
#: quote variants or partial pairs is not attempted (see that function's
#: docstring).
_QUOTE_CHARS: Final[tuple[str, ...]] = ('"', "'")


def _strip_surrounding_quotes(text: str) -> str:
    """Strip one matching layer of leading/trailing quote characters from *text*.

    Only a single layer is removed, and only when the first and last
    characters are the same character drawn from :data:`_QUOTE_CHARS`.
    Mismatched quotes (``"like this'``) or a lone quote character are left
    untouched rather than guessed at — this is a normalization of an
    otherwise-clean single-line answer, not a parser for arbitrary quoting.

    Args:
        text: Candidate rewritten-query text, already reduced to one line.

    Returns:
        *text* with one enclosing quote pair removed and the interior
        re-stripped of whitespace, or *text* unchanged if no matching pair
        is found.
    """
    if len(text) >= 2 and text[0] == text[-1] and text[0] in _QUOTE_CHARS:
        return text[1:-1].strip()
    return text


def _reduce_completion(completion: str) -> str:
    """Reduce a raw chat completion to rewritten-query text, or ``""`` if blank.

    A completion spanning more than one non-blank line is rejected rather
    than resolved by picking a line: :data:`DEFAULT_REWRITE_PROMPT`
    explicitly asks for a single line, so multiple non-blank lines mean the
    model violated that contract, and there is no principled way to tell
    "the real line" from "leftover preamble" without guessing — this repo's
    convention across every provider boundary is rejection over coercion for
    malformed output (SPEC.md §2; see ``providers/embeddings.py``'s
    ``_coerce_vector`` for the same choice made at a different seam). Purely
    blank leading/trailing lines (a trailing newline, for instance) are not
    a violation and are dropped silently before this check runs.

    Args:
        completion: The raw string returned by ``ChatProtocol.complete``.

    Returns:
        The single reduced line, with surrounding whitespace and one layer
        of surrounding quotes stripped, or ``""`` if *completion* held no
        non-blank content.

    Raises:
        QueryRewriteError: If *completion* contains more than one non-blank
            line.
    """
    stripped = completion.strip()
    if not stripped:
        return ""

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if len(lines) > 1:
        raise QueryRewriteError(
            f"query rewrite completion had {len(lines)} non-blank lines; expected a single "
            "line and refusing to guess which one is the intended rewrite"
        )

    return _strip_surrounding_quotes(lines[0])


class QueryRewriter:
    """Rewrites a retrieval query via an injected chat provider.

    Optional and skippable (SPEC.md §2): nothing constructs or calls this
    class unless a caller opts in, and retrieval works identically without
    it. Fails closed at every stage rather than ever falling back to the
    original query — see the module docstring for why a silent fallback
    would corrupt any measured rewrite-quality delta.

    Args:
        chat: The chat provider used to produce the rewrite. Any object
            structurally satisfying
            :class:`~groundkit.providers.protocols.ChatProtocol`.
        prompt_template: A ``str.format``-style template containing the
            literal placeholder ``{query}``, sent whole as the ``prompt``
            argument to ``chat.complete`` (``system`` is left at its
            default). Defaults to :data:`DEFAULT_REWRITE_PROMPT`.
    """

    def __init__(
        self, chat: ChatProtocol, *, prompt_template: str = DEFAULT_REWRITE_PROMPT
    ) -> None:
        self._chat = chat
        self._prompt_template = prompt_template

    async def rewrite(self, query: str) -> str:
        """Rewrite *query* for combined lexical and semantic retrieval.

        Args:
            query: The original retrieval query. Must contain non-whitespace
                content; checked before any provider call is made.

        Returns:
            The rewritten query, stripped of surrounding whitespace and one
            layer of surrounding quote characters.

        Raises:
            QueryRewriteError: If *query* is empty or whitespace-only
                (raised before any provider call), if the completion is
                empty or whitespace-only, or if the completion spans more
                than one non-blank line (:func:`_reduce_completion`).
                Messages carry lengths and counts only, never the query or
                completion text itself.
            ChatError: Propagated unmodified from the underlying provider —
                never swallowed, never turned into a fallback to *query*.
            ChatProviderNotConfiguredError: Propagated unmodified from the
                underlying provider (a ``ChatError`` subclass; covered by
                the same guarantee as above).
        """
        if not query.strip():
            raise QueryRewriteError(
                f"query rewrite input was empty or whitespace-only ({len(query)} chars)"
            )

        prompt = self._prompt_template.format(query=query)
        completion = await self._chat.complete(prompt)

        rewritten = _reduce_completion(completion)
        if not rewritten:
            raise QueryRewriteError(
                f"query rewrite completion was empty or whitespace-only "
                f"({len(completion)} chars received)"
            )
        return rewritten

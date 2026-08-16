"""Anonymization at the LLM boundary (Phase 5, SPEC.md §2): a redaction pass
that runs before any text leaves the process for a cloud provider — names to
tokens, with configurable patterns. Local mode sends nothing anywhere, so this
module exists for the boundary that does egress; the egress hook itself
(query rewrite, synthesis) is designed elsewhere, and this module is
deliberately boundary-agnostic — pure, deterministic, no I/O.

**No built-in person-name pattern, by design.** Free-text name detection by
regex is unreliable (false positives on ordinary capitalized words, false
negatives on names it wasn't tuned for), and shipping one under the label
"redacts names" would be a false promise SPEC.md §2's honesty rule
("no number in any doc that wasn't generated", generalized to "no claim of
a capability this repo cannot back") does not let this module make quietly.
:data:`DEFAULT_PATTERNS` covers only deterministic, structurally-recognizable
values: email addresses, phone numbers, IPv4 addresses, and long opaque
secret-like tokens. Redacting person names is a configuration a caller
supplies via their own :class:`RedactionPattern` entries, not a default this
module ships.

Fail-closed points, each load-bearing rather than defensive:

- A pattern whose regex does not compile, or that can match the empty
  string, is rejected at :class:`RedactionPattern` construction — never at
  first use. A zero-width pattern would leave a scan re-emitting the same
  position, and a construction-time check is the only place that can catch
  it before any real text is ever processed.
- Two patterns whose names collide once uppercased are rejected at
  :class:`RedactionConfig` construction: the token format uses the
  uppercased name as its category, so a same-cased-when-uppercased pair
  (``"Email"`` and ``"EMAIL"``) would otherwise generate indistinguishable
  tokens for two different pattern definitions.
- :meth:`Redactor.restore` raises :class:`UnknownRedactionTokenError` for a
  bracketed substring shaped like a token *of a category this instance
  knows*, whose specific counter this instance never issued — instead of
  silently leaving it as a dangling placeholder. A bracketed substring whose
  category this instance does not recognize at all is left untouched: there
  is no basis for treating an unrecognized shape as a misuse signal.

Never log or embed matched document text in an exception message: every
error this module raises carries pattern names, category names, or counters
— never the value a pattern matched.
"""

from __future__ import annotations

import re
from typing import Final, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Explicit re-export: these lived here before their promotion into the
# central hierarchy, and the tests (and any external caller) legitimately
# import them from the module whose behavior raises them.
from groundkit.errors import RedactionError as RedactionError
from groundkit.errors import UnknownRedactionTokenError as UnknownRedactionTokenError

#: Character class for a :class:`RedactionPattern` name: a letter followed by
#: letters, digits, or underscores. "Uppercase-able" means exactly this —
#: every character in the class is meaningful after ``str.upper()``, so the
#: derived token category (``name.upper()``) never collides with characters
#: the name itself didn't have.
_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

#: Recognizes a token this module could have emitted: ``[CATEGORY_n]``. Used
#: by :meth:`Redactor.restore` to find candidate tokens to invert. The
#: category group is greedy, so for a category name that itself ends in
#: digits before an underscore (e.g. ``PHONE_E164``), backtracking still
#: resolves the split correctly: the engine only accepts the final
#: ``_(\d+)]`` at the rightmost position where the rest of the string still
#: matches, which is exactly the counter suffix this module appends.
_TOKEN_SHAPE_PATTERN: Final[re.Pattern[str]] = re.compile(r"\[([A-Za-z][A-Za-z0-9_]*)_(\d+)\]")


class RedactionPattern(BaseModel):
    """One configured redaction rule: a name and the regex it matches.

    Attributes:
        name: Identifier for this pattern's token category. Must match
            ``^[A-Za-z][A-Za-z0-9_]*$`` — a letter followed by letters,
            digits, or underscores. Uppercased to form the token category
            (e.g. ``name="email"`` produces tokens ``[EMAIL_1]``,
            ``[EMAIL_2]``, ...).
        regex: A pattern compiled with :func:`re.compile`. Must compile, and
            must not be able to match the empty string — both checked here,
            at construction, never at first use against real text.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    regex: str

    @model_validator(mode="after")
    def _validate(self) -> RedactionPattern:
        if not _NAME_PATTERN.fullmatch(self.name):
            raise ValueError(
                f"redaction pattern name {self.name!r} must match "
                f"{_NAME_PATTERN.pattern!r} (a letter, then letters/digits/underscores)"
            )
        try:
            compiled = re.compile(self.regex)
        except re.error as exc:
            raise ValueError(
                f"redaction pattern {self.name!r} has a regex that does not compile: {exc}"
            ) from exc
        # The only string on which a regex can match with zero width at every
        # possible position is the empty string itself — so "can this regex
        # match empty" and "does it match against ''" are the same question.
        if compiled.search("") is not None:
            raise ValueError(
                f"redaction pattern {self.name!r} can match an empty string; a "
                "zero-width match would loop when scanned, so it is rejected "
                "at construction rather than at first use"
            )
        return self

    @property
    def token_category(self) -> str:
        """The uppercased category this pattern's tokens are named under."""
        return self.name.upper()


#: Deterministic, structurally-recognizable patterns only — no free-text name
#: detection (see module docstring). Names are already uppercase so each
#: pattern's declared name and its token category are identical.
DEFAULT_PATTERNS: Final[tuple[RedactionPattern, ...]] = (
    RedactionPattern(
        name="EMAIL",
        regex=r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}",
    ),
    RedactionPattern(
        # E.164-ish: a leading '+', a nonzero first digit, then 7-14 more
        # digits (8-15 digits total, the E.164 bound).
        name="PHONE_E164",
        regex=r"\+[1-9]\d{7,14}",
    ),
    RedactionPattern(
        # US-formatted: area code (bare or parenthesized) plus two more
        # digit groups, each pair joined by a space/dot/dash separator.
        # Bare 10-digit runs with no separator are deliberately not matched
        # here — that shape collides with too much else (e.g. a 32+ digit
        # secret token stops well short of this pattern, but a bare 10-digit
        # ID would false-positive with no separator requirement at all).
        name="PHONE_US",
        regex=r"(?<!\d)(?:\(\d{3}\)[ .-]?|\d{3}[ .-])\d{3}[ .-]\d{4}(?!\d)",
    ),
    RedactionPattern(
        name="IPV4",
        regex=r"(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)",
    ),
    RedactionPattern(
        # 32+ character run drawn from the base64 alphabet, with optional
        # '=' padding. A pure-hex run (0-9a-f) is a subset of this alphabet,
        # so this single pattern covers both cases named in the design brief
        # without a second, redundant hex-only pattern.
        name="SECRET_TOKEN",
        regex=r"[A-Za-z0-9+/]{32,}={0,2}",
    ),
)


class RedactionConfig(BaseModel):
    """A set of redaction patterns to apply, keyed by distinct token category.

    Attributes:
        patterns: The configured patterns, applied together by one
            :class:`Redactor`. Defaults to :data:`DEFAULT_PATTERNS`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    patterns: tuple[RedactionPattern, ...] = Field(default_factory=lambda: DEFAULT_PATTERNS)

    @model_validator(mode="after")
    def _validate_unique_categories(self) -> RedactionConfig:
        seen: set[str] = set()
        for pattern in self.patterns:
            category = pattern.token_category
            if category in seen:
                raise ValueError(
                    f"duplicate redaction pattern name {pattern.name!r}: its token "
                    f"category {category!r} collides with another pattern's, "
                    "case-insensitively"
                )
            seen.add(category)
        return self


class RedactionResult(BaseModel):
    """The output of one :meth:`Redactor.redact` call.

    Attributes:
        text: ``text`` with every matched span replaced by its token.
        mapping: ``(token, original)`` pairs for the tokens that appear in
            ``text`` from *this* call — not the :class:`Redactor` instance's
            full accumulated history (which ``restore()`` uses internally
            and which callers do not need to interpret one result). A
            ``tuple`` of pairs rather than a ``dict``: this model is frozen,
            and a ``dict`` value stays mutable and unhashable underneath a
            frozen wrapper, which would let a caller mutate shared state
            through it and would make the model itself unhashable. A tuple
            of ``(str, str)`` pairs is both immutable and hashable, matching
            every other frozen model in this repo, sorted here for
            deterministic equality between two results built from the same
            input.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    mapping: tuple[tuple[str, str], ...] = Field(default_factory=tuple)


class _Span(NamedTuple):
    """One pattern match found in a text, before overlap resolution."""

    start: int
    end: int
    category: str
    value: str


class Redactor:
    """Applies a :class:`RedactionConfig` to text, tokenizing matches and
    reversing the tokenization later.

    Not a frozen Pydantic model, unlike everything else in this module:
    a ``Redactor`` is inherently stateful. It accumulates a token/original
    mapping across every ``redact()`` call made on one instance, both so
    the same matched value always gets the same token within that instance
    (stable across multiple calls, e.g. across turns of one conversation)
    and so :meth:`restore` can invert tokens from any prior call on this
    instance, not just the most recent one.

    Args:
        config: The patterns to apply.
    """

    def __init__(self, config: RedactionConfig) -> None:
        self._config = config
        self._compiled: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
            (pattern.token_category, re.compile(pattern.regex)) for pattern in config.patterns
        )
        self._known_categories: frozenset[str] = frozenset(
            category for category, _ in self._compiled
        )
        self._counters: dict[str, int] = {}
        self._token_to_original: dict[str, str] = {}
        self._original_to_token: dict[tuple[str, str], str] = {}

    def redact(self, text: str) -> RedactionResult:
        """Replace every matched span in ``text`` with a stable token.

        Overlap tie-break — **leftmost-longest, deterministic**: every
        configured pattern's matches are gathered into one list, then
        sorted by ``(start ascending, length descending)`` and walked
        greedily, keeping a match only if it starts at or after the end of
        the last kept match. The effect: the earliest-starting match always
        wins; when two matches start at the same position, the longer one
        wins; when two matches share both start and length (two different
        patterns matching the identical span), the pattern declared earlier
        in ``config.patterns`` wins, because the sort is stable and matches
        are gathered in pattern-declaration order before it runs.

        **Token-format collision, by design, not closed here:** token
        numbering is purely a function of match order and is never checked
        against what the input text already contains. If ``text`` already
        contains a literal substring shaped exactly like a token this call
        will also produce — e.g. the literal text ``"[EMAIL_1]"`` is
        already present, and the first real email match also becomes
        ``EMAIL_1`` — the two become byte-identical in the output, and
        :meth:`restore` (which matches on token shape and string value, not
        provenance) replaces both occurrences with the real email,
        corrupting what was previously inert literal text. This is a known,
        narrow hazard of a human-readable bracket token format; closing it
        would need escaping pre-existing bracket-shaped text or a
        non-guessable token alphabet, either of which is a bigger design
        decision than this pass makes.

        Args:
            text: The text to redact. Not logged or embedded in any
                exception this method raises.

        Returns:
            The redacted text, plus the ``(token, original)`` pairs
            introduced or reused in producing it.
        """
        spans: list[_Span] = []
        for category, compiled in self._compiled:
            for match in compiled.finditer(text):
                spans.append(_Span(match.start(), match.end(), category, match.group()))

        selected = self._select_leftmost_longest(spans)

        pieces: list[str] = []
        cursor = 0
        call_mapping: dict[str, str] = {}
        for span in selected:
            pieces.append(text[cursor : span.start])
            token = self._token_for(span.category, span.value)
            pieces.append(token)
            call_mapping[token] = span.value
            cursor = span.end
        pieces.append(text[cursor:])

        return RedactionResult(text="".join(pieces), mapping=tuple(sorted(call_mapping.items())))

    def restore(self, text: str) -> str:
        """Replace every token this instance recognizes with its original value.

        A bracketed substring shaped like ``[CATEGORY_n]``:

        - whose ``CATEGORY`` is not one this instance's config declares is
          left untouched — there is no basis for treating an unrecognized
          shape as a token at all;
        - whose ``CATEGORY`` is recognized but whose specific ``n`` this
          instance never issued raises :class:`UnknownRedactionTokenError`
          (fail closed);
        - whose ``CATEGORY`` and ``n`` this instance did issue is replaced
          with the original value it was produced from.

        Args:
            text: Text potentially containing tokens this instance (or an
                unrelated one) produced.

        Returns:
            ``text`` with every token this instance issued replaced by its
            original value.

        Raises:
            UnknownRedactionTokenError: ``text`` contains a token whose
                category this instance knows but whose specific counter it
                never issued.
        """

        def _replace(match: re.Match[str]) -> str:
            category, counter = match.group(1), match.group(2)
            if category not in self._known_categories:
                return match.group(0)
            original = self._token_to_original.get(match.group(0))
            if original is None:
                raise UnknownRedactionTokenError(
                    f"restore() saw a token in category {category!r} (a category "
                    f"this Redactor knows) with counter {counter!r}, which this "
                    "instance never issued"
                )
            return original

        return _TOKEN_SHAPE_PATTERN.sub(_replace, text)

    @staticmethod
    def _select_leftmost_longest(spans: list[_Span]) -> list[_Span]:
        """Resolve overlapping spans: leftmost start wins; longest breaks a tie."""
        ordered = sorted(spans, key=lambda span: (span.start, -(span.end - span.start)))
        selected: list[_Span] = []
        next_allowed_start = 0
        for span in ordered:
            if span.start >= next_allowed_start:
                selected.append(span)
                next_allowed_start = span.end
        return selected

    def _token_for(self, category: str, value: str) -> str:
        """Return the stable token for ``value`` in ``category``, allocating
        a fresh one on first sight."""
        key = (category, value)
        existing = self._original_to_token.get(key)
        if existing is not None:
            return existing

        index = self._counters.get(category, 0) + 1
        self._counters[category] = index
        token = f"[{category}_{index}]"
        self._original_to_token[key] = token
        self._token_to_original[token] = value
        return token

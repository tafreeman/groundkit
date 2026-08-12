"""Golden eval corpus: schema, loading, and quote-to-span resolution (Phase 2).

``chunk_id``/``document_id`` are ``uuid.uuid4().hex`` (``contracts.py:38,64``),
regenerated on every ingest, so a committed judgment cannot reference them
directly, and chunk offsets move whenever chunking config changes. Instead a
:class:`Judgment` names a corpus-relative document path plus a distinctive
verbatim quote (:class:`GoldSpan`); :func:`resolve_gold_span` resolves that
quote to a character span in the document's current text, and the relevant
chunk set is every chunk whose ``[start_offset, end_offset)`` overlaps that
span (:func:`chunk_overlaps_span`).

The authoring contract for corpus documents and judgments lives in
``evals/README.md``, derived from the models here so the two cannot drift.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from groundkit.errors import EvalError
from groundkit.utils.path_safety import ensure_within_base

#: The four judgment categories the golden corpus must cover (SPEC.md §6).
Category = Literal["normal", "ambiguous", "no_answer", "adversarial"]

#: Corpus-integrity size floors — the *literal* numbers are re-asserted in
#: ``tests/test_corpus_integrity.py`` (SPEC.md §6: the test is the
#: authoritative number, not this constant or the README); this constant only
#: guards against the README/tests drifting *below* what is asserted.
MIN_CORPUS_DOCS: int = 8
MIN_CORPUS_JUDGMENTS: int = 40

#: Ceiling on :func:`query_gold_token_overlap` for answerable judgments — a
#: query that shares too many tokens with its own gold quote lets lexical
#: overlap (not genuine relevance) win recall@1, silently inflating the
#: BM25-only baseline. See ``evals/README.md``'s paraphrase rule.
MAX_QUERY_GOLD_TOKEN_OVERLAP: float = 0.7

#: Lowercase, distinctive prompt-injection phrases an ``adversarial``
#: document's distractor text must contain (case-insensitively matched).
INJECTION_MARKERS: tuple[str, ...] = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "disregard the above",
    "you are now in developer mode",
    "new instructions:",
    "system prompt:",
)

#: Rejects a doc path containing a Windows drive letter (``C:``, ``d:``), the
#: other half of "absolute path" alongside a leading ``/``.
_DRIVE_LETTER_PATTERN: re.Pattern[str] = re.compile(r"^[A-Za-z]:")

#: Kebab-case query IDs: lowercase alphanumerics, hyphen-separated segments.
_QUERY_ID_PATTERN: str = r"^[a-z0-9]+(-[a-z0-9]+)*$"


class GoldSpan(BaseModel):
    """One authored gold answer: a corpus-relative document path plus a
    verbatim quote to resolve within it.

    Attributes:
        doc: Path to a document under ``evals/corpus/``, relative and
            forward-slashed (e.g. ``"guides/setup.md"``). Validated at the
            string level only — filesystem containment is a second, separate
            layer applied at read time by :func:`read_corpus_doc` via
            :func:`~groundkit.utils.path_safety.ensure_within_base`, mirroring
            how :class:`~groundkit.index.metadata.SQLiteMetadataStore`
            validates a collection name before applying its own path
            barrier.
        quote: A short, distinctive substring of the document's text. Must
            appear exactly once in the document (:func:`resolve_gold_span`
            fails closed on zero or multiple matches).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    doc: str
    quote: str = Field(min_length=1)

    @field_validator("doc")
    @classmethod
    def _validate_doc(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("doc must not be empty or whitespace-only")
        if "\\" in value:
            raise ValueError(f"doc must use forward slashes, not backslashes: {value!r}")
        if value.startswith("/") or _DRIVE_LETTER_PATTERN.match(value):
            raise ValueError(f"doc must be a relative path, not absolute: {value!r}")
        if ".." in value.split("/"):
            raise ValueError(f"doc must not contain '..' path segments: {value!r}")
        return value


class Judgment(BaseModel):
    """One labeled query -> gold-answer judgment.

    Attributes:
        query_id: Unique, kebab-case identifier (e.g. ``"setup-timeout"``).
            The corpus file is kept sorted by this field so diffs stay
            minimal; :func:`load_judgments` enforces both uniqueness and
            ascending order.
        query: The natural-language query text.
        category: One of :data:`Category`.
        gold: Authored gold answers. Empty iff ``category == "no_answer"``;
            ``ambiguous`` requires at least 2 (distinct authored answers — a
            single quote that happens to resolve to multiple chunks by
            boundary overlap does not qualify).
        notes: Optional free-text authoring notes.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    query_id: str = Field(pattern=_QUERY_ID_PATTERN)
    query: str = Field(min_length=1)
    category: Category
    gold: list[GoldSpan] = Field(default_factory=list)
    notes: str | None = None

    @model_validator(mode="after")
    def _validate_gold_invariants(self) -> Judgment:
        if self.category == "no_answer":
            if self.gold:
                raise ValueError("category 'no_answer' judgments must have an empty gold list")
            return self

        if not self.gold:
            raise ValueError(f"category {self.category!r} judgments require at least one gold span")

        if self.category == "ambiguous" and len(self.gold) < 2:
            raise ValueError(
                "category 'ambiguous' requires at least 2 gold spans (distinct authored "
                "answers) — a single quote that happens to resolve to 2 chunks by boundary "
                "overlap does not qualify"
            )
        return self


def load_judgments(path: Path) -> list[Judgment]:
    """Parse a JSONL judgments file into validated :class:`Judgment` objects.

    One ``Judgment`` per non-blank line. The file must already be sorted by
    ``query_id`` (ascending, unique) so diffs stay minimal — this function
    enforces that ordering rather than silently re-sorting.

    Args:
        path: Path to the ``evals/judgments.jsonl`` file (or a synthetic
            fixture built the same way).

    Returns:
        Judgments in file order (== ``query_id`` order).

    Raises:
        EvalError: The file cannot be read or is not valid UTF-8; or a line
            is not valid JSON, fails :class:`Judgment` validation, repeats a
            ``query_id`` already seen, or has a ``query_id`` that does not
            sort strictly after the previous one. Line-level messages name
            the 1-based line number.
    """
    judgments: list[Judgment] = []
    seen_ids: set[str] = set()
    previous_id: str | None = None

    # OSError and UnicodeDecodeError are wrapped for the same reason the
    # loader and citation resolver wrap them: UnicodeDecodeError is a
    # ValueError, not an OSError, so catching only OSError would let the raw
    # builtin escape past every caller that handles GroundkitError.
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvalError(f"Cannot read judgments file {str(path)!r}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise EvalError(f"Judgments file {str(path)!r} is not valid UTF-8: {exc}") from exc
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvalError(f"{path}:{line_number}: invalid JSON: {exc}") from exc

        try:
            judgment = Judgment.model_validate(payload)
        except ValidationError as exc:
            raise EvalError(f"{path}:{line_number}: invalid judgment: {exc}") from exc

        if judgment.query_id in seen_ids:
            raise EvalError(f"{path}:{line_number}: duplicate query_id {judgment.query_id!r}")
        if previous_id is not None and judgment.query_id <= previous_id:
            raise EvalError(
                f"{path}:{line_number}: query_id {judgment.query_id!r} is not in ascending "
                f"order after {previous_id!r} (the file must be kept sorted by query_id)"
            )

        seen_ids.add(judgment.query_id)
        previous_id = judgment.query_id
        judgments.append(judgment)

    return judgments


def read_corpus_doc(docs_dir: Path, doc: str) -> str:
    """Read a corpus document's text, contained within ``docs_dir``.

    Uses ``encoding="utf-8"`` — this must match
    :meth:`~groundkit.ingestion.loaders.FileLoader._read_text`, which also
    reads with ``encoding="utf-8"``. A ``utf-8-sig`` mismatch on either side
    would silently shift every offset in a BOM'd document and misresolve
    every quote in it.

    Args:
        docs_dir: The corpus documents root (``evals/corpus/``).
        doc: A corpus-relative path, as validated string-level by
            :class:`GoldSpan`.

    Returns:
        The document's full text.

    Raises:
        EvalError: ``doc`` escapes ``docs_dir``, the file cannot be read
            (missing, not a file, permission error, ...), or its bytes are
            not valid UTF-8.
    """
    try:
        path = ensure_within_base(docs_dir / doc, docs_dir)
    except ValueError as exc:
        raise EvalError(str(exc)) from exc

    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvalError(f"failed to read corpus doc {doc!r}: {exc}") from exc
    except UnicodeDecodeError as exc:
        # UnicodeDecodeError is a ValueError, not an OSError, so it needs its
        # own except clause — the repo already fixed exactly this bug in
        # groundkit.ingestion.loaders.FileLoader.load.
        raise EvalError(f"corpus doc {doc!r} is not valid UTF-8: {exc}") from exc


def resolve_gold_span(document_text: str, quote: str) -> tuple[int, int]:
    """Resolve a verbatim quote to a ``[start, end)`` character span.

    Fails closed in both directions: a quote that is not found is an error,
    and so is a quote that matches more than once (ambiguous — silently
    taking the first match is exactly the coercion SPEC.md §2 bans; the
    judgment's author must add surrounding context to disambiguate instead).

    Args:
        document_text: The full text to search within.
        quote: The verbatim substring to locate.

    Returns:
        ``(start, end)`` such that ``document_text[start:end] == quote``.

    Raises:
        EvalError: ``quote`` does not appear in ``document_text``, or
            appears more than once.
    """
    first = document_text.find(quote)
    if first == -1:
        raise EvalError(f"quote not found in document: {quote!r}")

    second = document_text.find(quote, first + 1)
    if second != -1:
        raise EvalError(
            f"quote appears more than once in document (ambiguous — add context to "
            f"disambiguate): {quote!r}"
        )

    return first, first + len(quote)


def chunk_overlaps_span(chunk_start: int, chunk_end: int, span: tuple[int, int]) -> bool:
    """Return True when ``[chunk_start, chunk_end)`` overlaps ``span``.

    Half-open interval overlap: touching-but-not-overlapping ranges (e.g.
    ``chunk_end == span_start``) are False.

    Args:
        chunk_start: Chunk's ``start_offset``.
        chunk_end: Chunk's ``end_offset``.
        span: A resolved gold span, ``(start, end)``.

    Returns:
        True iff the two half-open intervals overlap.
    """
    span_start, span_end = span
    return chunk_start < span_end and span_start < chunk_end


def tokenize(text: str) -> list[str]:
    """Tokenize text by lowercasing and extracting word characters.

    Deliberately duplicates ``groundkit.index.bm25.BM25Index``'s private
    ``_tokenize`` (``re.findall(r"\\w+", text.lower())``) rather than
    importing it: :func:`query_gold_token_overlap`'s circularity guard is
    only meaningful when measured with the exact tokenizer BM25 scores
    with, so this must track that function's behavior, not merely import
    something with the same name.

    Args:
        text: The input text to tokenize.

    Returns:
        Lowercase word tokens with punctuation removed.
    """
    return re.findall(r"\w+", text.lower())


def query_gold_token_overlap(query: str, quote: str) -> float:
    """Fraction of the query's distinct tokens that also appear in the quote.

    Args:
        query: The judgment's query text.
        quote: The gold span's verbatim quote text.

    Returns:
        ``len(query_tokens & quote_tokens) / len(query_tokens)``, or ``0.0``
        for an empty (or all-punctuation) query.
    """
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    quote_tokens = set(tokenize(quote))
    return len(query_tokens & quote_tokens) / len(query_tokens)

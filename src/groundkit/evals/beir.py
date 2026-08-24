"""Adapt BEIR datasets to groundkit's quote-anchored evaluation format.

Placed in :mod:`groundkit.evals` rather than a top-level ``datasets`` package
because it imports :mod:`groundkit.evals.corpus` to reuse the harness's own
judgment loader. GK-021 (``tests/test_deterministic_core.py``) bars every
module *outside* ``groundkit.evals`` from importing the harness, so that the
harness stays droppable from a plain library install; an adapter that exists
only to feed the harness belongs inside that boundary rather than being an
exception to it.

Validation order is the load-bearing property here. Every identifier is
checked -- against *both* this module's path-safety class and the harness's
own stricter contract -- before a single byte is written, so a dataset the
harness would later reject leaves nothing behind on disk. An earlier shape of
this module wrote the corpus first and re-validated afterwards, which turned
a rejected dataset into a half-populated output directory.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from groundkit.errors import EvalError
from groundkit.evals.corpus import _QUERY_ID_PATTERN, load_judgments
from groundkit.utils.path_safety import ensure_within_base

#: Path-safety class for a BEIR identifier used as a file name. Deliberately
#: narrower than "any string": the first character must be alphanumeric, which
#: structurally excludes ``..``, a leading separator, a drive prefix and a UNC
#: prefix without enumerating them.
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

#: The harness's own query-id contract, compiled once. ``load_judgments``
#: enforces this via ``Judgment.query_id``'s ``Field(pattern=...)``; checking
#: it here as well is what makes the adapter fail *before* writing rather
#: than after. Imported from :mod:`groundkit.evals.corpus` rather than
#: restated, so the two cannot drift.
_HARNESS_QUERY_ID = re.compile(_QUERY_ID_PATTERN)

#: BEIR splits are directory-name components interpolated into a path
#: (``qrels/<split>.tsv``), so the same character-class discipline applies.
_SAFE_SPLIT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

#: Suffix every adapted document file carries. Named once so the length bound
#: in ``_validate_identifier`` and the writers below cannot disagree about what
#: the final filename is.
_DOCUMENT_SUFFIX = ".txt"

#: Longest filename, in bytes, this adapter will produce. 255 is the component
#: limit on ext4, APFS and NTFS alike; a corpus needing more is malformed for
#: every target, not just this one. Bounded here rather than discovered at
#: ``write_text``, which would raise mid-loop after the alphabetically earlier
#: documents had already been written -- exactly the half-populated output this
#: module validates up front to avoid.
_MAX_FILENAME_BYTES = 255

#: Stems Windows reserves for devices. The reservation applies to the stem, so
#: ``CON.txt`` is reserved as surely as ``CON``.
_WINDOWS_RESERVED_STEMS = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{digit}" for digit in range(1, 10)}
    | {f"LPT{digit}" for digit in range(1, 10)}
)

_TITLE_SEPARATOR = "\n\n"


class BeirAdaptationReport(BaseModel):
    """Counts and paths produced by :func:`adapt_beir_dataset`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_dir: str
    output_dir: str
    split: str
    document_count: int = Field(ge=0)
    query_count: int = Field(ge=0)
    relevance_pair_count: int = Field(ge=0)
    total_characters: int = Field(ge=0)
    corpus_dir: str
    judgments_path: str


def adapt_beir_dataset(
    source_dir: Path, output_dir: Path, *, split: str = "test"
) -> BeirAdaptationReport:
    """Convert a BEIR directory without downloading or filtering its data.

    BEIR qrels are document-level.  Each emitted gold quote is therefore the
    complete adapted document text.  groundkit resolves the quote at offset
    zero and treats every overlapping chunk as relevant, preserving the qrel
    meaning without pretending that BEIR provides span-level annotations.

    The source directory is untrusted input -- a BEIR corpus is a third-party
    download, and a mirror or hand-off can supply any ``_id`` it likes -- so
    every identifier is validated as a path component before it reaches a
    path join, and every resulting path is containment-checked against
    ``output_dir`` as a second, independent barrier.

    Raises:
        EvalError: The source files are missing or malformed, an identifier is
            unsafe as a path component or violates the harness's own contract,
            the qrels reference an unknown document or query, the split
            contains no positive relevance rows, or the output corpus
            directory already holds files (which would silently mix two
            datasets -- remove it and re-run).
    """

    if not _SAFE_SPLIT.fullmatch(split):
        raise EvalError(
            f"unsafe BEIR split {split!r}; expected letters, digits, dot, underscore, or hyphen"
        )

    source = source_dir.resolve()
    destination = output_dir.resolve()
    corpus_records = _load_jsonl(source / "corpus.jsonl")
    query_records = _load_jsonl(source / "queries.jsonl")
    qrels = _load_qrels(source / "qrels" / f"{split}.tsv")

    texts: dict[str, str] = {}
    for position, record in enumerate(corpus_records, start=1):
        document_id = _required_string(record, "_id", context=f"corpus row {position}")
        _validate_identifier(document_id, subject="document", becomes_a_filename=True)
        if document_id in texts:
            raise EvalError(f"duplicate BEIR document id {document_id!r}")
        title = _optional_string(record, "title", context=f"corpus row {position}").strip()
        body = _optional_string(record, "text", context=f"corpus row {position}").strip()
        text = f"{title}{_TITLE_SEPARATOR}{body}" if title else body
        if not text:
            raise EvalError(f"BEIR document {document_id!r} has no title or text")
        texts[document_id] = text

    queries: dict[str, str] = {}
    for position, record in enumerate(query_records, start=1):
        query_id = _required_string(record, "_id", context=f"query row {position}")
        _validate_identifier(query_id, subject="query", becomes_a_filename=False)
        if query_id in queries:
            raise EvalError(f"duplicate BEIR query id {query_id!r}")
        query = _required_string(record, "text", context=f"query row {position}").strip()
        if not query:
            raise EvalError(f"BEIR query {query_id!r} is empty")
        queries[query_id] = query

    missing_documents = sorted(
        {doc_id for doc_ids in qrels.values() for doc_id in doc_ids} - texts.keys()
    )
    missing_queries = sorted(qrels.keys() - queries.keys())
    if missing_documents:
        raise EvalError(f"BEIR qrels reference missing documents: {missing_documents}")
    if missing_queries:
        raise EvalError(f"BEIR qrels reference missing queries: {missing_queries}")
    if not qrels:
        raise EvalError(f"BEIR qrels for split {split!r} contain no positive relevance rows")

    # The harness's query-id contract is stricter than `_SAFE_IDENTIFIER`
    # (kebab-case lowercase, so `MED-10` and `q_1` are path-safe but still
    # inadmissible). Checked here, before any write, so `load_judgments`
    # below can only fail on something this loop could not have known.
    unusable = sorted(qid for qid in qrels if not _HARNESS_QUERY_ID.fullmatch(qid))
    if unusable:
        raise EvalError(
            f"BEIR query ids are not admissible as groundkit judgment ids: {unusable}. "
            f"The harness requires kebab-case lowercase ({_QUERY_ID_PATTERN}); rename or "
            "map these ids before adapting, rather than writing a corpus the harness "
            "would reject on load."
        )

    _reject_case_colliding_document_ids(texts)

    corpus_dir = destination / "corpus"
    _require_empty_corpus_dir(corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for document_id, text in sorted(texts.items()):
        # `_validate_identifier` already makes an escape unrepresentable; the
        # containment check is the independent second barrier the rest of this
        # package applies wherever a path is built from outside input, and the
        # one static analysis recognizes as a path-injection sanitizer.
        # Contained against `destination`, not `corpus_dir`. Checking against
        # `corpus_dir` resolves the candidate *and* the base through the same
        # link, so a `corpus` symlink pointing outside the destination passes
        # trivially and every document lands wherever it points.
        # `_require_empty_corpus_dir` refuses such a link outright; this is the
        # independent second barrier, and the one that stays correct if that
        # check is ever relaxed.
        filename = f"{document_id}{_DOCUMENT_SUFFIX}"
        document_path = ensure_within_base(corpus_dir / filename, destination)
        document_path.write_text(text, encoding="utf-8", newline="\n")

    judgments_path = ensure_within_base(destination / "judgments.jsonl", destination)
    with judgments_path.open("w", encoding="utf-8", newline="\n") as handle:
        for query_id in sorted(qrels):
            row = {
                "query_id": query_id,
                "query": queries[query_id],
                "category": "normal",
                "gold": [
                    {"doc": f"{document_id}{_DOCUMENT_SUFFIX}", "quote": texts[document_id]}
                    for document_id in sorted(qrels[query_id])
                ],
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Reuse the harness's authoritative schema/path/order checks on our output.
    load_judgments(judgments_path)
    return BeirAdaptationReport(
        source_dir=str(source),
        output_dir=str(destination),
        split=split,
        document_count=len(texts),
        query_count=len(qrels),
        relevance_pair_count=sum(len(document_ids) for document_ids in qrels.values()),
        total_characters=sum(len(text) for text in texts.values()),
        corpus_dir=str(corpus_dir),
        judgments_path=str(judgments_path),
    )


def _reject_case_colliding_document_ids(texts: dict[str, str]) -> None:
    """Refuse document ids that are distinct keys but the same file name.

    ``_SAFE_IDENTIFIER`` admits upper case, so ``DOC-1`` and ``doc-1`` are two
    dictionary entries with two different texts -- and one file on Windows and
    on the default macOS filesystem. The second write silently overwrites the
    first, while the judgments still reference both names with both quotes, so
    one of them resolves against the wrong document's text.

    Refused on every platform, not only the case-insensitive ones. This is a
    benchmark adapter, and a corpus that adapts one way on Linux and another
    way on a laptop is not reproducible; a dataset whose meaning depends on
    the filesystem it landed on should fail loudly rather than differ quietly.
    Case folding catches the realistic collision class here (BEIR ids are
    ASCII); it does not attempt Unicode normalization forms.
    """
    by_folded: defaultdict[str, list[str]] = defaultdict(list)
    for document_id in texts:
        by_folded[document_id.casefold()].append(document_id)
    collisions = sorted(sorted(group) for group in by_folded.values() if len(group) > 1)
    if collisions:
        raise EvalError(
            "BEIR document ids collide when case is folded, so they would share one "
            f"file on a case-insensitive filesystem: {collisions}. Each would overwrite "
            "the other while the judgments still reference both, leaving quotes resolved "
            "against the wrong document. Rename or map these ids before adapting."
        )


def _require_empty_corpus_dir(corpus_dir: Path) -> None:
    """Refuse to write into a corpus directory that already holds files.

    Adapting a second dataset over the first silently unions them: the stale
    documents stay retrievable and are scored against judgments that never
    mentioned them. Fail closed and let the operator delete, matching
    ADR-0004 decision 5's delete-and-re-derive posture rather than guessing
    which files were meant to survive.
    """
    if corpus_dir.is_symlink():
        # An empty directory reached through a symlink passes the emptiness
        # check below while every write lands outside the destination the
        # caller named. Refused rather than followed, matching how the rest of
        # this package treats a link discovered where a real path was expected.
        raise EvalError(
            f"BEIR output corpus path {str(corpus_dir)!r} is a symbolic link; refused "
            "rather than followed, because documents written through it would land "
            "outside the output directory you named. Remove the link and re-run."
        )
    if not corpus_dir.exists():
        return
    existing = sorted(entry.name for entry in corpus_dir.iterdir())
    if existing:
        raise EvalError(
            f"BEIR output corpus directory {str(corpus_dir)!r} already contains "
            f"{len(existing)} entries (e.g. {existing[:3]}). Adapting into it would mix "
            "two datasets in one corpus and score stale documents against judgments that "
            "never referenced them. Remove the directory and re-run."
        )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EvalError(f"cannot read BEIR file {str(path)!r}: {exc}") from exc
    except UnicodeDecodeError as exc:
        # UnicodeDecodeError is a ValueError, not an OSError, so it would
        # otherwise escape this module's documented EvalError contract and
        # reach the CLI as a traceback. A third-party download being
        # mis-encoded is ordinary malformed input, not a bug.
        raise EvalError(f"BEIR file {str(path)!r} is not valid UTF-8: {exc}") from exc

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvalError(f"invalid JSON in {path} line {line_number}: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise EvalError(f"{path} line {line_number} must be a JSON object")
        records.append(value)
    return records


def _load_qrels(path: Path) -> dict[str, set[str]]:
    try:
        rows = [line.split("\t") for line in path.read_text(encoding="utf-8").splitlines() if line]
    except OSError as exc:
        raise EvalError(f"cannot read BEIR qrels {str(path)!r}: {exc}") from exc
    except UnicodeDecodeError as exc:
        # See `_load_jsonl`: a ValueError, so it needs its own clause.
        raise EvalError(f"BEIR qrels {str(path)!r} are not valid UTF-8: {exc}") from exc
    if not rows or rows[0][:3] != ["query-id", "corpus-id", "score"]:
        raise EvalError(f"unexpected or missing BEIR qrels header in {path}")

    relevance: defaultdict[str, set[str]] = defaultdict(set)
    for line_number, row in enumerate(rows[1:], start=2):
        if len(row) < 3:
            raise EvalError(f"invalid BEIR qrels row at {path} line {line_number}")
        query_id, document_id, raw_score = row[:3]
        _validate_identifier(query_id, subject="query", becomes_a_filename=False)
        _validate_identifier(document_id, subject="document", becomes_a_filename=True)
        try:
            score = int(raw_score)
        except ValueError as exc:
            raise EvalError(
                f"invalid integer relevance score {raw_score!r} at {path} line {line_number}"
            ) from exc
        if score not in (0, 1):
            # groundkit's judgment format cannot express a grade: a `GoldSpan`
            # is `(doc, quote)` and nothing else, so a 2 and a 1 would become
            # the same set membership. Flattening them is not a lossy
            # convenience, it is a silent change to the relevance signal -- and
            # the resulting artifact would report *binary* nDCG that reads as
            # comparable to the collection's published *graded* nDCG. Refused
            # rather than coerced, per this package's no-silent-coercion rule.
            # SciFact is unaffected: every one of its qrels rows is 0 or 1.
            raise EvalError(
                f"graded BEIR relevance score {score} at {path} line {line_number}: this "
                "adapter supports binary qrels only. groundkit judgments carry no "
                "relevance grade, so every positive grade would collapse to the same "
                "membership and the adapted set would report binary nDCG against a "
                "collection whose published numbers are graded. Use a binary collection "
                "(SciFact, NFCorpus's binary splits) or add graded relevance to the "
                "harness first."
            )
        if score == 1:
            relevance[query_id].add(document_id)
    return dict(relevance)


def _required_string(record: dict[str, Any], key: str, *, context: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise EvalError(f"{context} requires non-empty string field {key!r}")
    return value


def _optional_string(record: dict[str, Any], key: str, *, context: str) -> str:
    value = record.get(key, "")
    if not isinstance(value, str):
        raise EvalError(f"{context} field {key!r} must be a string when present")
    return value


def _validate_identifier(value: str, *, subject: str, becomes_a_filename: bool) -> None:
    """Validate one BEIR identifier.

    ``becomes_a_filename`` gates the checks that are only about the
    filesystem. A document id is written as ``<id>.txt``; a query id is only
    ever a JSON string inside ``judgments.jsonl`` and never reaches a path
    join, so applying a filesystem-specific rule to it would reject datasets
    that are perfectly usable. It is a required keyword rather than a default
    so that a new call site has to state which kind it is passing.
    """
    if not _SAFE_IDENTIFIER.fullmatch(value) or value in {".", ".."}:
        raise EvalError(
            f"unsafe BEIR {subject} id {value!r}; expected letters, digits, dot, "
            "underscore, or hyphen"
        )
    if not becomes_a_filename:
        return
    encoded = len(f"{value}{_DOCUMENT_SUFFIX}".encode())
    if encoded > _MAX_FILENAME_BYTES:
        raise EvalError(
            f"BEIR {subject} id {value[:32]!r}... is {len(value)} characters; with "
            f"{_DOCUMENT_SUFFIX!r} that is {encoded} bytes, past the {_MAX_FILENAME_BYTES}-byte "
            "filename limit every common filesystem enforces. Refused here rather than at "
            "the write, which would fail partway through an otherwise-valid corpus."
        )
    # `CON`, `NUL`, `COM1` and friends name devices rather than files in the
    # Win32 namespace, and the reservation applies to the stem, so `CON.txt`
    # is reserved too. How that manifests depends on the Windows build --
    # measured on Windows 11 26340 it does not raise at all, and `NUL` instead
    # *accepts* the write and reads back empty, silently losing the document's
    # text. Refused on every platform for the same reason case-colliding ids
    # are: a benchmark corpus whose contents depend on the OS it was adapted
    # on is not a reproducible input, and a silent truncation is worse than a
    # refusal.
    if value.split(".", 1)[0].upper() in _WINDOWS_RESERVED_STEMS:
        raise EvalError(
            f"BEIR {subject} id {value!r} is a reserved Windows device name; it cannot "
            "be a file there, and on some builds the write silently succeeds while "
            "discarding the content. Rename or map this id before adapting."
        )

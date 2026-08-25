"""Adapt a BEIR dataset into the shape groundkit's eval harness already reads.

Tracked, alongside ``evals/perf/``. Run it once per dataset:

    uv run --no-sync python evals/beir/adapt_beir.py --src <dir>/scifact --out <dir>/scifact-gk

**This is a CLI wrapper, not an implementation.** All of the adaptation lives in
:func:`groundkit.evals.beir.adapt_beir_dataset`, which is tracked, typed, and
covered by ``tests/test_beir_dataset.py``. This file exists only to keep the
documented reproduce command working.

It used to carry its own copy of the logic, and the copy had drifted into a
security defect the library version never had: it interpolated the BEIR ``_id``
straight into a path (``corpus_dir / f"{doc_id}.txt"``) with no validation, so a
corpus row with an ``_id`` of ``"../../victim"`` -- or, on Windows, an absolute
``"C:\\..."`` , which ``pathlib`` resolves by *discarding* the left operand
entirely -- wrote outside ``--out``. A BEIR corpus is a third-party download, so
that made the documented reproduce command an arbitrary file write against
whatever mirror served the data. Delegating removes the second implementation
rather than patching it, which is also why the drift became possible.

Behaviour differences from the old copy, both deliberate:

- **A non-empty output corpus directory is now refused** rather than written
  into. Adapting a second dataset over the first silently unioned them, leaving
  stale documents retrievable and scored against judgments that never mentioned
  them. Delete the directory and re-run.
- **Every identifier is validated before anything is written.** The old copy
  wrote the corpus first and discovered an inadmissible id afterwards, leaving a
  half-populated output directory behind on failure.

The three BEIR-to-groundkit mismatches, and how the library resolves each:

**BEIR relevance is document-level; a groundkit ``GoldSpan`` is a verbatim quote.**
The quote is set to the document's *entire text*. ``resolve_gold_span`` then finds
it at offset 0, spanning the whole file, and ``chunk_overlaps_span`` marks every
chunk of that document gold -- which is exactly document-level relevance expressed
in the span vocabulary. No change to the harness, and nothing is faked: the quote
really is a verbatim substring, it is simply the maximal one.

**BEIR query ids are bare integers.** They already satisfy groundkit's kebab-case
pattern (``^[a-z0-9]+(-[a-z0-9]+)*$``), but ``load_judgments`` also requires the
file be sorted ascending by ``query_id`` *as a string*, so the sort is
lexicographic, not numeric. Do not "fix" that to a numeric sort. Note that a
dataset whose query ids are *not* kebab-case (uppercase, dots, underscores --
e.g. NFCorpus's ``MED-10``) is now rejected up front with a message saying so,
instead of being written to disk and then refused by ``load_judgments``.

**BEIR has no judgment categories.** Every row is written as ``normal``. That is
honest rather than convenient: ``no_answer`` would need queries with no relevant
document (BEIR has none -- every qrel row asserts relevance), and ``ambiguous``
means *authored* distinct answers, not merely several gold documents. The
category-level breakdown in the report will therefore be a single bucket.

One thing this deliberately does NOT do: dedupe, filter, or sample the corpus.
The whole point of the exercise is a real collection at real size.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from groundkit.errors import EvalError
from groundkit.evals.beir import adapt_beir_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt a BEIR dataset for groundkit's harness.")
    parser.add_argument("--src", type=Path, required=True, help="Extracted BEIR dataset dir.")
    parser.add_argument("--out", type=Path, required=True, help="Destination for the adapted set.")
    parser.add_argument("--split", default="test", help="qrels split to adapt (default: test).")
    args = parser.parse_args()

    try:
        report = adapt_beir_dataset(args.src, args.out, split=args.split)
    except EvalError as exc:
        # A refusal is the expected outcome for a malformed or inadmissible
        # dataset, so it prints as a message rather than a traceback.
        raise SystemExit(f"adapt_beir: {exc}") from exc

    gold_per_query = report.relevance_pair_count / report.query_count
    print(
        f"corpus     : {report.document_count:,} documents, "
        f"{report.total_characters:,} chars -> {report.corpus_dir}"
    )
    print(f"judgments  : {report.query_count:,} queries, {gold_per_query:.2f} gold docs/query")
    print(f"             -> {report.judgments_path}")


if __name__ == "__main__":
    sys.exit(main())

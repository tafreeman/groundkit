# ADR-0003 — Eval corpus and metrics: quote-anchored judgments, hit-rate recall, threshold-free abstention, JSONL

- **Status:** Accepted (owner, 2026-08-11)
- **Date:** 2026-08-11
- **Deciders:** Andy Freeman (owner)

## Context

SPEC.md §6 requires a labeled golden corpus and a deterministic metrics
engine (recall@k, MRR, nDCG@10) landing in Phase 2, before hybrid retrieval,
rerank, or synthesis — the harness exists to measure every later feature
against a BM25-only baseline, including when a feature fails to beat it.
Building that harness forced four decisions this ADR records together,
because they interlock: how a judgment identifies "the right answer" shapes
what recall@k can honestly mean, which shapes how abstention has to be
measured, which shapes what the judgment file format needs to tolerate
under hand-editing.

The identity question is forced, not chosen: `Chunk.chunk_id` and
`Document.document_id` are `uuid.uuid4().hex` (`contracts.py:38,64`),
regenerated on every ingest, and a chunk's character offsets move whenever
chunking configuration changes — which matters here specifically because
tuning chunk size is itself a variable the harness must be able to measure,
not a constant a judgment can assume stays fixed. `groundkit.evals.runner`
builds a throwaway index per run (never the repo tree), so a judgment
authored against one run's ids or offsets would already be meaningless
against the next run before any retrieval logic even changes.

## Decision

1. **Judgment identity is a gold quote, resolved to a document span, resolved
   to overlapping chunks — never a chunk id or an offset.** A `Judgment`
   (`groundkit/evals/corpus.py`) names a corpus-relative document path plus a
   short, distinctive, verbatim quote (`GoldSpan`). At run time,
   `resolve_gold_span` locates that quote as a `[start, end)` character span
   in the document's current text, and `chunk_overlaps_span` treats every
   persisted chunk whose own `[start_offset, end_offset)` overlaps that span
   as relevant. Resolution happens once, up front, against every judgment
   before any indexing work starts, so a broken quote fails the run closed
   before burning an ingest.
2. **`recall_at_k` is hit-rate — did any top-`k` ranked id land in the gold
   set — not set-recall (`|retrieved ∩ gold| / |gold|`).** A retriever that
   surfaces one of several gold chunks for a query scores identically to one
   that surfaces all of them; multi-passage credit is left entirely to
   `ndcg_at_k`.
3. **Abstention is measured as "returned zero results," with no score
   threshold anywhere in the harness.** `StageResult.no_answer_abstained_count`
   counts `no_answer`-category queries whose retrieval call returned an empty
   list — nothing else.
4. **Judgments are stored as JSONL** — `evals/judgments.jsonl`, one
   `Judgment` object per non-blank line, the file kept sorted ascending by
   `query_id`. `load_judgments` enforces both uniqueness and strict ordering
   as parse-time errors rather than silently re-sorting.

## Alternatives considered

- **Planted marker tokens** in place of quote-anchored spans (e.g. inserting
  a unique sentinel string at the answer location and matching on it).
  Rejected: a marker token enters BM25's own vocabulary through
  `_tokenize`'s `re.findall(r"\w+")` and changes the chunk's token length,
  which feeds directly into the `b` length-normalization parameter and the
  corpus-wide average document length that `BM25Index._score_document` scores
  every other chunk against — the label would contaminate the exact
  retrieval scores it exists to measure. It also has no way to express a gold
  answer that itself spans a chunk boundary: a marker sits at one position,
  and a boundary-straddled answer has no single point to mark.
- **Raw character offsets** (a judgment stores `[start, end]` integers
  directly). Rejected: offsets are invariant to nothing — any edit earlier in
  a document (a typo fix, an added sentence) silently shifts every later
  offset with no signal that a label broke, and a judgments file of bare
  integer pairs is unreviewable in a diff; a reviewer looking at an arbitrary
  `[start, end]` pair cannot tell whether it still points at the intended
  sentence without re-running the harness. The quote approach fails loudly
  instead: `tests/test_corpus_integrity.py` asserts every quote resolves to
  exactly one span, so a corpus edit that breaks a label is a red CI run,
  not a silent drift.
- **Set-recall** in place of hit-rate for `recall_at_k` (decision 2).
  Rejected for the reason given above: it would let a chunk-boundary
  artifact — how many chunks a single gold quote happens to straddle — stand
  in for retrieval quality. See also the Consequences below and the matching
  entry in `KNOWN_LIMITATIONS.md`: `ndcg_at_k`'s IDCG still carries this
  artifact, because SPEC.md §6 specifies the ideal-hit cap as
  `min(len(gold_ids), k)` with no span/chunk distinction, and reopening that
  is out of scope for this ADR.
- **A fixed BM25 score threshold** for detecting abstention (decision 3).
  Rejected: BM25 scores are unbounded above and corpus-dependent — there is
  no value that means "no real answer" across corpora of different sizes and
  vocabularies, so any fixed cutoff would report noise from an arbitrarily
  chosen number rather than a real measurement. The direct consequence is
  that a `no_answer` query must share zero vocabulary with the corpus (the
  tokenizer is a bare `\w+` split with no stopword removal, so a
  sentence-shaped query carries words like "what" or "with" that appear in
  ordinary prose and would score above zero) — recorded honestly as a
  caveat in `KNOWN_LIMITATIONS.md` rather than treated as a free lunch.
- **A pretty-printed JSON array** for `evals/judgments.jsonl` (decision 4).
  Rejected: inserting or editing one record in a formatted array reflows the
  surrounding brackets, commas, and indentation, so the dominant edit this
  file sees in practice — "add N judgments" — would produce diff noise on
  every line around the change, not just the lines that actually changed.
- **TOML**, array-of-tables, for the same file. Rejected: array-of-tables
  syntax is verbose per record and painful to hand-edit at the corpus's
  scale of homogeneous records — every judgment repeats the same handful of
  fields, which is exactly the shape JSONL was designed for and TOML's
  table syntax was not.

## Consequences

**Positive:** the corpus is frozen data. A future change to chunking
configuration — a smaller `chunk_size`, a different overlap — invalidates no
judgment, because a gold span is resolved fresh against the document's text
at run time rather than pinned to a chunk id or an offset that chunking
config would move. Judgments are reviewable in an ordinary diff: a change to
`evals/judgments.jsonl` shows up as added or removed lines a reviewer can
read on sight, not a change to opaque ids that requires re-running the
harness to interpret.

**Negative:** the corpus must be authored with deliberate cross-document
vocabulary competition, or none of the above buys anything. A corpus of
lexically disjoint documents — one on databases, one on cooking, one on
astronomy — passes every check this ADR's scheme can express: every quote
resolves exactly once, every id is unique, every category is represented.
It would still be a useless instrument, because any single matching term
would instantly identify the right document, BM25 would never have to
discriminate between competing candidates, and no later retrieval feature
(dense, hybrid, rerank) could ever demonstrate an improvement over a
baseline that was unbeatable by construction. Nothing in the `Judgment` or
`GoldSpan` schema, and nothing in the integrity suite, can enforce
vocabulary overlap structurally — `evals/README.md`'s distractor rule exists
precisely because this has to be authored by hand, not validated by code.

**Portability:** comparability between runs keys on `corpus_hash` and
`judgments_hash` (`RunMetadata`, `schema.py`), both SHA-256 over raw file
bytes. That only holds across machines if the bytes themselves are
identical, which is what `.gitattributes`'s `* text=auto eol=lf` guarantees.
Without it, a checkout with CRLF line endings would silently change both
hashes and make every run on that machine look incomparable to one from a
LF checkout, for a reason that has nothing to do with the corpus or the
retrieval code being measured.

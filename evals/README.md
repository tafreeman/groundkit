# Golden eval corpus — authoring guide

The retrieval eval harness (SPEC.md §6, Phase 2) scores BM25 (and later,
hybrid) retrieval against a committed golden corpus: documents plus labeled
query -> relevant-chunk judgments. This file is the authoring contract. The
schema below is derived from `src/groundkit/evals/corpus.py`'s Pydantic
models (`GoldSpan`, `Judgment`) — read that module first if a rule here
looks vague; the model is the precise source, this file is the prose gloss
on it, and the two are written to not drift apart.

## Location matters — read this before adding a file

**This file lives at `evals/README.md`, not inside `evals/corpus/`.** That
placement is itself a rule, not an accident:

> The ingest root is `evals/corpus/`, and `FileLoader` (the same loader
> `grk ingest` uses) ingests any `.md`/`.markdown`/`.txt` file it finds
> there. A stray `evals/corpus/README.md` would silently become a corpus
> **document** — indexed, chunked, and searchable like any other — and
> pollute the baseline with content nobody intended to be retrievable.
> `tests/test_corpus_integrity.py` asserts `evals/README.md` exists and
> `evals/corpus/README.md` does not, precisely to catch this mistake.

Do not add a `README.md`, `NOTES.md`, or any other stray `.md` file inside
`evals/corpus/`. If you need to leave a note for corpus authors, it goes
here.

## Layout

```
evals/
  README.md          this file
  corpus/*.md         documents only — nothing else, see above
  judgments.jsonl      one Judgment per line, sorted by query_id
  results/             generated eval output — gitignored, never committed
```

## The JSONL record schema

One JSON object per non-blank line in `evals/judgments.jsonl`, mirroring
`Judgment` and its nested `GoldSpan`:

| Field | Type | Notes |
|---|---|---|
| `query_id` | string | Unique, kebab-case (`^[a-z0-9]+(-[a-z0-9]+)*$`). The file is kept sorted by this field so diffs stay minimal — `load_judgments` rejects duplicates and out-of-order IDs. |
| `query` | string | The natural-language query text. |
| `category` | string | One of `"normal"`, `"ambiguous"`, `"no_answer"`, `"adversarial"`. |
| `gold` | array of `{doc, quote}` | Empty iff `category == "no_answer"`; every other category requires at least one entry; `"ambiguous"` requires at least 2 (see the ambiguous rule below). |
| `notes` | string or omitted | Optional free-text authoring notes. |

Each `gold` entry:

| Field | Type | Notes |
|---|---|---|
| `doc` | string | Path to a document under `evals/corpus/`, relative and forward-slashed (e.g. `"guides/setup.md"`). No absolute paths, no backslashes, no `..` segments. |
| `quote` | string | A verbatim substring of that document's text — see the quote rule below. |

Worked example (a `"normal"` judgment with one gold span):

```json
{"query_id": "wal-mode-purpose", "query": "why does the metadata store use write-ahead logging", "category": "normal", "gold": [{"doc": "storage/sqlite-design.md", "quote": "WAL mode allows concurrent readers"}]}
```

## Floors — the tests are the authoritative numbers

The corpus must have at least 8 documents and at least 40 judgments, with
every one of the four categories represented. **These floors are asserted
as literal numbers in `tests/test_corpus_integrity.py`**, not derived from
this README or from `MIN_CORPUS_DOCS`/`MIN_CORPUS_JUDGMENTS` in
`corpus.py` — those constants exist so the two cannot silently drift below
what the test enforces, but the test itself is the contract (SPEC.md §6: no
hardcoded metric value is authoritative outside a test or generated
artifact).

## The quote rule

A gold `quote` must appear **exactly once**, verbatim, in its `doc`.
`resolve_gold_span` fails closed in both directions: not found is an error,
and found more than once is also an error — silently taking the first match
is exactly the coercion SPEC.md §2 bans. If your quote is ambiguous, extend
it with more surrounding context until it is unique, rather than picking a
different, accidentally-unique substring.

Prefer **short, distinctive** quotes over long ones. A quote spanning an
entire document makes recall@1 trivially satisfiable (nearly any chunk from
that document "contains" the answer) and tests nothing about whether
retrieval found the *right* passage within it. A good quote is the smallest
verbatim span that actually answers the query.

## The paraphrase rule

Queries must **paraphrase** the gold answer, never quote it. A query that
reuses the gold quote's own wording lets lexical overlap do retrieval's job
for it — BM25 wins by definition, not by understanding the query. Every
answerable judgment's query-to-quote lexical overlap must stay at or below
`MAX_QUERY_GOLD_TOKEN_OVERLAP` (see `corpus.py` for the exact value —
intentionally not restated as a bare number here, so it cannot drift out of
sync with the constant). `tests/test_corpus_integrity.py` enforces this as
the circularity guard.

## The distractor rule

Corpus documents must **share vocabulary with each other**. A corpus of 8
lexically disjoint documents (one about databases, one about cooking, one
about astronomy, ...) passes every other check in this file while being a
useless instrument: any single matching word instantly identifies the right
document, BM25 does no real discrimination work, and no later retrieval
feature (dense, hybrid, rerank) can ever demonstrate an improvement over
that trivial baseline. Write documents on **overlapping topics** — several
documents touching the same subsystem, tool, or domain from different
angles — so a query's terms plausibly match more than one document and
ranking actually matters.

## The no-answer rule

A `"no_answer"` judgment's `query` must use vocabulary that appears
**nowhere** in the corpus. This is what makes abstention (returning zero
results, rather than a confident-but-wrong top hit) actually reachable:
BM25 already returns nothing for a query whose terms match no indexed
chunk, so a `no_answer` query only tests that behavior honestly if it truly
shares no vocabulary with anything indexed.
`test_no_answer_queries_share_zero_corpus_vocabulary` checks the token
intersection between every `no_answer` query and the full indexed
vocabulary is empty.

## The adversarial rule

An `"adversarial"` judgment's document must contain genuine
prompt-injection-styled text (e.g. "ignore previous instructions...") as
**distractor content** — text an LLM synthesis step might mistake for a
real instruction if it were naively concatenated into a prompt — while the
judgment itself still has a real, resolvable gold answer elsewhere in that
document. The point is not to test that injected text gets *filtered out*
(this repo's retrieval path is deterministic, non-LLM code — SPEC.md §2);
it is to prove retrieval finds the genuine answer in a document that also
happens to contain adversarial-looking text, and that the corpus-integrity
suite can detect the marker is present at all
(`test_adversarial_docs_contain_injection_marker`, matched against
`INJECTION_MARKERS` in `corpus.py`, case-insensitively).

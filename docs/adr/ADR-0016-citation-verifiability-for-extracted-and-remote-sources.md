# ADR-0016 — Citation verifiability for extracted and remote sources

- **Status:** Accepted (owner, 2026-08-15)
- **Date:** 2026-08-15
- **Deciders:** Andy Freeman (owner)

## Context

PDF/HTML loaders and URL ingestion are v1 scope (SPEC.md §4) with no phase assigned.
ADR-0001's `loaders.py` row records the precondition — *"Collapse the ~90%-duplicated
Markdown/Text loaders into one parametrized class before adding PDF/HTML"* — and Phase 1
satisfied it: `FileLoader` is already one class parametrized over an `extensions` tuple.
Nothing blocks the work except the question this ADR exists to answer.

**That question is not "which parser", it is what `verifiable` means.** SPEC.md §2 makes
citations the product claim: *"Every returned passage carries document ID, chunk ID, and
character offsets resolvable to source."* `resolve_citation` implements that literally —
`ensure_within_base`, then `path.read_text(encoding="utf-8")`, then slice by offsets. It
works today only because of a property the current loader quietly guarantees: for `.md`
and `.txt`, **`Document.content` *is* the file's decoded bytes**, so offsets into content
are offsets into the file, and re-reading the file reproduces them exactly.

Every format this ADR is about breaks that property, and each breaks it differently:

- **PDF.** `content` would be *extracted* text. Re-reading the file as UTF-8 does not
  merely mismatch, it raises. The offsets index a derived artifact that reading the source
  cannot reproduce without running the extractor again.
- **HTML.** Same as PDF *if* tags are stripped. If they are not, `content` is the file's
  decoded bytes and verification keeps working untouched — but then chunks carry markup
  into retrieval, and BM25 scores `<div>` as a term.
- **URL.** Two independent breaks. `Document.source` becomes a URL, and re-reading it is a
  network fetch whose result can differ from what was indexed — a server can change or
  remove the resource, so a "verification" against a re-fetch proves nothing about what
  this index actually returned. Worse, it is not currently *refused*: `_validate_path_input`
  rejects only empty and null-byte strings, so a URL flows into `os.path.realpath`, which
  resolves it as a **relative path under the current directory**. The containment check can
  therefore pass, and the failure surfaces later from `read_text` as a confusing
  file-not-found rather than as "this is not a path".

**What must not happen is the quiet option:** shipping loaders whose citations resolve
through the existing path and produce `RetrievalError` in normal operation, or worse,
resolve to wrong text. `fetch_chunk` (ADR-0014 decision 10) already returns a
`verified` / `drifted` / `unresolvable` verdict, so a degraded answer has somewhere honest
to go — but the verdict must mean something per source class rather than collapsing
"we cannot check this format" into "the source drifted".

One storage fact constrains the alternatives: **the full document text is not persisted.**
The `documents` table holds `document_id`, `source`, `content_hash`, `ingested_at` — no
content. Chunk rows hold content, and chunks tile the document with overlap, so the text is
approximately but not exactly recoverable from them. Any option that verifies against
stored text needs a schema change, and therefore a `SCHEMA_VERSION` bump with ADR-0004
decision 5's delete-and-re-ingest consequence.

## Decision

### 1. Verification is defined per source class, and the class is recorded at ingest

A citation's verification answers one question — *does the text we returned still match
what the source says?* — but "what the source says" is reached differently per class:

| Class | `content` is | Verified by | Extension examples |
|---|---|---|---|
| `text` | the file's decoded bytes | re-read, slice, compare | `.md`, `.markdown`, `.txt` |
| `extracted` | deterministic extractor output | re-extract, slice, compare | `.pdf`, `.html` |
| `snapshot` | the fetched bytes, stored locally | re-read the snapshot, compare | URL ingestion |

The class is a property of how the document was loaded, so it is recorded on the document
at ingest rather than re-derived from the extension at citation time. Deriving it later
would re-introduce the same drift ADR-0004 closed for embeddings: a document ingested by
one loader and verified against another's assumptions.

### 2. `extracted` sources carry an extractor identity, and a mismatch fails closed

This reuses ADR-0004's shape deliberately rather than inventing a second mechanism. An
embedding identity exists because two models produce two incompatible semantic spaces under
one name; an extractor identity exists because two extractor versions produce two
incompatible *offset* spaces under one name. A PDF text extractor that changes how it joins
hyphenated line breaks shifts every offset after the first hyphen in the document.

So an `extracted` document records `(extractor, version)`, and `resolve_citation` refuses —
`RetrievalError`, never a silent slice — when the active extractor's identity differs from
the one recorded. That is the same fail-closed rule as an embedding-identity mismatch, for
the same reason: the alternative is returning confidently wrong text.

**Consequence, stated rather than discovered:** upgrading a PDF library invalidates existing
`extracted` documents. Pre-1.0 the remedy is re-ingest, consistent with ADR-0004 decision 5.

### 3. HTML is `extracted`, and tags are stripped

The alternative — keep raw bytes so verification stays free — is rejected on retrieval
quality. Chunks would carry markup, BM25 would score tag names as terms, and the chunker's
separators would split on markup boundaries rather than prose. Paying an extractor identity
to keep retrieval meaningful is the better trade, and it keeps HTML and PDF on one path
rather than giving HTML a third set of rules.

### 4. URL ingestion snapshots the resource locally, and the snapshot is what verifies

A citation into a remote resource cannot be verified against a re-fetch: the fetch is a
different observation at a different time, so a mismatch cannot distinguish "the index is
stale" from "the server changed" from "the network lied". Verifying against nothing is
worse — it would make `verified` mean "we did not check".

So URL ingestion **stores the fetched bytes** under the index directory, `Document.source`
records the URL for provenance, and verification resolves against the local snapshot. This
makes URL ingestion honest at the cost of storage, and it is the only option under which
`verified` keeps the meaning it has for local files.

Three constraints on the fetch, none negotiable:

- **`ensure_safe_endpoint` from `utils/url_safety.py`** guards it, with
  `allow_private_endpoint=False` — the Ollama allowance is a provider-side `ClassVar` and is
  unreachable from here, which is exactly the scoping ADR-0014 decision 10 chose.
- **Redirects are refused**, matching the outbound policy already in place.
- **A URL is rejected before it can reach `ensure_within_base`.** The latent hazard above is
  closed by classifying the source first, not by hardening `path_safety` — a URL is not a
  path, and making the path helper URL-aware would blur a boundary that is currently sharp.

### 5. PDF and HTML ship as optional extras; URL ingestion does not need one

ADR-0015's rule decides this: *an extra may gate an option on a command, not the command
itself*. `grk ingest` works without PDF support; a `.pdf` file is an input the command may
be given, so its parser is an option. URL fetching uses `httpx`, already a base dependency,
so it needs no extra.

An unavailable extra fails closed with `ConfigurationError` naming the install, exactly as
`dense` and `rerank` do. No new exception types (`errors.py`'s existing classes suffice).

### 6. `fetch_chunk`'s verdict gains no new value; its `detail` carries the reason

`verified` / `drifted` / `unresolvable` already spans the outcomes. An extractor-identity
mismatch is `unresolvable` — the citation cannot be checked, which is different from having
been checked and failed — with `detail` naming the recorded and active identities. Adding a
fourth verdict would push every client into handling a case that is operationally identical
to the one it already handles.

## Alternatives considered

- **Persist the full extracted text in `documents`.** Verification then needs no extractor at
  read time, which is simpler and removes decision 2's upgrade hazard entirely. Rejected for
  now on two grounds: it duplicates content already stored chunk-wise, roughly doubling
  storage for every document, and "resolvable to source" (SPEC.md §2) becomes resolvable to
  *our copy of* the source, which cannot detect that the underlying file changed — the exact
  drift the citation check exists to catch. Worth revisiting if extractor-identity churn
  proves more painful in practice than the storage cost.
- **Keep HTML raw so verification stays free.** Rejected per decision 3 — free verification of
  text nobody would want retrieved.
- **Verify URL citations by re-fetching.** Rejected per decision 4: a re-fetch is a different
  observation, and a mismatch is uninterpretable. It also turns citation resolution into a
  network call with its own SSRF surface and latency.
- **Refuse to verify URL citations at all (always `unresolvable`).** Cheaper than snapshotting
  and honest, but it makes URL ingestion produce citations that can never be verified, which
  is a strange thing for this repo to ship given SPEC.md §2. Kept as the fallback if
  snapshot storage proves unacceptable.
- **Harden `path_safety` to reject URL-shaped strings.** Rejected per decision 4: a path
  helper that knows about URLs is a blurred boundary, and the real fix is that a `snapshot`
  document never reaches it with a URL in the first place.
- **Derive the source class from the file extension at citation time.** Rejected per
  decision 1 — it lets a document ingested one way be verified under another's assumptions.

## Consequences

- `Document` and the `documents` table gain the source class and, for `extracted`, the
  extractor identity. That is a schema change, so `SCHEMA_VERSION` goes to 3 and existing
  collections must be deleted and re-ingested (ADR-0004 decision 5).
- `resolve_citation` stops being a single code path. It gains a per-class branch and can now
  raise for a reason that is neither "escaped the base" nor "source changed", which
  `fetch_chunk` surfaces as `unresolvable` with a `detail`.
- URL ingestion consumes disk proportional to what is fetched, under the index directory.
  Retention and cleanup for those snapshots are **not** decided here and are owed before any
  deployment that is not a single user's local machine, in the same sense SPEC.md §7 already
  says of SQLite.
- The BOM caveat in `KNOWN_LIMITATIONS.md` becomes narrower: `extracted` and `snapshot`
  classes route through an extractor that can normalize, while `text` keeps today's
  behaviour.
- Upgrading a PDF library is now a breaking change for existing `extracted` documents, by
  design. It is loud rather than silent, which is the trade.

## References

- SPEC.md §2 (citations verifiable; fail closed), §4 (PDF/HTML/URL in v1 scope), §7 (SSRF
  guard on URL ingestion; content-bearing storage and the product decisions it owes), §8.
- [ADR-0001](ADR-0001-promote-vs-rewrite.md) — the `loaders.py` row, its
  collapse-before-PDF/HTML precondition (satisfied in Phase 1), and the confirmed
  no-directory-ingestion gap.
- [ADR-0004](ADR-0004-embedding-identity-binding.md) — the identity-binding shape decision 2
  reuses one layer out, and decision 5's pre-1.0 rebuild-not-migrate rule.
- [ADR-0014](ADR-0014-read-only-service-surface-and-outbound-endpoint-safety.md) — decision
  10's `utils/url_safety.py`, whose Ollama allowance is deliberately unreachable from
  ingestion; decision 11's redirect policy; and `fetch_chunk`'s verdict set.
- [ADR-0015](ADR-0015-service-dependencies-are-base-not-an-extra.md) — the option-vs-command
  rule decision 5 applies.
- `KNOWN_LIMITATIONS.md` — the unscheduled-loaders entry this ADR schedules, and the
  length-only drift check `fetch_chunk`'s byte comparison already strengthens.

# Extracted and remote source loaders — citation verifiability

Feature spec for the plumbing [ADR-0016](../adr/ADR-0016-citation-verifiability-for-extracted-and-remote-sources.md)
requires but does not itself build. PDF/HTML loaders and URL ingestion are v1
scope (SPEC.md §4) with **no phase number** — ADR-0016's own Context section
says so explicitly — so this workstream is tracked here rather than folded
into a numbered phase's spec.

Status: **Waves 1–2 not yet started on this branch.** PR #12 landed the
contract and dispatch half of ADR-0016 (see §2) ahead of this spec, which is
why §2 reads as ground truth rather than a plan. Waves 3–4 (§3) are not
scheduled.

Nothing here overrides SPEC.md or ADR-0016; where this document makes a
decision neither already contains, it says so and states the alternatives
considered, per SPEC.md §8.

## 1. What ADR-0016 already settled

Six decisions, accepted 2026-08-15, not open for relitigation here:

1. Verification is defined per `SourceClass` (`text` / `extracted` /
   `snapshot`), recorded at ingest rather than re-derived from the extension
   at citation time.
2. An `extracted` document carries an extractor identity; a mismatch between
   the identity recorded at ingest and the one active in the build fails
   closed.
3. HTML is `extracted` (tags stripped), not a fourth class.
4. URL ingestion snapshots the fetched bytes locally; verification resolves
   against the snapshot, never a re-fetch.
5. PDF/HTML ship behind an extra (ADR-0015's option-vs-command rule); URL
   fetching does not need one (`httpx` is already base).
6. `fetch_chunk`'s verdict set (`verified` / `drifted` / `unresolvable`)
   gains no fourth value. An extractor-identity mismatch is `unresolvable`,
   with `detail` naming the reason.

Read the ADR for the reasoning; it is not restated here.

## 2. What is already built, and what is not (ground truth, verified against the tree)

PR #12 landed more of ADR-0016 than the six decisions above might suggest.
Already present and correct:

- `SourceClass = Literal["text", "extracted", "snapshot"]` (`contracts.py`).
- `Document.source_class` / `Document.extractor`, with a two-way
  `model_validator` enforcing decision 2's pairing (an `extracted` document
  must carry an extractor; every other class must not).
- `Citation.source_class`, `Citation.extractor`, `RetrievalResult.source_class`,
  `RetrievalResult.extractor`, and `RetrievalResult.citation` propagating all
  four onto the `Citation` it builds.
- `resolve_citation` (`retrieval/citations.py`) dispatching on
  `citation.source_class` and refusing `extracted` and `snapshot` outright,
  with the URL-before-`ensure_within_base` ordering decision 4 requires.
- `verify_citation(citation, expected_content, allowed_base_dir) -> bool`.
- `tests/test_source_class.py` — the contract-level tests (validator pairing,
  dispatch refusal, the pinned `ensure_within_base`-resolves-a-URL hazard).

**None of that is the gap. The gap is that nothing persists the class, and
the failure mode is fail-*open*, not merely incomplete:**

- `documents` (`index/metadata.py`, `_SCHEMA`) has columns
  `document_id, source, content_hash, ingested_at` only — no `source_class`,
  no `extractor`.
- `indexer.py` has zero references to either field. `Indexer._persist_document`
  calls `store.replace_document(source=doc.source, document_id=doc.document_id,
  content_hash=doc_hash, chunks=chunks)` — `doc.source_class` and
  `doc.extractor` are read off the loaded `Document` and then dropped.
- `retrieval/search.py` has zero references to either field.
  `Retriever._resolve` builds every `RetrievalResult` with no
  `source_class=`/`extractor=` keyword, so both default to `Citation`'s and
  `RetrievalResult`'s Pydantic defaults — `"text"` and `None` — regardless of
  what was actually ingested.
- `service/tools.py::handle_fetch_chunk` builds its `Citation` the same way,
  from a `sources: dict[str, str]` (`get_document_sources()`) that carries a
  bare path string, no class.

The consequence: `Indexer` accepts a hand-constructed
`Document(source_class="extracted", extractor="x@1", ...)` today — the
contract permits it, keyword-only defaults ask for nothing more — chunks it,
persists it, and every read path (`search`, `fetch_chunk`) reports it back as
`source_class="text"`. A `text`-class citation routes through the plain
read-and-slice path ADR-0016 exists to keep `extracted` and `snapshot`
citations *out of*. **The current default is a silent downgrade, not an
absent feature**: nothing raises, nothing logs, and the citation looks
verifiable — it just verifies the wrong thing, because it verifies as if it
were byte-identical file content it never was.

No loader produces an `extracted` or `snapshot` `Document` yet (§3, Waves 3–4)
so this defect is unreachable from any shipped `grk` command today. It is
reachable from library code and from tests that construct a `Document`
directly, and it will become reachable from `grk ingest` the moment a Wave 3
loader lands — silently, unless this spec's Wave 1 closes it first.

## 3. The wave plan

| Wave | Scope | Status |
|---|---|---|
| 1 | Persist `source_class`/`extractor`; join them back through `search` and `fetch_chunk` | this spec, not started |
| 2 | Extractor-identity check at resolve time; typed `fetch_chunk` verdict (§5) | this spec, not started |
| 3 | PDF/HTML extractors behind a `pdf`/`html` extra (ADR-0016 decisions 3, 5) | **out of scope** |
| 4 | URL fetching + local snapshot storage (ADR-0016 decision 4) | **out of scope** |

Waves 3 and 4 are named so a reader finds a scheduled gap, not a forgotten
one. **Do not build a PDF or HTML extractor, do not add a `pdf`/`html`
extra, and do not implement URL fetching or snapshot storage as part of this
spec** — that is explicitly reserved. Wave 2 has to *behave correctly* in
their absence (§5.3), which is a smaller obligation than building them.

## 4. Wave 1 — persistence and the join

### 4.1 Schema: `SCHEMA_VERSION` 2 → 3

`documents` gains two columns:

```sql
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    source TEXT UNIQUE NOT NULL,
    content_hash TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    source_class TEXT NOT NULL DEFAULT 'text',
    extractor TEXT
);
```

`SCHEMA_VERSION: Final[int] = 3` in `index/metadata.py`. Confirmed with the
owner: the delete-and-re-ingest consequence is acceptable
([ADR-0004](../adr/ADR-0004-embedding-identity-binding.md) decision 5 —
pre-1.0, an unversioned or wrong-version store is a rebuild, not a
migration). **This bump is not additive the way 1→2 was.** ADR-0013's bump
added a *new* table (`collection_state`); a v1 store kept working for
BM25-only reads and writes and only lost dense-manifest capability. This
bump changes `documents` itself — the table every read and write already
goes through — so a v1 or v2 store cannot silently keep working for
*anything* that touches `source_class`/`extractor`. `CREATE TABLE IF NOT
EXISTS` is a no-op against an existing `documents` table, so an old store's
table genuinely lacks the two new columns; there is no `ALTER TABLE` here
(pre-1.0, no migrations).

**The guard stays narrow, matching the existing `_require_manifest_capable`
pattern** rather than refusing at `open()` for every store below v3:
`upsert_document`, `replace_document`, and the new `get_document_records`
(§4.3) are the only methods that read or write the two new columns, so they
are the only ones that need to check `self._schema_current` and refuse. Add
a sibling guard (or generalize the existing one — implementer's choice of
shape, not of behavior) that raises `StorageError` naming the delete-and-
re-ingest remedy, the same way `_require_manifest_capable` does today for
`IndexIdentityError`. Every other method — `get_document_hash`,
`get_document_id`, `get_document_sources`, `add_chunks`, `get_chunks`,
`get_chunk`, `delete_document`, the manifest and generation methods — is
untouched by this bump and keeps its existing schema-currency requirements
exactly as they are today. This is deliberately narrower than "the whole
store refuses below v3": `get_document_sources()`'s query
(`SELECT document_id, source FROM documents`) never references the new
columns, so it keeps working unchanged against a v1, v2, *or* v3 store —
there is no reason to make it fail on the two it currently serves fine.

Write-path signatures (frozen decision: new parameters are keyword-only with
a default, so no existing caller breaks):

```python
async def upsert_document(
    self,
    source: str,
    document_id: str,
    content_hash: str,
    *,
    source_class: SourceClass = "text",
    extractor: str | None = None,
) -> None: ...


async def replace_document(
    self,
    source: str,
    document_id: str,
    content_hash: str,
    chunks: list[Chunk],
    *,
    source_class: SourceClass = "text",
    extractor: str | None = None,
) -> None: ...
```

Both write methods gain the pair (not just `replace_document`, which is the
only one any production code calls today) so the protocol stays internally
consistent and `upsert_document`'s own tests can exercise it directly.
`MetadataStoreProtocol` (`index/protocols.py`) and its conformance test
(`tests/test_protocol_conformance.py`) gain the same two keyword-only
parameters on both methods.

`Indexer._persist_document` (`indexer.py`) passes the two fields through on
its one call site:

```python
await self._store.replace_document(
    source=doc.source,
    document_id=doc.document_id,
    content_hash=doc_hash,
    chunks=chunks,
    source_class=doc.source_class,
    extractor=doc.extractor,
)
```

### 4.2 The join: reading `source_class`/`extractor` back

`RetrievalResult.source_class`/`.extractor` and `Citation.source_class`/
`.extractor` must be populated from what was actually persisted, for *every*
document a search or fetch touches — not only the `extracted` ones. A mixed
collection (some `text`, some `extracted`) must report each result's real
class.

**New protocol method, not a widened `get_document_sources()`.**
`get_document_sources() -> dict[str, str]` has five call sites outside this
workstream — `Indexer`'s two prune paths (string equality against a resolved
path), `index/dense.py::verify_dense_side_present` (truthiness only),
`evals/runner.py` (two sites, one builds a corpus-relative path map via
`Path(source)`, the other only reads the keys) — none of which need
`source_class` or `extractor`. Widening its value type from `str` to a
record would force every one of those five sites to change for a field they
never use, and `Retriever.open()`'s and `Retriever._dense_candidates`'s uses
only need the *key set* anyway. Adding a sibling method confines the change
to the two call sites that actually need the richer read:

```python
# contracts.py, beside SourceClass
class DocumentRecord(BaseModel):
    """A stored document's provenance, projected for the citation join (ADR-0016).

    Read-only: this is a view over the `documents` row, not a second place
    that fact is asserted from. `source_class`/`extractor` default exactly as
    Document's own fields do, so a v3 store's rows written before this wave
    (none exist pre-launch, but the defaults are the same regardless) read
    back as `text`/None, matching what an old-style ingest actually produced.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    source: str
    source_class: SourceClass = "text"
    extractor: str | None = None
```

```python
# index/protocols.py — MetadataStoreProtocol
async def get_document_records(self) -> dict[str, DocumentRecord]:
    """Return {document_id: DocumentRecord} for every stored document.

    Requires a schema-v3 store (§4.1) — raises StorageError otherwise, naming
    the delete-and-re-ingest remedy, the same way write_manifest does for a
    pre-ADR-0004 store.
    """
```

Two call sites switch to it:

- `Retriever.search()` (`retrieval/search.py`): its one
  `sources = await self._store.get_document_sources()` read (used by every
  mode) becomes `records = await self._store.get_document_records()`.
  `_resolve`'s `sources: dict[str, str]` parameter widens to
  `dict[str, DocumentRecord]`, and every constructed `RetrievalResult` gains
  `source=record.source, source_class=record.source_class,
  extractor=record.extractor`. `_dense_candidates`'s and
  `_apply_snapshot_filter`'s `sources` parameter type widens to match for
  `mypy --strict`, even though their only use of it is `chunk.document_id in
  sources` — membership, unaffected by the value type.
  `Retriever.open()`'s `documents_at_open` snapshot keeps calling
  `get_document_sources()` (it only needs the key set, and there's no reason
  to pay for the richer read there).
- `handle_fetch_chunk` (`service/tools.py`): its
  `sources = await runtime.get_document_sources()` becomes
  `records = await runtime.get_document_records()`, and the `Citation` it
  builds gains `source_class=record.source_class,
  extractor=record.extractor` alongside the existing `source=record.source`.

`CollectionRuntime` (`runtime.py`) gains the matching thin proxy, mirroring
its existing `get_document_sources`:

```python
async def get_document_records(self) -> dict[str, DocumentRecord]:
    """Return {document_id: DocumentRecord} for every document in the collection."""
    self._require_open()
    return await self._store.get_document_records()
```

## 5. Wave 2 — extractor-identity check and the typed verdict

### 5.1 The extractor-identity check

`resolve_citation`'s current dispatch refuses every non-`text` class with one
generic message. Wave 2 splits `extracted` out into its own branch — decision
2's actual check — while `snapshot` keeps its exact current behavior
(Wave 4's job, untouched here, still refused).

```python
#: Identity strings of every extractor this build can re-run a citation
#: through. Empty in Waves 1-2 by design, not a TODO: no PDF/HTML extractor
#: is wired in yet (Wave 3, ADR-0016 decisions 3/5), so no extractor
#: identity is ever active, and every `extracted` citation correctly refuses
#: regardless of what it recorded. Wave 3 populates this per extractor it
#: ships; the membership check below does not change shape when it does.
_ACTIVE_EXTRACTOR_IDENTITIES: frozenset[str] = frozenset()
```

```python
if citation.source_class == "extracted":
    if citation.extractor not in _ACTIVE_EXTRACTOR_IDENTITIES:
        raise RetrievalError(
            f"cannot resolve an 'extracted' citation for {citation.source!r}: "
            f"the extractor identity recorded at ingest ({citation.extractor!r}) "
            "is not active in this build "
            f"({sorted(_ACTIVE_EXTRACTOR_IDENTITIES) or 'none registered'}). "
            "Re-deriving these offsets needs the exact extractor that produced "
            "them; refused rather than guessing (ADR-0016 decision 2).",
            verdict="unresolvable",
        )
    # Unreachable while _ACTIVE_EXTRACTOR_IDENTITIES is empty (Waves 1-2).
    # Wave 3 adds the re-extraction call here: re-run the active extractor
    # over citation.source, slice [start_offset:end_offset], and return it —
    # the same shape resolve_citation already has for `text`.
elif citation.source_class == "snapshot":
    raise RetrievalError(
        f"cannot resolve a 'snapshot' citation for {citation.source!r}: ...",
        verdict="unresolvable",
    )
```

The message must keep the substring `"extracted"` —
`tests/test_source_class.py::test_resolve_refuses_an_extracted_citation`
already asserts `pytest.raises(RetrievalError, match="extracted")` and stays
green under this change (the extractor it supplies, `"pdf-x/1"`, is not in
the empty active set, so it still refuses). The `snapshot` branch keeps its
existing message verbatim — `match="snapshot"` in the sibling test is
unaffected.

This is real, not placeholder, code: the membership check is exactly what
Wave 3 will exercise once it populates the set. Nothing about its shape
changes when an extractor is added — only the constant does.

### 5.2 THE verdict mechanism — replacing the string sniff

`fetch_chunk` (`service/tools.py`) currently classifies a `RetrievalError` by
searching its message for `"changed since indexing"`:

```python
verification = "drifted" if "changed since indexing" in str(exc) else "unresolvable"
```

That already only classifies two outcomes correctly by coincidence, and
decision 2's new extractor-mismatch refusal (§5.1) — which must map to
`unresolvable` — would classify correctly today only because it, too, lacks
that substring. Coincidence is not a mechanism; the ADR names this exact
prose-coupling as the thing to replace.

**Alternatives weighed:**

- **New `RetrievalError` subclasses** (e.g. a drift error and an
  unresolvable error). Rejected. ADR-0016 decision 5 states a preference for
  no new exception types where an existing one already covers the failure
  (there, in the narrower context of the extras — but the reasoning
  generalizes: `errors.py`'s taxonomy already has a type per *kind* of
  failure, not per *classification a specific caller needs*). More
  concretely: `evals/echo.py:491` calls `resolve_citation` directly and lets
  a failure propagate as a bare `except`-free call — it needs "did this
  fail", never "which of two ways". A subclass hierarchy would exist purely
  for `fetch_chunk`'s benefit while every other caller pays for it in import
  surface, and `fetch_chunk`'s own dispatch barely simplifies — `except
  CitationDriftError` / `except CitationUnresolvableError` is not meaningfully
  less classification logic than reading an attribute.
- **A structured return from a `citations.py` helper**, so `fetch_chunk` maps
  values instead of classifying exceptions. Rejected for this change,
  specifically because it changes `resolve_citation`'s existing contract:
  today it always raises on failure and returns `str` on success, and
  `evals/echo.py`'s citation-echo check depends on exactly that — it awaits
  `resolve_citation` with no local exception handling at all. Making it
  return a value for the "expected" failures (drift, unresolvable) while
  still raising for a genuine containment escape would need this spec to
  draw that line, and it would ripple into a currently-working, already-
  tested call site this workstream has no reason to touch. Worth
  reconsidering if a *third* caller ever wants the same classification
  `fetch_chunk` does — one call site does not justify inventing a second
  return protocol next to the raise-based one this module already has.
- **A typed attribute on `RetrievalError` carrying the verdict — chosen.**
  `resolve_citation` keeps its exact existing contract: it still always
  raises `RetrievalError` on failure, so `echo.py` is untouched. Every raise
  site inside `resolve_citation` states its own verdict once, at the point
  that already knows it best, instead of `fetch_chunk` re-deriving it from
  prose written for a human 200 lines away. No new exception type, matching
  decision 5's spirit. The attribute is available to any future caller that
  wants the same classification without re-parsing a message.

**The exact shape** (implement precisely — this is the seam between Wave 1
and Wave 2, and between whichever agents build them):

```python
# errors.py
from typing import Literal

#: The two failure verdicts a citation-resolution failure maps to. Distinct
#: from service.schemas.VerificationVerdict (which also has "verified" — a
#: success, never carried by an exception) — CitationVerdict values are a
#: literal subtype of it, so assigning one to a VerificationVerdict-typed
#: variable type-checks under mypy --strict with no cast.
CitationVerdict = Literal["drifted", "unresolvable"]


class RetrievalError(GroundkitError):
    """Error during retrieval or search.

    Attributes:
        verdict: For a citation-resolution failure raised by
            retrieval.citations.resolve_citation, which of fetch_chunk's two
            failure verdicts this maps to. None for every other
            RetrievalError (an index inconsistency, an empty query, an
            out-of-range top_k) — those have no fetch_chunk verdict to carry,
            and leaving this unset for them is the point: nothing downstream
            can read a guessed value for an error this attribute was never
            meant to describe.
    """

    def __init__(self, message: str, *, verdict: CitationVerdict | None = None) -> None:
        super().__init__(message)
        self.verdict = verdict
```

Every raise site inside `resolve_citation` (`retrieval/citations.py`) sets
it explicitly: the containment-escape `ValueError` branch and the
`OSError`/`UnicodeDecodeError` branches all pass `verdict="unresolvable"`;
the offset-overflow branch (`"source changed since indexing"`, the one
genuine drift case at the `text` class) passes `verdict="drifted"`; both new
branches in §5.1 pass `verdict="unresolvable"`, per decision 6.

`fetch_chunk`'s classification collapses to a map, not a search:

```python
try:
    resolved = await resolve_citation(citation, ctx.base_dir)
except RetrievalError as exc:
    verification = exc.verdict if exc.verdict is not None else "unresolvable"
    detail = str(exc)
else:
    ...
```

The `is not None else "unresolvable"` fallback is defensive, not load-bearing
— every path inside `resolve_citation` now sets a verdict, so it should
never trigger. It exists so a future `RetrievalError` raised inside
`resolve_citation` without a verdict fails toward the more conservative of
the two verdicts (`unresolvable` never claims a definite "the source
changed", `drifted` would) rather than crashing `fetch_chunk` outright.

**`grep -rn "changed since indexing" src/groundkit/service/tools.py` must
return nothing** once this lands — the string-sniff itself, not just its
symptom, must be gone.

### 5.3 What Wave 2 does not do

No re-extraction call exists or is stubbed with a `NotImplementedError` — the
membership check in §5.1 already produces the correct refusal without one.
No registry, plugin mechanism, or config surface for extractor identities is
built; `_ACTIVE_EXTRACTOR_IDENTITIES` is a plain empty constant, and inventing
an extension mechanism for zero registered extractors is exactly the kind of
scope creep §3 rules out. Wave 3 decides that shape when it has a real
extractor to register.

## 6. Done means

- [ ] `documents` has `source_class TEXT NOT NULL DEFAULT 'text'` and
      `extractor TEXT`; `SCHEMA_VERSION == 3`.
- [ ] `upsert_document` and `replace_document` accept
      `*, source_class: SourceClass = "text", extractor: str | None = None`;
      `MetadataStoreProtocol` and its conformance test agree.
- [ ] `Indexer._persist_document` passes `doc.source_class`/`doc.extractor`
      through on its one `replace_document` call.
- [ ] `get_document_records()` exists on the protocol, the SQLite store, and
      `CollectionRuntime`; refuses cleanly (`StorageError`, naming the
      remedy) on a pre-v3 store.
- [ ] `Retriever._resolve` and `handle_fetch_chunk` build their
      `RetrievalResult`/`Citation` from a joined `DocumentRecord`, not a bare
      source string — every result and every fetched chunk reports the class
      it was actually ingested under.
- [ ] `resolve_citation` has a real (currently always-refusing) extractor-
      identity check for `extracted`, gated on `_ACTIVE_EXTRACTOR_IDENTITIES`.
- [ ] `RetrievalError.verdict` exists; every raise site inside
      `resolve_citation` sets it; `fetch_chunk` reads it instead of
      string-matching. `"changed since indexing"` does not appear in
      `service/tools.py`.
- [ ] ADR-0016 decisions 1, 2, and 6 are closed: a document's recorded class
      round-trips through search and fetch_chunk unchanged (1); an
      extractor-identity mismatch fails closed rather than silently
      resolving as `text` (2); `fetch_chunk`'s verdict set stays three-valued
      with the reason in `detail` (6).
- [ ] A regression test demonstrates the fail-open downgrade described in §2
      is gone: construct a `Document(source_class="extracted",
      extractor="x@1", ...)` directly (no loader produces one yet — this is
      exactly what the keyword-only defaults are for), ingest it, and assert
      that `get_document_records()`, a `Retriever.search()` result's
      `source_class`/`extractor`, and a `fetch_chunk` call against one of its
      chunks all report `"extracted"`/`"x@1"` — none of the three silently
      reports `"text"`/`None`. Per SPEC.md §8, this test must be shown to
      fail against the pre-Wave-1 tree (`git stash` the persistence+join
      change, run, observe the failure, restore, run again, observe the
      pass) — both directions reported.
- [ ] A regression test proves the extractor-identity-mismatch path returns
      `"unresolvable"` with a `detail` naming the recorded identity, not
      `"drifted"` and not the old generic refusal text.
- [ ] Waves 3 and 4 remain unbuilt: no `pdf`/`html` extra, no PDF/HTML
      extractor, no URL fetching, no snapshot storage.

## 7. Risks and open questions

**R1 — The 2→3 bump is a harder break than 1→2, on purpose.** Any collection
with documents ingested before this lands cannot accept a further write
through `upsert_document`/`replace_document` afterward — not just its dense
side, as the 1→2 bump left intact for BM25-only collections, but *any*
mutation of `documents` at all, because the table itself changed shape. Reads
that don't touch the two new columns (`get_document_sources`, `get_chunks`,
etc.) keep working. This is the accepted consequence (§4.1), stated here so
it is not mistaken for scope creep when a reviewer notices how much wider it
is than the previous bump.

**R2 — `_ACTIVE_EXTRACTOR_IDENTITIES` is permanently empty until Wave 3, and
that is not dead code.** A reviewer scanning §5.1's branch might read an
always-false membership test as unfinished. It is finished: the check is
correct for a build with zero extractors, and Wave 3's job is populating the
constant, not changing the check's shape. Worth a code comment at the
constant's definition (already drafted above) so this doesn't get "fixed"
into a `NotImplementedError` or a bypassed check by someone who hasn't read
this spec.

**Q1 — Should `get_document_records()` eventually replace
`get_document_sources()` outright?** Not decided here. §4.2 argues the
narrower two-call-site change is right *now*, on blast-radius grounds — five
unrelated call sites would need to change for a field they don't use. If a
third or fourth caller ever needs `source_class`/`extractor`, that balance
may flip; revisit then rather than pre-emptively unifying the two methods
for two current callers.

**Q2 — Does `_ACTIVE_EXTRACTOR_IDENTITIES` want to be an injectable seam
(e.g. read from installed extractor plugins) once Wave 3 has a real
extractor?** Deliberately not decided here (§5.3) — there is nothing to
inject yet, and designing the registration mechanism now would be guessing
at Wave 3's shape before Wave 3 exists.

## 8. Verification

A reviewer should be able to confirm, without running anything:

- `index/metadata.py`'s `_SCHEMA` and `SCHEMA_VERSION` match §4.1 exactly.
- `grep -n "source_class\|extractor" src/groundkit/indexer.py
  src/groundkit/retrieval/search.py` returns real, non-default-only usages —
  the two files the ground truth in §2 named as having *zero* references
  today.
- `grep -n "changed since indexing" src/groundkit/service/tools.py` returns
  nothing.
- `MetadataStoreProtocol` in `index/protocols.py` declares
  `get_document_records`, and `tests/test_protocol_conformance.py` checks its
  signature against `SQLiteMetadataStore`'s.
- `errors.py` defines `CitationVerdict` and `RetrievalError.__init__` accepts
  `verdict`.

And by running the gates (orchestrator's job, not the spec's):

- `uv run ruff check . && uv run ruff format --check .`
- `uv run mypy` (strict; the new keyword-only parameters, `DocumentRecord`,
  and `RetrievalError.verdict` are all fully typed)
- `uv run pytest --cov && uv run coverage report` (whole-package gate) and
  the core-subset gate (`retrieval/*` already covers `citations.py` and
  `search.py`; `index/metadata.py` stays outside the subset, unchanged from
  today — see `pyproject.toml`'s existing note on why)
- The two regression tests named in §6, each shown failing against reverted
  source before being shown passing against the fix (SPEC.md §8).

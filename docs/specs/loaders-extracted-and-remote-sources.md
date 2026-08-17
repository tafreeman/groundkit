# Extracted and remote source loaders — citation verifiability

Feature spec for the plumbing [ADR-0016](../adr/ADR-0016-citation-verifiability-for-extracted-and-remote-sources.md)
requires but does not itself build. PDF/HTML loaders and URL ingestion are v1
scope (SPEC.md §4) with **no phase number** — ADR-0016's own Context section
says so explicitly — so this workstream is tracked here rather than folded
into a numbered phase's spec.

Status: **Waves 1–2 have landed** (`0bcb417` persists `source_class`/
`extractor` and adds `get_document_records`; `4b41c12` fixes `fetch_chunk` to
read the joined record instead of a bare source string) — the "not yet
started" note that used to sit here was stale and is corrected rather than
left wrong. PR #12 landed the contract and dispatch half of ADR-0016 (see §2)
ahead of Waves 1–2 proper, which is why §2 already read as ground truth; §4–§6
(Waves 1–2's own design) are now equally ground truth, verified against the
tree at `4b41c12`, not a plan that shipped unread.

**Wave 3 has now landed in part**, and the part is stated precisely because
the boundary is not the one §9 implies at first reading: `groundkit/extraction.py`
(the `ExtractorProtocol` seam, `PdfExtractor`, `HtmlExtractor` and the
`active_extractors()` registry), the `pdf`/`html` extras, and
`resolve_citation`'s re-extraction branch are built and tested. The ingest-side
PDF/HTML **loaders** are not — they need the multi-loader dispatch §9.6
explicitly declines to design, so building them would have meant guessing at
exactly the seam that section refuses to guess at. Wave 4 remains **designed**
(§10) and **not built**. §3's status column carries the same split.

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
| 1 | Persist `source_class`/`extractor`; join them back through `search` and `fetch_chunk` | **landed** (`0bcb417`) |
| 2 | Extractor-identity check at resolve time; typed `fetch_chunk` verdict (§5) | **landed** (`0bcb417`, `4b41c12`) |
| 3 | PDF/HTML extractors behind a `pdf`/`html` extra (ADR-0016 decisions 3, 5) | **landed**, except the ingest-side loaders |
| 4 | URL fetching + local snapshot storage (ADR-0016 decision 4) | **designed (§10), not built** |

Wave 3's exception is not a shortfall against §9 but a consequence of §9.6:
`groundkit/extraction.py`, both extras and `resolve_citation`'s re-extraction
branch are built and tested, so an `extracted` citation produced by a
`pypdf`/`beautifulsoup4` identity this build can run now verifies rather than
refusing. What is still absent is a `.pdf`/`.html` **loader** reachable from
`grk ingest`, because routing extensions to loaders within one invocation
needs the multi-loader dispatch §9.6 declines to design. The practical shape
of that gap: nothing in-tree yet *produces* an `extracted` document, so the
re-extraction path is exercised by tests rather than by a live ingest.

Wave 4 is named so a reader finds a scheduled gap, not a forgotten one. §10
settles the seams ADR-0016 deliberately left open — precisely enough that an
implementer should not need to make a judgment call this document doesn't
already state and justify — but **building it is still someone else's job**:
no URL fetching and no snapshot storage exist in `src/groundkit/` as of this
revision. Wave 2 already had to *behave correctly* in their absence (§5.3),
which was a smaller obligation than building them, and remains true.

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

**Resolved by §9 below**, now that Wave 3 has a real shape to design against:
the module constant is replaced by `groundkit.extraction.active_extractors()`,
a lazily-memoized accessor (not an import-time probe) that independently
try/excepts each candidate extractor so one missing extra never blanks out
the other. See §9.2 for the exact mechanism and the reasoning for choosing
lazy-and-memoized over an eager module-load-time probe.

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

## 9. Wave 3 design — the extractor registry, the PDF/HTML libraries, and re-extraction

This section was written as the design, before the implementation. Everything
in it except the ingest-side loaders (§9.5, §9.6) is now built and tested in
`src/groundkit/extraction.py` and `src/groundkit/retrieval/citations.py`; the
text is left in its original design voice rather than rewritten into a
description, so the reasoning stays legible as reasoning — §3 carries the
status. It answers, precisely enough that three implementers
building in parallel should reach the same shape independently: which
libraries, where the shared extractor code lives, how `resolve_citation`
finds an extractor from an identity string, and what happens when
re-extraction itself fails. Where ADR-0016 already decided something (HTML is
`extracted` with tags stripped — decision 3; an identity mismatch fails
closed — decision 2; a missing extra is `ConfigurationError` naming the
install — decision 5), this section cites it and does not re-argue it. Where
it makes a call ADR-0016 leaves open, that is stated explicitly, with the
alternatives considered.

### 9.1 Library choices

**PDF: [`pypdf`](https://pypi.org/project/pypdf/), pinned `pypdf>=6.15,<7`.**
Pure Python (no compiled extension, so no per-platform wheel risk on any of
3.11/3.12/3.13), BSD-3-Clause licensed (no license friction against this
repo's MIT), actively maintained, and exposes its version through the
standard `importlib.metadata` mechanism like any other installed
distribution — no library-specific version attribute to special-case. The
bound style is this repo's usual one (`lancedb>=0.13,<1`,
`sentence-transformers>=3,<6` — a floor near a known-good release, a ceiling
at the next major so an unreviewed breaking change never enters silently).

**The floor is `6.15`, not the `5.1` this section originally specified, and
the reason is security rather than taste.** That bound was written against a
snapshot in which "the latest release is 5.6.0". By the time Wave 3 was built,
`pip-audit` reported **37 advisories against the 5.x line** (`pypdf 5.9.0`,
the newest release the original `<6` ceiling admitted), whose fixes land
across 6.7.1 through 6.15.0 — so `>=6.15` is the lowest floor that clears all
of them, and no bound under `<6` is shippable at all. CI's audit job would
have failed on the original pin, which is the gate behaving exactly as
ADR-0015 decision 4 intends. The API this module depends on
(`PdfReader`, `page.extract_text(extraction_mode=...)`) is unchanged across
the major, verified by `tests/test_extraction.py` passing against 6.16.1 with
no edit. A version bound justified by a point-in-time "latest release" has a
shelf life; this one expired between design and implementation.

Rejected alternatives:

- **`pdfminer.six`** — also pure Python, also viable. Not chosen because
  `pypdf` has the simpler, more directly documented plain-text extraction API
  for this repo's need (extract text per page, join it — no layout
  reconstruction), and one fewer transitive dependency.
- **`PyMuPDF` (`fitz`)** — faster, wheel-available, but dual-licensed
  AGPL/commercial. An AGPL dependency pulled in by an *optional* extra of an
  MIT-licensed library is a real question a design document should not answer
  by default; rejected to avoid raising it at all.
- **`pdfplumber`** — built on `pdfminer.six`, heavier (adds table/layout
  analysis this repo has no use for), rejected on the same "no more than what
  citation-verifiable plain-text extraction needs" grounds SPEC.md §2's
  determinism requirement already argues for: fewer moving parts is fewer
  ways for extraction to become accidentally non-deterministic.

**HTML: [`beautifulsoup4`](https://pypi.org/project/beautifulsoup4/) (import
name `bs4`), pinned `beautifulsoup4>=4.12,<5`, backed explicitly by the
**stdlib** `"html.parser"`.** Pure Python (its own dependency, `soupsieve`,
is also pure Python — no native code anywhere in the chain), MIT licensed, and
as of this writing the latest release is 4.15.0, requiring Python `>=3.7`.

Two determinism-critical decisions, spelled out because getting either wrong
produces code that *looks* correct and silently isn't:

- **The parser must be passed explicitly: `BeautifulSoup(text, "html.parser")`,
  never `BeautifulSoup(text)`.** Omitting it lets bs4 auto-select the "best
  available" parser, which depends on what else happens to be installed in
  the environment (`lxml`, `html5lib`) — the same bytes could parse
  differently in CI than on an operator's machine, which is precisely the
  non-determinism ADR-0016 decision 2's whole mechanism exists to make
  detectable rather than silent. Pinning `"html.parser"` (the stdlib parser,
  always present, never a second dependency) removes the environment as a
  variable entirely.
- **Decode as UTF-8 before handing text to `BeautifulSoup`, not raw bytes.**
  Passed raw bytes, bs4 runs its own encoding sniffer (`UnicodeDammit`),
  which can consult optional `chardet`/`charset-normalizer` packages if
  present — again making the result a function of what else is installed,
  not just of the file's bytes. This repo already assumes UTF-8 uniformly
  (`FileLoader._read_text`, `resolve_citation`'s `path.read_text("utf-8")`);
  `HtmlExtractor` keeps that assumption rather than introducing a second,
  environment-sensitive decoding path. A file that fails to decode as UTF-8
  raises `IngestionError`, mirroring `FileLoader`'s `UnicodeDecodeError`
  handling exactly.

Version string: `importlib.metadata.version("beautifulsoup4")` — the
**distribution** name. `importlib.metadata.version("bs4")` raises
`PackageNotFoundError`: the import name and the distribution name differ for
this package, and it is an easy, silent-until-runtime mistake to use the
wrong one here. Called out explicitly so no implementer discovers it by
watching `pdf_extractor()`'s sibling work and `html_extractor()`'s not.

Rejected alternatives:

- **`lxml`** — faster, wheel-available on all three CI Pythons, but a native
  C extension (libxml2-backed) where bs4 + `html.parser` is pure Python with
  zero platform-wheel exposure ever. Nothing this repo needs (fault-tolerant
  SGML repair, XPath) justifies taking on a native dependency for "strip
  tags, keep text" (ADR-0016 decision 3's literal scope).
- **`selectolax`** — fast, but its wheel coverage for newer CPython releases
  has historically lagged upstream Python releases, which is precisely the
  "wheel-available on all three CI Pythons" criterion this choice must not
  fail silently on some future CI matrix bump. Not verified against 3.13 at
  the time of this design; rejected on that uncertainty rather than
  confirmed-and-still-rejected.
- **`trafilatura`** — a boilerplate-removal/main-content-extraction library,
  which is a different (and heavier, heuristic-driven) problem than "strip
  tags deterministically." Its whole value proposition — guessing what part
  of the page is the "real" content — is in tension with determinism:
  different versions tune those heuristics, which is a much larger surface
  for two versions to disagree than "same tags stripped the same way."
- **`html2text`** — converts to Markdown syntax rather than plain text, which
  is a content *transformation* beyond ADR-0016 decision 3's literal
  "tags stripped."

### 9.2 `groundkit/extraction.py` — the shared extractor module

**Answering Q2's "protocol an extractor satisfies," "how the set is
populated," and "how `resolve_citation` gets from an identity string to the
extractor."**

A new top-level leaf module, `src/groundkit/extraction.py` — not
`ingestion/protocols.py` alongside `LoaderProtocol`, and not
`retrieval/citations.py` itself. The reasoning is the one already written
down for `identity.py`, quoted because it transfers verbatim: extraction is
"placement follows `groundkit.identity`: a module outside whichever caller
happened to need it first, so sharing it creates no dependency between
ingest and retrieval, and nothing imports it back." An ingest-time PDF/HTML
loader and `retrieval/citations.py`'s re-extraction call need the *literal
same* extractor object — ADR-0016 decision 2's whole point is that "the same
extractor that produced it" is what re-extraction verifies against, so a
separate ingest-side copy and verify-side copy of "what pypdf extraction
means" would be exactly the seam that could silently drift. Putting both the
protocol and the two concrete implementations in one shared module makes
that identity hold *by construction*: there is only one `PdfExtractor` class,
imported by both sides, not two classes that happen to agree today.

This is a deliberate, named departure from this repo's other four protocol
modules (`ingestion/protocols.py`, `index/protocols.py`,
`retrieval/protocols.py`, `providers/protocols.py`), each of which holds a
protocol whose implementations live in the *same* package as the code that
needs them. `ExtractorProtocol` has no such single owning package — that is
exactly `identity.py`'s situation, not theirs, so it gets `identity.py`'s
placement, not theirs.

```python
# src/groundkit/extraction.py
"""Deterministic content extraction shared between ingest and citation
resolution (ADR-0016 decisions 2, 3). Both PdfExtractor and HtmlExtractor are
imported by exactly two callers: the Wave 3 PDF/HTML loaders (to build
Document.content) and retrieval.citations.resolve_citation (to re-derive it
at citation-resolution time). One class per format, used from both sides, is
what makes "the same extractor that produced it" (decision 2) hold by
construction rather than by two implementations staying in sync by hand.

Neither `pypdf` nor `beautifulsoup4` is imported at module level — importing
this module must never require either extra, matching the established
pattern in index/dense.py (_import_lancedb) and retrieval/rerank.py
(_import_reranker_backend).
"""

from __future__ import annotations

import importlib.metadata
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from collections.abc import Mapping

from groundkit.errors import ConfigurationError, IngestionError


@runtime_checkable
class ExtractorProtocol(Protocol):
    """Deterministically converts a file's bytes into extracted text.

    Satisfied by every extractor this build can run — both to produce a
    freshly-ingested Document.content and to re-derive that same text at
    citation-resolution time (ADR-0016 decision 2). Same bytes in, same text
    out: no extractor implementation may consult wall-clock time, randomness,
    environment-dependent library auto-selection, or anything outside the
    file's own bytes and this object's own pinned configuration.
    """

    @property
    def identity(self) -> str:
        """This extractor's identity string: ``"<distribution>/<version>"``.

        Derived at runtime from the installed package's own distribution
        metadata via `importlib.metadata.version` — never hardcoded — so a
        library upgrade changes this string automatically and a mismatch
        against a citation's recorded identity (ADR-0016 decision 2) is
        detected without a separate manual version-tracking step.
        """
        ...

    async def extract(self, path: Path) -> str:
        """Return the deterministic extracted text for the file at ``path``.

        ``path`` is already containment-checked by the caller (the loader's
        own `ensure_within_base`, or `resolve_citation`'s) — this method
        performs no path-safety check of its own.

        Raises:
            IngestionError: The file cannot be read, or its bytes cannot be
                parsed as this extractor's format.
        """
        ...


#: Page separator for joined PDF text. Named and pinned rather than an inline
#: literal — the join behavior is part of what "deterministic" and "the same
#: extractor" (ADR-0016 decision 2) must mean identically at ingest and at
#: resolve time.
_PDF_PAGE_SEPARATOR = "\n\n"

#: Separator BeautifulSoup.get_text() inserts between what were separate
#: elements, and whether surrounding whitespace is stripped. Named for the
#: same reason as _PDF_PAGE_SEPARATOR above.
_HTML_TEXT_SEPARATOR = " "
_HTML_TEXT_STRIP = True


def _import_pypdf() -> Any:
    """Import ``pypdf`` on demand; never at module import time.

    Raises:
        ConfigurationError: ``pypdf`` is not installed (ADR-0016 decision 5).
    """
    try:
        import pypdf
    except ImportError as exc:
        raise ConfigurationError(
            "PDF extraction requires the optional 'pdf' extra: install with "
            "`pip install groundkit[pdf]` (provides pypdf)"
        ) from exc
    return pypdf


def _import_bs4() -> Any:
    """Import ``bs4`` on demand; never at module import time.

    Raises:
        ConfigurationError: ``beautifulsoup4`` is not installed (ADR-0016
            decision 5).
    """
    try:
        import bs4
    except ImportError as exc:
        raise ConfigurationError(
            "HTML extraction requires the optional 'html' extra: install with "
            "`pip install groundkit[html]` (provides beautifulsoup4)"
        ) from exc
    return bs4


class PdfExtractor:
    """Deterministic PDF text extraction via pypdf (ADR-0016 decisions 2, 3).

    Construct via :func:`pdf_extractor`, not directly — that accessor is what
    guarantees ``pypdf`` already imported successfully before ``__init__``
    reads its version, and is what makes every caller share one instance
    (and therefore one ``identity`` computed once, not recomputed per call).
    """

    def __init__(self) -> None:
        self._identity = f"pypdf/{importlib.metadata.version('pypdf')}"

    @property
    def identity(self) -> str:
        return self._identity

    async def extract(self, path: Path) -> str:
        import asyncio

        pypdf = _import_pypdf()
        return await asyncio.to_thread(self._extract_sync, pypdf, path)

    def _extract_sync(self, pypdf: Any, path: Path) -> str:
        """Runs off the event loop. Pinned to plain-text extraction mode —
        never pypdf's default, in case a future pypdf release changes it —
        because pypdf's "layout" mode is documented to depend on system font
        metrics, which would make extraction a function of the machine it
        runs on rather than of the file's bytes alone.
        """
        try:
            reader = pypdf.PdfReader(str(path))
            pages = [page.extract_text(extraction_mode="plain") or "" for page in reader.pages]
        except Exception as exc:
            # pypdf's own exception hierarchy (PdfReadError and friends) is
            # not part of this repo's typed error surface; wrap unconditionally
            # rather than let a third-party exception type escape this
            # module's documented contract.
            raise IngestionError(f"Failed to extract PDF text from {path.name!r}: {exc}") from exc
        return _PDF_PAGE_SEPARATOR.join(pages)


class HtmlExtractor:
    """Deterministic HTML tag-stripping via BeautifulSoup + html.parser
    (ADR-0016 decisions 2, 3). Construct via :func:`html_extractor`, not
    directly — see :class:`PdfExtractor`'s docstring for why.
    """

    def __init__(self) -> None:
        self._identity = f"beautifulsoup4/{importlib.metadata.version('beautifulsoup4')}"

    @property
    def identity(self) -> str:
        return self._identity

    async def extract(self, path: Path) -> str:
        import asyncio

        bs4 = _import_bs4()
        return await asyncio.to_thread(self._extract_sync, bs4, path)

    def _extract_sync(self, bs4: Any, path: Path) -> str:
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise IngestionError(f"Failed to read {path.name!r}: {exc}") from exc
        except UnicodeDecodeError as exc:
            raise IngestionError(f"{path.name!r} is not valid UTF-8: {exc}") from exc
        # "html.parser" pinned explicitly — see §9.1's determinism discussion.
        soup = bs4.BeautifulSoup(raw, "html.parser")
        return soup.get_text(separator=_HTML_TEXT_SEPARATOR, strip=_HTML_TEXT_STRIP)


@lru_cache(maxsize=1)
def pdf_extractor() -> PdfExtractor:
    """The process's one PdfExtractor, built on first use and memoized.

    Raises:
        ConfigurationError: The 'pdf' extra is not installed.
    """
    _import_pypdf()
    return PdfExtractor()


@lru_cache(maxsize=1)
def html_extractor() -> HtmlExtractor:
    """The process's one HtmlExtractor, built on first use and memoized.

    Raises:
        ConfigurationError: The 'html' extra is not installed.
    """
    _import_bs4()
    return HtmlExtractor()


@lru_cache(maxsize=1)
def active_extractors() -> Mapping[str, ExtractorProtocol]:
    """Every extractor this process can actually run, keyed by identity string.

    Replaces retrieval.citations._ACTIVE_EXTRACTOR_IDENTITIES (a plain
    frozenset constant, Waves 1-2) with a lazily-memoized accessor. Called on
    the first `extracted`-class citation resolved in this process — never at
    module import time — so `import groundkit.extraction` (and therefore
    `import groundkit.retrieval.citations`, which imports it) never requires
    either extra.

    Each candidate is probed independently via `pdf_extractor()` /
    `html_extractor()`, catching only ConfigurationError (never a broader
    except): one missing extra must never blank out the other, and a
    genuine bug inside an installed extractor's construction must not be
    silently swallowed and reported as "not registered."
    """
    registry: dict[str, ExtractorProtocol] = {}
    for accessor in (pdf_extractor, html_extractor):
        try:
            extractor = accessor()
        except ConfigurationError:
            continue
        registry[extractor.identity] = extractor
    return registry
```

**Why lazy-and-memoized, not an import-time probe (Q2, and §7's now-resolved
Q2):** an eager probe at module load — `try: import pypdf; except ImportError:
...` at the top of `extraction.py` — would make `import groundkit.extraction`
itself sensitive to which extras happen to be installed, which is exactly the
constraint `index/dense.py` and `retrieval/rerank.py` already rejected for
the same reason (`_import_lancedb`'s docstring: "Never called at module
import time... so `import groundkit.index.dense` never requires the optional
`dense` extra"). Since `retrieval/citations.py` (and every text-class
citation resolution) must keep working on a base install with neither extra,
`extraction.py` follows the same discipline its two precedents already
established. `@lru_cache(maxsize=1)` gives "compute once, lazily" without a
hand-rolled `if _cache is None:` guard, and reflects only what was
importable *the first time it was asked* — consistent with this being a
per-process, not per-request, question (the same tradeoff `pdf_extractor`/
`html_extractor`/`active_extractors` all make identically).

**`resolve_citation`'s answer to "identity string → extractor":**
`extraction.active_extractors().get(citation.extractor)` — a dict lookup, not
a scan. `_ACTIVE_EXTRACTOR_IDENTITIES` (Waves 1-2's frozenset, membership
only) is deleted; every reference to it in `retrieval/citations.py` becomes
`extraction.active_extractors()` (membership: `citation.extractor not in
extraction.active_extractors()`; the identity list for the refusal message:
`sorted(extraction.active_extractors())`, which iterates dict keys
identically to how it iterated frozenset members before — the refusal
message's exact wording is otherwise unchanged, so
`tests/test_source_class.py::test_resolve_refuses_an_extracted_citation`
(pinned to `match="extracted"`, extractor `"pdf-x/1"`, which will never
coincidentally collide with a real `"pypdf/…"` or `"beautifulsoup4/…"`
identity) stays green under this change with no edit).

### 9.3 `resolve_citation`'s Wave 3 restructuring

The Wave 2 comment left in `retrieval/citations.py` — "Wave 3 adds the
re-extraction call here: re-run the active extractor over `citation.source`,
slice `[start_offset:end_offset]`, and return it — the same shape
`resolve_citation` already has for `text` below" — is accurate about the
*outcome* but, read literally as "insert code at this exact point," would
duplicate the offset/drift check across two branches. This section
supersedes that literal reading with the actual shape: **the containment
check and the offset/drift check are shared between `text` and `extracted`**
(both resolve a local path under `allowed_base_dir`); only the "how do we
turn that path into text" step differs.

```python
def _slice_verified_span(text: str, citation: Citation) -> str:
    """Slice `citation`'s offsets out of `text`, or raise the drift verdict.

    Shared by every branch that has *obtained* text by whatever means its
    source class requires (a raw read, a fresh extraction, or a snapshot
    read) — once `text` exists, checking and slicing the offsets is
    identical regardless of how it was produced.

    Raises:
        RetrievalError: `citation.end_offset` exceeds `len(text)` —
            `verdict="drifted"`.
    """
    if citation.end_offset > len(text):
        raise RetrievalError(
            f"Cited span [{citation.start_offset}:{citation.end_offset}] exceeds "
            f"source length ({len(text)}) for {citation.source!r} — source changed "
            "since indexing",
            verdict="drifted",
        )
    return text[citation.start_offset : citation.end_offset]


async def resolve_citation(
    citation: Citation, allowed_base_dir: Path, *, snapshot_dir: Path | None = None
) -> str:
    if citation.source_class == "snapshot":
        return await _resolve_snapshot(citation, snapshot_dir)  # §10.2

    # text and extracted share containment; they differ only in how `text`
    # is obtained from the (already-contained) path.
    try:
        path = ensure_within_base(citation.source, allowed_base_dir)
    except ValueError as exc:
        raise RetrievalError(str(exc), verdict="unresolvable") from exc

    if citation.source_class == "extracted":
        extractor = extraction.active_extractors().get(citation.extractor)
        if extractor is None:
            raise RetrievalError(
                f"cannot resolve an 'extracted' citation for {citation.source!r}: "
                f"the extractor identity recorded at ingest ({citation.extractor!r}) "
                "is not active in this build "
                f"({sorted(extraction.active_extractors()) or 'none registered'}). "
                "Re-deriving these offsets needs the exact extractor that produced "
                "them; refused rather than guessing (ADR-0016 decision 2).",
                verdict="unresolvable",
            )
        try:
            text = await extractor.extract(path)
        except IngestionError as exc:
            raise RetrievalError(
                f"cannot resolve an 'extracted' citation for {citation.source!r}: "
                f"re-extraction with {extractor.identity} failed: {exc}",
                verdict="unresolvable",
            ) from exc
    else:  # "text"
        try:
            text = await asyncio.to_thread(path.read_text, "utf-8")
        except OSError as exc:
            raise RetrievalError(
                f"Cannot read cited source {citation.source!r}: {exc}", verdict="unresolvable"
            ) from exc
        except UnicodeDecodeError as exc:
            raise RetrievalError(
                f"Cited source {citation.source!r} is not valid UTF-8: {exc}",
                verdict="unresolvable",
            ) from exc

    return _slice_verified_span(text, citation)
```

The refusal message's wording is preserved verbatim from Waves 1-2 except for
substituting `extraction.active_extractors()` for
`_ACTIVE_EXTRACTOR_IDENTITIES` — `match="extracted"` in the existing
regression test is unaffected.

Two notes on what actually landed, since the block above reaches into Wave 4
to show the finished shape. The `snapshot_dir` keyword and `_resolve_snapshot`
are **§10.2's**, not Wave 3's: the shipped signature is still
`resolve_citation(citation, allowed_base_dir)` and the `snapshot` branch still
raises its Wave-2 refusal, so Wave 4 adds the parameter when it adds the
thing the parameter is for. And one behaviour change this restructuring
carries that is easy to miss: the containment check now runs **before** the
extractor lookup, where Waves 1-2 refused an unknown identity first. An
`extracted` citation whose source escapes `allowed_base_dir` is therefore now
reported as a path violation rather than as an extractor-identity problem —
the right order, since containment is the security boundary, and pinned by
`test_containment_is_checked_before_the_extractor_lookup`.

### 9.4 What "extracted" verification does, and what happens when re-extraction fails

Per ADR-0016 decision 1's table: **re-extract, slice, compare.**
`resolve_citation` re-runs the *same* extractor instance (by identity match,
enforced in §9.3) over the source file, exactly as it would at ingest time,
then applies the identical offset/drift check `text` uses. There is no
separate "compare" step beyond that: the caller (`fetch_chunk`) already does
byte comparison against the stored chunk content once `resolve_citation`
returns text, exactly as it does for `text`-class citations today
(`service/tools.py`'s existing `resolved == chunk.content` check, untouched
by this wave).

**When re-extraction itself fails** — a corrupted PDF, a file that no longer
parses under the recorded extractor version, a truncated download — the
extractor raises `IngestionError` (per `ExtractorProtocol.extract`'s
contract), and `resolve_citation` catches it and re-raises as
`RetrievalError(verdict="unresolvable")`, symmetric with how a `text`
citation's `OSError`/`UnicodeDecodeError` is handled. This is deliberately
**not** `"drifted"`: `drifted` means text was successfully obtained and
disagrees in length with what was indexed; a parse failure means the check
never got far enough to compare anything, which is what `unresolvable`
means (ADR-0016 decision 6). No fourth verdict is introduced, matching
decision 6 exactly.

### 9.5 Ingest-time missing-extra failure

Separate from §9.4 (which is about *resolving* an already-ingested
citation), this is about *ingesting* a `.pdf`/`.html` file when its extra
isn't installed. Per ADR-0016 decision 5 (citing ADR-0015's command/option
rule): `grk ingest` itself must keep working; a `.pdf`/`.html` file is an
input the command may be given, so only encountering one triggers the check.
The Wave 3 PDF/HTML loaders' `.load()` methods call `extraction.pdf_extractor()`
/ `extraction.html_extractor()` directly (not `active_extractors()`, which
would silently skip an unavailable format rather than naming it) — so a
`.pdf` file with the `pdf` extra absent raises `ConfigurationError` naming
`pip install groundkit[pdf]`, propagating out of `.load()` and failing that
one ingest with a message an operator can act on, exactly as `dense`/
`rerank` do today for their own missing extras (ADR-0016 decision 5's
literal words — note this differs from the *exact classes* `dense`/`rerank`
actually raise today, `StorageError` and `RerankerNotConfiguredError`
respectively, not `ConfigurationError`; ADR-0016's own prose calls them
"exactly as," which is imprecise about the literal class name but not about
the shape — typed, existing, fail-closed, naming the install string. This
spec follows ADR-0016's literal `ConfigurationError` instruction rather than
the two precedents' literal classes, since ADR-0016 is the more specific,
directly-on-point authority for this wave.)

`pyproject.toml` gains two new extras, matching the existing bound style:

```toml
pdf = [
  "pypdf>=5.1,<6",
]
html = [
  "beautifulsoup4>=4.12,<5",
]
```

Neither is mirrored into the dev group's *hard requirement* the way `dense`
is — both are pure-Python and small, so there is no `rerank`-style
multi-gigabyte reason to keep them out of CI's default job, but the sole
proof of each backend must still be a non-`continue-on-error` job per
SPEC.md §3; which job that is (a new gated suite vs. the default suite with
the extras installed) is left to the implementer, since it depends on
`uv.lock`/CI wiring this spec does not own.

### 9.6 What is not decided here

**Multi-loader dispatch is not designed here.** `IngestionPipeline` and
`Indexer` are wired for exactly one `loader: LoaderProtocol` per run today
(`ingestion/pipeline.py`, `Indexer.__init__`); routing `.md`/`.txt` to
`FileLoader`, `.pdf` to a PDF loader, and `.html` to an HTML loader within
one `grk ingest` invocation needs either a dispatching composite loader or a
CLI-level per-extension routing decision, neither of which ADR-0016 or the
four questions this section answers speaks to. Flagged explicitly rather
than guessed at, per this document's own standing rule (§1's closing
sentence) that an undecided question gets named, not silently resolved.

## 10. Wave 4 design — snapshot storage and reaching it from `resolve_citation`

### 10.1 Where snapshots live, and how the path is derived (Q1)

**The problem, stated precisely:** `resolve_citation(citation, allowed_base_dir)`
receives one containment root, `allowed_base_dir` — the corpus root
(`ctx.base_dir` in the service, `corpus_dir` in `evals/echo.py`). A
`snapshot` document's bytes live under the **index directory**
(`ctx.index_dir`, a wholly separate, non-nested directory an operator points
at independently — confirmed at `cli.py`'s `_cmd_serve`, which validates
`--index-dir` and `--base-dir` as two unrelated paths with no containment
relationship asserted between them). `resolve_citation` has no reference to
the index directory today, and `Citation` carries no `collection` field
either — so the resolver cannot, on its own, know *which collection's*
snapshot store to look in even if it had the index directory.

**Decision: a new keyword-only parameter, `snapshot_dir: Path | None = None`,
supplied by the caller** — the *per-collection* snapshot containment root,
already resolved by whoever knows the collection name, exactly mirroring how
`allowed_base_dir` is already computed by the caller today rather than
derived inside `resolve_citation` from some larger config object. This keeps
`resolve_citation` a pure function of what it is given (this repo's
"deterministic core" principle, SPEC.md §2) and — the deciding factor over
the alternatives below — needs no change to `Citation`, `RetrievalResult`,
or `SCHEMA_VERSION` (already 3; Wave 4 does not bump it again).

**Path shape:** `<index_dir>/<collection>.snapshots/<document_id>` — a
directory per collection, sibling of `<collection>.sqlite3` and
`<collection>.lance`, matching that existing `<index_dir>/<collection>.<suffix>`
convention exactly (`.snapshots`, plural noun, because it holds many files
rather than being one file/table itself). One file per document, named by
`document_id` — the join key every other part of this system already uses
(`chunks.document_id REFERENCES documents(document_id)`, `Citation.document_id`)
— rather than inventing a second identifier (a content hash, say) that would
need its own persistence and its own consistency argument for no benefit:
`document_id` is already unique per collection (`documents.document_id` is
the primary key) and is already on every `Citation`.

Two small shared, pure functions carry this convention — defined once so
the write side (Wave 4's URL loader) and the read side (`resolve_citation`)
cannot independently drift on what the path means, the same "shared, not
duplicated" argument §9.2 makes for the extractor classes:

```python
# src/groundkit/snapshots.py
"""Where a URL-ingested document's local snapshot lives on disk (ADR-0016
decision 4). Pure path arithmetic, no I/O — shared by the Wave 4 URL loader
(which writes) and retrieval.citations.resolve_citation (which reads), so
the naming convention is asserted from exactly one place.
"""

from __future__ import annotations

from pathlib import Path


def snapshot_dir_for(index_dir: Path, collection: str) -> Path:
    """The containment root for one collection's stored snapshots.

    Sibling of `<index_dir>/<collection>.sqlite3` and `.lance`, following the
    same per-collection-suffix convention.
    """
    return index_dir / f"{collection}.snapshots"


def snapshot_path_for(snapshot_dir: Path, document_id: str) -> Path:
    """The snapshot file for one document within its collection's snapshot_dir."""
    return snapshot_dir / document_id
```

**Containment is still enforced**, exactly as it is for `citation.source`
today: `document_id` is a plain string field with no character-class
restriction the way `collection` names have
(`index/metadata.py::_COLLECTION_NAME_PATTERN`), so a hand-constructed or
malicious `document_id` (e.g. `"../../etc/passwd"`) must not be trusted
implicitly. `resolve_citation` calls `ensure_within_base(snapshot_path_for(
snapshot_dir, citation.document_id), snapshot_dir)` before reading — the same
barrier pattern used for `citation.source`, applied here per this repo's
"path containment everywhere a path is read" rule (SPEC.md §7,
`utils/path_safety.py`'s module docstring). This is defense-in-depth
regardless of what the Wave 4 loader itself guarantees about `document_id`
shape (it should leave `document_id` at `Document`'s own
`uuid.uuid4().hex` default, exactly as `FileLoader` already does — never
hand-supplying one derived from the URL).

**Call sites — verified against the tree, correcting a premise in the design
brief:**

- **`service/tools.py::handle_fetch_chunk` — must change.** It already has
  everything needed: `ctx.index_dir` (on `ServiceContext`) and
  `request.collection` (on `FetchChunkRequest`, defaulting to `"default"`).
  It computes `snapshot_dir = snapshots.snapshot_dir_for(ctx.index_dir,
  request.collection)` and passes `snapshot_dir=snapshot_dir` on its existing
  `resolve_citation(citation, ctx.base_dir)` call. No `ServiceContext` or
  request-schema change needed.
- **`evals/echo.py` — does not need to change.** `_evaluate_echo_case` calls
  `resolve_citation(positive_citation, allowed_base_dir)` positionally; the
  new parameter is keyword-only with a default of `None`, so this call keeps
  compiling and behaving identically — matching the exact "keyword-only and
  defaulted so no existing caller breaks" pattern Wave 1 already used for
  `source_class`/`extractor` (§4.1). It is also, independently, never
  reachable through the `snapshot` branch: `build_echo_case` always
  constructs its two `RetrievalResult`s at `source_class`'s `"text"` default
  (verified — `evals/echo.py`'s `RetrievalResult(...)` calls at lines 208 and
  217 never set `source_class`), so it never needs one.
- **`retrieval/search.py` — does not call `resolve_citation` at all, and
  therefore needs no change.** This corrects the design brief's premise that
  all three modules call it: `grep -rn "resolve_citation" src/groundkit`
  returns hits only in `retrieval/citations.py` itself (the definition),
  `service/tools.py`, and `evals/echo.py`. `Retriever._resolve` builds
  `RetrievalResult`s (which carry an *unresolved* `citation` computed field)
  but never reads a citation back — resolution is `fetch_chunk`'s job,
  invoked separately by a client on a hit it intends to quote, per
  `handle_fetch_chunk`'s own docstring ("this is the step a client performs
  on a hit it intends to quote").
- **`retrieval/citations.py::verify_citation`** — the thin wrapper — gains
  the identical keyword-only passthrough:

  ```python
  async def verify_citation(
      citation: Citation,
      expected_content: str,
      allowed_base_dir: Path,
      *,
      snapshot_dir: Path | None = None,
  ) -> bool:
      return (
          await resolve_citation(citation, allowed_base_dir, snapshot_dir=snapshot_dir)
          == expected_content
      )
  ```

  Not currently called anywhere in production (`grep -rn "verify_citation"
  src/groundkit` shows only its own definition and the `__init__.py`
  re-export), so this is a signature-consistency change with zero behavioral
  call sites to verify today.

**Alternatives considered and rejected:**

- **Persist an absolute snapshot path as a new `documents` column.** This is
  the portability hazard the design brief names explicitly: an absolute path
  baked into a row is wrong the moment the collection's SQLite file is
  copied to a different machine or a different `--index-dir`, which is a
  normal operation for this repo (SPEC.md's local-first, portable-index
  posture). It would also force another `SCHEMA_VERSION` bump (4), with
  ADR-0004 decision 5's delete-and-re-ingest consequence, for information
  that is fully *derivable* from `index_dir` + `collection` + `document_id`
  — all three already available wherever a citation is resolved. Deriving it
  is strictly better: no new column, no new schema version, and the path is
  always correct relative to wherever the index currently lives.
- **Widen `Citation`/`RetrievalResult` with a `collection` field.** Would let
  `resolve_citation` derive `snapshot_dir` internally from `index_dir` alone
  (still a needed new parameter) plus the citation's own collection. Rejected
  as disproportionate: it touches `contracts.py` (every `Citation`/
  `RetrievalResult` construction site across the whole codebase, including
  every test fixture) to solve a problem the `snapshot_dir` parameter already
  solves with zero contract changes. A collection is already implicit in
  "which store this citation came from," and every existing caller that
  would supply it already knows it independently (`handle_fetch_chunk` has
  `request.collection` right there) — so widening the contract buys nothing
  the parameter didn't already buy, at a much larger blast radius.
- **Have `resolve_citation` accept `index_dir: Path | None` instead of a
  pre-computed `snapshot_dir`, and derive the `.snapshots` suffix internally.**
  Considered and rejected in favor of the pre-computed form for the same
  reason `allowed_base_dir` itself is pre-computed by the caller rather than
  derived inside `resolve_citation` from some larger `ServiceContext`-shaped
  object: `resolve_citation` stays a function of exactly the paths it needs,
  not of an index-directory-plus-collection-name pair it would have to
  re-derive the convention from. It also avoids `resolve_citation` needing a
  `collection: str | None` parameter in addition to `snapshot_dir` — the
  collection name folds into the one `Path` the caller already computed.

### 10.2 `resolve_citation`'s snapshot branch — and a call this document makes explicitly

**A refinement ADR-0016 does not resolve at the level of detail needed to
implement it, stated as a decision with reasoning, per this document's own
"say so explicitly" rule (§1):**

ADR-0016 decision 4 says a `snapshot` document's local copy is "the fetched
bytes." Read literally, that is the raw HTTP response body. But decision 3
requires HTML to have its tags stripped for retrieval quality *regardless of
where the HTML came from* — a URL that serves an HTML page has exactly the
same "BM25 scores `<div>` as a term" problem a local `.html` file does. If
the stored snapshot were the *raw* response body while `Document.content`
were the *tag-stripped* text, verification would need to re-run
`HtmlExtractor` over the snapshot at resolve time — reintroducing an
extractor-identity dependency into the `snapshot` class that decision 4's
own "the snapshot is what verifies" design was meant to avoid, and, worse,
requiring `Citation` to carry an `extractor` field for a `snapshot`-class
document, which `contracts.py`'s existing model validator currently forbids
outright (`extractor is only meaningful for an 'extracted' document`) —
a `contracts.py` change this spec's Wave 3/4 scope should not need.

**This spec's call: the snapshot is a copy of `Document.content` itself — the
fully-processed text that was actually indexed — not the raw wire bytes.**
Concretely: if the fetched resource is HTML-shaped, the Wave 4 URL loader
extracts it through the identical `extraction.html_extractor()` Wave 3
built (raising the identical `ConfigurationError` if the `html` extra is
absent — Wave 3 and Wave 4 compose here rather than duplicating the check),
*before* writing the snapshot; if it's already plain text, no extraction
step runs and the decoded text is written as-is. Either way, the byte-for-byte
same string that becomes `Document.content` is what lands at
`snapshot_path_for(snapshot_dir, document_id)`. The document's
`source_class` stays `"snapshot"` — never `"extracted"` — and `extractor`
stays `None`, exactly as `contracts.py`'s existing validator already
requires: `source_class` encodes *where verification looks* (a local
original file, a re-run extractor, or a local snapshot copy), not whether
extraction happened somewhere upstream in the ingest pipeline. This keeps
`SourceClass` genuinely three-way exclusive with **zero `contracts.py`
changes**, and — the concrete payoff — makes `resolve_citation`'s `snapshot`
branch a pure read-and-compare with no extractor dependency at all, exactly
as simple as the `text` branch:

```python
async def _resolve_snapshot(citation: Citation, snapshot_dir: Path | None) -> str:
    if snapshot_dir is None:
        raise RetrievalError(
            f"cannot resolve a 'snapshot' citation for {citation.source!r}: no "
            "snapshot_dir was supplied to resolve_citation. A snapshot citation "
            "verifies against the local copy URL ingestion stores at ingest time "
            "(ADR-0016 decision 4), under <index_dir>/<collection>.snapshots/ — "
            "the caller must compute that directory (snapshots.snapshot_dir_for) "
            "and pass it. source is a URL, not a path, so it is never passed to "
            "ensure_within_base, which would resolve it as a relative path under "
            "the current directory rather than reject it as not-a-path.",
            verdict="unresolvable",
        )
    try:
        snapshot_path = ensure_within_base(
            snapshots.snapshot_path_for(snapshot_dir, citation.document_id), snapshot_dir
        )
    except ValueError as exc:
        raise RetrievalError(str(exc), verdict="unresolvable") from exc

    try:
        text = await asyncio.to_thread(snapshot_path.read_text, "utf-8")
    except OSError as exc:
        raise RetrievalError(
            f"cannot read the local snapshot for {citation.source!r}: {exc}",
            verdict="unresolvable",
        ) from exc
    except UnicodeDecodeError as exc:
        raise RetrievalError(
            f"the local snapshot for {citation.source!r} is not valid UTF-8: {exc}",
            verdict="unresolvable",
        ) from exc

    return _slice_verified_span(text, citation)  # §9.3
```

**Consequence, stated rather than left implicit:** unlike `extracted`,
`snapshot` verification at resolve time never re-runs an extractor and
therefore has no "re-extraction failed" case at all — extraction, if the
fetched resource needed any, already happened exactly once, at ingest time,
and its result is frozen into both the chunk rows and the snapshot file. A
`snapshot` citation can only fail for `text`-shaped reasons: the snapshot
file is missing, unreadable, or not UTF-8 (`unresolvable`), or its length no
longer covers the cited offsets (`drifted` — the snapshot itself would have
to have been altered or deleted out from under the index, since URL
ingestion never rewrites a snapshot after the fact within one ingest).

**Rejected alternative:** store the raw HTTP response body verbatim and
carry an `extractor` on `snapshot` documents too (widening `contracts.py`'s
validator to permit the `("snapshot", extractor=...)` pairing). Rejected for
the reasons stated above — it reintroduces an identity dependency and a
`contracts.py` change that storing the already-processed content avoids
entirely, for no compensating benefit: nothing in ADR-0016 or SPEC.md asks
for the raw wire bytes to be recoverable, only for the *indexed* text to be
verifiable against a stable local copy, which storing `content` itself
satisfies precisely.

### 10.3 The fetch itself: SSRF guard, redirect refusal, and the ordering constraint

Restating ADR-0016 decision 4's three non-negotiables with the exact
mechanics, since "guard the fetch" and "refuse redirects" are correct at the
ADR's level of abstraction but not yet precise enough to implement
identically three times:

```python
async def load(self, source: str) -> list[Document]:
    # (a) Classify BEFORE any path-safety call — source is a URL, never
    #     handed to ensure_within_base. This ordering is decision 4(c)
    #     itself, not a detail: os.path.realpath resolves a URL string as a
    #     RELATIVE PATH, so containment could silently pass on a string that
    #     was never a path at all.
    validate_endpoint_shape(source)  # utils/url_safety.py — sync, no DNS

    # (b) The SSRF guard, per request (not once at construction — this
    #     loader has no fixed endpoint the way an embedder's base_url is
    #     fixed; every `load()` call targets a different host). The Ollama
    #     private-endpoint allowance is a provider-side ClassVar
    #     (`_allow_private_endpoint`) that this loader never sets and never
    #     reads — allow_private_endpoint=False is the only value passed here,
    #     unconditionally.
    await ensure_safe_endpoint(source, allow_private_endpoint=False)

    # (c) Redirects refused, not merely un-followed: a 3xx is an error, not
    #     a value the caller silently gets zero content for.
    async with httpx.AsyncClient(follow_redirects=False) as client:
        response = await client.get(source)
    if 300 <= response.status_code < 400:
        raise IngestionError(
            f"refusing a redirect ({response.status_code}) fetching a URL source "
            "(ADR-0016 decision 4) — the destination is never followed automatically"
        )
    response.raise_for_status()

    # ... decode, extract if HTML-shaped (§10.2), write the snapshot, build
    # the Document at source_class="snapshot".
```

No `pdf`/`html`-shaped extra gates this loader's *existence* — `httpx` is
already a base dependency (ADR-0016 decision 5, ADR-0015's command/option
rule) — only the extraction step inside it, if the fetched resource turns
out to be HTML, defers to `extraction.html_extractor()` and can raise the
identical `ConfigurationError` §9.5 already specifies. A URL serving a
`.pdf`-shaped response would do the same through `extraction.pdf_extractor()`.

**Noted, not designed here:** `utils/url_safety.py`'s module docstring
currently opens with "URL safety helpers for validating outbound
**embedding-provider** endpoints." Wave 4 makes it that module's second
caller, so that opening line becomes stale the moment this loader exists —
worth a one-line docstring update in the same change, though it is not this
spec's file to edit.

**Not decided here:** how `grk ingest` accepts a URL as its `path` argument
at the CLI layer (today `_cmd_ingest` unconditionally does `Path(args.path)`
and branches on `path.is_dir()`, neither of which is meaningful for a URL
string) is a CLI-wiring question this spec's four questions do not cover,
named explicitly per §9.6's same standing rule rather than guessed at.

## 11. Design summary (Waves 3–4, at a glance)

| Question | Answer | Section |
|---|---|---|
| How does `resolve_citation` reach a snapshot? | New keyword-only `snapshot_dir: Path \| None = None` param, caller-computed via `snapshots.snapshot_dir_for(index_dir, collection)`; snapshot file is `snapshot_dir / document_id`, containment-checked like any other read path. No `Citation`/`RetrievalResult`/`SCHEMA_VERSION` change. | §10.1 |
| Extractor registry shape | `ExtractorProtocol` (`identity: str`, `async extract(path) -> str`) + concrete `PdfExtractor`/`HtmlExtractor` in a new shared leaf module `groundkit/extraction.py` (mirrors `identity.py`'s placement). Populated by `active_extractors()`, a lazily-memoized accessor — never an import-time probe — that independently try/excepts each format. | §9.2 |
| PDF library | `pypdf>=5.1,<6` — pure Python, BSD-3-Clause, `importlib.metadata.version("pypdf")`. | §9.1 |
| HTML library | `beautifulsoup4>=4.12,<5` with the stdlib `"html.parser"` pinned explicitly, UTF-8-decoded before parsing. `importlib.metadata.version("beautifulsoup4")` — note the distribution name differs from the import name (`bs4`). | §9.1 |
| What `extracted` verification does | Re-extract via the identity-matched extractor, then the same offset/drift check `text` uses (`_slice_verified_span`, shared). | §9.3, §9.4 |
| What happens when re-extraction fails | `IngestionError` from the extractor → `RetrievalError(verdict="unresolvable")` — never `"drifted"`, since no comparison text was obtained to disagree with anything. | §9.4 |
| What a `snapshot` verifies against | A local copy of `Document.content` itself (post-extraction, if any was needed) — not the raw wire bytes — so resolution is a plain read-and-compare with no extractor dependency, ever. | §10.2 |

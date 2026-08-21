# Backlog

Remediation backlog opened 2026-08-18, after the v0.1.0 release, from a nine-dimension
review of `c9647eb` (the commit tag `v0.1.0` points at). Every item below is therefore a
**shipped** defect or gap, not a release gate.

This file is a working backlog, not a report: it is edited in place as items land. It
lives at the repo root rather than under `docs/` deliberately — `mkdocs.yml` sets
`validation.omitted_files: warn` and CI builds `--strict`, so any new page under `docs/`
that is not also added to `nav` fails the build. Root-level documents
(`SPEC.md`, `KNOWN_LIMITATIONS.md`, `CHANGELOG.md`) sit outside `docs_dir` and carry no
such obligation.

`SPEC.md` remains the contract and `KNOWN_LIMITATIONS.md` remains the honest record of
what is presently broken. This file records only what is **planned to change**, and each
item is deleted from here — not marked done and left forever — once it has landed and its
consequence is reflected in whichever of those two documents owns it.

---

## How to work an item

1. Pick the lowest-numbered `todo` item whose `Depends on` are all `done`.
2. Set its status to `wip` and branch: `fix/gk-NNN-<slug>` (or `docs/`, `chore/`, `test/`
   to match the change type).
3. **Write the regression test and watch it fail first.** This is not optional here —
   `SPEC.md` §8 requires it, and every defect in Phase A/B below is on a path where a
   green suite cannot distinguish a real regression test from a decorative one:

   ```bash
   git stash push <the source file(s) you fixed>
   uv run pytest -k <your new test> -v      # MUST fail
   git stash pop
   uv run pytest -k <your new test> -v      # MUST pass
   ```

   Record both directions in the PR body. A test that passes against the unfixed code is
   testing something other than the defect.
4. Run the gates before opening the PR — all of them, not a subset:

   ```bash
   uv run ruff check . && uv run ruff format --check .
   uv run mypy
   uv run pytest --cov && uv run coverage report
   ```

   The core-subset gate is a second, separate `coverage report --include=...` step in
   `ci.yml`, parsed from `pyproject.toml`'s `[tool.groundkit.coverage].core_subset` at run
   time. Check it too if you touched anything in that list.
5. Update `KNOWN_LIMITATIONS.md` if the item changes what is broken, `CHANGELOG.md` under
   `[Unreleased]` if it changes behaviour, and **delete the item from this file**.

### Rules that apply to every item

- **An ADR is required before deviating from `SPEC.md`.** Items below marked `ADR: yes`
  need one first. Filename is `docs/adr/ADR-NNNN-<slug>.md`, four digits, one decision,
  alternatives recorded — and it must be added to **both** `docs/adr/index.md` and
  `mkdocs.yml`'s `nav`, or the strict docs build fails. Claim the number by checking
  `git status --untracked-files=all docs/adr/` as well as `origin`, since a number can be
  held by an uncommitted file.
- **No ungenerated numbers in any document.** Metrics come from generated eval artifacts
  or live badges, or are omitted. This applies to anything you add to `README.md`,
  `SPEC.md` or the docs site while closing an item.
- Effort is a coarse bucket, never an estimate presented as a measurement:
  **XS** under an hour · **S** a few hours · **M** a day or two · **L** about a week.
- `Verified` marks findings re-confirmed first-hand against source, the live GitHub API,
  the installed SDK, or by executing the code — as opposed to reported by a review agent
  and not independently checked.

### Status values

`todo` · `wip` · `blocked` · `done` (delete on next pass) · `wontfix` (move to the
Declined section with the reason)

---

## Index

| ID | Item | Sev | Phase | Effort | Status | Depends on |
|---|---|---|---|---|---|---|
| GK-020 | The staleness cache stops working during an ingest | MED | G | L | todo | — |

---

## Phases A, B and C — closed 2026-08-18

All six items (GK-001, GK-003, GK-004, GK-005, GK-006, GK-007) landed together on
`fix/backlog-phase-abc` and are deleted from this file per the rule above. Their
consequences now live where those documents own them: ADR-0024 and `SECURITY.md` for the
`Host` validation that closed GK-001, `CHANGELOG.md` under `[Unreleased]` for every
behaviour change, `KNOWN_LIMITATIONS.md` for the five residuals the work exposed, and
`SPEC.md` §9 for the Phase 7 status GK-005 corrected.

Two of the three exit criteria were met as written. The third — Phase A's "`SECURITY.md`
accurate about what the bind does and does not protect" — was met only after adversarial
re-verification: the first `Host` fix branched on *is a loopback literal* while its own ADR
and `SECURITY.md` justified the unrestricted branch with *once the socket is routable*, so
`grk serve --host localhost --allow-remote-access` bound a non-routable socket with
validation disabled, reinstating the CRITICAL on the one bind where it is exploitable. The
predicate now resolves the host and fails closed. Recorded here because the lesson is
reusable and outlives the items: a security control whose stated premise and actual
predicate differ passes every test written against the predicate.

---

## Phase D — closed 2026-08-18

GK-008, GK-009 and GK-010 landed together on `fix/backlog-phase-d` and are deleted from
this file per the rule above. Their consequences live in `CHANGELOG.md` under
`[Unreleased]`, `KNOWN_LIMITATIONS.md` for the residuals each leaves standing,
`SECURITY.md` for the concurrency cap's scope, and `infra/k8s/deployment.yaml`, whose
memory-limit comment depended on the cap that did not exist.

Both exit criteria were met, one of them differently than expected. "No single request can
stall every other" holds: BM25's whole-corpus scan and `InMemoryVectorStore`'s were the
last CPU-bound steps still running on the loop, and both now dispatch to a worker. "A
concurrent writer waits rather than failing" was **already true** — GK-010 asserted
SQLite's own `busy_timeout` default of 0 was in force, but `sqlite3.connect` passes
`timeout=5.0` unless told otherwise and `_connect` never told it otherwise. Measured, a
contending write waited five seconds and then raised. The real defect was provenance, not
behaviour: the value was inherited from a stdlib default, named nowhere, tested nowhere,
and one keyword away from being replaced by an unrelated edit. It is now a module constant
with an injection test. What still does not exist is a retry, and that is recorded in
`KNOWN_LIMITATIONS.md` rather than quietly closed.

The reusable lesson repeats Phase A's, from the other direction: there, a fix's predicate
disagreed with its own stated premise; here, a finding's premise disagreed with the running
code. Both were only visible by executing the thing rather than reading about it.

---

## Phase E — closed 2026-08-19

GK-011, GK-012, GK-013 and GK-014 landed together on `fix/backlog-phase-e` and are deleted
from this file per the rule above. All four were coverage-only: the code each one covers
was already correct, so none of the four changes production behaviour, and neither
`CHANGELOG.md` nor `KNOWN_LIMITATIONS.md` owns any consequence of this phase. Each new test
was instead demonstrated the way `tests/test_service_api.py` and `tests/test_service_errors.py`
already demonstrate a guard with no pre-fix version to revert to: by injecting the exact
violation it exists to catch and watching it fail for that reason, then restoring. GK-011's
"three resources closed" criterion narrowed under execution to two: when `build_chat` is
what raises, `chat` itself was never built, so `_maybe_aclose(None)` is a no-op by
construction rather than a third closeable resource — what the test proves for `chat` is
that the `finally` block still reaches that no-op rather than being skipped entirely.

Both exit criteria were met. Every CLI verb is now driven through `main()` at least once,
and the dark surface named by all four items — the `--chat-*` path end to end, the
unauthenticated 500 boundary, and `add_chunks`' orphaned guard copy — is dark no longer.

---

## Phase F — closed 2026-08-19

GK-015, GK-016, GK-017 and GK-029 landed together on `fix/backlog-phase-f` and are
deleted from this file per the rule above. GK-016's consequence is structural (`Retriever`
now depends on `LexicalIndexProtocol`, unblocking Phase G); GK-029's lives in ADR-0025,
`SECURITY.md` and `KNOWN_LIMITATIONS.md` (one of the four `Host` residuals closed); GK-015
and GK-017 change no behaviour a doc describes.

Both exit criteria were met. One item grew in scope under execution: `assert_signature_parity`
(`tests/test_protocol_conformance.py`) could not see a classmethod member at all —
`inspect.isfunction` is `False` for a raw `classmethod` object, the same blind spot
`__call__` once was before PR #23 — so `LexicalIndexProtocol.from_store` would have passed
conformance vacuously. The helper itself was extended first (classmethod unwrapping, a
kind-mismatch check, a return-annotation exemption specific to construction factories) and
proven against the unfixed helper before `LexicalIndexProtocol` was written, mirroring the
same helper's own `__call__` fix in PR #23.

GK-017's fix was adversarially reviewed (as instructed) and the review found three real gaps
the initial fix missed: `json.dumps`'s stdlib default accepts `NaN`/`Infinity` and would have
let a non-finite metadata float pass construction as "JSON-serializable" while still breaking
the REST/MCP read surface; a sufficiently deep metadata value overflows `json.dumps`'s
recursion rather than raising `TypeError`, and pydantic does not auto-wrap a bare
`RecursionError` the way it does `ValueError`, so it would have escaped every typed-error
boundary a caller has. Both are fixed (`allow_nan=False`; `RecursionError` caught
alongside `TypeError`/`ValueError`) and regression-tested. A third, informational finding —
`copy.deepcopy` preserves a tuple as a tuple where a real JSON round-trip would return a
list — has no live caller today and is left as a documented comment rather than fixed
pre-emptively.

GK-029 surfaced its own regression during verification, caught by re-running the full suite
rather than only the newly-touched test file: `tests/test_service_api.py`'s six `create_app(...)`
call sites relied on the old unrestricted default while testing unrelated REST-surface
behaviour, and `TestClient`'s default `Host: testserver` does not satisfy the new
`LOOPBACK_HOST_ALLOW_LIST` default. Fixed by passing `host_allow_list=UNRESTRICTED_HOST_ALLOW_LIST`
explicitly at each site — that file tests the REST surface, not `Host` validation, which is
`tests/test_service_host_validation.py`'s job and the one file that varies the argument.

**`fix/backlog-phase-e` was still open, unmerged, when this branch was cut from `main`.**
This file therefore still shows GK-011..014 as `todo` above — accurate as of this branch's
base commit, not current. Merging both PRs will conflict on this file; resolve by rebasing
whichever merges second, not by discarding either closure note.

---

## Phase G — closed in part 2026-08-20; GK-020 remains open

**Goal:** remove the complexity-class limits, not just their symptoms.
**Exit:** query cost is proportional to matching chunks, not corpus size.

The measurement prerequisite is **done** and is no longer a gate:
`scripts/measure_retriever_open.py` now times `search`, the document read and a
warm-vs-rebuild acquire, plus an ingest window with and without a concurrent
acquire loop. ADR-0002's revisit trigger asked for a measurement rather than a
guess, and the ordering that measurement produced - GK-019 first, because its
waste ratio is `documents / top_k` and no corpus or query shape makes it small,
where GK-018's win is bounded by the candidate fraction the script prints - is
recorded here rather than left in an agent's report.

GK-018 and GK-019 are **closed** and deleted from this file per the rule above.
GK-018's consequence is an erratum on ADR-0002 decision 2, which named three
structures where the code had two; GK-019's is the optional
`DocumentRecordStoreProtocol` widening plus `tests/metadata_store_doubles.py`,
the shared base that actually removes the test-maintenance constraint the
`isinstance` fork was built around.

The exit criterion is **partly** met, and saying so is the point of having one.
A query no longer materializes the `documents` table, and no longer scores
chunks holding none of its terms. It is still not proportional to matching
chunks in two ways, both recorded in `KNOWN_LIMITATIONS.md`: an unselective
query has most of the corpus in its postings union, and `BM25Index.from_store`
still rebuilds in O(corpus) at open.

### GK-020 — The staleness cache stops working during an ingest

- **Severity** MEDIUM · **Effort** L (criteria 2-3 only) - **ADR** ADR-0026
  (Accepted) · **Verified**
- **Where** `src/groundkit/index/metadata.py` (bump per document);
  `src/groundkit/runtime.py` (cache validity)

**Criteria 1 and 4 landed 2026-08-20; the cliff itself is not fixed.**
`index_status` now reports `retriever_acquires`, `retriever_rebuilds`,
`rebuild_seconds_total` and `last_rebuild_seconds`, so the hit rate is read
rather than inferred from latency, and ADR-0026 records the decision and
re-defers ADR-0002's persisted-postings alternative against a trigger the new
measurement can satisfy.

The defect is unchanged: the generation bumps once per *document* and any bump
invalidates the cache, so an ingest over N changed files commits N
invalidations, and each rebuild is a full-corpus `get_chunks()` holding the same
lock the ingest writer needs. During an ingest the hit rate approaches zero and
the fallback is the reopen-per-request baseline ADR-0013 rejected on
measurement, plus contention that also slows the ingest.

Criteria 2 and 3 were deliberately **not attempted** rather than half-landed.
`SCHEMA_VERSION` is at 3, and v3 adds columns to an existing table, which
`CREATE TABLE IF NOT EXISTS` cannot supply to an older store - which is why it
refuses *writes* on a pre-v3 store. A watermark column has exactly that shape,
so a partial v4 would strand real indexes. The other hard half is
`remove_document` on the lexical index: a watermark cannot represent a row that
is *gone*, and `BM25Index`'s postings map is keyed by position in its chunk
list, so removing a chunk renumbers every posting after it - an index redesign
carrying GK-018's score-identity obligation, not an added method.

**Acceptance criteria**

- [x] Observability first: a rebuild counter and duration visible via
      `index_status`, so the cliff is measurable rather than inferred.
- [ ] Then incremental rebuild: a monotonic per-document watermark
      (`ingested_at` is a wall-clock string and unsuitable), a
      `get_chunks_since`, and a `remove_document` on the lexical index, which
      does not exist today - `index_chunks` is accumulate-only.
- [ ] A schema bump, with the delete-and-re-ingest consequence recorded per
      ADR-0004 decision 5.
- [x] ADR recording the decision and closing out ADR-0002's deferred
      alternative explicitly - ADR-0026 re-defers it against a new trigger: a
      recorded reading of `index_status`'s counters under a concurrent ingest.

**Start here:** take the reading before building anything. The counters and the
measurement script both exist so that the next attempt is triggered by a
measurement rather than by this file's argument.

---

## Phase H — closed 2026-08-20

GK-021, GK-022, GK-024, GK-025, GK-026, GK-027 and GK-028 all landed and are
deleted from this file per the rule above. Notes on the three that did not land
exactly as scoped:

- **GK-025 was decided, not built.** Option (b): the seam is recorded in
  `KNOWN_LIMITATIONS.md` with the trigger that would justify wiring it up, and
  no source file changed. The argument for not exposing it is worth keeping:
  `index/bm25.py` has no filter at all, so a surface-level `metadata_filter`
  would apply in `dense` mode, have nothing to apply to in the default `bm25`
  mode, and in `hybrid` mode filter one candidate list while RRF fused the
  survivors with an unfiltered lexical list - excluded chunks re-entering the
  ranking by a depth-dependent amount. A filter honoured by one of three modes
  and silently leaky in a second is worse than an unexposed seam. The item's own
  "**Where** `index/dense.py` only" was wrong as a change footprint: the
  footprint was documentation.
- **GK-028 sub-item 2 was declined**, with the reason recorded in
  `contracts.py` rather than only here. `content_hash` stays an uncached
  `computed_field`. The instruction was to justify a change on correctness
  rather than an unmeasured performance claim, and correctness argues the other
  way: on a frozen model, assignment to a `cached_property` succeeds and writes
  into `__dict__`, where assignment to a field raises `ValidationError`. Caching
  would open a way to decouple the hash from the content it hashes. Pinned by
  `tests/test_contracts.py::test_content_hash_cannot_be_decoupled_from_content_by_assignment`.
- **GK-028 sub-item 4 landed on the write side only.** The snapshot *read* path
  kept the same check-then-use gap, and it was the more exploitable half - the
  write side could corrupt a file, the read side returns one to a service
  caller. Carried forward as GK-030 and closed on
  `fix/gk-030-snapshot-read-nofollow`.

---

## Net-new from the Phase G/H review

Found by the adversarial review of this fan-out rather than by the original audit. Two
of its findings landed as fixes in the same branch - a partial
`DocumentRecordStoreProtocol` implementer being silently downgraded to the `text` source
class, and `EMLINK` being reported as a planted symlink on platforms where it means "too
many links" - each with a regression test shown to fail first. The third — the snapshot
read path's symlink race — was carried forward as GK-030 and has since landed on
`fix/gk-030-snapshot-read-nofollow`, so nothing from this review remains open.

Closing it turned up a second defect on the same line of code, which is why the read is
now a byte read and its own decode rather than only an `O_NOFOLLOW` open:
`Path.read_text` defaults to universal-newline mode, so a snapshot served with CRLF came
back one character shorter per line break and every offset past the first was wrong —
`fetch_chunk` returning a shifted span, or `drifted` on a snapshot that had not drifted.
It was invisible to every existing test because the fixtures are LF-only, and invisible
to the write side because that side already pinned byte-exactness with `newline=""` and
`O_BINARY` — the asymmetry was the bug. Recorded here because the lesson outlives the
item: a round trip is only verified by a fixture that differs between the two
representations.

## Declined

Nothing yet. Items moved here keep their ID and gain a one-line reason.

---

## Corrections carried in from the prior audit

Recorded so the earlier work order is not re-followed as written.

- **M6 is closed, not open.** `ADR-0023` (Accepted 2026-08-17) decides snapshot retention:
  lifetime bound to the document row, cleanup in the `Indexer`, implemented. What remains
  is narrower and is GK-025-adjacent: there is no `delete_collection` anywhere, so nothing
  removes a collection's `.snapshots/` directory or its `.lance` sibling. `SPEC.md` §7
  names all three artifacts.
- **M2 is fixed.** `README.md`'s core-subset list matches `pyproject.toml` exactly,
  including `runtime.py`.
- **The `# pragma: no cover` count is two, not thirteen.** Both are `AssertionError`
  guards on exhaustive `Literal` matches that mypy already proves unreachable. The rest
  live under `tests/`, which `[tool.coverage.run] source` never measures.
- **The version-parity gap is not real.** It is closed twice: by
  `tests/test_smoke.py::test_version_matches_pyproject` per PR, and again at release by
  `release-gates.yml`'s clean-wheel install step, which asserts the built distribution's
  metadata equals `groundkit.__version__`. See GK-028.
- **Six of the nine documentation-drift claims were already repaired** before this review.
  The egress inventory, the deployment guide's Jaeger paragraph, `KNOWN_LIMITATIONS.md`'s
  internal contradiction, and the installation dependency list are all current. What
  survives is GK-005, GK-006 and GK-007.

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
| GK-001 | DNS rebinding defeats the loopback bind | CRIT | A | S | todo | — |
| GK-003 | Rerank drops `source_class` / `extractor` | HIGH | B | S | todo | — |
| GK-004 | Caller metadata overwrites the authoritative `source` | HIGH | B | S | todo | — |
| GK-005 | Docs state a live package is unpublished | HIGH | C | M | todo | — |
| GK-006 | Two rendered docstrings claim Phase 3 is unbuilt | MED | C | XS | todo | — |
| GK-007 | `CLAUDE.md` is frozen at end-of-Phase-2 | MED | C | M | todo | GK-005 |
| GK-008 | BM25 scoring blocks the event loop | HIGH | D | S | todo | — |
| GK-009 | Unauthenticated O(corpus) work, no concurrency cap | HIGH | D | S | todo | — |
| GK-010 | `busy_timeout` unset; a concurrent writer fails instead of waiting | MED | D | S | todo | — |
| GK-011 | `grk answer` has no end-to-end test | HIGH | E | S | todo | — |
| GK-012 | `resolve_chat_config` has no test | MED | E | S | todo | — |
| GK-013 | The unauthenticated error boundary is never driven over HTTP | MED | E | S | todo | — |
| GK-014 | `add_chunks`' document-id mismatch branch is untested | LOW | E | XS | todo | — |
| GK-015 | `SPEC.md` §2 has no structural guard | MED | F | S | todo | — |
| GK-016 | `BM25Index` has no protocol seam | MED | F | S | todo | — |
| GK-017 | Frozen models alias nested caller metadata | MED | F | S | todo | — |
| GK-018 | BM25 has no postings list | MED | G | S | todo | GK-016 |
| GK-019 | Three read paths materialize a table to answer a keyed question | MED | G | M | todo | — |
| GK-020 | The staleness cache stops working during an ingest | MED | G | L | todo | GK-016, GK-019 |
| GK-021 | `answer.py` imports the eval harness | MED | H | S | todo | — |
| GK-022 | `extraction.py` omitted from the coverage core subset | MED | H | XS | todo | — |
| GK-024 | Blocking filesystem I/O inside `async def` | MED | H | XS | todo | — |
| GK-025 | `metadata_filter` has no caller | LOW | H | XS | todo | — |
| GK-026 | ADR-0013 decision 7 was never implemented | LOW | H | S | todo | — |
| GK-027 | `run_eval` is one long, deeply nested function | LOW | H | M | todo | — |
| GK-028 | Assorted small correctness debt | LOW | H | S | todo | — |

---

## Phase A — Contain what shipped

**Goal:** close the findings that are exploitable or that govern everything else.
**Exit:** GK-001 done; `SECURITY.md` accurate about what the bind does and does not
protect.

> The governance half of this phase has landed: `main` is now enforced by a repository
> ruleset requiring nine status checks, with no direct pushes and no bypass actors. What
> it enforces, and why zero approvals is deliberate rather than an oversight, is recorded
> in `CONTRIBUTING.md`. Note the ruleset targets the default branch only — any other
> long-lived branch is unprotected.

### GK-001 — DNS rebinding defeats the loopback bind on both transports

- **Severity** CRITICAL · **Effort** S · **ADR** yes (amends ADR-0014's threat model) ·
  **Verified**
- **Where** `src/groundkit/service/mcp_server.py:459` (`create_session_manager`);
  `src/groundkit/service/api.py` (`create_app`)

ADR-0014's access-control argument is that the `127.0.0.1` bind *is* the control, so no
authentication is needed. That holds only if the bind cannot be reached from off-box. It
can: a page on any site the victim visits can re-point its own hostname at `127.0.0.1`
with a short-TTL DNS answer; the browser still treats the request as same-origin, so no
CORS preflight applies and the response is fully readable. Nothing in groundkit inspects
`Host`.

The MCP SDK ships the mechanism and defaults it off for backwards compatibility
(`TransportSecurityMiddleware.__init__`), but its own convenience server auto-enables it
for exactly this case:

```python
# mcp/server/fastmcp/server.py — the SDK's own default
if transport_security is None and host in ("127.0.0.1", "localhost", "::1"):
    transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*"], ...
    )
```

groundkit uses the lower-level `Server` / `StreamableHTTPSessionManager` API, which
bypasses that, and passes no `security_settings`. The REST half has no
`TrustedHostMiddleware` and no `Host` check at all. The module docstring defers this
because "the allowed `Host` values depend on the bind address, which lives in the CLI" —
that is plumbing, and the CLI already computes `--host` and `--allow-remote-access`.

**Acceptance criteria**

- [ ] `create_session_manager` receives a `TransportSecuritySettings` derived from the
      serve-time host decision, not a constant.
- [ ] `create_app` installs `TrustedHostMiddleware` with the same derived allow-list.
- [ ] `--allow-remote-access` widens both allow-lists consistently; without it, both
      refuse a non-loopback `Host`.
- [ ] Regression test, shown to fail first: a request carrying a forged `Host` is refused
      on the REST surface **and** on `/mcp`, and a loopback `Host` still succeeds.
- [ ] ADR amending ADR-0014 records that the bind alone was insufficient and why the
      no-authentication position still holds once `Host` is validated.
- [ ] `SECURITY.md` lists inbound rebinding as closed, or as a named residual until it is.

---

## Phase B — Correct the shipped defects

**Goal:** fix the two silent-wrong-answer defects in the published artifact.
**Exit:** both fixed with must-fail-first regression tests; a `0.1.1` is releasable.

### GK-003 — `rerank_by_logits` drops `source_class` and `extractor`

- **Severity** HIGH · **Effort** S · **ADR** no · **Verified**
- **Where** `src/groundkit/retrieval/rerank.py:170`

Three places in production rebuild a `RetrievalResult`. Two pass both fields explicitly —
`retrieval/search.py:594`, whose docstring calls the omission "the fail-open defect this
closes", and `providers/context_assembly.py:294`. This one does not, so every reranked
result silently reverts to the contract defaults `"text"` / `None`.

Phase 3 code that was never revisited when ADR-0016 grew the contract underneath it. It is
reachable today: `service/schemas.py:61` exposes `rerank` on every REST and MCP search
request and `service/tools.py:197` applies it. Against a collection holding URL-ingested
documents, every citation returns labelled `"text"` when it is actually `"snapshot"`,
routing later verification down the plain read-and-slice path ADR-0016 exists to keep
snapshots out of. It fails closed rather than returning wrong content, but the snapshot
verification mechanism is silently gone.

**Acceptance criteria**

- [ ] `rerank_by_logits` passes `source_class=result.source_class` and
      `extractor=result.extractor`.
- [ ] `tests/test_rerank.py`'s `_result()` helper accepts both fields.
- [ ] Regression test, shown to fail first: a `snapshot`-class result survives rerank with
      its class intact; likewise an `extracted` result with its extractor identity.
- [ ] A check that no *fourth* rebuild site can reappear silently — either a test that
      enumerates `RetrievalResult(` construction sites in `src/`, or the field made
      non-defaulting so omission is a type error. Prefer the latter if it does not break
      `evals/echo.py`'s synthetic fixtures.

### GK-004 — Caller metadata silently overwrites the authoritative `source`

- **Severity** HIGH · **Effort** S · **ADR** no · **Verified**
- **Where** `src/groundkit/ingestion/chunking.py:92`

```text
metadata={"source": document.source, **document.metadata},
```

(Fenced as `text`, not `python`: `ruff format` formats Python blocks inside Markdown, and
this line is a keyword-argument fragment rather than a statement. Ruff parses it as a
tuple assignment and rewrites it into something that means something else entirely, which
fails the `lint` job's `ruff format --check` step. Quote source fragments as `text`.)

Dict-literal ordering applies the caller's keys *after* the seeded one. Confirmed by
execution: a `Document` carrying `metadata={"source": "..."}` produces chunks whose
`metadata["source"]` is the caller's value, not the real path.

Citations are unaffected — they join through `documents.source` in SQLite, which is why
this is HIGH and not CRITICAL. But `metadata_filter={"source": ...}` on dense search is a
real, tested capability (`tests/test_dense.py:801`), and against such a document it
returns zero or wrong results with nothing raised anywhere.

**Acceptance criteria**

- [ ] Merge order reversed to `{**document.metadata, "source": document.source}`.
- [ ] Regression test, shown to fail first: a `Document` whose metadata carries a
      colliding `source` key still produces chunks whose `metadata["source"]` equals
      `document.source`.
- [ ] Decide and record whether a colliding key should instead be rejected at ingest —
      silently winning and silently losing are both defensible, silently *swapping* is
      not. A `KNOWN_LIMITATIONS.md` line is sufficient if the answer is "last write wins,
      deliberately".

---

## Phase C — Make the documentation true

**Goal:** a live package whose own documentation describes it accurately.
**Exit:** no document claims groundkit is unpublished; `SPEC.md` §9 shows Phase 7 done.

### GK-005 — Documentation states that a published package is unpublished

- **Severity** HIGH · **Effort** M · **ADR** no · **Verified**
- **Where** `SPEC.md:288`; `README.md:120`; `docs/guides/mcp-clients.md:26`;
  `CHANGELOG.md` (`[0.1.0]` heading undated)

v0.1.0 published to PyPI and GitHub Releases on 2026-08-18; tag `v0.1.0` points at
`c9647eb`. All of the above still say the release has not happened and that
`pip install groundkit` does not work. This is worse than ordinary doc lag — it steers
real users away from the correct install path on the project's most-read pages, and
`SPEC.md` §9 is the document this repo designates as the phase authority over every other
statement of status.

**Acceptance criteria**

- [ ] `SPEC.md` §9 Phase 7 row reads done, with the release date.
- [ ] `README.md`'s status block and install guidance make `pip install groundkit` the
      primary path.
- [ ] `docs/guides/mcp-clients.md` drops the not-on-PyPI caveat.
- [ ] `docs/getting-started/installation.md` leads with the PyPI install.
- [ ] `CHANGELOG.md`'s `[0.1.0]` heading carries its date, per the Keep a Changelog format
      the file says it follows. Add the inverted-span rejection landed in `04384db` under
      Retrieval — a contract behaviour change currently with no changelog trace.
- [ ] The three badges `README.md`'s own comment block reserves — PyPI version, Python
      versions, License — added now that each reports something true. No hand-written
      values; live endpoints only.

### GK-006 — Two rendered docstrings claim Phase 3 is unbuilt

- **Severity** MEDIUM · **Effort** XS · **ADR** no
- **Where** `src/groundkit/config.py:220`; `src/groundkit/index/protocols.py:235`

"LanceDB table arrives in Phase 3" and "implementations arrive in Phase 3". Both shipped
on 2026-08-15 (`InMemoryVectorStore`, `LanceDBVectorStore`). These render on the published
docs site via mkdocstrings, so a site reader is told an existing capability does not exist.

**Acceptance criteria**

- [ ] Both docstrings describe what exists, naming the classes.
- [ ] Grep for other `arrives in Phase`/`Phase N` futures in source docstrings; fix any
      whose phase has closed.

### GK-007 — `CLAUDE.md` is frozen at end-of-Phase-2

- **Severity** MEDIUM · **Effort** M · **ADR** no · **Depends on** GK-005
- **Where** `CLAUDE.md` (gitignored at `.gitignore:40`)

Because it is gitignored it is invisible to CI, absent from a fresh clone, and unreachable
by any doc sweep — and it is what every future agent session reads as ground truth. Its
stale claims include: three CLI verbs (there are six — `ingest, search, answer, eval,
serve, serve-mcp`); six modules described as docstring-only stubs (all implemented); a
three-entry coverage core subset (there are five); five CI jobs (there are seven); and an
"open Phase 3 decision" that is closed.

**Acceptance criteria**

- [ ] Regenerated against `cli.py`, `pyproject.toml`, `ci.yml` and the actual module set.
- [ ] States near the top that `SPEC.md` §9 is the phase authority and that this file is
      gitignored and therefore unverifiable by CI.
- [ ] Decide whether it should stay gitignored at all. If it stays, note in
      `CONTRIBUTING.md` that it exists and is untracked, so its absence in a fresh clone
      is not mistaken for a missing file.

---

## Phase D — Harden the service under concurrency

**Goal:** the service degrades rather than collapses under concurrent or hostile load.
**Exit:** no single request can stall every other; a concurrent writer waits rather than
failing.

### GK-008 — BM25 scoring runs synchronously on the event loop

- **Severity** HIGH · **Effort** S · **ADR** no · **Verified**
- **Where** `src/groundkit/retrieval/search.py:327` and `:339`; no `asyncio.to_thread`
  anywhere in `src/groundkit/index/bm25.py`

`self._bm25.search(...)` is a synchronous, CPU-bound, whole-corpus scan invoked directly
from the async handler. The same codebase applies `asyncio.to_thread` for exactly this
reason at `retrieval/rerank.py:504` and throughout `index/dense.py` — so the one CPU-bound
path never moved off the loop is the default, always-available retrieval mode.
`grk serve` runs a single uvicorn worker on one loop, so one slow query stalls every other
in-flight request, including `index_status` and `fetch_chunk`.

This is a multiplier on GK-018 rather than an independent limit, and it appears in no ADR
and no limitations entry.

**Acceptance criteria**

- [ ] Both call sites wrapped in `asyncio.to_thread`.
- [ ] `InMemoryVectorStore.search`'s scoring loop likewise — it is `async def` today with
      no `await` in its body.
- [ ] Regression test, shown to fail first: a search does not block a concurrently issued
      second request. Assert on thread identity, as
      `test_list_collections_does_not_block_the_event_loop` does, not on timing.
- [ ] `KNOWN_LIMITATIONS.md` records the residual: total CPU cost is unchanged, only the
      stalling is.

### GK-009 — Unauthenticated O(corpus) operations with no concurrency cap

- **Severity** HIGH · **Effort** S · **ADR** no
- **Where** `src/groundkit/runtime.py:357` (`chunk_count`);
  `src/groundkit/index/metadata.py:664` (`get_chunks`); `src/groundkit/cli.py:1361`
  (uvicorn config); `infra/k8s/deployment.yaml`

`index_status` reads as a cheap status call and is wired to `get_chunks()`, which
materializes every chunk's full text to call `len()` on the list. `search` is whole-corpus
per call regardless of `top_k`. uvicorn is started with no `limit_concurrency` and there
is no per-request semaphore. The Kubernetes manifest sets a hard memory limit on a single
replica with `strategy: Recreate` — no capacity to fail over to — and its comment assumes
an OOMKill would mean the corpus outgrew the limit, which concurrent callers invalidate.
Combined with GK-001, the caller need not even be on the network.

**Acceptance criteria**

- [ ] `limit_concurrency` set on the uvicorn config, value as a named constant.
- [ ] `chunk_count` no longer materializes every chunk. `runtime.py`'s docstring declines
      to widen `MetadataStoreProtocol` with a `COUNT(*)` on conformance-test grounds —
      that argument does not require using the slowest available internal implementation.
      Put the count on the optional `DocumentRecordStoreProtocol`, or run the aggregate
      inside a dedicated `_op` without widening the shared protocol.
- [ ] Regression test that `chunk_count` does not read chunk content.
- [ ] `SECURITY.md`'s rate-limiting paragraph updated to say what now exists.

### GK-010 — `busy_timeout` is unset, so a concurrent writer fails instead of waiting

- **Severity** MEDIUM · **Effort** S · **ADR** no · **Verified**
- **Where** `src/groundkit/index/metadata.py` `_connect` — sets `foreign_keys` and
  `journal_mode`, nothing else

The string `busy_timeout` occurs exactly once in the repository: in ADR-0013's own note
flagging it *for* ADR-0014. ADR-0014 never took it up, and Phase 4 then shipped the
concurrent surface the note anticipated. SQLite's default busy timeout is 0, so the losing
writer fails immediately rather than retrying.

The trigger is ordinary: an operator runs `grk ingest` while the service is up, a client
makes its first request for that collection, and the store-open's own `commit()` collides
with the ingest's write transaction. The request fails with a `StorageError` and surfaces
as a 5xx. Retrying works; nothing retries.

This is the second instance of the same failure mode as the (now closed) snapshot
retention deferral: a named, dated obligation handed to a later ADR that never picked it
up. Worth a moment's thought about whether deferrals need a tracked home rather than a
sentence inside a record nobody re-reads.

**Acceptance criteria**

- [ ] `PRAGMA busy_timeout` set in `_connect`, value a named module constant.
- [ ] Regression test, shown to fail first: hold a write transaction on a second
      connection and assert `open` succeeds rather than raising.
- [ ] The chosen value and its reasoning recorded — a `KNOWN_LIMITATIONS.md` entry is
      enough; an ADR is not required for a pragma.

---

## Phase E — Close the test gaps

**Goal:** the dark part of the covered surface contains no user-facing entry point.
**Exit:** every CLI verb is driven through `main()` at least once.

### GK-011 — `grk answer` has no end-to-end test

- **Severity** HIGH · **Effort** S · **ADR** no · **Verified**
- **Where** `src/groundkit/cli.py:809` (`_cmd_answer`)

No `"answer"` invocation exists in `tests/test_cli.py` or `tests/test_cli_serve.py`, and
`_cmd_answer` appears nowhere under `tests/`. `main(["search", ...])` is exercised
repeatedly by comparison. Only the pure printer is driven, explicitly "rather than through
`main()`" per its own docstring.

Untested: the mode/embed-flag mutual-exclusion guard, the store → embedder → vector store
→ retriever → chat wiring, the JSON/text branch, and the `finally` block that closes three
resources in order. A cleanup-ordering bug — an exception during `build_chat` leaving the
store or embedder unclosed — is exactly the class a unit test of the pipeline alone cannot
see.

**Acceptance criteria**

- [ ] A happy-path `main(["answer", ...])` test using the existing offline doubles.
- [ ] A test asserting the `--mode bm25` + embed-flags combination is rejected.
- [ ] A test asserting all three resources are closed when `build_chat` raises.

### GK-012 — `resolve_chat_config` has no test

- **Severity** MEDIUM · **Effort** S · **ADR** no · **Verified**
- **Where** `src/groundkit/config.py:180`

Its sibling `resolve_embedding_config`, one function above it in the same file and
documented as "the same peer ... for the same reasons", has a full test class including
the `ValidationError` → `ConfigurationError` translation. Chat config has none. With
GK-011, the entire `--chat-*` path is dark end to end.

**Acceptance criteria**

- [ ] The embedding-config test class ported to chat config, including the error
      translation case.

### GK-013 — The unauthenticated error boundary is never driven over HTTP

- **Severity** MEDIUM · **Effort** S · **ADR** no
- **Where** `src/groundkit/service/api.py:383`; `tests/test_service_errors.py:217`

`unexpected_error_rendering()` is called directly as a function with no request in flight.
No test raises a bare non-`GroundkitError` through `create_app` and inspects the actual
response. This is the boundary that stops an internal exception message reaching an
unauthenticated caller; a reordering of the two `except` clauses, or a refactor that
includes `str(exc)` in the body, would go undetected.

**Acceptance criteria**

- [ ] An integration test with a handler monkeypatched to raise a bare exception carrying
      a distinctive string, asserting the response status, the generic detail, and the
      absence of that string from the body.

### GK-014 — `add_chunks`' document-id mismatch branch is untested

- **Severity** LOW · **Effort** XS · **ADR** no
- **Where** `src/groundkit/index/metadata.py` (`add_chunks`)

`add_chunks` has no production caller — `indexer.py` uses `replace_document` exclusively —
but remains a required `MetadataStoreProtocol` member. `replace_document` carries a
structurally separate copy of the same document-id guard and *is* tested; this copy is
not. A third-party implementation, or a refactor of `add_chunks` alone, could drop it
silently.

**Acceptance criteria**

- [ ] A test calling `add_chunks` with a mismatched `chunk.document_id`, mirroring the
      existing `test_add_chunks_rejects_source_with_no_document`.

---

## Phase F — Structural guards and seams

**Goal:** the invariants this project rests on are enforced by mechanism, not by review.
**Exit:** GK-015 landed; GK-016 landed so Phase G can proceed without touching `Retriever`.

### GK-015 — `SPEC.md` §2's central claim has no structural guard

- **Severity** MEDIUM · **Effort** S · **ADR** no · **Verified**
- **Where** the only AST import scan is `tests/test_service_tools.py:184`, scoped to
  `service/`

"Deterministic core, LLM at the boundary" currently holds — nothing under `retrieval/`,
`index/` or `ingestion/` imports the LLM boundary. But it is enforced by review alone,
while a strictly *narrower* property already has an AST scan whose docstring reads "Guard,
demonstrated by injection." Meanwhile `answer.py` composes retrieval with synthesis,
rewrite and the judge one import away, so the obvious next request — make query rewrite an
option on `Retriever.search` so `grk search` benefits — is a two-line change that violates
the project's central claim and that nothing would catch.

This is the cheapest high-value item in the backlog, and it is the structural answer to the
process gap that produced GK-003: a module grew a contract underneath it and no mechanism
noticed.

**Acceptance criteria**

- [ ] `test_deterministic_core_imports_no_llm` AST-walks `src/groundkit/{retrieval,index,
      ingestion}/*.py` plus `indexer.py`, `contracts.py` and `identity.py`.
- [ ] Bars `groundkit.providers.{llm,synthesis,query_rewrite}` and `groundkit.evals.*`;
      allows `groundkit.providers.protocols`, since the embedding seam legitimately sits in
      the retrieval path.
- [ ] Demonstrated by injection: add a barred import, watch it fail, remove it.

### GK-016 — `BM25Index` has no protocol seam

- **Severity** MEDIUM · **Effort** S · **ADR** no
- **Where** `src/groundkit/retrieval/search.py:105` takes the concrete class

Seams exist for loaders, chunkers, metadata stores, vector stores, embedders, chat and
rerankers. `VectorStoreProtocol` and `RerankerProtocol` were even defined before their
implementations existed, so Phase 3 could fill them in without touching callers. The
lexical side — the one component carrying a named revisit trigger in ADR-0002 — got no such
treatment, which makes GK-018 and GK-020 more expensive than they need to be, since both
would otherwise mean editing `Retriever.open` directly.

**Acceptance criteria**

- [ ] `LexicalIndexProtocol` defined with exactly `size`, `index_chunks`, `search` and the
      `from_store` factory.
- [ ] A `tests/test_protocol_conformance.py` entry using `assert_signature_parity`.
- [ ] `Retriever`'s annotation changed; no behaviour change, no test changes beyond the
      new conformance entry.

### GK-017 — Frozen models alias nested caller metadata

- **Severity** MEDIUM · **Effort** S · **ADR** no · **Verified**
- **Where** `src/groundkit/contracts.py` — `metadata: dict[str, Any]` on `Document`,
  `Chunk`, `RetrievalResult`, `SearchResponse`

Confirmed by execution: Pydantic's `frozen=True` blocks attribute rebinding and rebuilds
the *outer* dict, but nested mutable values remain the same objects the caller holds, so a
caller mutation lands inside the frozen model. Dormant today only because every shipped
loader writes scalars.

The chain that would survive is real, not hypothetical: `BM25Index.from_store` builds each
`Chunk` once and holds it for the index's lifetime, and `CollectionRuntime` caches one
`Retriever` across every request in a running server, so the same objects reach the service
boundary on every query. `providers/context_assembly.py:304` defensively copies the dict
while `retrieval/rerank.py:178` beside it does not — and that copy is itself shallow, so it
closes nothing.

Per-call-site discipline is the wrong fix; PDF/HTML loaders will add call sites.

**Acceptance criteria**

- [ ] Immutability enforced at the contract boundary — a validator that deep-copies or
      converts to immutable equivalents on construction — so new call sites are safe by
      default.
- [ ] Consider narrowing the type: metadata must already survive `json.dumps` at
      persistence, so a recursive `JSONValue` alias would fail closed at construction
      rather than at the far-away persistence boundary.
- [ ] Regression test, shown to fail first: mutating a nested value the caller still holds
      does not change the model's view of it.

---

## Phase G — Scale

**Goal:** remove the complexity-class limits, not just their symptoms.
**Exit:** query cost is proportional to matching chunks, not corpus size.

> Before starting this phase, extend `scripts/measure_retriever_open.py` (or add a sibling)
> to time `search` and a warm-vs-rebuild acquire. ADR-0002's revisit trigger is explicitly
> *"that measurement is the trigger, not a guess made now"* — and the ordering of GK-018,
> GK-019 and GK-020 should follow measurement rather than this file's guess.

### GK-018 — BM25 has no postings list

- **Severity** MEDIUM · **Effort** S · **ADR** yes (erratum to ADR-0002) ·
  **Depends on** GK-016 · **Verified**
- **Where** `src/groundkit/index/bm25.py:113` —
  `for doc_idx in range(len(self._chunks))`

The class docstring says it builds an inverted index, and ADR-0002 asserts that "the
postings, document frequencies, and length statistics are pure functions of the persisted
chunk set." Two of those three exist. `_doc_freqs` is a document-frequency counter used
only for IDF; there is no `term → [doc_idx]` map, so every query scores every chunk
regardless of selectivity.

The fix is score-identical rather than an approximation: chunks with no matching term
already score exactly `0.0` and are discarded by the existing filter, so restricting the
loop to the union of the query terms' postings changes no output, including the tie-break.
It needs no new dependency and no persisted-format change, so ADR-0002's no-shadow-state
invariant is untouched.

**Acceptance criteria**

- [ ] A postings map populated in `index_chunks` alongside the existing counter.
- [ ] `search` iterates the union of the query terms' postings.
- [ ] An equivalence test over the golden corpus asserting byte-identical result lists
      before and after, including ties.
- [ ] The class docstring corrected, and an ADR-0002 erratum recording that the postings
      the ADR described were not built until now — following the ADR-0022 `Indexer.run`
      erratum precedent.

### GK-019 — Three read paths materialize a table to answer a keyed question

- **Severity** MEDIUM · **Effort** M · **ADR** no
- **Where** `src/groundkit/retrieval/search.py:322`; `src/groundkit/service/tools.py:244`;
  `src/groundkit/runtime.py:357`

`Retriever.search` calls `get_document_records()` on **every** query — a full `documents`
scan building a validated model per row — to look up at most `top_k` IDs. This read must be
live rather than cached at open, so deletions fail closed; the cost is therefore
load-bearing, not incidental. `handle_fetch_chunk`, the tool a client is expected to call
per result, does the same for a single ID. `chunk_count` does it for chunks (see GK-009).

The recorded reason in all three places is test maintenance: widening
`MetadataStoreProtocol` would break hand-built doubles across the suite. The discipline is
right and has become a design constraint, and the workaround it produced is an `isinstance`
capability fork in the retrieval hot path with a silent downgrade branch.

**Acceptance criteria**

- [ ] `get_document_record(document_id)` and count aggregates added to the **optional**
      `DocumentRecordStoreProtocol`, which only the real store implements — no test double
      is affected.
- [ ] `_resolve` / `_apply_snapshot_filter` look up per hit; `handle_fetch_chunk` uses the
      keyed form; the existing full-table path stays as the fallback branch.
- [ ] Separately: hand-built store doubles replaced by one shared base in `tests/`, so the
      next legitimate protocol widening is not blocked by test cost. This is the item that
      actually removes the constraint.

### GK-020 — The staleness cache stops working during an ingest

- **Severity** MEDIUM · **Effort** L · **ADR** yes · **Depends on** GK-016, GK-019
- **Where** `src/groundkit/index/metadata.py:659` (bump per document);
  `src/groundkit/runtime.py:229` (cache validity)

The generation bumps once per *document*, and any bump invalidates the cache. So an ingest
over N changed files commits N invalidations, and each rebuild is a full-corpus
`get_chunks()` holding the same lock the ingest writer needs. During an ingest the hit rate
approaches zero and the fallback is the reopen-per-request baseline ADR-0013 rejected on
measurement, plus contention that also slows the ingest. Waiters cannot be served stale by
design, so concurrent requests each absorb a full rebuild.

ADR-0013's per-commit bump is correct as specified; what is unrecorded is this
consequence, and that ADR-0002's deferred alternative is still open with its trigger now
satisfied.

**Acceptance criteria**

- [ ] Observability first: a rebuild counter and duration visible via `index_status`, so
      the cliff is measurable rather than inferred.
- [ ] Then incremental rebuild: a monotonic per-document watermark (`ingested_at` is a
      wall-clock string and unsuitable), a `get_chunks_since`, and a `remove_document` on
      the lexical index, which does not exist today — `index_chunks` is accumulate-only.
- [ ] A schema bump, with the delete-and-re-ingest consequence recorded per ADR-0004
      decision 5.
- [ ] ADR recording the decision and closing out ADR-0002's deferred alternative
      explicitly, either by adopting or by re-deferring with a new trigger.

---

## Phase H — Hygiene and smaller debt

**Goal:** clear the long tail. No ordering constraint between items.

### GK-021 — `answer.py` imports the eval harness

- **Severity** MEDIUM · **Effort** S · **Verified**
- **Where** `src/groundkit/answer.py:85` — `from groundkit.evals.judge import ...`

A production module depends on the eval package, so `evals` can never become an extra or be
dropped from a runtime install. `FaithfulnessJudge` is a `ChatProtocol` consumer like
`Synthesizer` and `QueryRewriter` and belongs beside them. It would also force a special
case in GK-015's guard, which is the signal that it is on the wrong side of the line.

- [ ] `FaithfulnessJudge` / `FaithfulnessVerdict` moved to `providers/`;
      `evals/synthesis_eval.py` imports from there.
- [ ] `answer.py`'s docstring claim that "every collaborator is injected" made true, or
      narrowed — the instances are injected, the types are concrete.

### GK-022 — `extraction.py` omitted from the coverage core subset

- **Severity** MEDIUM · **Effort** XS
- **Where** `pyproject.toml` `[tool.groundkit.coverage].core_subset`

The table's own convention — restated in `CONTRIBUTING.md` — is that every module in the
subset *and every module left out of it* is argued in the comment above the table. Four
modules get exactly that. `extraction.py` sits at the package root where no glob catches
it, gets no note in either direction, and is real citation-resolution code called directly
by `resolve_citation`. `snapshots.py` is a weaker case (pure path arithmetic) and a
one-line exclusion note may be the right answer for it.

- [ ] `extraction.py` added to `core_subset` with a note in house style, or excluded with
      an argued note. Check its current coverage before choosing.
- [ ] `snapshots.py` given a note either way, so the convention holds with no silent
      omissions left.
- [ ] `README.md`'s core-subset list kept in step.

### GK-024 — Blocking filesystem I/O inside `async def`

- **Severity** MEDIUM · **Effort** XS
- **Where** `src/groundkit/ingestion/url_loader.py` — `_write_snapshot` and the scratch
  write in `_extract_html`

The only content-sized I/O in the repo not wrapped in `asyncio.to_thread`, against a
convention followed at every comparable site. Bounded today because URL ingestion is
CLI-only and nothing else is scheduled on that loop, but this is exactly the code a
service-side ingest tool would reuse verbatim.

- [ ] Both writes wrapped.
- [ ] Also `evals/runner.py`'s judgments-hash read, which is unwrapped while the
      corpus-hash read beside it is wrapped — the comment above them asserts a symmetry the
      code does not have.

### GK-025 — `metadata_filter` has no caller

- **Severity** LOW · **Effort** XS · **Verified**
- **Where** `src/groundkit/index/dense.py` only

A `SPEC.md` §5.3 obligation met at the seam and never connected to a surface: no CLI flag,
no `SearchRequest` field, no call site passes it. Its enabled branch is the most expensive
code in the dense path, is gated by the coverage core subset, and is unreachable in
production — so its cost has never been paid and its correctness is proved only by unit
tests. Not a bug; a decision never written down.

- [ ] Decide: expose it (a `SearchRequest` field plus a `Retriever.search` parameter, which
      per ADR-0014 decision 6 must not carry any resolution-shaped key), or record in
      `KNOWN_LIMITATIONS.md` that the seam exists with no caller, and name the trigger to
      wire it up.

### GK-026 — ADR-0013 decision 7 was never implemented

- **Severity** LOW · **Effort** S · **Verified**
- **Where** `src/groundkit/cli.py:780`, `:825`

The ADR, status Accepted, states that `_cmd_search` is absorbed into the runtime so "three
prospective new copies collapse into one", and claims the resulting benefit that the
default suite then exercises the runtime's open path on every CLI test. Neither is true:
four `Retriever.open` sites exist, `cli.py` imports only `CollectionRegistry`, and no CLI
test references `CollectionRuntime`. Phase 5's `_cmd_answer` added a fourth lifecycle the
ADR never counted.

- [ ] Either route `_cmd_search` and `_cmd_answer` through the runtime (behaviour-identical
      — the cache never hits in a one-shot process), or amend ADR-0013 with an erratum
      stating decision 7 was not implemented and why.

### GK-027 — `run_eval` is one long, deeply nested function

- **Severity** LOW · **Effort** M
- **Where** `src/groundkit/evals/runner.py:199`

Against the repo's own guidance of roughly 50 lines and four levels, it mixes corpus
hashing, store lifecycle, gold-truth resolution, a stage × judgment double loop, synthesis
and report assembly. Its test file is large as a direct consequence — though notably that
file is well organized into single-concern classes, not copy-paste, so the cost is
concentrated in the source. Worth noting the other large functions in the repo are benign
flat dispatch tables; this is the only genuine offender.

- [ ] Split into corpus/store setup, the per-stage scoring loop, and report assembly.
- [ ] Behaviour-preserving: the existing tests should pass unchanged.

### GK-028 — Assorted small correctness debt

- **Severity** LOW · **Effort** S

Grouped because each is a few lines and none justifies its own branch.

- [ ] `ingestion/pipeline.py:195` — `asyncio.gather` without `return_exceptions=True`.
      `indexer.py:452` does the opposite with a comment explaining why. No torn-write risk
      here since the pipeline never writes, but it is exported public API, so a third-party
      host inherits unsupervised background work after the first failure.
- [ ] `contracts.py:141` — `content_hash` is an uncached `computed_field` recomputing
      SHA-256 on every access, used as a sort tie-break per candidate per query in
      `fusion.py` and `dense.py`. `bm25.py:75` already caches it once at build time.
      `cached_property` works on a frozen model. Profile before prioritizing.
- [ ] `ingestion/url_loader.py` — no explicit total-request timeout, so httpx's
      per-operation default lets a slow server hold a connection well past the intended
      bound. Mirror `EmbeddingConfig.timeout_seconds`.
- [ ] `url_loader.py` / `snapshots.py` — check-then-use between the containment check and
      the open. Narrow (`document_id` is an unguessable uuid4, and the read path returns
      nothing unless bytes match) but `O_NOFOLLOW` costs nothing.
- [ ] `tests/test_smoke.py` — the docstring claiming nothing but that test holds
      `__version__` to `pyproject.toml` is wrong; `release-gates.yml`'s clean-wheel step
      asserts it independently. Narrow the claim.
- [ ] `requirements-audit.txt` — both `ci.yml` and `release-gates.yml` re-export it
      immediately before auditing, so the committed copy is never read and has already
      drifted once. Gitignore it, or add a step asserting it matches a fresh export.

---

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

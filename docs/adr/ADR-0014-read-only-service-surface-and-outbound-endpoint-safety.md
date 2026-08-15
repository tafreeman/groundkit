# ADR-0014 — Phase 4 is a read-only service surface, and outbound endpoints are guarded

- **Status:** Accepted (owner, 2026-08-15)
- **Date:** 2026-08-15
- **Deciders:** Andy Freeman (owner)

## Context

SPEC.md §1.2 names the MCP server's tools exactly: `search`, `fetch_chunk`,
`list_collections`, `index_status` — "real and installable, not a demo" — over stdio
and streamable HTTP. §4 requires a FastAPI REST surface mirroring them. **Ingest is not
among the four**, and §1.2 is cited here as the authority so that a later reader does
not "restore" it as an oversight.

SPEC.md §7 requires that mutating REST routes **and mutating MCP operations** sit behind
a shared-secret header with a constant-time compare, disabled when the secret is unset,
and that the service bind 127.0.0.1 by default — the rule being per *operation*, not per
transport. All four named tools are reads. That requirement can therefore be satisfied
two ways: ship a guard over a surface with nothing to guard, or ship no guard and record
why it is met vacuously. Choosing by omission is the outcome this ADR exists to prevent.

Three facts make "no mutating operation" the right scope rather than a convenience.
SPEC.md §7 names product decisions Phase 4 has not made — deletion behavior (including
that deleting a collection means deleting its `.sqlite3` *and* its `.lance` sibling,
"neither inferable from the other's absence"), file permissions, backup scope, and
retention. Ingest is path-based and contained by `FileLoader(allowed_base_dir=...)`; an
HTTP ingest route widens that boundary from "what this operator typed" to "what any
caller reaching the port can name." And with no mutation, the residual exposure is
disclosure, which the loopback bind bounds and a shared secret would not.

Two Phase 3 constraints must be respected rather than routed around. ADR-0012 decision 2
keeps rerank out of `Retriever.search` and out of `SearchMode`, deferring it to "Phase 4,
where a service boundary makes 'rerank this request' a per-request option."
`RerankerNotConfiguredError` already exists so a missing extra can never look like a
successful rerank.

**Outbound, the guard SPEC.md §7 requires is genuinely absent.**
`EmbeddingConfig.base_url` is a free-form operator-controlled `str` with no validator,
and `OpenAICompatibleEmbedder` concatenates it into `{base_url}/v1/embeddings` and POSTs
with an `Authorization: Bearer` header. KNOWN_LIMITATIONS and SECURITY.md both argue
exploitability is low because "no network-facing caller can set `base_url`; the REST/MCP
surface that would expose one is a Phase-4 stub." Phase 4 builds that surface. The first
half stays true by decision 6 below, but "the exploit path does not exist yet" is no
longer available as a sentence.

**One claimed defect in that area was investigated and does not exist.** It was proposed
that `_post_json` accepts a 3xx response whose body is valid JSON, on the reasoning that
`raise_for_status()` raises only for 4xx and 5xx. That was tested directly against
unmodified source with an `httpx.MockTransport` returning a 302 carrying a well-formed
embedding body: the call **refused it**, raising `HTTPStatusError("Redirect response
'302 Found' … Redirect location: …")`. Modern httpx raises on an unfollowed redirect.
SPEC.md §7's `redirect: error` semantics are therefore **already satisfied**, and no
redirect defect exists to fix. What remains is hardening only (decision 11).

**One claimed defect is real and was confirmed by execution.** `_sanitize_url` redacts
every query-parameter value but rebuilds with `urlunsplit((scheme, netloc, path, …))`,
and `netloc` includes `user:password@`. Run against unmodified source,
`https://user:hunter2@api.example.com/v1/embeddings?api-key=sk-live-123` sanitizes to
`https://user:hunter2@api.example.com/v1/embeddings?api-key=***` — the query credential
redacted, the password verbatim. This is the ADR-0001 hazard 6 class the function exists
to close, missed on one URL component.

Finally, a defect found while designing: `SQLiteMetadataStore.open` **creates**
`<index_dir>/<collection>.sqlite3` when absent. A naive `index_status?collection=anything`
handler mutates the filesystem on every request for an unknown name.

## Decision

### 1. Every Phase 4 operation is read-only on both transports, and the shared-secret guard is not built

SPEC.md §7's clause is satisfied vacuously: the set of mutating operations is empty, so
the set requiring the header is empty. **This is a scope decision, not a scope
reduction** — SPEC.md §1.2's four tools are the complete named surface and ingest was
never among them. It is recorded here, in SECURITY.md and in KNOWN_LIMITATIONS.md so it
reads as a decision rather than a missing feature. Any later phase adding a mutating
operation must build the header, the constant-time compare and the unset-secret disable
**in the same change**, must make SPEC.md §7's four product decisions first, and must
supersede this ADR rather than amend it.

### 2. Read-only is enforced structurally by five independent checks

1. **A declarative registry is the only way to register an operation.** `service/tools.py`
   holds `TOOLS: Final[tuple[ToolSpec, ...]]`, and `ToolSpec.side_effect` is a closed
   `Literal["read_only"]` with exactly one member. Adding a mutating operation requires
   widening that `Literal`, which `mypy --strict` blocks until someone does it visibly.
2. **Registry↔route parity.** A test enumerates the app's `APIRoute`s, excludes the
   framework-owned doc paths, and asserts the `(path, method)` set equals the set derived
   from `TOOLS`. A route added with a bare decorator fails.
3. **Registry↔MCP parity.** The same assertion over the MCP server's registered tool
   names. The rule is per operation, not per transport, so an MCP-only tool is caught
   exactly as a REST-only route is.
4. **Handlers can only reach a read-only facade**, exposing search, chunk lookup,
   document sources, manifest and counts — nothing else. A test asserts its public member
   set against an exact allow-list, so a rename does not defeat it.
5. **An import scan over the service package** fails on any import of
   `groundkit.indexer`, `groundkit.ingestion.loaders`, or `groundkit.ingestion.pipeline`
   — the anti-dependency-scan pattern the portfolio already uses, applied one package
   down. It fires three steps upstream of the route that would have exposed a write.

### 3. A request can never open a collection that does not exist

The service resolves `<index_dir>/<collection>.sqlite3` and returns a not-found response
when absent; `SQLiteMetadataStore.open` is never called on a non-existent collection.
`--index-dir` must exist at startup or serving fails; the service never creates it.
Collection names pass the existing validation *before* the existence check, so a
traversal attempt is a validation failure rather than a not-found that leaked a probe
result. The test asserts both the response **and** that the index directory's contents
are unchanged.

This composes with ADR-0013 decision 9, which places the same refusal in the registry;
the service-side check is the boundary translation, not a second mechanism.

### 4. "Read-only" means no document, chunk, or manifest state changes — not that the process writes no bytes

Opening a WAL database updates its `-shm`/`-wal` sidecars, and `SQLiteMetadataStore.open`
runs `CREATE TABLE IF NOT EXISTS` and a best-effort chmod on every open. The claim is
scoped to durable content and is enforced in Python: the process holds a read-write file
handle. Opening with `mode=ro` was rejected (alternatives). Filesystem-level enforcement
is deferred to Phase 6, where a container can mount the index directory read-only.

### 5. One runtime, two transports, one registry

`grk serve` runs uvicorn over the FastAPI app and mounts the MCP streamable-HTTP
transport on the same app; `grk serve-mcp` runs the stdio transport. Both construct the
same read-only facade over the same `CollectionRuntime` registry (ADR-0013).

*Differs deliberately:* framing, error rendering, process lifecycle, and binding — only
the HTTP path binds a socket, so decision 7 applies to it alone. One stdio-specific rule
is load-bearing: **stdout carries JSON-RPC frames and nothing else**; logging goes to
stderr, and a test captures stdout across a tool call and asserts every line parses as
JSON-RPC.

*Must not differ:* the operation set (checks 2 and 3), request and response models — the
MCP tool input schema is **generated** from the same Pydantic model rather than
hand-written a second time, which is precisely the seam drift ADR-0001 hazard 4 exists to
catch — read-only-ness, the error code for a given exception type, and config provenance.

### 6. Embedder, reranker, index directory and containment root come from serve-time resolution, unreachable from a request

The service receives a fully-constructed `EmbeddingConfig`. Resolution happens at serve
time through the **promoted** `_resolve_embedding_config` — moved out of `cli.py` into a
shared module rather than imported as a private CLI helper, and rather than copied. The
argparse adapter stays in `cli.py`; the shared function takes explicit keyword parameters
typed as `EmbeddingConfig`'s own field types, preserving the property the original
docstring defends (namespace attributes are `Any`, so the provider `Literal` needs no
cast at the call site) while keeping both rules it enforces: defaults come from a fresh
`EmbeddingConfig()` rather than a second copy of its field defaults, and `ValidationError`
is translated to `ConfigurationError` at that single construction site.

**This makes the constraint hold by construction:** there is no resolution path inside
`service/` for a request to reach. It is nonetheless made **executable**, because a
structural argument decays as the surface grows. A test asserts that **no MCP tool input
model and no REST route schema accepts any `base_url`, `index_dir`, `base_dir`, or
`embed_*` field** — walking the generated OpenAPI component schemas *and* every registered
MCP tool's generated input schema, so models added later are covered without the test
being told about them. Every request model is `frozen=True, extra="forbid"`, so an unknown
field is a rejection rather than a silent ignore. That test is what keeps SECURITY.md's
sentence "no network-facing caller can set `base_url`" true.

`--base-dir` is **required** for both serve commands: it is the containment root
`resolve_citation` checks against, and a service that cannot verify any citation should
not start, given that verifiable citations are the product claim (SPEC.md §2).

### 7. The service binds loopback, and this is load-bearing rather than a default

`--host` defaults to a named `DEFAULT_SERVE_HOST = "127.0.0.1"` and `--port` to a named
`DEFAULT_SERVE_PORT`. At startup the host is classified with `ipaddress`; a non-loopback
host is refused unless `--allow-remote-access` is also passed, in which case a warning
names what is exposed.

**Keeping this is not optional and its rationale is independent of the shared secret.**
SPEC.md §7 states the bind in the same sentence as the header, so dropping the guard
makes it easy to drop the bind with it. With decision 1 there is **no authentication of
any kind**, and SPEC.md §7 records that SQLite is content-bearing: a `search` response
carries document text and absolute source paths, and `index_status` and
`list_collections` carry collection topology. The bind is therefore the service's only
access control, and KNOWN_LIMITATIONS names it as load-bearing rather than as a default.
Tests pin the default, the refusal, the acknowledged override, and both IPv4 and IPv6
loopback.

Note the polarity inversion against decision 10, so nobody "unifies" them: the same
`ipaddress` classification **requires** loopback for the inbound bind and **rejects** it
for outbound provider endpoints.

### 8. Response models reuse `contracts.py`; exactly two thin wrappers are new

`search` returns the existing **`SearchResponse`** unchanged, so a client parsing
`grk search --json` and one parsing the REST route parse the same shape.
`list_collections` returns a list of names and needs no model at all.

Two wrappers are irreducible, and the reason each is irreducible is recorded so neither
is mistaken for a parallel DTO:

- **`index_status`** nests the existing **`CollectionManifest`** for the identity half —
  `contracts.py` already documents it as identity minus operational settings, exactly the
  leak-safe subset a status endpoint needs. What no contract models is *counts*, so the
  wrapper adds document and chunk counts, dense availability, schema version, and the
  ADR-0013 generation. It returns no document sources, no chunk content, no queries, no
  index directory, no containment root, no `base_url`, no `api_key_env`.
- **`fetch_chunk`** nests the existing **`Citation`** and adds content plus a verification
  verdict. It cannot reuse `RetrievalResult`: that model requires `score: float =
  Field(ge=0.0)`, and a fetch has no score — supplying `0.0` invents a number SPEC.md §2
  forbids and drives `is_high_confidence` off a fabricated value. `Chunk` cannot serve
  either: it carries no `source`, which only `Document` holds (ADR-0006).

`ToolSpec` and the request models are likewise new, but they model the transport, not the
domain; nothing in `contracts.py` describes an operation registry or a request.

### 9. Bounds and errors reuse what exists; no new exception types

**`MAX_TOP_K` is imported from `retrieval/search.py`**, never redeclared. ARP's own
`tools.py` carries a private `_MAX_TOP_K = 50`; porting that constant alongside the code
would create a second source of truth for one bound, which is the drift `MAX_TOP_K`'s
single definition already prevents — `cli.py`, `retrieval/__init__.py` and
`evals/runner.py` all import the one constant.

**`MAX_QUERY_LEN` is genuinely new** — nothing in groundkit bounds query length today. It
lands as a named constant beside `MAX_TOP_K` in `retrieval/search.py`, never inline
(SPEC.md §5.2: thresholds are named constants, not inline literals).

**`errors.py`'s existing hierarchy is sufficient and the service defines no new exception
type.** It maps the existing types to HTTP status codes and JSON-RPC error codes through
one shared function used identically by both transports, matched subclass-first — a
mis-ordered chain is a real bug class and a test pins the order. Only an allow-list of
types has its own message returned to a caller; every other type returns a fixed detail.

Four points that are decisions rather than table entries:

- **`RetrievalError` maps to a conflict rather than a bad request, and that is only
  correct because the request schema makes the caller-error cases unreachable.**
  `Retriever.search` raises it both for an empty query and out-of-range `top_k` — caller
  errors — and for index inconsistency, a server-side data fault. The type does not
  distinguish them and message-matching is not an option. The request model therefore
  pins `query` to a minimum length and at most `MAX_QUERY_LEN`, and `top_k` to
  `1..MAX_TOP_K`, so those never reach the handler. A test pins that precondition; if it
  ever stops holding, the mapping becomes wrong, and that test is what says so.
- **`RerankerNotConfiguredError` maps to not-implemented**, not bad-request or
  unavailable: the request is well-formed and would be servable on an install with the
  extra, and a missing optional extra is not transient.
- **`ConfigurationError` maps to bad-request** only because every reachable one in Phase 4
  is caused by a request field — an invalid collection name, or a dense mode against a
  collection with no manifest (ADR-0008). Startup config faults exit non-zero before the
  socket binds. An endpoint rejection (decision 10) is the operator-fault case, and it
  does not reach this mapping: it is raised inside the embedding call path, where the
  existing `_raise_embedding_error` boundary already converts it, so it surfaces as the
  embedding-backend mapping with a fixed detail.
- **Credential safety is by construction, not by scrubbing at the edge.** No
  `EmbeddingError` message reaches a response body — that class returns a fixed detail —
  so the sanitized-URL and scrubbed-secret text stays server-side. The mapper never reads
  `__cause__`, `__context__`, `repr(exc)`, or a traceback: the ADR-0001 hazard 6
  chain-severing discipline applied at the egress boundary, with a test that plants a
  sentinel in a `__cause__` and asserts it appears in neither body nor headers. Every
  response carries a request id, and the full unscrubbed exception is logged server-side
  against it. Query text is DEBUG only; a test asserts the INFO access log carries id,
  route, status, latency and result count, and never the query (SPEC.md §3).

### 10. Outbound endpoints are validated by a new `utils/url_safety.py`, shaped as `path_safety`'s peer

It carries over the `ensure_*` naming, the validate-at-the-boundary/raise-typed/never-coerce
contract, and the docstring discipline of stating *why* a check is spelled as it is. Two
divergences are stated rather than mimicked: `path_safety` inlines its containment check
into both public helpers because that is the barrier pattern CodeQL recognizes for
`py/path-injection`, and there is no comparably recognized sanitizer for
`py/request-forgery`, so the check lives in one place; and no non-raising `is_safe_*` peer
ships, because nothing needs one.

**It raises `ConfigurationError`** — an endpoint is operator configuration, which is that
type's documented meaning. No new exception type is introduced.

**The check runs in two parts.** *Shape*, at embedder construction, no DNS: scheme
allow-list, non-empty host, and rejection of userinfo of any kind (a credential does not
belong in a config URL, and `http://expected.example.com@evil.example/` is the classic
parser-confusion vector), a query string or fragment (the module concatenates paths onto
`base_url`), and a host that looks like an IPv4 address but does not parse as one. The
message never echoes the URL, because the URL may hold the credential being rejected.
*Address*, per request immediately before the POST, resolved through a thread so a
blocking resolver cannot stall the event loop, with an injectable resolver seam so no
unit test touches the network. Construction-time-only checking was rejected: a service
binds once and serves for days, which is the widest possible window between check and
connect.

**Classification uses `ipaddress`, never a regex, and unmaps before classifying.**
`IPv6Address("::ffff:127.0.0.1").is_loopback` is **False** — the IPv6 loopback test is
equality with `::1` — so a classifier reading `.is_loopback` on the un-unmapped address
admits the exact IPv4-mapped spelling SPEC.md §7 names. `.is_private` does consult
`ipv4_mapped` in current CPython, but that behavior has moved across point releases, so
relying on it makes the guard's correctness a function of the interpreter patch version.
Mapped, 6to4 and Teredo forms are unmapped explicitly, then rejected on loopback,
private, link-local, multicast, reserved or unspecified, returning which predicate fired.

**Every resolved address is checked, not the literal only**, and that step is what closes
non-decimal IPv4 spellings. A host that already parses as a literal is classified without
DNS, so the default loopback endpoint costs no resolution. Otherwise every address the
resolver returns is classified and any unsafe one refuses. The mechanism is worth stating
because it shows the two steps are not redundant: Python's `ipaddress` deliberately
**rejects** `0177.0.0.1`, `0x7f.1` and `2130706433`, so they fall through as hostnames,
while glibc's resolver **accepts** all three and returns `127.0.0.1`. The resolver check,
not the literal parser, is the barrier.

**The Ollama exception is scoped around, as a class attribute.** `_HttpEmbedder` gains a
`ClassVar` defaulting to `False`; `OllamaEmbedder` sets it `True`;
`OpenAICompatibleEmbedder` inherits `False`. A `ClassVar` rather than a constructor
keyword, because a keyword would make requesting the exception for any provider a legal
thing to write — scoping *over* the guard rather than around it. There is no parameter,
so there is nothing to pass. It is named for permitting private endpoints generally, not
loopback specifically, because SPEC.md §9's Phase 6 compose topology reaches Ollama at a
bridge-network address; a flag named for loopback that also admits RFC1918 is the
name/behaviour mismatch this repo keeps catching. It does **not** relax the shape check.

### 11. Redirect handling is hardened, not fixed

`follow_redirects=False` is pinned explicitly on the embedding client, which today
inherits it as an httpx default. **This closes no present hole** — an unfollowed 3xx
already raises through `raise_for_status()`, verified by execution against unmodified
source — and the docstring must say so rather than implying otherwise. It is a pin against
a future httpx default change and against an injected test client that sets it `True`.

### 12. `service/*` and `utils/url_safety.py` stay outside the coverage core subset

SPEC.md §8 defines the subset as retrieval, chunking, scoring and citation resolution; a
transport layer is none of those, and widening it to mean "anything important" dissolves
what it measures. Both remain under the whole-package gate, and the security-load-bearing
behaviour is proved by the *named* tests in decisions 2, 3, 6, 7 and 9 — a stronger
control, since a coverage number cannot distinguish an enforcement test from a decorative
one. Recorded for the same reason `pyproject.toml` records why `index/metadata.py` stays
out. Note `src/groundkit/runtime.py` **is** added to the subset, explicitly, per ADR-0013
decision 8.

### 13. Any new Protocol seam gets a signature-parity conformance entry

Any `typing.Protocol` this surface introduces gains an entry in
`tests/test_protocol_conformance.py` using `assert_signature_parity`, never `isinstance`
— which compares member names only and would pass through the `query` → `_query` rename
that caused ARP's signature drift (ADR-0001 hazard 4).

## Alternatives considered

- **Ship the shared-secret guard anyway over read-only routes.** Rejected: a
  constant-time compare protecting nothing makes the next reader believe the mutation
  question was answered, and establishes a half-built control a later mutating route
  could be attached to without re-deriving SPEC.md §7's four product decisions.
- **Include one mutating operation — ingest — behind the secret.** Rejected: it is not in
  SPEC.md §1.2's tool list, and it widens containment from a local operator's typed
  argument to any caller reaching the port. The CLI covers the use case with a strictly
  smaller surface.
- **Enforce read-only by HTTP method — every route a GET.** Rejected for a specific
  reason: access logs record the request line at INFO, so a query in a query string
  writes every query into an INFO log, directly against SPEC.md §3. `search` is therefore
  a POST with a body, which means method alone cannot certify read-only-ness and the
  registry has to.
- **Open SQLite `mode=ro` for OS-level enforcement.** Rejected for Phase 4: a read-only
  connection to a WAL database still needs the `-shm` file and cannot perform WAL
  recovery, so it fails in exactly the situations a restart after unclean shutdown
  produces. Trading a reliable code-level guard for an OS-level one that intermittently
  refuses to start is the wrong trade; the clean fix is a read-only mount in Phase 6.
- **A fourth `SearchMode` for rerank.** Rejected — ADR-0012 decision 2, unchanged.
- **Expose `rerank_candidates` as a request field.** Rejected for ADR-0012 decision 1a's
  reason: candidate depth decides whether the `@k` metrics could move at all, and a
  per-request knob makes two responses silently incomparable in the dimension a reader is
  least likely to check.
- **New response DTOs for every operation.** Rejected: `SearchResponse`, `Citation` and
  `CollectionManifest` already model the domain, and a parallel DTO layer would let the
  HTTP surface drift from `grk search --json`. Only the two irreducible wrappers in
  decision 8 are new.
- **A new `UnsafeEndpointError` type.** Rejected: `errors.py`'s existing hierarchy is
  sufficient, an endpoint is operator configuration, and the embedding-path boundary
  already produces the right transport mapping without a new type.
- **Strip the stale `metadata["source"]` copy from responses.** Rejected: it would make
  the HTTP response diverge from `grk search --json` for the same query. The clean fix —
  removing the ingest-time copy — is already an open KNOWN_LIMITATIONS item.
- **Two processes, one per transport.** Rejected: SPEC.md §7 states the rule is per
  operation because Phase 4 ships both surfaces over one runtime, and two runtimes means
  two caches over one index, doubling the rebuild and giving two snapshots that drift.
- **A regex address classifier**, and **classifying with `.is_private` alone**. Rejected
  per decision 10.
- **A constructor keyword for the Ollama allowance.** Rejected per decision 10.
- **`trust_env=False` to close the proxy bypass.** Rejected: it also disables
  `SSL_CERT_FILE` and `.netrc`, which operators legitimately need, to close a bypass in
  the same trust domain as `base_url` itself.
- **Extend the guard to URL ingestion now.** Rejected: URL ingestion is unscheduled
  (KNOWN_LIMITATIONS), so it would be code with no caller and no test that could fail.

## Consequences

- **The service discloses document content and absolute filesystem paths to anyone who
  can reach the port, with no authentication.** That is the honest cost of decision 1 and
  the entire justification for decision 7. An operator passing `--allow-remote-access`
  has published their corpus, and SECURITY.md says so in those words.
- Every future phase adding an operation pays a five-check tax. That is the intent, and
  it is also real friction: a Phase 5 author adding cited synthesis — a read — touches
  the registry, both parity tests and the schema module before their route runs.
- The `RetrievalError` mapping rests on a schema precondition rather than on the type
  system. One test guards it; that is thinner than the rest of this ADR and is named as
  such.
- Reranking a hybrid search returns a ranking no `grk search --mode hybrid` produces,
  because RRF is not depth-invariant and the set fused at the wider candidate depth
  differs in membership. The response records the input depth so a client can tell.
- **The Ollama allowance has no test-visible override**, so existing tests pointing
  `OpenAICompatibleEmbedder` at a loopback mock will start failing and must be re-pointed
  at a public-looking hostname with an injected resolver. This is a real, non-trivial
  test-suite change, not a footnote.
- **DNS rebinding is not closed.** Between the guard's resolution and httpx's own
  connect-time resolution the answer can change. The window is two resolutions in one
  process — small, not zero. Closing it needs connections pinned to the validated address
  with the original host in `Host` and SNI, which is not built. Pretending otherwise would
  be a false claim in SECURITY.md.
- **The proxy bypass is not closed** (decision 9's alternatives). The threat model is
  operator misconfiguration, not a hostile operator.
- `EmbeddingProtocol.embed` can now raise `ConfigurationError` from the guard, so
  `_HttpEmbedder.embed`'s `Raises:` docstring must be updated or it becomes wrong.
- **Most tests this ADR owes cannot be shown to fail against unfixed source**, because
  `service/api.py` and `service/mcp_server.py` are docstring-only stubs and
  `utils/url_safety.py` does not exist. That is stated plainly in the implementation notes
  rather than implied away.

## Amendment (2026-08-15) — decision 10's `.is_loopback` claim is version-dependent

Decision 10 states that `IPv6Address("::ffff:127.0.0.1").is_loopback` is
**False**, and uses that to argue a classifier reading the property admits the
IPv4-mapped spelling. **That is true only of older CPython patch releases.** On
Python 3.11.9 it is `False`; on the later 3.11, 3.12 and 3.13 patches CI runs it
is `True`, because the property now consults `ipv4_mapped` — the same migration
decision 10 already documented for `.is_private`.

**No implementation changes and no hole is opened.** `service/binding.py` and
`utils/url_safety.py` both unmap explicitly *before* classifying, precisely so
the verdict is not a function of the interpreter patch version, and both behave
identically under either stdlib behaviour. Decision 10's *reasoning* — do not
depend on these properties' treatment of mapped addresses — is strengthened by
this, not weakened; only its parenthetical statement of fact needs the caveat.

Found the way such things should be: a test asserted the stdlib behaviour
directly, passed on the developer's 3.11.9, and failed on all three CI versions.
That assertion has been removed — the test now pins the **guard's** behaviour and
says why asserting the interpreter's would be pinning something that drifts.

## Amendment (2026-08-15) — the not-found/bad-request boundary, settled

Decision 9 maps `ConfigurationError` to a 400/`invalid_request`. Decision 3
calls a request against a non-existent collection a *not-found*.
`CollectionRegistry` itself raises `ConfigurationError` for exactly that
case. Read against each other without their call sites, those two decisions
look like they conflict.

They do not, because decision 3 was never describing an exception path.
**The not-found is produced structurally, not by discrimination between
exception types.** `service/errors.py`'s `check_collection` runs as a
precondition at the transport boundary, before any handler — and therefore
before the registry — is reached: it validates the collection name, then
checks `<index_dir>/<collection>.sqlite3` for existence, and returns a 404
`ErrorRendering` directly when it is absent. The registry's own
`ConfigurationError` for an unknown collection is defense-in-depth that a
well-formed request never reaches, not a second opinion the mapper has to
arbitrate between. No message is matched against the exception and no new
exception type was introduced to distinguish the two cases — the two
answers come from two different places in the call, which is why this is
robust rather than a string comparison waiting to rot.

**Name validation runs before the existence check**, and the order is
itself the security property: a traversal attempt in the collection name is
reported as a validation failure, not as a not-found that would have
confirmed whether the traversed-to path exists. A probe learns nothing
either way.

**The residual, recorded rather than left to omission: a missing
`chunk_id` surfaces as 400, not 404.** `handle_fetch_chunk` raises
`ConfigurationError` for it, and there is no `check_chunk` precondition
analogous to `check_collection` — a chunk cannot be checked for existence
without first doing the lookup the handler exists to perform. Separating
that case from an invalid collection name would need either a new
exception type (forbidden by decision 9) or message matching (forbidden
for fragility), and this ADR takes neither. 400 is also the reading
decision 9's own rationale already gives: "every reachable one in Phase 4
is caused by a request field," and `chunk_id` is a request field exactly as
a collection name is.

This amendment revises neither decision 3 nor decision 9; it records how
they compose at the one call site where they meet.
`src/groundkit/service/errors.py`'s module docstring states the same
resolution for the same reason — a reader who hits a 400 on a `GET` for a
missing chunk is meant to find it in either place.

## References

- SPEC.md **§1.2** (the four tools, cited as the authority that ingest is not among them),
  §2, §3, §4, §5.2 (named constants), §7, §8, §9.
- [ADR-0012](ADR-0012-rerank-eval-stage-reorders-upstream-stage.md) — decision 2's
  deferral, discharged here; decision 1a's candidate depth adopted verbatim.
- [ADR-0013](ADR-0013-collection-runtime-persisted-staleness-marker.md) — the runtime this
  surface consumes, and its registry-side refusal that decision 3 translates.
- [ADR-0007](ADR-0007-default-retrieval-mode.md),
  [ADR-0008](ADR-0008-dense-search-requires-a-dense-collection.md),
  [ADR-0006](ADR-0006-dense-seam-returns-chunk-score-pairs.md).
- [ADR-0001](ADR-0001-promote-vs-rewrite.md) hazard 4 (the drift the generated MCP schema
  avoids) and hazard 6 (the `__cause__` chain, and the userinfo component `_sanitize_url`
  missed).
- `src/groundkit/utils/path_safety.py` — the design peer, and the two deliberate
  divergences.
- `src/groundkit/contracts.py` — `SearchResponse`, `Citation`, `CollectionManifest`,
  reused rather than mirrored.

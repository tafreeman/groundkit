# Security

## Reporting

Report suspected vulnerabilities via GitHub private vulnerability reporting on
this repository. Please do not open public issues for security reports.

## Operational scope — honest statement (Phase 6)

BM25 and hybrid retrieval, a persisted SQLite (+ optional LanceDB) index, and
citation-bearing search work end-to-end locally against file/directory
input, contained to an allowed base directory via ported `path_safety`
(`ensure_within_base`). The REST API and MCP server are real and
installable, not stubs: `grk serve` runs a FastAPI REST surface and the MCP
streamable-HTTP transport on one process, and `grk serve-mcp` runs the
stdio transport. Both mirror the same four tools SPEC.md §1.2 names —
`search`, `fetch_chunk`, `list_collections`, `index_status` — and nothing
else (ADR-0014).

**Every Phase 4 operation is read-only on both transports, and the
shared-secret header SPEC.md §7 requires for mutating routes is deliberately
not built.** SPEC.md §7's requirement is satisfied *vacuously*: the set of
mutating operations is empty, so the set of operations requiring the header
is empty. This is a **scope decision, not a scope reduction** — SPEC.md
§1.2's four tools are the complete named surface, and ingest was never among
them (ADR-0014 decision 1). Any later phase that adds a mutating operation
must build the header, the constant-time compare, and the unset-secret
disable **in the same change** that adds the operation, and must supersede
ADR-0014 rather than amend it.

**The service discloses document content and absolute filesystem paths to
anyone who can reach the port, with no authentication of any kind.** An
operator passing `--allow-remote-access` has published their corpus. The
127.0.0.1 default bind is therefore the service's only access control
(ADR-0014 decision 7) and it is load-bearing, not a default one can safely
override without consequence: a non-loopback `--host` is refused unless
`--allow-remote-access` is also passed, and passing it prints a warning
naming what is exposed.

**Outbound endpoint safety.** `utils/url_safety.py` validates every
cloud-provider embedding endpoint in two parts (ADR-0014 decision 10):
*shape*, checked once at embedder construction — a scheme allow-list, a
non-empty host, and rejection of userinfo, a query string, or a fragment on
the URL; and *address*, resolved immediately before every request, with
every address the resolver returns (not just a literal) classified with
`ipaddress` — never a regex — and unmapped before classification so an
IPv4-mapped IPv6 literal cannot slip past a `.is_loopback` check that reads
`False` on the mapped form. Loopback, private, link-local, multicast,
reserved, and unspecified addresses are all rejected, with one named
exception: local Ollama traffic, scoped as a `ClassVar` on `OllamaEmbedder`
rather than a constructor parameter, so requesting the exception for any
other provider is not a legal thing to write. **Note the polarity inversion
against the inbound bind, so it is never "unified" with it**: the same
`ipaddress` classification *requires* loopback for the inbound bind and
*rejects* it for outbound provider endpoints.

**URL ingestion shares the address check, not the shape check.**
`grk ingest <url>` (`UrlLoader`, ADR-0016 Wave 4) is now a third caller of
this module's *address* validation: every fetch calls the same
`ensure_safe_endpoint(..., allow_private_endpoint=False)` immediately before
the request, unconditionally — `UrlLoader` has no private-endpoint exception
the way `OllamaEmbedder` does. Its *shape* check is its own, not the
embedder's: an embedding endpoint URL is rejected for carrying any query
string at all, while a document URL legitimately needs one for identity, so
`UrlLoader` instead rejects userinfo and rejects a query parameter whose name
is credential-shaped — `token`, `key`, `secret`, `password`, and the rest of
`CREDENTIAL_QUERY_PARAMS`, via the shared `credential_query_params` helper.
That check is a denylist, not a closed set — `code` and `state` are excluded
as ambiguous (OAuth credentials in one context, an ordinary country or
US-state code in another), so a credential under an unlisted parameter name
still reaches the fetch. Redirects are refused rather than followed
(`follow_redirects=False` per request; a 3xx status raises rather than being
chased), and the response body is bounded by a byte cap and refused, never
truncated, past it.

The userinfo credential leak in `_sanitize_url` is **fixed**. It previously
rebuilt the sanitized URL's `netloc` including `user:password@`, redacting
the query parameter's value while leaving the password verbatim in the
string that reached logs. Confirmed against unmodified source before the
fix: `https://user:hunter2@api.example.com/v1/embeddings?api-key=sk-live-123`
sanitized to `https://user:hunter2@api.example.com/v1/embeddings?api-key=***`
— the query credential redacted, the password not.

**Two things this does not close, for either outbound caller, stated plainly
rather than implied away:**

- **DNS rebinding is not closed.** Between `url_safety`'s resolution and
  httpx's own connect-time resolution, the answer can change — the window
  is two resolutions in one process, small, not zero. This applies equally
  to a cloud embedding endpoint and to a URL fetched by `UrlLoader`, since
  both resolve through the same `ensure_safe_endpoint` and then let httpx
  make its own connection. Closing it needs connections pinned to the
  validated address with the original host preserved in `Host` and SNI,
  which is not built.
- **The proxy bypass is not closed.** `trust_env=False` was rejected because
  it also disables `SSL_CERT_FILE` and `.netrc`, which operators
  legitimately need, to close a bypass that sits in the same trust domain as
  `base_url` itself. The threat model here is operator misconfiguration, not
  a hostile operator, and it applies to `UrlLoader`'s client construction the
  same way it applies to the embedding client.

**Redirect handling is a hardening pin, not a fix.** `follow_redirects=False`
is now pinned explicitly on the embedding client. This closes no present
hole: tested directly against unmodified source with an `httpx.MockTransport`
returning a well-formed embedding body behind a 302, the call already
refused it — an unfollowed 3xx raises through `raise_for_status()`. The pin
guards against a future httpx default change and against an injected test
client that sets it `True` (ADR-0014 decision 11).

**No network-facing caller can set `base_url`, `index_dir`, `base_dir`, or
any `embed_*` field.** Every request model is `frozen=True, extra="forbid"`,
and a test walks the generated OpenAPI component schemas *and* every
registered MCP tool's generated input schema to confirm none accepts them
(ADR-0014 decision 6). Embedder, reranker, index directory, and containment
root all come from serve-time resolution and are unreachable from a
request. This sentence used to be true because the surface did not exist;
it is now true because the surface exists and is tested.

This section is rewritten as each phase lands real surface, per SPEC.md §7:

- Rate limiting, when it arrives, is process-local — not a distributed or
  DoS-grade control, and it will be documented as such.
- Redaction at the LLM boundary (structural patterns by default, names only
  via configured patterns) shipped in Phase 5 and runs on every cloud **chat**
  call (`build_chat` wraps it in `RedactingChat`, no operator opt-out); local
  mode sends nothing anywhere. It deliberately does not run on the cloud
  **embedding** boundary — a recorded deviation, ADR-0017 decision 5, detailed
  in [the LLM boundary doc](https://tafreeman.github.io/groundkit/architecture/llm-boundary/)
  — so redaction
  is pattern-based, covers one boundary, and does not guarantee removal of
  all sensitive content even there.

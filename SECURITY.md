# Security

**What this page is:** groundkit's honest, current statement of what is and
isn't guarded, plus how to report a problem. **Who needs it:** anyone
deploying groundkit anywhere other than their own laptop, anyone about to
expose its network surface beyond `127.0.0.1`, or anyone who wants the
precise boundary of what "local-first" and "read-only" actually guarantee
before relying on either. Below is a security policy in the ordinary sense
— how to report a vulnerability — and then this project's most exacting
document: every claim after that is either enforced by a test or explicitly
marked as not yet closed, never left implied. If terms like *MCP*,
*citation*, or *local-first* are unfamiliar,
[Concepts](https://tafreeman.github.io/groundkit/concepts/) explains each
in plain language; this page assumes you know what groundkit does and is
precise about what it does not guard.

## Reporting

Report suspected vulnerabilities via GitHub private vulnerability reporting on
this repository. Please do not open public issues for security reports.

## Operational scope — honest statement (Phase 6)

BM25 and hybrid retrieval, a persisted SQLite (+ optional LanceDB) index, and
[citation-bearing search](https://tafreeman.github.io/groundkit/concepts/#grounding-and-why-verified-beats-asserted)
work end-to-end locally against file/directory
input, contained to an allowed base directory via ported `path_safety`
(`ensure_within_base`). The REST API and
[MCP server](https://tafreeman.github.io/groundkit/concepts/#mcp-how-an-assistant-like-claude-searches-your-documents)
are real and
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

**Inbound DNS rebinding is closed, on both transports** (ADR-0024). The bind
alone was not a boundary against a browser: a page on any site the victim
visits can re-point its own hostname at `127.0.0.1` with a short-TTL answer,
after which the browser treats this service as same-origin — no CORS
preflight, response fully readable — while the connection genuinely arrives
from loopback. The `Host` header is the only part of such a request that
still names the attacker, and nothing inspected it. Both surfaces now do,
from **one** allow-list derived at serve time from the bind address:
Starlette's `TrustedHostMiddleware` on the REST app (which also covers the
mounted `/mcp` sub-path) and the MCP SDK's own
`TransportSecuritySettings` on the streamable-HTTP transport, which the SDK
defaults *off* for backwards compatibility and which the lower-level API
groundkit uses does not auto-enable. Only this machine's own names —
`127.0.0.1`, `localhost`, `[::1]`, with or without a port, **plus the address
actually bound** if it is not one of those three — are accepted, and the MCP
transport additionally refuses any `Origin` outside the same set (an absent
`Origin`, which is what a non-browser client sends, is allowed). The bound
address is on the list because loopback is a whole `/8`: `grk serve --host
127.0.0.2` is a legal loopback bind that none of the three canonical spellings
names, and a fixed list would start that server and then refuse every
legitimate client on both transports. A forged `Host` is refused on the REST
surface *and* on `/mcp`, shown to fail against unfixed source in both
directions, in `tests/test_service_host_validation.py`.

**Whether the check is enforced is decided by resolving the bind host, not by
whether it was typed as a loopback literal.** Those are different questions
for exactly one input — a hostname that resolves to loopback — and that one
input is where the check matters most: `--host localhost
--allow-remote-access` binds a socket nothing off-box can route to, which is
precisely the situation DNS rebinding exists to attack. The derivation
therefore resolves a name and keeps the check on when every answer is
loopback, and keeps it on when the name does not resolve at all, since
switching it off requires positive evidence that the socket is routable. An
address literal is never resolved. The bind guard itself still refuses to
resolve, and the asymmetry is deliberate: a wrong answer there would publish a
corpus, while a wrong answer in the allow-list costs at most a refused client
(ADR-0024 decision 3). Stated plainly rather than implied: this leaves the same
kind of two-resolution window the **outbound** direction has below, since the
ASGI server resolves the name again to bind it. Only one direction of it
matters — a name answering routable to the derivation and loopback to the bind
would leave an unrestricted list on a loopback socket — and reaching it takes
control of the DNS for the operator's *own* bind name plus
`--allow-remote-access`, which is deeper access than the attack this closes
needs. It does not apply to an address literal, and so not to the default bind.

Three residuals of that fix, named rather than implied:

- **Starlette admits any `[::…]`-shaped `Host` on the REST surface.** It
  strips the port by splitting on the *first* colon, so `[::1]:8765` reduces
  to `"["` — which the allow-list must therefore contain for any IPv6
  loopback client to be served at all, and which also matches other bracketed
  literals. Not reachable by rebinding: a bracketed address literal is never
  the product of a DNS answer, so no browser can be made to send one.
- **Neither matcher normalises `Host` for letter case or a trailing root
  dot.** `Host` is case-insensitive and the trailing-dot FQDN form is legal,
  but Starlette compares with `==` and the MCP SDK with `in`, so `LOCALHOST`,
  `LocalHost:8765` and `localhost.` are refused (400 and 421). This fails
  **closed** — it is a compatibility gap, never a widening — and is left open
  on purpose: the trailing dot is one extra entry, but case is not enumerable,
  so covering the cheap half would read as normalisation while being none. The
  fix, if a real client ever needs it, is a normalising matcher, which
  ADR-0024 weighs and rejects. A test pins the current behaviour so a change
  to it is deliberate.
- **A bind that is genuinely routable disables the `Host` check entirely**,
  together with the bind widening `--allow-remote-access` authorises, and logs
  a warning saying so. This is deliberate (ADR-0024 decision 3): once the port
  is routable, `Host` validation stops anyone who could not already connect
  directly, and a restrictive list would break every reverse-proxy and
  overlay-network deployment — including the one way an operator can put
  authentication in front of a service that ships none. Note what this is
  *not*: passing `--allow-remote-access` does not by itself disable the check.
  A bind that stays on loopback — including a hostname that resolves there —
  keeps it enforced, and says so in the log rather than claiming a publication
  that did not happen.

This does not change the **outbound** rebinding gap recorded further down;
that is the opposite direction and remains open.

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

- **Outbound DNS rebinding is not closed.** Between `url_safety`'s resolution and
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

# ADR-0024 — The loopback bind is not a boundary against a browser, so `Host` is validated on both transports

- **Status:** Accepted (owner, 2026-08-18)
- **Amends:** ADR-0014 decision 7 (the threat model behind it, not the bind guard itself)
- **Date:** 2026-08-18
- **Deciders:** Andy Freeman (owner)

## Context

ADR-0014 decision 1 ships **no authentication of any kind**, and decision 7 supplies the
argument for why that is acceptable: every operation is read-only, and the `127.0.0.1`
bind is the access control. `service/binding.py` enforces the bind and says so in its own
docstring — "the address the process binds is the entire boundary between that corpus and
everyone else."

That argument has an unstated premise: *a loopback socket can only be reached by something
already on the box.* At the IP layer that is true. Through a browser it is not.

**DNS rebinding.** A victim visits any page the attacker controls, on `rebind.example`.
The page's own name is served with a short TTL; after the page loads, the name is
re-answered as `127.0.0.1`. Every subsequent request the page makes to `rebind.example` is
**same-origin** as far as the browser is concerned — no CORS preflight is sent, no CORS
response header is required, and the response body is fully readable by the attacker's
JavaScript. The TCP connection genuinely originates from loopback. The corpus, the
document text, and the absolute filesystem paths in a `search` response all leave the
machine.

The one part of such a request that still names the attacker is the `Host` header: the
browser sends the name it thinks it is talking to, which is `rebind.example`, never
`127.0.0.1`. **Nothing in groundkit inspected it.**

Both transports were exposed, for different reasons:

- **MCP.** The SDK ships the mechanism (`TransportSecuritySettings`,
  `enable_dns_rebinding_protection`) and defaults it **off** for backwards compatibility.
  Its own convenience server auto-enables it whenever the host is loopback — for exactly
  this case. groundkit uses the lower-level `Server` / `StreamableHTTPSessionManager` API,
  which bypasses that override, and passed no `security_settings` at all.
  `create_session_manager`'s docstring *deferred* the question, arguing the allowed `Host`
  values "depend on the bind address, which lives in the CLI". That was plumbing described
  as a design constraint; the CLI already computes both `--host` and
  `--allow-remote-access`.
- **REST.** No `TrustedHostMiddleware`, no `Host` check of any kind.

## Decision

**The serve-time bind decision derives one `Host` allow-list, and both transports enforce
it.**

`service/binding.py` gains `derive_host_allow_list(host)`, returning a frozen
`HostAllowList`. It lives in that module for the reason the module already gives about the
bind constant: inbound access control is enforced in one named place, so widening it is a
visible edit rather than a default nobody reads. `cli._serve_http` calls it once, next to
the address it is about to bind, and hands the same object to `create_app` (Starlette's
`TrustedHostMiddleware`) and to `create_session_manager` (the SDK's
`TransportSecuritySettings`).

Four things follow, and each was a real choice:

### 1. The two matchers get two dialects derived from one list, not two hand-written lists

They do not agree on shape, and this was verified against the installed source of both
rather than assumed:

- The MCP SDK compares the whole `Host` header, port included, and understands a trailing
  `:*` implemented as `host.startswith(base + ":")`.
- `TrustedHostMiddleware` compares `headers["host"].split(":")[0]` — the port is stripped
  before the comparison, and there is no port wildcard.

So the Starlette list is *derived* from the MCP list by applying Starlette's own
reduction (`_trusted_host_pattern`), and a test asserts the derivation rather than the
literal contents. Two hand-written lists would drift, and drift between two guards is
decided by whichever is more permissive.

Both spellings of each address are listed on the MCP side — `127.0.0.1` **and**
`127.0.0.1:*` — because a request to port 80 omits the port from `Host` entirely and the
wildcard pattern does not match a portless value. An allow-list that refuses legitimate
clients is the failure mode where a security control gets switched off rather than
narrowed.

### 2. `localhost` is an allowed `Host` even though it is a refused bind

`_is_loopback_literal` refuses `localhost` as a `--host` value because a bind must be
classifiable without a resolver whose answer can change. A `Host` header is the opposite
kind of value: it is what a *client* claims, and a browser cannot be made to send
`Host: localhost` without having resolved `localhost` — which no attacker-controlled DNS
zone answers. The two verdicts are consistent because they are about different actors.

### 3. A **routable** bind is unrestricted — and a hostname is resolved to find out whether it is one

`derive_host_allow_list` returns `UNRESTRICTED_HOST_ALLOW_LIST` when the bind is routable.
Because `ensure_bindable_host` runs first and refuses every non-loopback *literal* the
operator did not acknowledge with `--allow-remote-access`, the bind address alone already
carries that flag's answer — which is why the derivation takes one parameter and cannot be
handed an address nobody consented to publish.

Unrestricted is the decision, not an omission. Once the socket is routable, `Host`
validation protects nothing: rebinding exists to bridge a browser onto a service the
attacker cannot route to, and an attacker who *can* route to it simply connects — to a
service with no authentication either way. It is not a silent no-op: it flips
`enable_dns_rebinding_protection` to `False` and `allowed_hosts` to `["*"]` together, and
emits a WARNING naming the address and stating that any `Host` is now accepted.

**The first draft of this decision branched on "is this a loopback address literal?" while
justifying itself with "once the socket is routable", and those two predicates differ at
exactly one input: a hostname that resolves to loopback.** So
`grk serve --host localhost --allow-remote-access` bound a socket nothing off-box could
route to and switched `Host` validation off on *both* transports — reinstating this
record's own CRITICAL on the one kind of bind where DNS rebinding is reachable at all. It
was worse than a silent hole: `ensure_bindable_host`'s error text steered the operator
into exactly that combination, and both warnings then stated the opposite of what had
happened, announcing a published corpus and calling `Host` validation "no protection at
all against anyone who can already route to this port" when nobody could route to it and
the browser path was the only one left. The decision is unchanged; what changed is that
the code now asks the question the decision is about.

**So the allow-list derivation resolves a hostname while the bind guard still refuses to,
and that asymmetry is the crux of the correction.** Both would be asking whether the
socket is reachable only from this machine, and a resolver's answer is mutable either way
— the ASGI server resolves the name again moments later, so the address the derivation saw
need not be the address bound. What differs is the *cost of being wrong*, and it is not
symmetric:

- For the **bind**, being wrong publishes a corpus. Resolution says loopback, the socket
  lands on a routable interface, and the operator was told the opposite by the guard whose
  whole job is to tell them. Hence literals only — unchanged, and `_is_loopback_literal`
  keeps its argument verbatim.
- For the **allow-list**, the two errors are different sizes. Resolution says loopback
  while the socket is really routable: the list is merely too *narrow*, legitimate clients
  get a 400 or a 421, and the operator meets it on their first request. Resolution says
  routable while the socket is really loopback: validation is off on the one bind
  rebinding can reach.

Resolution **errs closed** in the derivation and would **err open** in the bind guard,
which is why it is used in one and refused in the other. The same reasoning settles the
resolver-failure case: **a name that does not resolve stays restricted.** Going
unrestricted requires positive evidence that the socket is routable, and "the resolver did
not answer" is not evidence of anything. No competing error is invented for it either —
the bind that follows resolves the same name through the same resolver and fails there, in
the ASGI server's own words. Address literals are never resolved in either direction: a
literal is both what the derivation classifies and what actually gets bound, so a resolver
could only add a way to be wrong, plus a DNS round trip on the default bind's startup path.
A test pins that none is attempted for one.

The container default `--host 0.0.0.0 --allow-remote-access` (ADR-0021) is untouched:
`0.0.0.0` and `::` are literals, are not loopback, and stay unrestricted — which is what
the reverse-proxy argument under *Alternatives* requires. A regression test guards it.

Both warnings now say what happened. An acknowledged bind that resolves to loopback logs
that validation **stays enforced** and that nothing was published; `ensure_bindable_host`
separates a routable literal (which does publish) from a hostname (whose verdict it defers
to the derivation, and says so) rather than asserting the same exposure for both.

### 4. `Origin` is checked on the MCP transport, where the SDK offers it

The same object carries `mcp_allowed_origins`, populated with the `http` loopback origins
(`https` is excluded: this server speaks plain HTTP, and a page served over `https` cannot
fetch an `http` endpoint at all). The SDK treats an absent `Origin` as allowed, so
non-browser clients are unaffected, while a rebinding page — whose origin is its own name
and port — is refused a second time, independently of the `Host` check.

## Consequences

**The no-authentication position of ADR-0014 decision 1 still holds, and this is what
repairs its argument rather than replacing it.** The claim was never "loopback packets are
trustworthy"; it was "only software already running on this machine can reach this port."
Rebinding was a counterexample to that claim, not to the decision resting on it. With
`Host` validated, a request reaching the corpus must either originate from software on the
box that addressed this machine by one of its own names, or from a browser the user
themselves pointed at `127.0.0.1`. That is the boundary decision 1 assumed it had. A
shared-secret header would still be required the moment a mutating operation appears — that
obligation is unchanged and is ADR-0014's, not this record's.

**Two refusals, two status codes, deliberately not unified.** On the HTTP path the
middleware refuses first (Starlette: 400); mounted elsewhere, the transport refuses on its
own (the SDK: 421 for `Host`, 403 for `Origin`). The duplication is intentional — a
transport whose protection depends on the app someone wrapped it in has no protection of
its own — and each matcher renders its own refusal rather than being re-implemented to
agree. This is framing, which ADR-0014 decision 5 already classifies as *differs
deliberately* between transports.

**The allow-list is derived per bind, not a fixed tuple — because the bind guard accepts
more addresses than three spellings can name.** `_is_loopback_literal` accepts all of
`127.0.0.0/8` and `::ffff:127.0.0.0/104`, while the first draft's list named only
`127.0.0.1`, `localhost` and `[::1]`. The result was that `grk serve --host 127.0.0.2`
started without complaint and then refused *every* legitimate client — 400 on REST, 421 on
MCP — and that `--host ::ffff:127.0.0.1` was served by the REST matcher (only via the
`"["` residual below, i.e. by accident) while the MCP matcher had no entry for it at all:
the two surfaces disagreeing about the same request. A comment above the pattern tuple
asserted that "the ones that were not bound are unreachable", which is sound for narrowing
and silently assumed the bound address was one of the three; it now says what the tuple
actually is, a floor. `_restricted_allow_list` appends the bound address's own `Host`
spellings — bare and `:*`, bracketed for IPv6, plus the matching `http` origins — and the
Starlette list keeps being *derived* from the MCP one, so an addition cannot land on one
surface and be forgotten on the other. Two details are load-bearing and are pinned by
tests: the literal is listed **as typed** as well as canonically, because
`str(ip_address("::ffff:127.0.0.1"))` is `::ffff:7f00:1` while a client's `Host` carries
the dotted form; and a name that is not made of hostname characters contributes nothing,
because `--host '*'` would otherwise reduce to `"*"` — Starlette's allow-any wildcard —
and produce an "enforced" list that enforces nothing.

**Residual of resolving at all: a two-resolution window, in one direction.** The
derivation resolves the name and the ASGI server resolves it again to bind it, so the two
answers can differ — the same shape as the outbound window `SECURITY.md` records for
`utils/url_safety.py`, and it is not closed here either. Only one direction of the
disagreement is a security matter: routable to the derivation and loopback to the bind
leaves an unrestricted list on a loopback socket, which is Finding 1's shape reached by a
race. The other direction is an availability failure the operator meets immediately.
Reaching the dangerous one requires control of DNS for the operator's *own* bind name plus
`--allow-remote-access` — deeper access than the attack this record closes needs, and
strictly narrower than the pre-fix behaviour, which handed the same result to anyone who
typed `localhost`. It does not apply to an address literal, and therefore not to the
default bind. Closing it would mean binding to the resolved address rather than the name,
which is the ASGI server's decision to make, not this module's.

**Residual, unchanged and now recorded rather than implied: neither matcher normalises
`Host` for case or a trailing root dot.** `Host` is case-insensitive and the trailing-dot
FQDN form is legal, but Starlette compares with `==` and the MCP SDK with `in`, so
`LOCALHOST`, `LocalHost:8765` and `localhost.` are all refused — 400 and 421 respectively.
It fails **closed**, so this is a compatibility gap and never a widening. It is left open
deliberately: the trailing dot is one extra spelling per entry, but case is not enumerable
(an *n*-letter name has 2^n spellings), so adding the cheap half would produce a list that
reads as normalisation while being none, and would still need this paragraph. The real fix
is the normalising matcher the *Alternatives* section rejects below, and its trigger is a
real client that cannot be configured around it, not tidiness. Pinned by a test that
asserts the refusal, so a future change to it is visible rather than incidental. What
groundkit emits *is* canonicalised — an operator-supplied name is listed as typed and
lowercased — since that costs nothing and is this process's own spelling to choose.

**Residual, stated plainly: Starlette admits any `[::…]`-shaped `Host`.** Its reduction
splits on the *first* colon, so `[::1]:8765` and a bare `[::1]` both reduce to `"["`, and
`"["` is therefore what the allow-list must contain for an IPv6 loopback client to be
served at all. The consequence is that any other bracketed literal beginning `[::` also
passes the REST check. This is not reachable by the attack this record closes: a bracketed
address literal is never the product of a DNS answer, so no rebinding page can cause a
browser to send one. It is pinned by name in `tests/test_service_host_validation.py` and
listed in `SECURITY.md` rather than left to be rediscovered. One knock-on worth naming: an
IPv6 bind's appended spellings also reduce onto `"["`, so for those binds the per-bind
addition does its real work on the MCP list, which compares the whole header.

**Residual: the library constructors default to unrestricted.** `create_app` and
`create_session_manager` default `host_allow_list` to
`UNRESTRICTED_HOST_ALLOW_LIST`. They build objects; *serving* is what makes a `Host`
decision meaningful, and a caller embedding the app behind its own front door is not helped
by a check keyed to an address it never binds. `grk serve` always passes the derived list,
and the regression tests assert the property against the CLI-assembled app rather than
against those defaults — testing the defaults would assert the opposite of the security
property. `cli._build_mcp_mount` is CLI plumbing rather than a library seam, so it defaults
to the restricted list.

**Outbound rebinding is a different, still-open hazard.** `SECURITY.md` records that
`utils/url_safety.py` resolves an endpoint and then lets httpx resolve it again, leaving a
two-resolution window. That is the *outbound* direction and is untouched here; nothing in
this record narrows or widens it. The polarity note in `service/binding.py` applies as it
always did — the two modules classify the same addresses with opposite verdicts and must
not be unified.

## Alternatives considered

**Do nothing; rely on the bind.** Rejected: the premise is false, and the exposure is the
full corpus plus absolute filesystem paths, readable by any page the user visits. This is
the finding, not a hypothetical.

**Accept the bound address plus loopback when `--allow-remote-access` is passed.** Tighter
on its face, and rejected because it is wrong in every deployment that uses the flag. A
published service is reached through whatever name its clients, reverse proxy, or overlay
network use, and this process cannot know that name; binding `0.0.0.0` would mean requiring
`Host: 0.0.0.0`, which no client sends. It would break the reverse-proxy deployment
specifically — the one way an operator can put authentication in front of a service that
ships none — and a control that makes the documented escape hatch unusable gets removed
downstream rather than obeyed.

**Refuse `--host <hostname> --allow-remote-access` outright, so the ambiguous case never
arises.** This is the other way to make the predicate and the premise agree, and it is the
tempting one: with no name to classify, the derivation needs no resolver and decision 3's
"once the socket is routable" is true by construction. Rejected because the combination it
forbids is legitimate. `--host myhost.internal --allow-remote-access` is how an operator
binds one specific interface of a multi-homed host by the name their infrastructure knows
it by, and it is a *narrower* bind than the `0.0.0.0` the container default already
blesses — refusing it would push exactly that operator onto `0.0.0.0`, a strictly wider
exposure, in the name of a check. It also would not remove the resolver from the picture:
the ASGI server resolves the name either way, so the refusal buys certainty about a value
this process had already stopped depending on. Resolving in the derivation costs one
lookup on a path that is about to do the same lookup, and fails closed when it cannot.

**Add authentication instead.** Out of scope and, on its own, not a fix: a rebinding page
reaching an authenticated endpoint is refused, but so is every legitimate unauthenticated
client, and ADR-0014's vacuous-satisfaction argument for the shared-secret header is a
separate decision with its own trigger (the first mutating operation). `Host` validation is
what restores the premise ADR-0014 was already relying on.

**Write groundkit's own `Host` middleware, matching both dialects identically.** Rejected.
It would close the `"["` residual **and** the case/trailing-dot one recorded under
*Consequences*, at the cost of a hand-written security matcher this repo would then own and
have to keep correct against IPv6 spellings, ports, trailing dots, and case — the class of
parsing the module docstring already argues should be left to code that specializes in it.
Neither residual it would close is reachable by the threat: one is a widening no browser
can be made to exercise, the other a refusal. This is the fix if either ever stops being
acceptable, and the trigger is a real report, not tidiness.

**Pass `TransportSecuritySettings` on the MCP side only, and let the REST surface be
covered by it.** Rejected on the plain fact that it is not: the SDK's middleware runs
inside the streamable-HTTP transport and sees nothing addressed to `/v1/...`.

# ADR-0025 — The library constructors default `Host` validation on, not off

- **Status:** Accepted (owner, 2026-08-19)
- **Amends:** ADR-0024's Consequences section (the "library constructors default to
  unrestricted" residual specifically, not the rest of that record)
- **Date:** 2026-08-19
- **Deciders:** Andy Freeman (owner)

## Context

ADR-0024 gave `create_app` and `create_session_manager` a `host_allow_list` parameter
defaulting to `UNRESTRICTED_HOST_ALLOW_LIST`, and recorded that choice as a residual
rather than an oversight: "`create_app` builds an app object, while *serving* it is what
makes a `Host` decision meaningful, and `grk serve` always passes the derived list. A
caller that constructs the app for an in-process client, a test, or behind its own front
door is not helped by a check keyed to an address it never binds."

A post-merge security review of PR #26 flagged the same fact and drew a different
conclusion, confirmed by executing the code rather than reading the ADR alone: that
argument supports letting such a caller **opt out** of `Host` validation; it does not
support opting them out by default. A caller who embeds either constructor without ever
supplying `host_allow_list` — exactly the shape ADR-0024 names as the intended
beneficiary of the unrestricted default — gets no `Host` check at all, in a codebase
whose stated rule for every other unconfigured provider is to raise, never fall back
(`errors.py`'s `ProviderNotConfiguredError` family). `cli._build_mcp_mount` already
demonstrates the safer shape and always has: it is CLI plumbing, not a public library
seam, and it has defaulted to `LOOPBACK_HOST_ALLOW_LIST` since ADR-0024 landed — an
asymmetry that record never explained.

`LOOPBACK_HOST_ALLOW_LIST` already exists (`service/binding.py`), derived once from
`DEFAULT_SERVE_HOST` for exactly this purpose. Flipping the two service constructors to
it is not a new mechanism, only a new default.

## Decision

**`create_app`'s and `create_session_manager`'s `host_allow_list` parameter defaults to
`LOOPBACK_HOST_ALLOW_LIST`, not `UNRESTRICTED_HOST_ALLOW_LIST`.**

The public signature is unchanged — both parameters keep the same name, type, and
keyword-only position — so an embedder who already passes
`host_allow_list=UNRESTRICTED_HOST_ALLOW_LIST` explicitly keeps exactly today's
behavior, and `grk serve` is unaffected either way, since `cli._serve_http` always passes
the bind-derived list regardless of what the constructor would otherwise default to.
What changes is the behavior of a caller who passes neither: instead of silently
accepting any `Host`, they now get the same fail-closed posture the CLI's own default
bind has.

`cli._build_mcp_mount`'s default (`LOOPBACK_HOST_ALLOW_LIST`) is untouched — it already
had the safe default; the two service constructors now match it, closing the asymmetry
rather than introducing a new value.

## Consequences

**Nothing observable changes for the shipped binary.** `grk serve` derives its own
`Host` allow-list from the bind address and passes it explicitly to both constructors on
every call (`cli._serve_http`); it never relies on either default. The entire blast
radius of this decision is third-party code that embeds `create_app` or
`create_session_manager` directly and does not pass `host_allow_list`.

**A caller relying on the old default silently gets a stricter app, not a broken one.**
Before this decision, embedding either constructor bare produced an app that accepted
any `Host` header. After it, the same call produces an app that only accepts the
loopback spellings `LOOPBACK_HOST_ALLOW_LIST` names — `127.0.0.1`, `localhost`, `[::1]`,
and their `:*`-suffixed forms. A caller genuinely serving from a non-loopback address
behind a reverse proxy, under a name this process cannot predict, now gets 400s (REST)
or 421s (MCP) until they pass `host_allow_list=UNRESTRICTED_HOST_ALLOW_LIST` explicitly.
This is the correct failure mode for the class of bug ADR-0024's own no-authentication
argument names: an unconfigured security control should refuse traffic, not admit it.

**This closes one of ADR-0024's four named residuals, not all of them.** The other
three — case/trailing-dot normalization, Starlette's `[::…]`-shaped `Host` admission,
and the two-resolution window on a hostname bind — are untouched by this decision and
remain recorded in `SECURITY.md` and `KNOWN_LIMITATIONS.md`.

## Alternatives considered

**Leave the default as ADR-0024 recorded it.** Rejected: the argument for it only
supports an opt-out, and this codebase's own stated rule for every other unconfigured
provider is to raise or refuse, never silently fall back to the more permissive
behavior. Leaving it unrestricted keeps a footgun in the public API surface that this
repo's own test suite could never catch, because the CLI never exercises the default.

**Remove the default entirely; require every caller to pass `host_allow_list`
explicitly.** Rejected as unnecessarily strict for a library seam that already has a
well-named, safe value to default to (`LOOPBACK_HOST_ALLOW_LIST`) — the same value
`cli._build_mcp_mount` has defaulted to since ADR-0024. Forcing every embedder to think
about `Host` validation before their first successful request is a worse developer
experience than defaulting to the common case (a loopback embed) and letting the
uncommon one (a routable embed behind a proxy) opt in.

**Introduce a third sentinel value, e.g. `None` meaning "infer from context".**
Rejected: there is no context to infer from inside `create_app` / `create_session_manager`
— unlike `grk serve`, a library caller never tells either constructor what address it
intends to bind, so there is nothing to derive an allow-list from. A named constant that
states its own meaning is simpler than a sentinel whose meaning would have to be
documented separately.

# Service

The read-only surface: a FastAPI app, an MCP server over stdio and streamable
HTTP, and the registry both dispatch through (ADR-0014).

Neither transport defines an operation of its own. Both render `TOOLS`, and a
parity test on each asserts its registered operations are exactly that tuple —
so an operation added directly to
a router or an MCP server, bypassing the registry, fails rather than ships. That
is the mechanism by which "Phase 4 is read-only" is *enforced* rather than
asserted: `ToolSpec.side_effect` is a closed `Literal` with one member, and
widening it is a change `mypy --strict` blocks until someone makes it visibly.

There is no authentication, on either transport. Two controls stand in its
place and they stop different callers: the bind address decides who can reach
the port at all, and `Host` validation (ADR-0024) decides which requests are
answered once they arrive — the second exists because a browser can be walked
onto a loopback socket by DNS rebinding without its owner ever routing to it.
Neither is authentication, and a deliberately published bind
(`--allow-remote-access`) turns the second off by design. See
[Security](../security.md) and [Known limitations](../limitations.md) for what
that leaves open, [the deployment guide](../guides/deployment.md) for choosing
a bind, and [the MCP client guide](../guides/mcp-clients.md) for how to connect
a client.

## Operations

::: groundkit.service.tools

## Request and response models

::: groundkit.service.schemas

## Error mapping

::: groundkit.service.errors

## REST surface

::: groundkit.service.api

## MCP server

::: groundkit.service.mcp_server

## Host binding

::: groundkit.service.binding

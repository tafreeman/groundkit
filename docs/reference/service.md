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

There is no authentication. The loopback bind is the only access control — see
[Security](../security.md) and [Known limitations](../limitations.md), and
[the MCP client guide](../guides/mcp-clients.md) for how to connect one.

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

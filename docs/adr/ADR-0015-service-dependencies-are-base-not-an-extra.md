# ADR-0015 — Service dependencies are base requirements, not an optional extra

- **Status:** Accepted (owner, 2026-08-15)
- **Date:** 2026-08-15
- **Deciders:** Andy Freeman (owner)

## Context

Phase 4 needs FastAPI, an ASGI server, and the MCP SDK. `pyproject.toml` records that
"runtime dependencies land with the phase that needs them," so the question is not
whether to add them but *where*: `[project] dependencies`, or a `[project
.optional-dependencies]` extra alongside `dense` and `rerank`.

**This repo has an established precedent for gating a backend behind an extra, and it
must be addressed rather than ignored.** Two already exist, and both are gated
fail-closed rather than degrading:

- **`dense`** (`lancedb`) — optional so a BM25-only install stays light, but mirrored
  into the dev group because CI must genuinely exercise the backend rather than skip it
  (SPEC.md §3: no job that is the sole proof of a backend may be `continue-on-error`).
- **`rerank`** (`sentence-transformers`, which pulls torch) — deliberately **not**
  mirrored into the dev group, because a multi-gigabyte install must never land in CI's
  default job. Its absence raises `RerankerNotConfiguredError`, never a silent
  passthrough.

Both are reached the same way: an **option on a command**. `grk ingest --dense`,
`grk search --mode hybrid`, `grk eval --rerank`. The command exists and works on a
default install; the flag is what requires the extra, and supplying it without the
backend is a typed error naming the remedy.

SPEC.md §1.2 requires the MCP server be "real and installable, not a demo," and SPEC.md
§9's Phase 4 done criterion is that `grk ingest ./docs && grk serve-mcp` connects from
Claude Desktop or Claude Code with documented client config. SPEC.md §4 lists
`serve | serve-mcp` in the CLI's own verb set.

## Decision

### 1. `fastapi`, the ASGI server, and `mcp` are base runtime dependencies

The distinguishing principle, stated so it decides future cases rather than only this
one: **an extra may gate an option on a command; it may not gate the command itself.**

`--dense` and `--rerank` are options. `serve-mcp` is a command, and it is the command
SPEC.md §9 names as the phase's deliverable. Behind an extra, `grk serve-mcp` on a
default install could only ever raise a `ConfigurationError` — which would make the
headline Phase 4 deliverable fail on every install that did not opt in, and would make
"real and installable, not a demo" false in the most literal way available. A user
following the documented client config would hit an error, not a server.

The bounds style follows the existing pins (`pydantic>=2.7,<3`, `httpx>=0.27,<1`,
`lancedb>=0.13,<1`): a floor near the current release and a ceiling at the next major.

### 2. The torch boundary is unchanged, and this decision does not weaken it

`rerank` stays an extra, out of the base install and out of the dev group. Nothing here
touches that boundary, and the per-request rerank flag (ADR-0014) fails closed through
the existing `RerankerNotConfiguredError` on an install without it. The distinction is
not "important things go in base" — it is the command/option line in decision 1, and
torch sits on the option side.

The three added packages are pure-Python and comparatively small; the argument is the
command/option line, not weight. But weight is why this decision would be wrong for
torch even under the same reasoning: were a *command* ever to require torch, the right
answer would be to reconsider the command, not to move torch into base.

### 3. `uv.lock` regenerates in the same commit as the `pyproject.toml` change

CI runs `uv sync --locked`, which fails on a drifted lockfile, so a `pyproject.toml`
edit without a regenerated lock breaks **every** job rather than one. The two files
change together or not at all. This is not a process nicety: the lockfile-parity gate is
what makes the pinned-dependency claim in SECURITY.md true.

### 4. The `pip-audit` surface grows, and that is the accepted cost

The audit job exports `requirements-audit.txt` and audits it. These three direct
dependencies pull a transitive set that did not previously exist in this project —
Starlette, anyio, h11, click, and a websockets implementation among them — so the
exported file regenerates larger and the audit's surface grows correspondingly.

Recorded rather than discovered: a future advisory against any of those transitives will
now fail this repo's CI, and that is the intended behaviour of the gate, not a
regression to route around. `--no-emit-package` is **not** used to shrink the file: the
portfolio has already been bitten by an export exclusion silently removing a real
dependency from audit scope, and the whole point of the job is that the surface it
audits matches the surface that ships.

## Alternatives considered

- **A `service` extra, mirroring `dense`.** Rejected per decision 1: it gates a command
  rather than an option, so the documented Phase 4 deliverable would fail on a default
  install. The `dense` precedent does not transfer, because `grk ingest` works without
  `dense` while `grk serve-mcp` does not work without FastAPI or the MCP SDK.
- **A `service` extra plus a fail-closed `ConfigurationError` naming the install
  command.** Rejected: it is the same failure with better prose. SPEC.md §9's criterion
  is that the server *connects* from a real client, not that it explains why it cannot.
- **Base-install the MCP SDK only, extra for FastAPI/ASGI.** Rejected: it splits one
  runtime across two installability tiers, so `serve-mcp` works and `serve` does not,
  even though ADR-0014 decision 5 mounts both transports on one app. It also makes the
  streamable-HTTP transport — half of SPEC.md §1.2's requirement — the part that
  silently vanishes.
- **Vendor a minimal ASGI server to avoid the dependency.** Rejected outright: writing a
  production HTTP server to avoid an audited, pinned, widely-reviewed one inverts the
  risk it claims to reduce.
- **Defer the REST surface to keep the dependency set smaller.** Rejected: SPEC.md §4
  puts REST in v1 scope and ADR-0014 mirrors the four tools across both transports from
  one runtime. Deferring REST would not remove the ASGI dependency anyway, since the
  streamable-HTTP MCP transport needs it.

## Consequences

- A default `pip install groundkit` grows by the three packages and their transitives.
  For a library whose headline deliverable is a server, that is the correct default, and
  a BM25-only library user pays it.
- The `pip-audit` job now covers Starlette, anyio, h11, click and websockets among
  others; an advisory against any of them fails CI.
- `uv.lock` and `requirements-audit.txt` both regenerate in the Phase 4 change, so the
  diff is larger than the source change alone suggests, and a reviewer should expect it.
- The command/option line is now a stated rule with a citation, so the next dependency
  question is decided by it rather than re-argued.
- Nothing about the torch boundary changes; `rerank` remains an extra outside the dev
  group, and CI's default jobs still never pull it.

## References

- SPEC.md **§1.2** ("real and installable, not a demo"), §3 (no job that is the sole
  proof of a backend may be `continue-on-error`; pinned deps and lockfile parity), §4
  (the CLI verb set including `serve` and `serve-mcp`), §9 (Phase 4's done criterion).
- [ADR-0014](ADR-0014-read-only-service-surface-and-outbound-endpoint-safety.md) —
  decision 5 mounts both transports on one runtime, which is why the ASGI dependency is
  not avoidable by dropping REST.
- [ADR-0012](ADR-0012-rerank-eval-stage-reorders-upstream-stage.md) — the torch boundary
  this decision leaves untouched.
- `pyproject.toml` — the `dense` and `rerank` extras whose precedent decision 1
  distinguishes, and the existing bound style decision 1 follows.

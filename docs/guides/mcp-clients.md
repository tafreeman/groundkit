# MCP clients

groundkit ships a real MCP server, not a demo of one: `grk serve-mcp` speaks
stdio and `grk serve` mounts the same operations over streamable HTTP,
alongside a REST mirror. SPEC.md §9's Phase 4 done criterion is that
`grk ingest ./docs && grk serve-mcp` connects from Claude Desktop or Claude
Code using config someone else could follow without having seen this repo.
This page is that config.

## Prerequisites

groundkit is not yet published to PyPI, so `pip install groundkit` does not
work today — it arrives with the v0.1.0 release. Until then, install from a
clone (see [Installation](../getting-started/installation.md) for the full
walkthrough):

```bash
git clone https://github.com/tafreeman/groundkit
cd groundkit
uv sync
```

No extra is needed for the server. FastAPI, its ASGI server, and the MCP SDK
are base runtime dependencies, not an opt-in like `dense` or `rerank` —
[ADR-0015](../adr/ADR-0015-service-dependencies-are-base-not-an-extra.md)
draws the line as "an extra may gate an option on a command, never the
command itself," and `serve-mcp` is the command SPEC.md §9 names as the
phase's deliverable. A plain `uv sync` with no extras is enough for a client
to launch the server.

A collection has to exist before you point a client at it — the server never
creates one:

```bash
grk ingest ./docs --index-dir .groundkit
```

That writes a collection named `default` under `.groundkit/` (the same
default every request below uses when it omits a collection name). See the
[Quickstart](../getting-started/quickstart.md) if you want dense or hybrid
retrieval available to the client too — the same "vectors have to be there
from the first ingest" rule applies here as anywhere else in groundkit.

## Claude Desktop

Claude Desktop reads server definitions from `claude_desktop_config.json`:

=== "macOS"

    ```
    ~/Library/Application Support/Claude/claude_desktop_config.json
    ```

=== "Windows"

    ```
    %APPDATA%\Claude\claude_desktop_config.json
    ```

Add a `groundkit` entry to `mcpServers`:

```json
{
  "mcpServers": {
    "groundkit": {
      "command": "/absolute/path/to/groundkit/.venv/bin/grk",
      "args": [
        "serve-mcp",
        "--index-dir", "/absolute/path/to/.groundkit",
        "--base-dir", "/absolute/path/to/docs"
      ]
    }
  }
}
```

!!! warning "`command` must be a path the client can actually execute"

    A bare `"command": "grk"` only works if `grk` is on the **PATH that the
    client process inherits** — which is not the same PATH as your shell. If
    groundkit lives in a virtual environment, as it does for anyone who
    installed it with `uv sync` from a checkout, `grk` is on no PATH at all and
    the client fails at launch. The error surfaces as the server disappearing
    or "failed to start", which reads like a groundkit bug and is not one.

    Use the absolute path to the entry point:

    - **venv (Linux/macOS):** `<repo>/.venv/bin/grk`
    - **venv (Windows):** `<repo>\.venv\Scripts\grk.exe`
    - **`uv tool install groundkit` or a system-wide `pip install`:** bare `grk`
      is fine, since those put it on a PATH the client inherits.

    Check with `which grk` / `Get-Command grk`. If that prints nothing, you need
    the absolute path.

**Use absolute paths for the arguments too.** Claude Desktop launches the server with a working
directory of its own choosing, not the directory you ingested from — a
relative `--index-dir .groundkit` resolves against whatever that unpredictable
working directory turns out to be, not against your project. `--base-dir` is
required on both `serve` and `serve-mcp`: it is the containment root every
citation is checked against before a chunk is returned, so an absolute path
there is not a style preference, it is the value the server refuses to start
without.

Restart Claude Desktop after editing the file for it to pick up the new
server.

## Claude Code

The equivalent registration from the command line:

```bash
claude mcp add groundkit --scope project -- \
  /absolute/path/to/groundkit/.venv/bin/grk serve-mcp \
  --index-dir /absolute/path/to/.groundkit \
  --base-dir /absolute/path/to/docs
```

The same `command` rule from the Claude Desktop section applies: the executable
after `--` must be a path the client can run, not a name it would have to
resolve on a PATH it does not share.

`--scope project` writes the entry to a `.mcp.json` at the project root
instead of your personal, machine-local config, so a team can commit it
alongside the corpus it points at. The file `claude mcp add` produces is the
same shape as the Claude Desktop block above:

```json
{
  "mcpServers": {
    "groundkit": {
      "command": "/absolute/path/to/groundkit/.venv/bin/grk",
      "args": [
        "serve-mcp",
        "--index-dir", "/absolute/path/to/.groundkit",
        "--base-dir", "/absolute/path/to/docs"
      ]
    }
  }
}
```

The same absolute-path reasoning from the Claude Desktop section applies here
— `.mcp.json` is checked out at whatever path a teammate clones the repo to,
but the server process is still launched with an unpredictable working
directory, so the paths inside the config need to be absolute on whichever
machine runs it.

## The four tools

SPEC.md §1.2 names exactly four operations, and the server exposes exactly
those four — nothing more, nothing fewer:

| Tool | What it does | Returns |
|---|---|---|
| `search` | Runs a query against one collection — BM25, dense, or hybrid — optionally reranked. | A `SearchResponse`: the same shape `grk search --json` prints, with citation-bearing results. |
| `fetch_chunk` | Fetches one chunk by id and re-verifies its citation by re-reading the source file, not the indexed copy. | A chunk, its citation, and a verification verdict (`verified`, `drifted`, or `unresolvable`) — content is included only when verified. |
| `list_collections` | Lists the collections present in the server's index directory. | A list of collection names. |
| `index_status` | Reports one collection's document and chunk counts and its embedding identity. | Counts, the embedding manifest when the collection is dense, and the schema version — no document content, no source paths. |

**Ingest is deliberately not among them.** SPEC.md §1.2 is the authority
cited so this reads as a decision rather than an omission: every operation
here is read-only, and adding a write means adding an authentication story
that Phase 4 has not built (see Security posture, below). `grk ingest` stays
a local, operator-typed command; nothing reachable over a transport can build
or modify a collection.

## Streamable HTTP

`grk serve` runs the FastAPI app and mounts the MCP streamable-HTTP transport
on it at `/mcp`, alongside the same four operations as REST routes under
`/v1/` for callers that would rather speak plain HTTP than MCP. Point a
streamable-HTTP-capable client at `http://<host>:<port>/mcp`, using whatever
`--host` and `--port` you passed to `grk serve` (loopback by default — see
below).

Both transports — stdio from `grk serve-mcp` and streamable HTTP from
`grk serve` — dispatch through the same operation registry
([ADR-0014](../adr/ADR-0014-read-only-service-surface-and-outbound-endpoint-safety.md)
decision 5), so a client talking to either one is exercising the same four
operations against the same running collections. What differs is framing,
error rendering, and — only for the HTTP path — the socket it binds.

## Security posture

!!! danger "Phase 4 ships no authentication of any kind"

    Every operation above is read-only, and that emptiness is what lets the
    shared-secret header SPEC.md §7 asks for be skipped rather than built:
    there is nothing mutating to guard. The **loopback bind is the only
    access control this server has.** By default it binds `127.0.0.1` and
    refuses to bind anything else unless you pass `--allow-remote-access` —
    and that flag does exactly what it says: it publishes your document
    content and absolute filesystem paths to anyone who can reach the port.
    `search` responses carry source text; `index_status` and
    `list_collections` carry collection topology. None of it is behind a
    credential.

    There is no ingest and no delete over either transport, MCP or REST.
    Everything reachable from a client is a read.

Read [ADR-0014](../adr/ADR-0014-read-only-service-surface-and-outbound-endpoint-safety.md)
for the full reasoning, and [Security](../security.md) and
[Known limitations](../limitations.md) for the rest of the posture. If you
need remote access, put a real reverse proxy with authentication in front of
the loopback bind rather than passing `--allow-remote-access` directly to the
internet — this server was not designed to be the thing facing untrusted
callers.

## Troubleshooting

**`--base-dir` is required and the server will not start without it, on
either `serve` or `serve-mcp`.** It is the containment root
`fetch_chunk`'s citation-verification step checks every resolved path
against. A service that could not verify a citation would be shipping the
opposite of groundkit's product claim, so there is no default to fall back
to.

**`--index-dir` must already exist, and the server never creates it.** Point
it at a directory you have already run `grk ingest` against. If the client
reports the server exited immediately, check that the directory exists and
is spelled the same way it was ingested — a fresh empty directory is not the
same as a missing one, but a nonexistent one at startup is refused rather
than silently created for you.

**Asking for a collection that does not exist returns not-found, not a new,
empty collection.** `list_collections` is the way to check what is actually
there before calling `search`, `fetch_chunk`, or `index_status` on a name you
are not certain about.

**If the client reports it cannot launch `grk` at all**, the process is
started with a different `PATH` than your interactive shell has — see the
`command` warning in the Claude Desktop section above. This is the single most
common setup failure, and it is indistinguishable from a server crash in most
clients' error reporting.

**Test the command line before involving a client.** Every launch failure above
is reproducible from a terminal, where the error is legible instead of being
reported as "server failed to start". Run the exact `command` and `args` from
your config by hand and send it one frame:

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"probe","version":"0"}}}' \
  | /absolute/path/to/.venv/bin/grk serve-mcp --index-dir /abs/.groundkit --base-dir /abs/docs
```

A JSON object on stdout naming `groundkit` means the server is fine and the
problem is in the client config. Anything else — a traceback, a "not found", or
silence — is the real error the client was hiding. Note that **stdout carries
JSON-RPC frames and nothing else**: logs go to stderr, so any non-JSON line on
stdout is itself a bug worth reporting.

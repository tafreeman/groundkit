"""Tests for the MCP transport (ADR-0014 decisions 2, 3, 5, 6, 9).

Most of these are enforcement tests, and none of them can be shown to fail by
reverting source: ``service/mcp_server.py`` was a docstring-only stub before
this phase, so there is no unfixed version to run them against. Each is
demonstrated instead by **injecting the violation it exists to catch**, which is
the meaningful direction for a guard and is explicitly *not* the SPEC.md §8
revert procedure. Every such test says so in its own docstring rather than
leaving a reader to assume a revert was performed.

The exception is the leak test: the SDK's ``except Exception as e: return
self._make_error_result(str(e))`` is live code in the installed ``mcp`` package,
so removing this module's own ``except`` clauses genuinely does make it fail.
That one is a real regression test, and its docstring names the source lines to
delete to see it fail.
"""

from __future__ import annotations

import asyncio
import dataclasses
import io
import json
import logging
import os
import sys
import time
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Final

import mcp.types as types
import pytest

from groundkit.contracts import Chunk
from groundkit.errors import StorageError
from groundkit.index.metadata import SQLiteMetadataStore
from groundkit.runtime import CollectionRegistry
from groundkit.service import mcp_server
from groundkit.service.errors import GENERIC_DETAIL
from groundkit.service.mcp_server import (
    LIST_COLLECTIONS_RESULT_KEY,
    MCP_HTTP_PATH,
    build_mcp_server,
    list_tool_definitions,
    run_stdio,
)
from groundkit.service.schemas import ListCollectionsRequest
from groundkit.service.tools import TOOL_NAMES, TOOLS, ServiceContext, ToolSpec
from groundkit.telemetry import JsonLogFormatter

_MCP_LOGGER = mcp_server.__name__

if TYPE_CHECKING:
    from pathlib import Path

    from mcp.server.lowlevel import Server

#: Planted in an exception message and in its ``__cause__``. If either reaches a
#: client, the SDK's stringify path is live.
_SENTINEL: Final[str] = "leaked-secret-9d41c0"

#: Server config keys no generated input schema may accept (ADR-0014 decision 6).
_FORBIDDEN_SCHEMA_PROPERTIES: Final[frozenset[str]] = frozenset(
    {"base_url", "index_dir", "base_dir", "api_key_env"}
)

#: initialize response + tools/call response. The ``initialized`` notification
#: draws no reply.
_EXPECTED_STDOUT_FRAMES: Final[int] = 2

_STDIO_DEADLINE_SECONDS: Final[float] = 30.0
_POLL_INTERVAL_SECONDS: Final[float] = 0.01


async def _seed(tmp_path: Path) -> tuple[Path, Path, str]:
    """Write a real source file and index it so citations genuinely resolve."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    text = "Turbine maintenance intervals depend on load factor and ambient temperature."
    (corpus / "a.md").write_text(text, encoding="utf-8")

    index_dir = tmp_path / "index"
    index_dir.mkdir()
    store = await SQLiteMetadataStore.open(index_dir, "default")
    try:
        chunk = Chunk(
            chunk_id="c1",
            document_id="doc-1",
            chunk_index=0,
            content=text,
            start_offset=0,
            end_offset=len(text),
        )
        await store.replace_document(str(corpus / "a.md"), "doc-1", "h1", [chunk])
    finally:
        await store.close()
    return index_dir, corpus, "c1"


def _context(index_dir: Path, corpus: Path) -> ServiceContext:
    return ServiceContext(
        registry=CollectionRegistry(index_dir),
        index_dir=index_dir,
        base_dir=corpus,
    )


async def _call(
    server: Server[Any, Any], name: str, arguments: dict[str, Any]
) -> types.CallToolResult:
    """Invoke a tool through the SDK's real ``tools/call`` dispatch.

    Deliberately not a direct call to the module's handler: the whole point of
    the leak test below is what the SDK does with what the handler returns (and
    with what escapes it), so the test has to go through the SDK's wrapper.
    """
    handler = server.request_handlers[types.CallToolRequest]
    result = await handler(
        types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(name=name, arguments=arguments),
        )
    )
    payload = result.root
    assert isinstance(payload, types.CallToolResult)
    return payload


def _structured(result: types.CallToolResult) -> dict[str, Any]:
    assert result.structuredContent is not None
    return result.structuredContent


def _property_names(schema: object) -> set[str]:
    """Every property name anywhere in a JSON schema, ``$defs`` included."""
    names: set[str] = set()
    if isinstance(schema, dict):
        properties = schema.get("properties")
        if isinstance(properties, dict):
            names.update(str(key) for key in properties)
        for value in schema.values():
            names |= _property_names(value)
    elif isinstance(schema, list):
        for item in schema:
            names |= _property_names(item)
    return names


async def _unused_handler(ctx: ServiceContext, request: ListCollectionsRequest) -> list[str]:
    """Stand-in handler for injected registry entries. Never invoked."""
    del ctx, request
    return []


#: A fifth tool, for the injected-violation demonstration below. Named for what
#: it would be if this guard failed: a mutating operation that never appeared in
#: SPEC.md §1.2's list.
_EXTRA_SPEC: Final[ToolSpec] = ToolSpec(
    name="hidden_ingest",
    summary="An operation registered without passing through SPEC.md §1.2.",
    request_model=ListCollectionsRequest,
    handler=_unused_handler,
    side_effect="read_only",
    rest_path="/v1/hidden",
    rest_method="POST",
)


# -- Registry parity and generated schemas ---------------------------------


def test_registered_tools_are_exactly_the_registry(tmp_path: Path) -> None:
    """ADR-0014 decision 2 check 3: registry↔MCP parity.

    Reads the tools back through the server's *registered* ``tools/list``
    handler, not by re-deriving them from ``TOOLS`` — re-deriving would compare
    the registry against itself and pass even if registration were broken.
    """

    async def run() -> None:
        index_dir, corpus, _ = await _seed(tmp_path)
        ctx = _context(index_dir, corpus)
        try:
            names = {tool.name for tool in await list_tool_definitions(ctx)}
            assert names == set(TOOL_NAMES)
        finally:
            await ctx.registry.aclose()

    asyncio.run(run())


def test_an_extra_registered_tool_breaks_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same guard, demonstrated by injection.

    This is **not** the SPEC.md §8 revert procedure — there is no unfixed
    version of this module to revert to. It injects the violation the parity
    test exists to catch (a fifth tool registered on the MCP surface that
    SPEC.md §1.2 never named) and asserts the parity assertion above would
    fail, which is what makes that assertion load-bearing rather than
    decorative.
    """

    async def run() -> None:
        index_dir, corpus, _ = await _seed(tmp_path)
        ctx = _context(index_dir, corpus)
        try:
            monkeypatch.setattr(mcp_server, "TOOLS", (*TOOLS, _EXTRA_SPEC))
            names = {tool.name for tool in await list_tool_definitions(ctx)}
            assert names != set(TOOL_NAMES)
            assert "hidden_ingest" in names
        finally:
            await ctx.registry.aclose()

    asyncio.run(run())


def test_input_schemas_are_generated_from_the_request_models(tmp_path: Path) -> None:
    """The anti-drift test (ADR-0001 hazard 4, ADR-0014 decision 5).

    A hand-written schema and a Pydantic model are two descriptions of one
    contract with nothing forcing them to agree; equality against
    ``model_json_schema()`` is what makes "generated, never restated" checkable.
    """

    async def run() -> None:
        index_dir, corpus, _ = await _seed(tmp_path)
        ctx = _context(index_dir, corpus)
        try:
            by_name = {tool.name: tool for tool in await list_tool_definitions(ctx)}
            for spec in TOOLS:
                assert by_name[spec.name].inputSchema == spec.request_model.model_json_schema()
        finally:
            await ctx.registry.aclose()

    asyncio.run(run())


def test_no_generated_input_schema_accepts_server_configuration(tmp_path: Path) -> None:
    """ADR-0014 decision 6, made executable on the MCP side.

    Walks every registered tool's generated schema including ``$defs``, rather
    than a list of models someone has to remember to update, so a nested model
    added later is covered without the test being told about it. This is what
    keeps SECURITY.md's sentence "no network-facing caller can set base_url"
    true now that the surface exposing one exists.

    Guard, demonstrated by injection: add ``base_url: str | None = None`` to
    ``SearchRequest`` and this fails.
    """

    async def run() -> None:
        index_dir, corpus, _ = await _seed(tmp_path)
        ctx = _context(index_dir, corpus)
        try:
            for tool in await list_tool_definitions(ctx):
                for name in _property_names(tool.inputSchema):
                    assert name not in _FORBIDDEN_SCHEMA_PROPERTIES, (
                        f"{tool.name}.{name} exposes server configuration"
                    )
                    assert not name.startswith("embed"), (
                        f"{tool.name}.{name} exposes embedder configuration"
                    )
        finally:
            await ctx.registry.aclose()

    asyncio.run(run())


def test_tools_advertise_themselves_as_read_only(tmp_path: Path) -> None:
    """``readOnlyHint`` is derived from ``side_effect``, never hardcoded.

    A hardcoded ``True`` would keep advertising "read" through the very change
    that widened ``SideEffect`` to admit a mutating operation.
    """

    async def run() -> None:
        index_dir, corpus, _ = await _seed(tmp_path)
        ctx = _context(index_dir, corpus)
        try:
            for tool in await list_tool_definitions(ctx):
                assert tool.annotations is not None
                assert tool.annotations.readOnlyHint is True
        finally:
            await ctx.registry.aclose()

    asyncio.run(run())


def test_http_mount_path_is_exported() -> None:
    """The CLI mounts what this module names, so the two cannot drift."""
    assert MCP_HTTP_PATH.startswith("/")


# -- Dispatch --------------------------------------------------------------


def test_every_tool_dispatches_through_call_tool(tmp_path: Path) -> None:
    """All four tools answer through the SDK's real dispatch path."""

    async def run() -> None:
        index_dir, corpus, chunk_id = await _seed(tmp_path)
        ctx = _context(index_dir, corpus)
        server = build_mcp_server(ctx)
        try:
            listed = await _call(server, "list_collections", {})
            assert listed.isError is False
            assert _structured(listed)[LIST_COLLECTIONS_RESULT_KEY] == ["default"]

            searched = await _call(server, "search", {"query": "turbine"})
            assert searched.isError is False
            assert _structured(searched)["results"]

            fetched = await _call(server, "fetch_chunk", {"chunk_id": chunk_id})
            assert fetched.isError is False
            assert _structured(fetched)["verification"] == "verified"

            status = await _call(server, "index_status", {"collection": "default"})
            assert status.isError is False
            assert _structured(status)["chunk_count"] == 1
        finally:
            await ctx.registry.aclose()

    asyncio.run(run())


def test_access_and_failure_logs_promote_structured_fields_under_json_formatting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The MCP transport's ``request_id``-carrying lines, as a JSON-log collector sees them.

    Mirrors ``test_service_api.py``'s equivalent test for the REST surface's
    access log. Both transports log the same shape (``tool=%s
    request_id=%s ...``) as positional format arguments, so both needed the
    same fix: pass the same values via ``extra=`` too, or
    :class:`~groundkit.telemetry.JsonLogFormatter` has nothing to promote and
    every field stays trapped inside the ``message`` string.
    """

    async def boom(ctx: ServiceContext, request: ListCollectionsRequest) -> list[str]:
        del ctx, request
        raise StorageError("boom") from RuntimeError("cause")

    async def run() -> None:
        index_dir, corpus, _ = await _seed(tmp_path)
        ctx = _context(index_dir, corpus)
        poisoned = tuple(
            dataclasses.replace(spec, handler=boom) if spec.name == "list_collections" else spec
            for spec in TOOLS
        )
        server = build_mcp_server(ctx)
        try:
            with caplog.at_level(logging.INFO, logger=_MCP_LOGGER):
                ok_result = await _call(server, "search", {"query": "turbine"})
                assert ok_result.isError is False

                monkeypatch.setattr(mcp_server, "TOOLS", poisoned)
                failed_result = await _call(server, "list_collections", {})
                assert failed_result.isError is True
        finally:
            await ctx.registry.aclose()

        records = [r for r in caplog.records if r.name == _MCP_LOGGER]
        ok_records = [r for r in records if getattr(r, "tool", None) == "search"]
        assert len(ok_records) == 1, f"expected exactly one search access record, got {ok_records}"
        ok_payload = json.loads(JsonLogFormatter().format(ok_records[0]))
        assert ok_payload["tool"] == "search"
        assert ok_payload["status"] == "ok"
        assert ok_payload["request_id"]

        failed_records = [r for r in records if getattr(r, "tool", None) == "list_collections"]
        assert len(failed_records) == 1, (
            f"expected exactly one list_collections failure record, got {failed_records}"
        )
        failed_payload = json.loads(JsonLogFormatter().format(failed_records[0]))
        assert failed_payload["tool"] == "list_collections"
        assert failed_payload["status"] == "failed"
        assert failed_payload["request_id"]

    asyncio.run(run())


def test_list_collections_is_wrapped_so_structured_content_is_an_object(tmp_path: Path) -> None:
    """MCP's ``structuredContent`` is a JSON object; a bare list cannot ride there.

    The wrap is framing, which ADR-0014 decision 5 lists as differing
    deliberately between transports. Pinned so the key is a contract rather than
    an implementation detail a client discovered by reading a response.
    """

    async def run() -> None:
        index_dir, corpus, _ = await _seed(tmp_path)
        ctx = _context(index_dir, corpus)
        server = build_mcp_server(ctx)
        try:
            result = await _call(server, "list_collections", {})
            structured = _structured(result)
            assert set(structured) == {LIST_COLLECTIONS_RESULT_KEY}
            # The text block is built from the same dict, so a client reading
            # only content blocks sees exactly what a structured reader does.
            block = result.content[0]
            assert isinstance(block, types.TextContent)
            assert json.loads(block.text) == structured
        finally:
            await ctx.registry.aclose()

    asyncio.run(run())


def test_an_unknown_tool_name_is_an_error_result_not_a_crash(tmp_path: Path) -> None:
    """An unrecognized name must be answered, and must not echo itself back.

    The name is unvalidated caller-supplied text; a client that guessed wrong
    learns the real tool set from ``tools/list``.
    """

    async def run() -> None:
        index_dir, corpus, _ = await _seed(tmp_path)
        ctx = _context(index_dir, corpus)
        server = build_mcp_server(ctx)
        try:
            result = await _call(server, "ingest", {"path": "/etc"})
            assert result.isError is True
            assert _structured(result)["kind"] == "invalid_request"
            assert "ingest" not in result.model_dump_json()
        finally:
            await ctx.registry.aclose()

    asyncio.run(run())


def test_invalid_arguments_are_rejected_without_echoing_the_value(tmp_path: Path) -> None:
    """A pydantic rejection renders as ``invalid_request``, not a stringified traceback.

    Only each problem's location and machine-readable type are reported, never
    the submitted value: reflecting a caller's input back through a server that
    also logs it buys nothing a client did not already know.
    """

    async def run() -> None:
        index_dir, corpus, _ = await _seed(tmp_path)
        ctx = _context(index_dir, corpus)
        server = build_mcp_server(ctx)
        try:
            result = await _call(server, "search", {"query": "turbine", "top_k": 99999})
            assert result.isError is True
            structured = _structured(result)
            assert structured["kind"] == "invalid_request"
            assert "top_k" in structured["detail"]
            assert "99999" not in result.model_dump_json()

            missing = await _call(server, "search", {})
            assert missing.isError is True
            assert _structured(missing)["kind"] == "invalid_request"
        finally:
            await ctx.registry.aclose()

    asyncio.run(run())


# -- Error rendering -------------------------------------------------------


def test_a_failing_handler_leaks_nothing_through_the_sdk_stringify_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The SDK stringifies escaping exceptions; nothing may escape.

    ``Server.call_tool``'s generated handler ends in ``except Exception as e:
    return self._make_error_result(str(e))``, which places the raw message in
    the ``CallToolResult`` a client reads — bypassing ADR-0014 decision 9's
    allow-list, through which a ``StorageError``'s absolute database path would
    otherwise reach an unauthenticated caller.

    A real regression test, not an injected demonstration: the SDK path is live
    code in the installed package. Delete either ``except`` clause from
    ``_call_tool`` in ``service/mcp_server.py``, run this, and the sentinel
    appears in the result.

    ``StorageError`` is the chosen carrier because decision 9 *withholds* its
    message; an allow-listed type such as ``ConfigurationError`` returns its own
    message by design, so a sentinel there would prove nothing.
    """

    async def boom(ctx: ServiceContext, request: ListCollectionsRequest) -> list[str]:
        del ctx, request
        raise StorageError(f"database at /srv/{_SENTINEL}/default.sqlite3") from RuntimeError(
            f"cause carrying {_SENTINEL}"
        )

    async def run() -> None:
        index_dir, corpus, _ = await _seed(tmp_path)
        ctx = _context(index_dir, corpus)
        poisoned = tuple(
            dataclasses.replace(spec, handler=boom) if spec.name == "list_collections" else spec
            for spec in TOOLS
        )
        monkeypatch.setattr(mcp_server, "TOOLS", poisoned)
        server = build_mcp_server(ctx)
        try:
            result = await _call(server, "list_collections", {})
            assert result.isError is True
            assert _SENTINEL not in result.model_dump_json()
            structured = _structured(result)
            assert structured["kind"] == "storage_error"
            assert structured["detail"] == GENERIC_DETAIL
            assert structured["request_id"]
            assert structured["code"]
        finally:
            await ctx.registry.aclose()

    asyncio.run(run())


def test_a_non_groundkit_exception_is_rendered_by_the_same_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare ``Exception`` is a bug, and its message is whatever a library wrote.

    Caught by the same handler and rendered as ``internal_error`` with the fixed
    detail, so an arbitrary third-party message can never become a response
    body.
    """

    async def boom(ctx: ServiceContext, request: ListCollectionsRequest) -> list[str]:
        del ctx, request
        raise RuntimeError(f"third-party message quoting {_SENTINEL}")

    async def run() -> None:
        index_dir, corpus, _ = await _seed(tmp_path)
        ctx = _context(index_dir, corpus)
        poisoned = tuple(
            dataclasses.replace(spec, handler=boom) if spec.name == "list_collections" else spec
            for spec in TOOLS
        )
        monkeypatch.setattr(mcp_server, "TOOLS", poisoned)
        server = build_mcp_server(ctx)
        try:
            result = await _call(server, "list_collections", {})
            assert result.isError is True
            assert _SENTINEL not in result.model_dump_json()
            assert _structured(result)["kind"] == "internal_error"
            assert _structured(result)["detail"] == GENERIC_DETAIL
        finally:
            await ctx.registry.aclose()

    asyncio.run(run())


def test_unknown_collection_is_refused_and_creates_nothing(tmp_path: Path) -> None:
    """ADR-0014 decision 3, asserted on the response **and** on the directory.

    ``SQLiteMetadataStore.open`` creates and stamps a store it does not find, so
    without the precondition every request naming an arbitrary collection would
    leave an empty file behind — an unauthenticated read surface turned into a
    disk-fill primitive.
    """

    async def run() -> None:
        index_dir, corpus, _ = await _seed(tmp_path)
        ctx = _context(index_dir, corpus)
        server = build_mcp_server(ctx)
        try:
            before = sorted(path.name for path in index_dir.iterdir())
            result = await _call(server, "index_status", {"collection": "nonexistent"})
            assert result.isError is True
            assert _structured(result)["kind"] == "collection_not_found"
            assert sorted(path.name for path in index_dir.iterdir()) == before
        finally:
            await ctx.registry.aclose()

    asyncio.run(run())


def test_a_traversal_collection_name_is_a_validation_failure(tmp_path: Path) -> None:
    """Name validation runs before the existence check, and that order is the point.

    Reporting a traversal attempt as *not found* would answer the probe: the
    caller would learn whether the traversed-to path exists. As a validation
    failure it learns nothing either way.
    """

    async def run() -> None:
        index_dir, corpus, _ = await _seed(tmp_path)
        ctx = _context(index_dir, corpus)
        server = build_mcp_server(ctx)
        try:
            for name in ("../evil", "../../etc/passwd", ".."):
                result = await _call(server, "index_status", {"collection": name})
                assert result.isError is True
                assert _structured(result)["kind"] == "invalid_request", name
        finally:
            await ctx.registry.aclose()

    asyncio.run(run())


# -- stdio purity ----------------------------------------------------------


def _client_frames() -> bytes:
    """A minimal, well-formed JSON-RPC session ending in one tool call."""
    messages: list[dict[str, Any]] = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": types.LATEST_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "groundkit-tests", "version": "0"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "list_collections", "arguments": {}},
        },
    ]
    return "".join(f"{json.dumps(message)}\n" for message in messages).encode("utf-8")


def _emitted_lines(captured: io.BytesIO) -> list[str]:
    return [line for line in captured.getvalue().decode("utf-8").splitlines() if line.strip()]


async def _wait_for_frames(captured: io.BytesIO, expected: int) -> None:
    deadline = time.monotonic() + _STDIO_DEADLINE_SECONDS
    while time.monotonic() < deadline:
        if len(_emitted_lines(captured)) >= expected:
            return
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)
    raise AssertionError(
        f"stdio server wrote {_emitted_lines(captured)}, expected {expected} frames"
    )


class _PipeStdin:
    """Stand-in for ``sys.stdin`` exposing only what ``stdio_server`` reads."""

    def __init__(self, buffer: io.BufferedReader) -> None:
        self.buffer = buffer


class _CaptureBuffer(io.BytesIO):
    """A capture that survives its writers.

    ``stdio_server`` deliberately does not close the ``TextIOWrapper`` it builds
    over ``sys.stdout.buffer`` — it must not close the real process handle — so
    that wrapper is collected instead, and collection closes the buffer
    underneath it. Without this override the captured frames are unreadable by
    the time the assertions run.
    """

    def close(self) -> None:
        return None


def test_stdout_carries_only_json_rpc_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ADR-0014 decision 5's stdio rule, driven through a real session.

    A log handler bound to stdout is **planted before the server starts**, and
    the root level is lowered so the dispatch path's own INFO record is emitted
    during the tool call. If ``run_stdio`` did not re-point that handler at
    stderr, its record lands between two JSON-RPC frames and the parse below
    fails — which is the failure this test exists to produce, since a single
    stray line corrupts the protocol stream for the rest of the session.
    """

    async def run() -> None:
        index_dir, corpus, _ = await _seed(tmp_path)
        ctx = _context(index_dir, corpus)

        captured = _CaptureBuffer()
        fake_stdout = io.TextIOWrapper(captured, encoding="utf-8")
        read_fd, write_fd = os.pipe()
        stdin = _PipeStdin(os.fdopen(read_fd, "rb"))
        monkeypatch.setattr(sys, "stdout", fake_stdout)
        monkeypatch.setattr(sys, "stdin", stdin)

        root = logging.getLogger()
        planted = logging.StreamHandler(fake_stdout)
        previous_level = root.level
        root.addHandler(planted)
        root.setLevel(logging.INFO)

        server_task = asyncio.create_task(run_stdio(ctx))
        try:
            os.write(write_fd, _client_frames())
            await _wait_for_frames(captured, _EXPECTED_STDOUT_FRAMES)
        finally:
            os.close(write_fd)
            with suppress(TimeoutError):
                await asyncio.wait_for(server_task, _STDIO_DEADLINE_SECONDS)
            root.removeHandler(planted)
            root.setLevel(previous_level)
            stdin.buffer.close()
            await ctx.registry.aclose()

        responses: list[dict[str, Any]] = []
        for line in _emitted_lines(captured):
            try:
                responses.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise AssertionError(f"stdout carried a non-JSON-RPC line: {line!r}") from exc

        assert planted.stream is sys.stderr, "a stdout log handler survived run_stdio"
        assert all(frame["jsonrpc"] == "2.0" for frame in responses)
        tool_results = [
            frame["result"] for frame in responses if frame.get("id") == _EXPECTED_STDOUT_FRAMES
        ]
        assert tool_results
        assert tool_results[0]["structuredContent"] == {LIST_COLLECTIONS_RESULT_KEY: ["default"]}
        assert tool_results[0].get("isError", False) is False

    asyncio.run(run())

"""``grk serve`` / ``grk serve-mcp``: startup guards, wiring, and the bind refusal.

Nothing here binds a socket or starts a server. ``cli._serve_http`` and
``mcp_server.run_stdio`` are the two named seams the commands run through, and
substituting them is what lets every startup guard — the host classification,
the required containment root, the never-create-an-index rule, the reranker's
fail-closed wiring — be exercised for real while no port is ever opened.

The bind tests matter more than their size suggests: ADR-0014 decision 1 ships
**no authentication of any kind**, so the address this process binds is the
only thing between an indexed corpus and everyone who can route to the host.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import pytest

from groundkit import cli
from groundkit.contracts import RetrievalResult
from groundkit.errors import ConfigurationError, RerankerNotConfiguredError
from groundkit.providers import embeddings as embeddings_module
from groundkit.retrieval import rerank as rerank_module
from groundkit.service import mcp_server
from groundkit.service.binding import (
    DEFAULT_SERVE_HOST,
    DEFAULT_SERVE_PORT,
    ensure_bindable_host,
)
from groundkit.service.tools import MAX_CONCURRENT_CORPUS_SCANS, ServiceContext

#: A routable, non-loopback literal. RFC 5737 TEST-NET-1: documentation-only,
#: so it can never be a real host anyone reading this test might try to reach.
NON_LOOPBACK_HOST = "192.0.2.10"


@dataclass(frozen=True)
class _ServeCall:
    """What ``grk serve`` handed the (substituted) HTTP runner."""

    ctx: ServiceContext
    host: str
    port: int


@pytest.fixture
def index_dir(tmp_path: Path) -> Path:
    """An existing index directory. Empty: no command here needs a collection."""
    path = tmp_path / "idx"
    path.mkdir()
    return path


@pytest.fixture
def base_dir(tmp_path: Path) -> Path:
    """An existing containment root, the value ``--base-dir`` requires."""
    path = tmp_path / "docs"
    path.mkdir()
    return path


def _argv(command: str, index_dir: Path, base_dir: Path, *extra: str) -> list[str]:
    return [command, "--index-dir", str(index_dir), "--base-dir", str(base_dir), *extra]


def _patch_serve_http(monkeypatch: pytest.MonkeyPatch) -> list[_ServeCall]:
    """Substitute the HTTP runner, recording what serve-time assembly produced.

    Signature-identical to ``cli._serve_http`` (``host`` and ``port``
    keyword-only), so a change in how ``_cmd_serve`` calls it fails here rather
    than being absorbed by a permissive fake.
    """
    calls: list[_ServeCall] = []

    async def fake_serve_http(ctx: ServiceContext, *, host: str, port: int) -> None:
        calls.append(_ServeCall(ctx=ctx, host=host, port=port))

    monkeypatch.setattr(cli, "_serve_http", fake_serve_http)
    return calls


def _patch_run_stdio(monkeypatch: pytest.MonkeyPatch) -> list[ServiceContext]:
    """Substitute the stdio runner on its own module.

    Patched at ``groundkit.service.mcp_server.run_stdio`` rather than on
    ``cli``: ``_cmd_serve_mcp`` imports the name at call time, so this proves
    the command reaches *that* function and not a CLI-local indirection.
    """
    seen: list[ServiceContext] = []

    async def fake_run_stdio(ctx: ServiceContext) -> None:
        seen.append(ctx)

    monkeypatch.setattr(mcp_server, "run_stdio", fake_run_stdio)
    return seen


# --- The bind guard (ADR-0014 decision 7) ---------------------------------


def test_default_serve_host_is_loopback() -> None:
    """The default bind address is pinned, because it is a decision.

    SPEC.md section 7 states the loopback bind in the same sentence as the
    shared-secret header, and ADR-0014 decision 1 declines to build the header.
    That makes this constant the service's entire access control, so it is
    asserted literally rather than left to whatever ``--host`` happens to
    default to.
    """
    assert DEFAULT_SERVE_HOST == "127.0.0.1"
    # Returns None; the assertion is that it does not raise.
    ensure_bindable_host(DEFAULT_SERVE_HOST, allow_remote_access=False)


def test_serve_binds_the_default_host_and_port_when_neither_flag_is_given(
    index_dir: Path, base_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The named default reaches the runner, not just the parser."""
    calls = _patch_serve_http(monkeypatch)

    assert cli.main(_argv("serve", index_dir, base_dir)) == 0

    assert calls[0].host == DEFAULT_SERVE_HOST
    assert calls[0].port == DEFAULT_SERVE_PORT


@pytest.mark.parametrize(
    "host",
    [
        NON_LOOPBACK_HOST,
        # The all-interfaces bind: the classic mistake, and the one the flag
        # exists to make deliberate. Passed as data to the guard under test,
        # never bound — hence the scoped ignore of ruff's hardcoded-bind rule.
        "0.0.0.0",  # noqa: S104
        "::",
        # A hostname is refused too. See `_is_loopback_literal`: resolving it
        # would make the verdict depend on an answer that can change, and the
        # ASGI server resolves the name again at bind time regardless.
        "localhost",
    ],
)
def test_non_loopback_host_is_refused_without_the_acknowledgement(
    host: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refusal happens before anything is opened, and names the remedy.

    ``--index-dir`` points at a directory that does not exist, so if the host
    were classified anywhere other than first this would fail with the
    index-directory error instead — which is how the ordering is pinned rather
    than assumed.
    """
    calls = _patch_serve_http(monkeypatch)
    missing_index = tmp_path / "never-created"

    exit_code = cli.main(
        [
            "serve",
            "--index-dir",
            str(missing_index),
            "--base-dir",
            str(tmp_path),
            "--host",
            host,
        ]
    )

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "error:" in err
    assert "loopback" in err
    assert "--allow-remote-access" in err
    assert "--index-dir" not in err
    assert calls == []
    assert not missing_index.exists()


def test_non_loopback_host_is_accepted_with_the_acknowledgement_and_warns(
    index_dir: Path,
    base_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The override serves, and states in words what it published.

    The warning's *content* is asserted, not merely its presence: an operator
    who passes this flag has exposed document text and the absolute paths it
    was read from, to anyone who can reach the port, with no credential. A
    warning saying only "binding a public address" would leave the cost
    unstated.
    """
    calls = _patch_serve_http(monkeypatch)
    caplog.set_level(logging.WARNING, logger="groundkit.service.binding")

    exit_code = cli.main(
        _argv(
            "serve",
            index_dir,
            base_dir,
            "--host",
            NON_LOOPBACK_HOST,
            "--allow-remote-access",
        )
    )

    assert exit_code == 0
    assert calls[0].host == NON_LOOPBACK_HOST

    warnings = [r for r in caplog.records if r.name == "groundkit.service.binding"]
    assert warnings, "an acknowledged non-loopback bind must still warn"
    assert warnings[0].levelno == logging.WARNING
    message = warnings[0].getMessage()
    assert "NO AUTHENTICATION OF ANY KIND" in message
    assert "document content" in message
    assert "absolute filesystem paths" in message


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.53", "::1"])
def test_ipv4_and_ipv6_loopback_are_both_accepted(
    host: str, index_dir: Path, base_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole 127.0.0.0/8 block and the IPv6 loopback address serve.

    ``127.0.0.53`` is in the parametrization on purpose: systemd-resolved binds
    it routinely, and a guard comparing against the string ``"127.0.0.1"``
    instead of classifying with :mod:`ipaddress` would refuse it.
    """
    calls = _patch_serve_http(monkeypatch)

    assert cli.main(_argv("serve", index_dir, base_dir, "--host", host)) == 0
    assert calls[0].host == host


def test_ipv4_mapped_ipv6_loopback_is_accepted(
    index_dir: Path, base_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``::ffff:127.0.0.1`` is accepted, whatever the interpreter thinks of it.

    This test earns its keep because ``IPv6Address.is_loopback``'s answer for
    the IPv4-mapped form **is not stable across CPython patch releases**. On
    3.11.9 it is ``False`` (the test is equality with ``::1``); on later patches
    of 3.11, 3.12 and 3.13 it is ``True``, because the property now consults
    ``ipv4_mapped``. ADR-0014 decision 10 anticipated exactly this drift for
    ``.is_private`` and named it as the reason not to depend on the property —
    it turns out to apply to ``.is_loopback`` too, and decision 10's own
    parenthetical that ``.is_loopback`` is ``False`` here is therefore true only
    of older interpreters.

    That is precisely why :mod:`groundkit.service.binding` unmaps explicitly
    before classifying instead of reading the property: the guard's verdict must
    not be a function of the interpreter patch version. So this test asserts the
    GUARD's behaviour and deliberately does **not** assert the stdlib's — an
    earlier draft did, and it passed locally on 3.11.9 while failing on all
    three CI versions.

    The polarity note in :mod:`groundkit.service.binding` covers the other
    half: :mod:`groundkit.utils.url_safety` unmaps for the opposite reason, to
    *reject* this address on an outbound endpoint.
    """
    calls = _patch_serve_http(monkeypatch)
    assert cli.main(_argv("serve", index_dir, base_dir, "--host", "::ffff:127.0.0.1")) == 0
    assert calls[0].host == "::ffff:127.0.0.1"

    # Directly too, without the CLI in between: this must not raise.
    ensure_bindable_host("::ffff:127.0.0.1", allow_remote_access=False)


def test_sixtofour_address_embedding_loopback_is_still_refused() -> None:
    """A 6to4 address is not a loopback interface, however it is spelled.

    ``2002:7f00:1::`` encodes ``127.0.0.1`` in the 6to4 prefix, and
    ``url_safety.classify_address`` unmaps exactly that form — correctly, since
    unmapping there only ever widens *rejection*. Here it would widen
    *acceptance*, so this module unmaps the IPv4-mapped form and nothing else.
    Pinned so a later "unify the two classifiers" change fails loudly.
    """
    with pytest.raises(ConfigurationError, match="loopback"):
        ensure_bindable_host("2002:7f00:1::", allow_remote_access=False)


# --- Startup guards shared by both transports (ADR-0014 decisions 3 and 6) --


@pytest.mark.parametrize("command", ["serve", "serve-mcp"])
def test_base_dir_is_required_by_both_serve_commands(
    command: str, index_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No default containment root, on either transport.

    ``--base-dir`` is what ``resolve_citation`` checks against, so a server
    started without one could return results whose citations it can never
    verify — and verifiable citations are the product claim (SPEC.md section
    2). argparse refuses the invocation outright rather than the service
    picking a plausible root.
    """
    _patch_serve_http(monkeypatch)
    _patch_run_stdio(monkeypatch)

    with pytest.raises(SystemExit) as exc_info:
        cli.main([command, "--index-dir", str(index_dir)])

    assert exc_info.value.code != 0


@pytest.mark.parametrize("command", ["serve", "serve-mcp"])
def test_missing_index_dir_fails_at_startup_and_creates_nothing(
    command: str,
    base_dir: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The service never creates an index directory (ADR-0014 decision 3).

    Asserting the directory is still absent afterwards is the point: a non-zero
    exit would also be satisfied by a server that created the directory and
    then failed for some other reason, and "reading brings a collection into
    existence" is what would turn an unauthenticated surface into a disk-fill
    primitive.
    """
    calls = _patch_serve_http(monkeypatch)
    seen = _patch_run_stdio(monkeypatch)
    missing_index = tmp_path / "never-created"

    exit_code = cli.main([command, "--index-dir", str(missing_index), "--base-dir", str(base_dir)])

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "error:" in err
    assert "--index-dir" in err
    assert not missing_index.exists()
    assert calls == []
    assert seen == []


def test_missing_base_dir_fails_at_startup(
    index_dir: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A containment root that is not there can verify nothing, so serving stops."""
    calls = _patch_serve_http(monkeypatch)

    exit_code = cli.main(_argv("serve", index_dir, tmp_path / "no-such-root"))

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "error:" in err
    assert "--base-dir" in err
    assert calls == []


def test_serve_mcp_reaches_run_stdio_with_a_serve_time_context(
    index_dir: Path, base_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stdio transport is handed a context assembled from the flags.

    ``index_dir`` and ``base_dir`` arriving on the context — rather than on a
    request — is ADR-0014 decision 6 at the CLI end of the seam: there is no
    resolution path inside ``service/`` for a request to reach, because
    resolution already happened here.
    """
    seen = _patch_run_stdio(monkeypatch)

    assert cli.main(_argv("serve-mcp", index_dir, base_dir)) == 0

    assert len(seen) == 1
    ctx = seen[0]
    assert ctx.index_dir == index_dir
    assert ctx.base_dir == base_dir
    assert ctx.reranker is None


def test_top_k_flag_overrides_the_context_default_and_is_bounded(
    index_dir: Path,
    base_dir: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--top-k`` sets the default applied to requests that omit it.

    The unflagged case is compared against ``ServiceContext``'s own declared
    default rather than a literal, so the CLI cannot quietly grow a second copy
    of that number. Bounds are checked at startup rather than per request: out
    of range, one operator mistake would otherwise fail every request that
    omitted ``top_k``, one at a time, far from its cause.
    """
    calls = _patch_serve_http(monkeypatch)
    declared = next(f for f in fields(ServiceContext) if f.name == "default_top_k")

    assert cli.main(_argv("serve", index_dir, base_dir)) == 0
    assert calls[0].ctx.default_top_k == declared.default

    assert cli.main(_argv("serve", index_dir, base_dir, "--top-k", "3")) == 0
    assert calls[1].ctx.default_top_k == 3

    assert cli.main(_argv("serve", index_dir, base_dir, "--top-k", "0")) == 1
    assert "--top-k" in capsys.readouterr().err
    assert len(calls) == 2


# --- The seam itself: one app, both transports (ADR-0014 decision 5) -------


def test_serve_assembles_one_app_carrying_both_transports(
    index_dir: Path, base_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_serve_http`` really builds the app and really mounts MCP on it.

    Only ``uvicorn.Server`` is substituted, so ``create_app`` and
    ``create_session_manager`` run for real: this is the one test that would
    catch the CLI and the service package disagreeing about the mount seam,
    which no amount of testing either side alone can. Still no socket — the
    substituted server records its configuration instead of serving it.

    The mount path is asserted against ``MCP_HTTP_PATH`` rather than a literal
    ``"/mcp"``, because that constant is exported precisely so the mount point
    and the documented client configuration cannot drift apart.
    """
    import uvicorn
    from fastapi import FastAPI

    configs: list[uvicorn.Config] = []

    class _RecordingServer:
        def __init__(self, config: uvicorn.Config) -> None:
            self._config = config

        async def serve(self) -> None:
            configs.append(self._config)

    monkeypatch.setattr(uvicorn, "Server", _RecordingServer)

    assert cli.main(_argv("serve", index_dir, base_dir, "--port", "9001")) == 0

    assert len(configs) == 1
    config = configs[0]
    assert config.host == DEFAULT_SERVE_HOST
    assert config.port == 9001
    app = config.app
    assert isinstance(app, FastAPI)
    mounted = {getattr(route, "path", "") for route in app.routes}
    assert mcp_server.MCP_HTTP_PATH in mounted


def test_serve_sets_a_connection_backstop(
    index_dir: Path, base_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """uvicorn is given a connection ceiling, and it is only that.

    ``limit_concurrency`` trips on ``len(connections) >= limit or len(tasks) >=
    limit`` -- connections server-wide, idle keep-alive included -- and
    substitutes a 503 app *before* routing, so a trip answers every route, the
    Kubernetes probes included. An earlier version of this test asserted the
    same field while the constant was named for in-flight requests and set
    small, which is exactly the reading that made a stateful MCP transport's
    idle SSE sessions able to 503 a server doing no work.

    So this asserts only what the setting can actually promise -- that a
    connection ceiling exists and is generous. The bound on *work* is a
    semaphore inside the handlers, driven directly by
    ``tests/test_service_tools.py::test_concurrent_corpus_scans_are_bounded``,
    because a test that reads back a configured number cannot tell you what the
    number means.
    """
    import uvicorn

    configs: list[uvicorn.Config] = []

    class _RecordingServer:
        def __init__(self, config: uvicorn.Config) -> None:
            self._config = config

        async def serve(self) -> None:
            configs.append(self._config)

    monkeypatch.setattr(uvicorn, "Server", _RecordingServer)

    assert cli.main(_argv("serve", index_dir, base_dir)) == 0

    assert len(configs) == 1
    assert configs[0].limit_concurrency == cli.SERVE_MAX_CONNECTIONS
    # Generous relative to the work bound, and deliberately so: the two control
    # different resources, and sizing this one like a work bound is the defect
    # it replaced.
    assert cli.SERVE_MAX_CONNECTIONS > MAX_CONCURRENT_CORPUS_SCANS


def test_mcp_mount_wraps_the_session_manager_rather_than_passing_it_as_an_app(
    index_dir: Path, base_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mount carries the two halves ASGI keeps apart.

    ``StreamableHTTPSessionManager.run()`` owns the transport's lifecycle and
    ``handle_request`` is the per-connection entry point; the manager is not
    itself an ASGI app. Asserting the mount's ``app`` is a coroutine function
    of three parameters pins that the adapter exists rather than the manager
    having been handed over directly, which would fail only once a client
    connected.
    """
    import inspect

    captured = _patch_serve_http(monkeypatch)
    assert cli.main(_argv("serve", index_dir, base_dir)) == 0

    mount = cli._build_mcp_mount(captured[0].ctx)

    assert mount.path == mcp_server.MCP_HTTP_PATH
    assert inspect.iscoroutinefunction(mount.app)
    assert len(inspect.signature(mount.app).parameters) == 3
    assert callable(mount.lifespan)


# --- Rerank wiring: configured at serve time, refused at request time ------


def _retrieval_result() -> RetrievalResult:
    """One structurally valid result to hand a reranker."""
    return RetrievalResult(
        content="reciprocal rank fusion",
        score=0.5,
        document_id="doc-1",
        chunk_id="chunk-1",
        source="notes.md",
        start_offset=0,
        end_offset=22,
    )


def test_serve_without_rerank_leaves_the_context_reranker_unset(
    index_dir: Path, base_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``None`` is the refusal ``handle_search`` turns into a typed error."""
    calls = _patch_serve_http(monkeypatch)

    assert cli.main(_argv("serve", index_dir, base_dir)) == 0
    assert calls[0].ctx.reranker is None


def test_rerank_without_the_extra_fails_closed_rather_than_passing_through(
    index_dir: Path, base_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--rerank`` on a base install refuses at first use, never silently.

    The optional extra is deliberately absent from the dev group, so the
    missing-backend condition is simulated at its one boundary
    (``_import_cross_encoder``) rather than by uninstalling anything — which is
    also what stops a machine that *does* have the extra from downloading a
    multi-gigabyte model during the unit suite.

    What is asserted is the shape of the failure: the typed
    ``RerankerNotConfiguredError``, not an unreranked list. A reranker
    returning what it was given is indistinguishable from one that worked, so a
    passthrough here would report results a caller believes were reranked.
    """

    def _no_extra() -> tuple[Any, Any]:
        raise RerankerNotConfiguredError("simulated base install: the 'rerank' extra is absent")

    monkeypatch.setattr(rerank_module, "_import_cross_encoder", _no_extra)
    calls = _patch_serve_http(monkeypatch)

    # Starting succeeds: construction loads no model, so the server comes up
    # and the refusal lands on the request that actually asked to rerank.
    assert cli.main(_argv("serve", index_dir, base_dir, "--rerank")) == 0
    reranker = calls[0].ctx.reranker
    assert reranker is not None

    with pytest.raises(RerankerNotConfiguredError):
        asyncio.run(reranker.rerank("reciprocal rank fusion", [_retrieval_result()], top_k=1))


def test_rerank_model_without_rerank_fails_closed(
    index_dir: Path,
    base_dir: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirrors ``grk eval``: a flag configuring a path this process will not take."""
    calls = _patch_serve_http(monkeypatch)

    exit_code = cli.main(
        _argv("serve", index_dir, base_dir, "--rerank-model", "cross-encoder/some-model")
    )

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "requires --rerank" in err
    assert calls == []


def test_embed_flags_are_refused_rather_than_silently_ignored(
    index_dir: Path,
    base_dir: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ``--embed-*`` flag without ``--dense`` is a mistake to name, not to ignore.

    The same fail-closed convention ``grk ingest`` and ``grk search`` apply: a
    flag that configures a path this process will not take is refused, because
    silently accepting it lets an operator believe a setting took effect. Dense
    IS servable — ``--dense`` wires it — so this refusal is about the flag
    combination, not about a missing capability.
    """
    calls = _patch_serve_http(monkeypatch)

    exit_code = cli.main(_argv("serve", index_dir, base_dir, "--embed-provider", "inmemory"))

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "error:" in err
    assert "require --dense" in err
    assert calls == []


def test_dense_serve_builds_a_per_collection_vector_store_factory(
    index_dir: Path,
    base_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--dense`` wires a factory that is told WHICH collection to open.

    The registry's factory takes the collection name, so each served collection
    reads its own ``<index-dir>/<collection>.lance``. A collection-agnostic
    factory would search one collection's vectors and join the resulting chunk
    ids against another collection's SQLite — no error, just a silently thin
    result on a healthy collection (see
    ``test_runtime.test_each_collection_gets_its_own_vector_store``).

    Asserts on the path the factory resolves rather than on retrieval output:
    without a real dense corpus an empty result cannot distinguish the bug from
    a correctly empty collection, which is exactly the ambiguity that makes this
    failure mode hard to catch.
    """
    opened: list[Path] = []

    class _FakeLanceDB:
        @staticmethod
        async def open(path: Path) -> object:
            opened.append(path)
            return object()

    monkeypatch.setattr(cli, "LanceDBVectorStore", _FakeLanceDB)
    _patch_serve_http(monkeypatch)

    args = cli._build_parser().parse_args(
        _argv("serve", index_dir, base_dir, "--dense", "--embed-provider", "inmemory")
    )
    ctx, _embedder = cli._build_service_context(args)

    async def open_both() -> None:
        # Wrapped rather than calling `asyncio.run(factory(...))` directly: the
        # factory is typed as returning an `Awaitable`, and `asyncio.run`
        # requires a `Coroutine`.
        factory = ctx.registry._vector_store_factory
        assert factory is not None
        await factory("alpha")
        await factory("beta")

    try:
        asyncio.run(open_both())
    finally:
        asyncio.run(ctx.registry.aclose())

    assert opened == [index_dir / "alpha.lance", index_dir / "beta.lance"]


# --- Serve-time embedder ownership -----------------------------------------
#
# `CollectionRuntime.aclose` states outright that it does not close an embedder
# it was merely handed ("a runtime does not own what it was handed"), and
# `CollectionRegistry.aclose` only closes runtimes. That leaves exactly one
# owner for the embedder `_build_service_context` constructs: the command that
# asked for it. Before the fix neither serve command had one, so `grk serve
# --dense` and `grk serve-mcp --dense` returned with a live `httpx.AsyncClient`
# and its connection pool still open.
#
# These tests assert on `OllamaEmbedder`'s own client rather than on a fake's
# "was aclose called" flag, because the flag would pass against an `aclose`
# that did nothing. Constructing an `OllamaEmbedder` performs no I/O — only
# `validate_endpoint_shape` — so no network is involved here.


def _patch_embedder_capture(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Record every embedder `_build_service_context` builds, unmodified.

    Wraps the real ``build_embedder`` rather than substituting a fake, so what
    the assertions inspect afterwards is the actual provider object the command
    would have used, with its actual client.

    The wrapped function is taken from ``providers.embeddings``, its home
    module, while the patch is applied to ``cli`` -- the name ``cli`` binds is
    what ``_build_service_context`` actually calls, but reading it back off
    ``cli`` would be reaching through a re-export the module does not declare.
    """
    built: list[Any] = []
    real_build_embedder = embeddings_module.build_embedder

    def recording_build_embedder(config: Any) -> Any:
        embedder = real_build_embedder(config)
        built.append(embedder)
        return embedder

    monkeypatch.setattr(cli, "build_embedder", recording_build_embedder)
    return built


def test_serve_closes_the_embedder_it_built(
    monkeypatch: pytest.MonkeyPatch, index_dir: Path, base_dir: Path
) -> None:
    """`grk serve --dense` must not return with its HTTP client still open."""
    _patch_serve_http(monkeypatch)
    built = _patch_embedder_capture(monkeypatch)

    assert (
        cli.main(_argv("serve", index_dir, base_dir, "--dense", "--embed-provider", "ollama")) == 0
    )

    assert len(built) == 1, "the dense path must build exactly one embedder"
    assert built[0]._client.is_closed is True


def test_serve_mcp_closes_the_embedder_it_built(
    monkeypatch: pytest.MonkeyPatch, index_dir: Path, base_dir: Path
) -> None:
    """The stdio transport owns its embedder identically -- the leak was in
    both commands, and fixing only the one with a socket would leave the
    long-lived one (an MCP server runs for a whole client session) leaking.
    """
    _patch_run_stdio(monkeypatch)
    built = _patch_embedder_capture(monkeypatch)

    argv = _argv("serve-mcp", index_dir, base_dir, "--dense", "--embed-provider", "ollama")
    assert cli.main(argv) == 0

    assert len(built) == 1
    assert built[0]._client.is_closed is True


def test_serve_closes_the_embedder_when_serving_raises(
    monkeypatch: pytest.MonkeyPatch, index_dir: Path, base_dir: Path
) -> None:
    """The close is in a ``finally``, so a crash mid-serve still releases it.

    This is the case that matters operationally: a clean shutdown leaks a
    client into a process that is exiting anyway, while a crash inside a
    supervised, restarting service leaks one per restart.
    """
    built = _patch_embedder_capture(monkeypatch)

    async def exploding_serve_http(ctx: ServiceContext, *, host: str, port: int) -> None:
        del ctx, host, port
        raise RuntimeError("uvicorn fell over")

    monkeypatch.setattr(cli, "_serve_http", exploding_serve_http)

    args = cli._build_parser().parse_args(
        _argv("serve", index_dir, base_dir, "--dense", "--embed-provider", "ollama")
    )
    with pytest.raises(RuntimeError, match="uvicorn fell over"):
        asyncio.run(cli._cmd_serve(args))

    assert len(built) == 1
    assert built[0]._client.is_closed is True


def test_serve_without_dense_builds_no_embedder_to_close(
    monkeypatch: pytest.MonkeyPatch, index_dir: Path, base_dir: Path
) -> None:
    """The ``None`` branch: no embedder is built, and the added close is a
    no-op rather than an ``AttributeError`` on the default (BM25-only) path.
    """
    _patch_serve_http(monkeypatch)
    built = _patch_embedder_capture(monkeypatch)

    assert cli.main(_argv("serve", index_dir, base_dir)) == 0
    assert built == []

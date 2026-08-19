"""``Host`` validation on both service transports (ADR-0024, amending ADR-0014).

ADR-0014 decision 7 argued that the ``127.0.0.1`` bind *is* the access control,
so no authentication is needed. That argument holds only while the bind cannot
be reached from off-box, and a browser is exactly the thing that breaks it: a
page on any site the victim visits re-points its own hostname at ``127.0.0.1``
with a short-TTL answer, and every request the browser then makes to that name
is same-origin — no CORS preflight, response fully readable. The packet arrives
from loopback; the attacker is not on the box. ``Host`` is the only part of such
a request that still names them.

These tests are SPEC.md §8 regression tests in the strict sense — each was run
against the unfixed source and observed to fail — and the two halves are
independent on purpose. ``TrustedHostMiddleware`` on the FastAPI app also covers
the mounted ``/mcp`` sub-path, so a test that reached the MCP transport *through*
that app would pass with the MCP half reverted. The MCP tests below therefore
mount the session manager in a bare Starlette app with no middleware at all,
which is also the shape a caller mounting ``create_session_manager`` elsewhere
would get.

The second half of the file covers what the first version of that fix got wrong.
It branched on "is this an address literal naming loopback?" while arguing the
unrestricted branch from "once the socket is routable" — two predicates that
differ at exactly one place, a *name* that resolves to loopback. So
``--host localhost --allow-remote-access`` bound a non-routable socket and
switched ``Host`` validation off on both transports, on the one kind of bind
where rebinding is reachable at all. The same version derived the allow-list
from a fixed tuple of three spellings while accepting any of ``127.0.0.0/8`` as
a bind, so ``--host 127.0.0.2`` started and then refused every legitimate
client. Both are pinned below.
"""

from __future__ import annotations

import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.applications import Starlette
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.routing import Mount
from starlette.types import Receive, Scope, Send

from groundkit import cli
from groundkit.runtime import CollectionRegistry
from groundkit.service import mcp_server
from groundkit.service.api import create_app
from groundkit.service.binding import (
    DEFAULT_SERVE_HOST,
    LOOPBACK_HOST_ALLOW_LIST,
    UNRESTRICTED_HOST_ALLOW_LIST,
    HostAllowList,
    derive_host_allow_list,
)
from groundkit.service.mcp_server import MCP_HTTP_PATH, create_session_manager
from groundkit.service.tools import ServiceContext

#: The rebinding attacker's own name, as it would arrive in ``Host``. ``.example``
#: is reserved by RFC 2606, so this can never be a domain anyone owns.
FORGED_HOST = "rebind.example"

#: A routable, non-loopback literal (RFC 5737 TEST-NET-1), matching the constant
#: ``tests/test_cli_serve.py`` uses for the same purpose.
NON_LOOPBACK_HOST = "192.0.2.10"

#: A loopback bind that none of the three canonical spellings names.
#: ``127.0.0.0/8`` is loopback in its entirety and ``_is_loopback_literal``
#: accepts all of it, so this address starts a server; before the fix it then
#: refused every client on both transports.
SECONDARY_LOOPBACK_HOST = "127.0.0.2"

#: The IPv4-mapped loopback bind. Worth its own constant because its canonical
#: rendering is ``::ffff:7f00:1`` while a client's ``Host`` carries the dotted
#: form below — a list built only from ``str(ip_address(host))`` would name the
#: wrong one.
MAPPED_LOOPBACK_HOST = "::ffff:127.0.0.1"

#: A hostname whose resolution this file pins rather than trusts, used for the
#: cases no real name can be relied on to produce offline.
FAKE_HOSTNAME = "grk-test.example"

#: Every spelling of "this machine" a real client sends. The bare forms are not
#: padding: a request to port 80 omits the port from ``Host`` entirely, and the
#: MCP SDK's ``base:*`` pattern is ``host.startswith(base + ":")``, which does
#: **not** match them.
LOOPBACK_HOST_HEADERS = (
    "127.0.0.1:8765",
    "127.0.0.1",
    "localhost:8765",
    "localhost",
    "[::1]:8765",
    "[::1]",
)

#: A well-formed ``initialize`` call. Enough of a real request that a 200 means
#: the transport actually served it rather than rejecting it one check later.
_INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "groundkit-tests", "version": "0"},
    },
}

#: The streamable-HTTP transport requires both media types on ``Accept``.
_MCP_HEADERS = {
    "content-type": "application/json",
    "accept": "application/json, text/event-stream",
}

#: The mount path with its trailing slash, which is what these tests request.
#: A ``Mount`` answers the bare path with a 307 to the slashed one, and the
#: redirect's ``Location`` is built from the request's own ``Host`` — so the
#: test client would then try to connect to the header under test, and its
#: netloc parser splits ``[::1]`` on the wrong colon. Requesting the slashed
#: path skips the redirect and leaves the transport as the only thing deciding
#: the response, which is what is being measured.
_MCP_ENDPOINT = f"{MCP_HTTP_PATH}/"


# -- Fixtures --------------------------------------------------------------


@pytest.fixture
def ctx(tmp_path: Path) -> ServiceContext:
    """A context over an empty index directory.

    No collection is seeded because no test here needs a result — the question
    is whether a request is answered at all, and ``list_collections`` over an
    empty directory answers ``[]`` with a 200.
    """
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    return ServiceContext(
        registry=CollectionRegistry(index_dir), index_dir=index_dir, base_dir=index_dir
    )


def _mcp_only_app(ctx: ServiceContext, host_allow_list: HostAllowList) -> Starlette:
    """Mount the session manager with **no** middleware in front of it.

    Deliberately not ``create_app``: the FastAPI app's ``TrustedHostMiddleware``
    would refuse a forged ``Host`` before the transport ever saw it, so a test
    built on it could not tell whether the MCP half was doing anything.
    """
    manager = create_session_manager(ctx, host_allow_list=host_allow_list)

    async def mcp_app(scope: Scope, receive: Receive, send: Send) -> None:
        await manager.handle_request(scope, receive, send)

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        async with manager.run():
            yield

    return Starlette(lifespan=lifespan, routes=[Mount(MCP_HTTP_PATH, app=mcp_app)])


def _serve_and_capture_app(
    argv: list[str], monkeypatch: pytest.MonkeyPatch
) -> tuple[FastAPI, list[HostAllowList]]:
    """Run ``grk serve`` for real up to the socket, returning what it assembled.

    Only ``uvicorn.Server`` is substituted, so ``create_app``,
    ``_build_mcp_mount`` and ``create_session_manager`` all run — this is the
    shipped wiring, not a reconstruction of it. ``create_session_manager`` is
    wrapped rather than replaced so the allow-list it receives can be recorded
    without changing what it builds.
    """
    received: list[HostAllowList] = []
    real = mcp_server.create_session_manager

    def spy(ctx: ServiceContext, *, host_allow_list: HostAllowList) -> Any:
        received.append(host_allow_list)
        return real(ctx, host_allow_list=host_allow_list)

    monkeypatch.setattr(mcp_server, "create_session_manager", spy)

    configs: list[uvicorn.Config] = []

    class _RecordingServer:
        def __init__(self, config: uvicorn.Config) -> None:
            self._config = config

        async def serve(self) -> None:
            configs.append(self._config)

    monkeypatch.setattr(uvicorn, "Server", _RecordingServer)

    assert cli.main(argv) == 0
    assert len(configs) == 1
    app = configs[0].app
    assert isinstance(app, FastAPI)
    return app, received


def _trusted_host_kwargs(app: FastAPI) -> dict[str, Any]:
    """The options ``TrustedHostMiddleware`` was installed with, or fail.

    ``Middleware.cls`` is typed as a ``_MiddlewareFactory`` protocol rather than
    a class, so ``mypy --strict`` reads a direct identity check against a
    concrete class as non-overlapping. Widening to ``object`` first keeps the
    comparison an identity check — not a name comparison, which would pass for
    any middleware that happened to share the name.
    """
    for middleware in app.user_middleware:
        installed: object = middleware.cls
        if installed is TrustedHostMiddleware:
            return dict(middleware.kwargs)
    pytest.fail("the app installs no TrustedHostMiddleware")


def _pin_resolution(monkeypatch: pytest.MonkeyPatch, *, answers: tuple[str, ...] | None) -> None:
    """Make :func:`socket.getaddrinfo` answer *answers*, or fail to answer.

    ``derive_host_allow_list`` resolves a hostname, and two of the cases it
    distinguishes cannot be produced offline from a real name: an answer outside
    loopback needs a working resolver, and "does not resolve" would otherwise
    depend on the machine's DNS not hijacking unknown names. ``localhost`` is
    left un-pinned wherever it appears below, because it resolving to loopback
    is the one resolution result a test may assume — and it is the exact
    argument the finding was reported against.

    ``answers=None`` raises the error the resolver actually raises, so the
    ``except`` clause under test is the one a real failure would take.
    """

    def fake_getaddrinfo(*_args: object, **_kwargs: object) -> list[Any]:
        if answers is None:
            raise socket.gaierror("Name or service not known")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (answer, 0)) for answer in answers]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)


# -- The REST surface ------------------------------------------------------


def test_rest_surface_refuses_a_forged_host(ctx: ServiceContext) -> None:
    """A rebinding page's ``Host`` is refused; the response body is not reached.

    Shown to fail first by deleting the ``add_middleware`` call from
    ``create_app``: unfixed, this request is answered 200 with the collection
    listing, which is the disclosure ADR-0024 exists to close.
    """
    app = create_app(ctx, host_allow_list=LOOPBACK_HOST_ALLOW_LIST)
    with TestClient(app) as client:
        with_port = client.get("/v1/collections", headers={"host": f"{FORGED_HOST}:8765"})
        without_port = client.get("/v1/collections", headers={"host": FORGED_HOST})

    assert with_port.status_code == 400
    assert without_port.status_code == 400
    # The port is stripped before the comparison, so a forged host is refused
    # whether or not it carries one; asserting both is what pins that.
    assert "invalid host" in with_port.text.lower()


@pytest.mark.parametrize("host_header", LOOPBACK_HOST_HEADERS)
def test_rest_surface_serves_every_loopback_host_spelling(
    ctx: ServiceContext, host_header: str
) -> None:
    """A real local client is still served, in every spelling it might use.

    The IPv6 rows are the ones worth having. Starlette splits ``Host`` on the
    *first* colon, so ``[::1]:8765`` reduces to ``"["`` rather than to the
    address — an allow-list written by hand would omit that and lock out every
    IPv6 client, which is how a security control ends up switched off.
    """
    app = create_app(ctx, host_allow_list=LOOPBACK_HOST_ALLOW_LIST)
    with TestClient(app) as client:
        response = client.get("/v1/collections", headers={"host": host_header})

    assert response.status_code == 200
    assert response.json() == []


# -- The MCP streamable-HTTP transport -------------------------------------


def test_mcp_transport_refuses_a_forged_host(ctx: ServiceContext) -> None:
    """The transport refuses on its own, with no middleware in front of it.

    Shown to fail first by deleting ``security_settings=`` from
    ``create_session_manager``: unfixed, the SDK's default leaves DNS-rebinding
    protection off for backwards compatibility and this ``initialize`` is
    answered, opening a session for the attacker's page.
    """
    app = _mcp_only_app(ctx, LOOPBACK_HOST_ALLOW_LIST)
    with TestClient(app) as client:
        response = client.post(
            _MCP_ENDPOINT,
            json=_INITIALIZE,
            headers={**_MCP_HEADERS, "host": f"{FORGED_HOST}:8765"},
        )

    assert response.status_code == 421
    assert "invalid host" in response.text.lower()


@pytest.mark.parametrize("host_header", LOOPBACK_HOST_HEADERS)
def test_mcp_transport_serves_every_loopback_host_spelling(
    ctx: ServiceContext, host_header: str
) -> None:
    """Every loopback spelling still completes an ``initialize``.

    The MCP matcher and Starlette's disagree about shape — ports and a ``:*``
    wildcard here, a port-stripped hostname there — so both dialects are
    exercised against the same six headers rather than one list being assumed
    to imply the other.
    """
    app = _mcp_only_app(ctx, LOOPBACK_HOST_ALLOW_LIST)
    with TestClient(app) as client:
        response = client.post(
            _MCP_ENDPOINT, json=_INITIALIZE, headers={**_MCP_HEADERS, "host": host_header}
        )

    assert response.status_code == 200


def test_mcp_transport_refuses_a_cross_origin_browser_page(ctx: ServiceContext) -> None:
    """A page's ``Origin`` is refused even when its ``Host`` is loopback.

    This is the second half of the same request: a rebinding page's origin is
    its own name, so populating ``allowed_origins`` refuses it independently of
    the ``Host`` check. A non-browser client sends no ``Origin`` at all and is
    unaffected, which the loopback tests above already demonstrate.
    """
    app = _mcp_only_app(ctx, LOOPBACK_HOST_ALLOW_LIST)
    with TestClient(app) as client:
        response = client.post(
            _MCP_ENDPOINT,
            json=_INITIALIZE,
            headers={
                **_MCP_HEADERS,
                "host": "127.0.0.1:8765",
                "origin": f"http://{FORGED_HOST}",
            },
        )

    assert response.status_code == 403


# -- Loopback binds outside the three canonical spellings ------------------


def _status_on_both_surfaces(
    ctx: ServiceContext, allow_list: HostAllowList, host_header: str
) -> tuple[int, int]:
    """One ``Host`` header, both matchers, as ``(rest, mcp)``.

    The two dialects disagree by construction — Starlette compares a
    port-stripped string, the SDK compares the whole header and understands
    ``:*`` — so a spelling served by one and refused by the other is the
    expected shape of a bug here, not a surprise. Returning both statuses is
    what lets a test assert they agree.
    """
    with TestClient(create_app(ctx, host_allow_list=allow_list)) as rest:
        rest_status = rest.get("/v1/collections", headers={"host": host_header}).status_code
    with TestClient(_mcp_only_app(ctx, allow_list)) as mcp:
        mcp_status = mcp.post(
            _MCP_ENDPOINT, json=_INITIALIZE, headers={**_MCP_HEADERS, "host": host_header}
        ).status_code
    return rest_status, mcp_status


@pytest.mark.parametrize(
    "host_header", [f"{SECONDARY_LOOPBACK_HOST}:8765", SECONDARY_LOOPBACK_HOST]
)
def test_a_secondary_loopback_bind_serves_its_own_clients(
    ctx: ServiceContext, host_header: str
) -> None:
    """``--host 127.0.0.2`` serves the clients that reach it.

    Shown to fail first against the version whose allow-list was the fixed
    three-spelling tuple: the REST surface answered 400 and the MCP transport
    421 for every legitimate request, on a server that had started without
    complaint. systemd-resolved binds ``127.0.0.53`` routinely, so this is not
    a contrived address — ``tests/test_cli_serve.py`` already pins that the bind
    guard accepts the block.
    """
    derived = derive_host_allow_list(SECONDARY_LOOPBACK_HOST)

    assert _status_on_both_surfaces(ctx, derived, host_header) == (200, 200)


@pytest.mark.parametrize(
    "host_header", [f"[{MAPPED_LOOPBACK_HOST}]:8765", f"[{MAPPED_LOOPBACK_HOST}]"]
)
def test_the_ipv4_mapped_loopback_bind_agrees_across_the_two_matchers(
    ctx: ServiceContext, host_header: str
) -> None:
    """The two surfaces answer the same way for ``::ffff:127.0.0.1``.

    They did not. The REST surface served it — but only via the documented
    ``"["`` residual, which admits *any* bracketed literal and so was answering
    a different question by accident — while the MCP matcher had no entry for
    it at all. Agreement is what is asserted, because a guard that serves on one
    transport and refuses on the other is deciding by whichever the client
    happened to use.

    The dotted spelling is what a client sends, and it is not what
    ``str(ip_address(...))`` returns: the canonical rendering is
    ``::ffff:7f00:1``. Both are on the list; this asserts the one that arrives.
    """
    derived = derive_host_allow_list(MAPPED_LOOPBACK_HOST)

    assert _status_on_both_surfaces(ctx, derived, host_header) == (200, 200)


@pytest.mark.parametrize("host_header", ["LOCALHOST", "LocalHost:8765", "localhost.", "127.0.0.1."])
def test_case_and_trailing_dot_are_refused_and_that_is_a_recorded_residual(
    ctx: ServiceContext, host_header: str
) -> None:
    """Neither matcher normalises ``Host``, and this pins that it stays so.

    ``Host`` is case-insensitive and the trailing-dot FQDN form is legal, but
    Starlette compares with ``==`` and the MCP SDK with ``in``, neither
    lowercasing nor stripping the root dot. ADR-0024 already declined to own
    that parsing — its rejected "write our own matcher" alternative names
    "trailing dots, and case" — and this fix does not reverse that call: the
    trailing dot is one extra spelling per entry, but case is not enumerable,
    and a list that handled one legal spelling and not the other would read as
    normalisation while being none.

    It fails **closed** — 400 and 421, never a widening — so it is a
    compatibility gap, recorded as a residual in ADR-0024 and SECURITY.md and
    pinned here. If this test ever has to change, the fix is a normalising
    matcher, not more entries.
    """
    derived = derive_host_allow_list(DEFAULT_SERVE_HOST)

    assert _status_on_both_surfaces(ctx, derived, host_header) == (400, 421)


# -- The derivation itself -------------------------------------------------


def test_the_two_dialects_are_derived_from_one_list() -> None:
    """Starlette's list is a reduction of the MCP list, not a second hand-written one.

    The matchers genuinely differ: the SDK compares the whole header and
    understands ``base:*``, while ``TrustedHostMiddleware`` compares
    ``host.split(":")[0]``. Deriving one from the other is what stops a spelling
    being added to one surface and forgotten on the other — the failure mode
    where two guards disagree and the more permissive one decides.
    """
    reduced = {pattern.split(":")[0] for pattern in LOOPBACK_HOST_ALLOW_LIST.mcp_allowed_hosts}

    assert set(LOOPBACK_HOST_ALLOW_LIST.trusted_hosts) == reduced
    # Named explicitly because it looks like a bug and is not: three of the six
    # patterns collapse onto "[", which is what Starlette compares an IPv6
    # literal against. ADR-0024 records it as a residual — a bracketed literal
    # is never the product of a DNS answer, so no rebinding page can produce
    # one.
    assert "[" in LOOPBACK_HOST_ALLOW_LIST.trusted_hosts
    assert LOOPBACK_HOST_ALLOW_LIST.enforced


def test_the_default_bind_derives_exactly_the_named_constant() -> None:
    """``LOOPBACK_HOST_ALLOW_LIST`` is the derivation for the default host.

    Equality rather than identity, because the list is now built per bind. The
    constant is kept because ``cli._build_mcp_mount`` defaults to it, and this
    is what stops it drifting into a second hand-written copy of the same
    decision.
    """
    assert derive_host_allow_list(DEFAULT_SERVE_HOST) == LOOPBACK_HOST_ALLOW_LIST
    assert LOOPBACK_HOST_ALLOW_LIST.enforced


@pytest.mark.parametrize(
    "host", ["127.0.0.1", "::1", MAPPED_LOOPBACK_HOST, SECONDARY_LOOPBACK_HOST]
)
def test_every_loopback_literal_derives_an_enforced_list_that_names_itself(host: str) -> None:
    """A loopback bind is enforced *and* reachable, for every loopback literal.

    The second half is the one that was missing. ``_is_loopback_literal``
    accepts all of ``127.0.0.0/8`` and ``::ffff:127.0.0.0/104``, while the
    allow-list was a fixed tuple naming ``127.0.0.1``, ``localhost`` and
    ``[::1]`` — so a bind the guard called loopback could be one the allow-list
    could not express, and the server started and then refused everyone. The
    canonical three are asserted to survive alongside, because narrowing to just
    the bound address would break clients that reach a ``127.0.0.2`` bind by no
    name at all.
    """
    derived = derive_host_allow_list(host)

    assert derived.enforced
    assert set(LOOPBACK_HOST_ALLOW_LIST.mcp_allowed_hosts) <= set(derived.mcp_allowed_hosts)
    assert set(LOOPBACK_HOST_ALLOW_LIST.mcp_allowed_origins) <= set(derived.mcp_allowed_origins)
    # The spellings a client actually sends for *this* bind, both with a port
    # and without one, since the SDK's `base:*` pattern does not match a
    # portless value.
    expected = f"[{host}]" if ":" in host else host
    assert expected in derived.mcp_allowed_hosts
    assert f"{expected}:*" in derived.mcp_allowed_hosts
    assert f"http://{expected}" in derived.mcp_allowed_origins


def test_a_hostname_resolving_only_to_loopback_stays_enforced(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``localhost`` binds a non-routable socket, so validation stays on.

    **This is the finding.** The predicate was "is this a loopback address
    literal?", while ADR-0024 decision 3 and SECURITY.md both justified the
    unrestricted branch with "once the socket is routable". Those differ at
    exactly one input: a name that resolves to loopback. So
    ``--host localhost --allow-remote-access`` disabled ``Host`` validation on
    both transports while binding a socket nobody can route to — reinstating the
    rebinding hole on the only kind of bind rebinding can reach.

    The warning is asserted too, and not for completeness: the old one told the
    operator the corpus was published and that ``Host`` validation "is no
    protection at all against anyone who can already route to this port", when
    nothing was published and nobody could route to it.
    """
    with caplog.at_level("WARNING", logger="groundkit.service.binding"):
        derived = derive_host_allow_list("localhost")

    assert derived.enforced
    assert derived == LOOPBACK_HOST_ALLOW_LIST
    assert "ENFORCED" in caplog.text
    assert "published nothing" in caplog.text


def test_a_loopback_hostname_still_refuses_a_forged_host_on_both_surfaces(
    ctx: ServiceContext,
) -> None:
    """The enforced list is a real refusal, not a flag that reads as one.

    Paired with the test above for the reason
    ``test_an_unrestricted_app_accepts_the_host_a_restricted_one_refuses``
    exists: ``enforced is True`` is a boolean until some request is shown to be
    turned away because of it. Both surfaces are exercised, because the finding
    was that both were switched off together.
    """
    derived = derive_host_allow_list("localhost")

    with TestClient(create_app(ctx, host_allow_list=derived)) as rest:
        rest_response = rest.get("/v1/collections", headers={"host": f"{FORGED_HOST}:8765"})
    with TestClient(_mcp_only_app(ctx, derived)) as mcp:
        mcp_response = mcp.post(
            _MCP_ENDPOINT,
            json=_INITIALIZE,
            headers={**_MCP_HEADERS, "host": f"{FORGED_HOST}:8765"},
        )

    assert rest_response.status_code == 400
    assert mcp_response.status_code == 421


def test_a_hostname_resolving_off_loopback_is_unrestricted(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A name on a routable address gets ADR-0024 decision 3's answer.

    The decision survives the fix; only its predicate moved. A published
    deployment is reached through whatever name its clients, reverse proxy or
    overlay network use, which this process cannot know, so a restrictive list
    there would refuse everyone.
    """
    _pin_resolution(monkeypatch, answers=(NON_LOOPBACK_HOST,))

    with caplog.at_level("WARNING", logger="groundkit.service.binding"):
        derived = derive_host_allow_list(FAKE_HOSTNAME)

    assert derived == UNRESTRICTED_HOST_ALLOW_LIST
    assert "DISABLED" in caplog.text
    assert NON_LOOPBACK_HOST in caplog.text


def test_a_hostname_resolving_to_a_mix_is_unrestricted(monkeypatch: pytest.MonkeyPatch) -> None:
    """One routable answer is enough; loopback answers do not outvote it.

    A name resolving to both is bound to both, so the socket really is routable.
    Requiring *every* answer to be non-loopback would make a dual-stack name
    with a loopback entry look confined when it is not.
    """
    _pin_resolution(monkeypatch, answers=("127.0.0.1", NON_LOOPBACK_HOST))

    assert derive_host_allow_list(FAKE_HOSTNAME) == UNRESTRICTED_HOST_ALLOW_LIST


def test_an_unresolvable_hostname_fails_closed_to_the_restricted_list(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A name with no answer must not fall through to unrestricted.

    Going unrestricted requires positive evidence that the socket is routable,
    and "the resolver did not answer" is not evidence of anything. The two ways
    of being wrong are not the same size: too narrow is a 400 the operator sees
    on their first request, too wide is the finding above. The bind that follows
    resolves the same name through the same resolver, so an operator who really
    typed a dead name gets the ASGI server's error rather than one invented
    here.
    """
    _pin_resolution(monkeypatch, answers=None)

    with caplog.at_level("WARNING", logger="groundkit.service.binding"):
        derived = derive_host_allow_list(FAKE_HOSTNAME)

    assert derived.enforced
    assert f"{FAKE_HOSTNAME}:*" in derived.mcp_allowed_hosts
    assert "ENFORCED" in caplog.text


def test_a_wildcard_host_argument_cannot_reach_the_starlette_allow_list(
    ctx: ServiceContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--host '*'`` must not become Starlette's allow-any wildcard.

    ``_trusted_host_pattern`` reduces a value by splitting on the first colon,
    so a bare ``*`` survives it intact — and ``TrustedHostMiddleware`` reads
    ``"*"`` in its list as "serve everything", short-circuiting before it looks
    at a single header. Nothing rejects that argument earlier:
    ``ensure_bindable_host`` lets it through once ``--allow-remote-access`` is
    passed, and it fails only later, at the bind. So the derivation filters what
    it will name, and the assertion is on the served response rather than on the
    list, because the list containing ``"*"`` is only a problem via the
    behaviour it produces.
    """
    _pin_resolution(monkeypatch, answers=None)

    derived = derive_host_allow_list("*")

    assert "*" not in derived.trusted_hosts
    with TestClient(create_app(ctx, host_allow_list=derived)) as client:
        response = client.get("/v1/collections", headers={"host": FORGED_HOST})
    assert response.status_code == 400


def test_an_acknowledged_public_bind_derives_the_unrestricted_list(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A non-loopback bind accepts any ``Host``, and says so out loud.

    ``ensure_bindable_host`` has already refused every non-loopback address the
    operator did not acknowledge with ``--allow-remote-access``, so reaching
    this branch means the corpus was published deliberately. ``Host``
    validation would protect nothing there — an attacker who can route to the
    port simply connects, to a service with no authentication either way — and
    a restrictive list would break every reverse-proxy and overlay-network
    deployment. The decision is argued in ADR-0024; what is asserted here is
    that it is not silent.
    """
    with caplog.at_level("WARNING", logger="groundkit.service.binding"):
        derived = derive_host_allow_list(NON_LOOPBACK_HOST)

    assert derived == UNRESTRICTED_HOST_ALLOW_LIST
    assert not derived.enforced
    assert derived.trusted_hosts == ("*",)
    assert NON_LOOPBACK_HOST in caplog.text


@pytest.mark.parametrize(
    "host",
    [
        # The container default (ADR-0021). Passed as data, never bound — hence
        # the scoped ignore of ruff's hardcoded-bind rule, matching
        # `tests/test_cli_serve.py`.
        "0.0.0.0",  # noqa: S104
        "::",
    ],
)
def test_the_all_interfaces_bind_stays_unrestricted(host: str) -> None:
    """ADR-0021's container default must survive the Finding-1 fix unchanged.

    ``0.0.0.0`` is the one bind where the rejected "accept the bound address
    plus loopback" alternative is most obviously wrong: it would require
    ``Host: 0.0.0.0``, which no client sends. Neither address is a name, so
    neither is resolved; both are simply not loopback.
    """
    assert derive_host_allow_list(host) == UNRESTRICTED_HOST_ALLOW_LIST


def test_no_resolver_is_consulted_for_an_address_literal(monkeypatch: pytest.MonkeyPatch) -> None:
    """A literal is classified without DNS, in both directions.

    A literal is both what the derivation classifies and what actually gets
    bound, so resolving one could only add a way to be wrong — and it would put
    a network round trip on the startup path of the default bind. Pinned by
    making any resolution attempt fail the test outright rather than by
    asserting a call count, which a later refactor could satisfy while still
    resolving somewhere else.
    """

    def forbidden(*_args: object, **_kwargs: object) -> list[Any]:
        pytest.fail("derive_host_allow_list resolved an address literal")

    monkeypatch.setattr(socket, "getaddrinfo", forbidden)

    assert derive_host_allow_list(DEFAULT_SERVE_HOST).enforced
    assert not derive_host_allow_list(NON_LOOPBACK_HOST).enforced


# -- The shipped wiring ----------------------------------------------------


def test_serve_wires_the_restricted_list_into_both_transports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``grk serve`` derives one allow-list and gives it to both surfaces.

    Asserted against the app ``cli._serve_http`` actually built, because the
    library defaults on ``create_app`` and ``create_session_manager`` are
    unrestricted — the security property belongs to the product, and testing
    the defaults would assert the opposite of it.
    """
    index_dir = tmp_path / "idx"
    index_dir.mkdir()
    base_dir = tmp_path / "docs"
    base_dir.mkdir()

    app, mcp_lists = _serve_and_capture_app(
        ["serve", "--index-dir", str(index_dir), "--base-dir", str(base_dir)], monkeypatch
    )

    assert mcp_lists == [LOOPBACK_HOST_ALLOW_LIST]
    assert _trusted_host_kwargs(app)["allowed_hosts"] == list(
        LOOPBACK_HOST_ALLOW_LIST.trusted_hosts
    )


def test_allow_remote_access_widens_both_transports_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The acknowledged public bind widens **both** allow-lists, not one.

    A flag that widened the REST surface and left MCP refusing (or the reverse)
    would be worse than either answer alone: the operator would be told the
    service is published while half of it silently was not.
    """
    index_dir = tmp_path / "idx"
    index_dir.mkdir()
    base_dir = tmp_path / "docs"
    base_dir.mkdir()

    app, mcp_lists = _serve_and_capture_app(
        [
            "serve",
            "--index-dir",
            str(index_dir),
            "--base-dir",
            str(base_dir),
            "--host",
            NON_LOOPBACK_HOST,
            "--allow-remote-access",
        ],
        monkeypatch,
    )

    assert mcp_lists == [UNRESTRICTED_HOST_ALLOW_LIST]
    assert _trusted_host_kwargs(app)["allowed_hosts"] == ["*"]


def test_serve_keeps_validation_enforced_for_a_hostname_bound_to_loopback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--host localhost --allow-remote-access`` publishes nothing, so it enforces.

    The Finding-1 regression at the level it was reported: the real
    ``cli.main`` with only ``uvicorn.Server`` stubbed, so ``create_app``,
    ``_build_mcp_mount`` and ``create_session_manager`` all run. Before the fix
    this invocation exited 0 with ``allowed_hosts=['*']`` on the REST surface
    and rebinding protection off on MCP, while binding a socket nothing off-box
    can route to.

    The combination is not hypothetical: ``ensure_bindable_host``'s own error
    text steers an operator into it, telling them a hostname is refused and that
    ``--allow-remote-access`` is how to proceed.
    """
    index_dir = tmp_path / "idx"
    index_dir.mkdir()
    base_dir = tmp_path / "docs"
    base_dir.mkdir()

    app, mcp_lists = _serve_and_capture_app(
        [
            "serve",
            "--index-dir",
            str(index_dir),
            "--base-dir",
            str(base_dir),
            "--host",
            "localhost",
            "--allow-remote-access",
        ],
        monkeypatch,
    )

    assert mcp_lists == [LOOPBACK_HOST_ALLOW_LIST]
    assert mcp_lists[0].enforced
    assert _trusted_host_kwargs(app)["allowed_hosts"] != ["*"]
    assert _trusted_host_kwargs(app)["allowed_hosts"] == list(
        LOOPBACK_HOST_ALLOW_LIST.trusted_hosts
    )


def test_an_unrestricted_app_accepts_the_host_a_restricted_one_refuses(
    ctx: ServiceContext,
) -> None:
    """The widened list is a real widening, demonstrated on the same request.

    Pairing the two directions in one test is what makes the previous two
    meaningful: without it, ``allowed_hosts == ["*"]`` is a string comparison
    rather than a statement about who gets served.
    """
    unrestricted = create_app(ctx, host_allow_list=UNRESTRICTED_HOST_ALLOW_LIST)
    with TestClient(unrestricted) as client:
        response = client.get("/v1/collections", headers={"host": f"{FORGED_HOST}:8765"})

    assert response.status_code == 200


# -- GK-029 / ADR-0025: the library constructors' own defaults --------------
#
# Every REST test above passes host_allow_list explicitly, matching what
# grk serve actually does — so none of them would notice a regression in
# the bare default a caller gets by omitting the argument entirely. These
# two are the ones that construct create_app / create_session_manager with
# no host_allow_list at all.


def test_create_app_default_refuses_a_forged_host(ctx: ServiceContext) -> None:
    """ADR-0025 amends ADR-0024's residual: the library default is now
    :data:`~groundkit.service.binding.LOOPBACK_HOST_ALLOW_LIST`, not
    :data:`~groundkit.service.binding.UNRESTRICTED_HOST_ALLOW_LIST`. A caller
    who embeds ``create_app`` without ever passing ``host_allow_list`` -- the
    shape ADR-0024 itself named as the intended use of that default -- no
    longer gets an app that accepts any ``Host``.

    Shown to fail first: reverting ``create_app``'s default parameter to
    ``UNRESTRICTED_HOST_ALLOW_LIST`` turns this 400 into a 200.
    """
    app = create_app(ctx)
    with TestClient(app) as client:
        response = client.get("/v1/collections", headers={"host": f"{FORGED_HOST}:8765"})

    assert response.status_code == 400


def test_create_session_manager_default_refuses_a_forged_host(ctx: ServiceContext) -> None:
    """The MCP-side peer of the REST test above, mounted bare like
    ``_mcp_only_app`` so nothing else could be doing the refusing.

    Shown to fail first: reverting ``create_session_manager``'s default
    parameter to ``UNRESTRICTED_HOST_ALLOW_LIST`` turns this 421 into a 200.
    """
    manager = create_session_manager(ctx)

    async def mcp_app(scope: Scope, receive: Receive, send: Send) -> None:
        await manager.handle_request(scope, receive, send)

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        async with manager.run():
            yield

    app = Starlette(lifespan=lifespan, routes=[Mount(MCP_HTTP_PATH, app=mcp_app)])
    with TestClient(app) as client:
        response = client.post(
            _MCP_ENDPOINT,
            json=_INITIALIZE,
            headers={**_MCP_HEADERS, "host": f"{FORGED_HOST}:8765"},
        )

    assert response.status_code == 421

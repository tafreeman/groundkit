"""Tests for the generated FastAPI surface (ADR-0014).

Like ``test_service_tools.py``, the enforcement tests here cannot be shown to
fail by reverting source — ``service/api.py`` was a docstring-only stub before
this phase, so there is no unfixed version to run against. Each is demonstrated
instead by **injecting the violation it exists to catch**, which is the
meaningful direction for a guard and is explicitly NOT the SPEC.md §8 revert
procedure. Every such test says so in its own docstring rather than leaving a
reader to assume a revert was performed.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from starlette.responses import PlainTextResponse
from starlette.types import Receive, Scope, Send

from groundkit.contracts import Chunk, RetrievalResult, SearchResponse
from groundkit.errors import EmbeddingError
from groundkit.index.metadata import SQLiteMetadataStore
from groundkit.retrieval.search import MAX_QUERY_LEN, MAX_TOP_K
from groundkit.runtime import CollectionRegistry
from groundkit.service import api as api_module
from groundkit.service.api import DOC_PATHS, REQUEST_ID_HEADER, McpMount, create_app
from groundkit.service.schemas import ChunkFetchResponse, IndexStatusResponse, SearchRequest
from groundkit.service.tools import TOOLS, ServiceContext
from groundkit.telemetry import JsonLogFormatter

if TYPE_CHECKING:
    from groundkit.service.tools import ToolHandler

#: Logger the surface writes its access and failure lines to. Named once so the
#: caplog filters below and the module cannot drift apart.
_API_LOGGER = "groundkit.service.api"


# -- Fixture helpers -------------------------------------------------------


async def _seed(tmp_path: Path) -> tuple[Path, Path, str]:
    """Write a real source file and index it so citations genuinely resolve."""
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True)
    text = "Turbine maintenance intervals depend on load factor and ambient temperature."
    (corpus / "a.md").write_text(text, encoding="utf-8")

    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True)
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


def _context(index_dir: Path, base_dir: Path) -> ServiceContext:
    return ServiceContext(
        registry=CollectionRegistry(index_dir), index_dir=index_dir, base_dir=base_dir
    )


def _make_app(tmp_path: Path) -> tuple[FastAPI, Path, Path]:
    """Seed a real collection and build an app over it.

    The store is seeded in its own ``asyncio.run`` and closed before the app
    exists, so nothing outlives that loop. The app's own lifespan closes the
    registry on shutdown, which is why every test below drives it through
    ``with TestClient(app)`` rather than calling the client bare — the context
    manager is what runs startup and shutdown.
    """
    index_dir, corpus, _ = asyncio.run(_seed(tmp_path))
    return create_app(_context(index_dir, corpus)), index_dir, corpus


def _empty_context(index_dir: Path) -> ServiceContext:
    """A context over an empty index directory, for route-shape tests only."""
    index_dir.mkdir(parents=True, exist_ok=True)
    return _context(index_dir, index_dir)


def _with_handler(name: str, handler: ToolHandler) -> tuple[Any, ...]:
    """Return :data:`TOOLS` with one spec's handler swapped.

    ``ToolSpec`` is frozen, so the substitution builds a new tuple rather than
    mutating the registry. Tests then point ``api.TOOLS`` at it before calling
    ``create_app``, which is the only way to reach the transport's error and
    logging boundary with a failure of a chosen shape — the real handlers
    cannot be made to raise an ``EmbeddingError`` without a provider.
    """
    return tuple(replace(spec, handler=handler) if spec.name == name else spec for spec in TOOLS)


def _route_pairs(app: FastAPI) -> set[tuple[str, str]]:
    """The ``(path, method)`` pairs the app actually serves, docs excluded.

    ``APIRoute.methods`` is typed optional upstream; ``or ()`` satisfies that
    rather than hiding anything, since a route carrying no method is a route no
    request can reach and therefore no surface to compare.
    """
    return {
        (route.path, method)
        for route in app.routes
        if isinstance(route, APIRoute) and route.path not in DOC_PATHS
        for method in route.methods or ()
    }


def _registry_pairs() -> set[tuple[str, str]]:
    return {(spec.rest_path, spec.rest_method) for spec in TOOLS}


def _assert_registry_route_parity(app: FastAPI) -> None:
    """ADR-0014 decision 2 check 2, factored so the injection test reuses it."""
    assert _route_pairs(app) == _registry_pairs()


def _without_computed_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop the serialization-only fields pydantic will not accept back.

    ``RetrievalResult`` declares ``is_high_confidence`` and ``citation`` as
    ``@computed_field``: they are written on dump and are *not* accepted on
    validate, and ``extra="forbid"`` turns "not accepted" into a rejection. The
    asymmetry is pydantic's, not this surface's — the REST body is the same
    shape ``grk search --json`` prints, which is the property ADR-0014
    decision 8 is about.

    The names are read off the models rather than hardcoded, so a *non*-computed
    field appearing in the body would survive this stripping and be caught by
    the round-trip assertion instead of silently discarded here.
    """
    stripped = {k: v for k, v in payload.items() if k not in SearchResponse.model_computed_fields}
    results: list[Any] = stripped.get("results", [])
    stripped["results"] = [
        {k: v for k, v in item.items() if k not in RetrievalResult.model_computed_fields}
        for item in results
    ]
    return stripped


# -- Enforcement -----------------------------------------------------------


def test_routes_are_exactly_the_registry(tmp_path: Path) -> None:
    """ADR-0014 decision 2 check 2: no route exists that the registry did not declare.

    This is the check that makes "Phase 4 is read-only" decidable on the REST
    side. ``side_effect`` being a one-member ``Literal`` only constrains
    operations that go *through* the registry; a route added with a bare
    decorator would bypass it entirely, and this is what notices.
    """
    app = create_app(_empty_context(tmp_path / "index"))
    assert _route_pairs(app), "the app registered no routes at all"
    _assert_registry_route_parity(app)


def test_a_hand_added_route_breaks_parity(tmp_path: Path) -> None:
    """Injected-violation demonstration, NOT the SPEC.md §8 revert procedure.

    ``service/api.py`` had no pre-fix version to revert to, so the guard is
    proved in the only direction available: commit the violation it exists to
    catch — one route registered with a bare decorator instead of through
    ``TOOLS`` — and assert the same assertion the test above passes now fails.
    A parity test that cannot be made to fail is decoration.
    """
    app = create_app(_empty_context(tmp_path / "index"))
    _assert_registry_route_parity(app)

    @app.get("/v1/documents")
    async def _smuggled() -> list[str]:
        return []

    with pytest.raises(AssertionError):
        _assert_registry_route_parity(app)


# -- Behaviour over HTTP ---------------------------------------------------


def test_all_four_operations_work_over_http(tmp_path: Path) -> None:
    """SPEC.md §1.2's four tools, each reached through its declared route shape."""
    app, _, corpus = _make_app(tmp_path)
    with TestClient(app) as client:
        search = client.post("/v1/search", json={"query": "turbine"})
        assert search.status_code == 200
        assert search.json()["results"]

        collections = client.get("/v1/collections")
        assert collections.status_code == 200
        assert collections.json() == ["default"]

        status = client.get("/v1/collections/default")
        assert status.status_code == 200
        parsed_status = IndexStatusResponse.model_validate(status.json())
        assert parsed_status.document_count == 1
        assert parsed_status.chunk_count == 1
        assert parsed_status.dense_search_available is False

        fetched = client.get("/v1/collections/default/chunks/c1")
        assert fetched.status_code == 200
        parsed_chunk = ChunkFetchResponse.model_validate(fetched.json())
        assert parsed_chunk.verification == "verified"
        assert parsed_chunk.citation.chunk_id == "c1"
        assert parsed_chunk.citation.source == str(corpus / "a.md")

        # Every response carries the id the server logs its failures against.
        for response in (search, collections, status, fetched):
            assert response.headers[REQUEST_ID_HEADER]


def test_search_body_round_trips_through_the_existing_contract(tmp_path: Path) -> None:
    """ADR-0014 decision 8: ``search`` returns ``SearchResponse``, not a parallel DTO.

    The body is re-validated into the contract and re-serialized, and the
    result must equal the bytes the route produced. That is the strong form of
    "unchanged": it would fail if the route added a field, dropped one, or
    renamed one, none of which a shape-free ``200`` assertion would notice.
    """
    app, _, _ = _make_app(tmp_path)
    with TestClient(app) as client:
        response = client.post("/v1/search", json={"query": "turbine", "top_k": 3})
    assert response.status_code == 200

    body = response.json()
    assert set(body) == set(SearchResponse.model_fields) | set(SearchResponse.model_computed_fields)

    restored = SearchResponse.model_validate(_without_computed_fields(body))
    assert restored.query == "turbine"
    assert restored.results
    assert restored.total_results == len(restored.results)
    assert json.loads(restored.model_dump_json()) == body


def test_an_unknown_collection_is_not_found_and_writes_nothing(tmp_path: Path) -> None:
    """ADR-0014 decision 3, both halves.

    ``SQLiteMetadataStore.open`` *creates* the file it does not find, so a naive
    handler turns an unauthenticated read surface into a disk-fill primitive.
    Asserting the status alone would pass against that bug — the directory
    listing is the half that catches it.
    """
    app, index_dir, _ = _make_app(tmp_path)
    with TestClient(app) as client:
        before = sorted(p.name for p in index_dir.iterdir())

        via_path = client.get("/v1/collections/ghost")
        assert via_path.status_code == 404
        assert via_path.json()["kind"] == "collection_not_found"

        via_body = client.post("/v1/search", json={"query": "turbine", "collection": "ghost"})
        assert via_body.status_code == 404
        assert via_body.json()["kind"] == "collection_not_found"

        assert sorted(p.name for p in index_dir.iterdir()) == before


def test_an_invalid_collection_name_is_a_validation_failure_not_a_not_found(
    tmp_path: Path,
) -> None:
    """Order matters: a probe must not learn whether the traversed-to path exists.

    Name validation runs *before* the existence check, so a traversal attempt
    is reported as a bad request. Reversing the two would answer "does
    ``../../etc/passwd`` exist" with a status code, which is a disclosure the
    404 above would otherwise make free.
    """
    app, _, _ = _make_app(tmp_path)
    with TestClient(app) as client:
        traversal = client.post("/v1/search", json={"query": "turbine", "collection": "../evil"})
        assert traversal.status_code == 400
        assert traversal.json()["kind"] == "invalid_request"

        illegal_character = client.get("/v1/collections/evil$name")
        assert illegal_character.status_code == 400
        assert illegal_character.json()["kind"] == "invalid_request"


def test_rerank_without_a_reranker_is_not_implemented(tmp_path: Path) -> None:
    """501, and never a 200 carrying an unreranked list.

    A reranker that quietly returns what it was given is indistinguishable from
    one that worked, so the refusal is the whole reason
    ``RerankerNotConfiguredError`` exists. 501 rather than 400 or 503 because
    the request is well-formed and would be servable on an install carrying the
    extra, and a missing optional dependency is not transient.
    """
    app, _, _ = _make_app(tmp_path)
    with TestClient(app) as client:
        response = client.post("/v1/search", json={"query": "turbine", "rerank": True})
    assert response.status_code == 501
    assert response.json()["kind"] == "reranker_not_configured"


def test_a_credential_in_an_exception_cause_never_reaches_the_caller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """ADR-0014 decision 9's credential rule, injected end to end.

    A scrubbed message with an unscrubbed ``__cause__`` chained behind it is the
    exact shape that leaked credentials in the ported code this repo replaced
    (ADR-0001 hazard 6). The planted secret must appear in neither the body nor
    any header — and *must* appear in the server log, because withholding it
    from the operator too would be a different bug: the request id in the
    response is the only thing joining the two halves.

    Injected-violation demonstration, not a revert: render ``str(exc.__cause__)``
    into the body and the first assertions fail.
    """
    sentinel = "sk-live-DO-NOT-LEAK-0123456789"

    async def leaky(_ctx: ServiceContext, _request: SearchRequest) -> SearchResponse:
        cause = RuntimeError(f"POST https://user:{sentinel}@api.example.com/v1/embeddings")
        raise EmbeddingError("the embedding backend rejected the request") from cause

    monkeypatch.setattr(api_module, "TOOLS", _with_handler("search", leaky))
    app, _, _ = _make_app(tmp_path)

    with TestClient(app) as client, caplog.at_level(logging.ERROR, logger=_API_LOGGER):
        response = client.post("/v1/search", json={"query": "turbine"})

    assert response.status_code == 502
    assert response.json()["kind"] == "embedding_backend_error"
    assert sentinel not in response.text
    assert all(
        sentinel not in name and sentinel not in value for name, value in response.headers.items()
    )

    failures = [record for record in caplog.records if record.name == _API_LOGGER]
    assert failures, "the failure was not logged server-side at all"
    formatter = logging.Formatter()
    assert any(sentinel in formatter.format(record) for record in failures), (
        "the unscrubbed cause must reach the operator's log, keyed by the request id"
    )


def test_the_access_log_carries_the_request_id_and_never_the_query(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """SPEC.md §3 and ADR-0014 decision 9: id, route, status, latency, count — no query.

    This is also why ``search`` is a POST: a query in a query string would be
    written into the access log's request line by any ASGI server, before this
    surface got a say.
    """
    app, _, _ = _make_app(tmp_path)
    # Distinctive enough that a substring match cannot pass by coincidence.
    query = "zephyrine cardamom telemetry"

    with TestClient(app) as client, caplog.at_level(logging.INFO, logger=_API_LOGGER):
        response = client.post("/v1/search", json={"query": query})

    assert response.status_code == 200
    request_id = response.headers[REQUEST_ID_HEADER]

    lines = [record.getMessage() for record in caplog.records if record.name == _API_LOGGER]
    assert lines, "nothing was logged at INFO"
    access = [line for line in lines if request_id in line]
    assert len(access) == 1, f"expected exactly one access line for {request_id}, got {access}"

    line = access[0]
    assert "method=POST" in line
    assert "route=/v1/search" in line
    assert "tool=search" in line
    assert "status=200" in line
    assert "latency_ms=" in line
    assert "results=" in line

    assert all(query not in candidate for candidate in lines), (
        "the query text reached an INFO log line; it is DEBUG-only"
    )


def test_the_access_log_promotes_structured_fields_under_json_formatting(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The same access line, but as a collector under ``GROUNDKIT_LOG_FORMAT=json`` sees it.

    Regression test: the access line carried every field spelled ``key=value``
    inside the *message* string from the start, and the previous test already
    held that to its promise. What it did not catch is that
    :class:`~groundkit.telemetry.JsonLogFormatter` only promotes a field to a
    top-level JSON key when the call site passes it via ``extra=`` —
    ``_log_access`` passed everything as positional ``%s``/``%d`` arguments
    instead, so under JSON formatting every one of those fields stayed
    embedded in an opaque ``message`` string, unindexable by any collector
    that parses JSON logs rather than the human-readable line. This is
    exactly the shape of trace produced against the real Terraform-provisioned
    instance during the 2026-08-16 verification.
    """
    app, _, _ = _make_app(tmp_path)

    with TestClient(app) as client, caplog.at_level(logging.INFO, logger=_API_LOGGER):
        response = client.post("/v1/search", json={"query": "turbine"})

    assert response.status_code == 200
    request_id = response.headers[REQUEST_ID_HEADER]

    access_records = [
        record
        for record in caplog.records
        if record.name == _API_LOGGER and getattr(record, "request_id", None) == request_id
    ]
    assert len(access_records) == 1, (
        f"expected exactly one access record for {request_id}, got {access_records}"
    )
    record = access_records[0]

    payload = json.loads(JsonLogFormatter().format(record))
    assert payload["request_id"] == request_id
    assert payload["method"] == "POST"
    assert payload["route"] == "/v1/search"
    assert payload["tool"] == "search"
    assert payload["status"] == 200
    assert isinstance(payload["latency_ms"], float)
    assert isinstance(payload["results"], int)


def test_out_of_range_requests_are_rejected_before_the_handler_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The precondition ADR-0014 decision 9's ``RetrievalError`` → 409 mapping rests on.

    ``Retriever.search`` raises ``RetrievalError`` both for caller errors (empty
    query, out-of-range ``top_k``) and for a server-side index inconsistency,
    and the type does not distinguish them. Mapping it to 409 is correct *only*
    while the schema rejects the caller-error cases first, so this test spies on
    the handler rather than reading a status code: a 422 produced *after* the
    handler ran would satisfy a status-only assertion and still leave the
    mapping wrong.

    The final leg is a positive control — the same spy must be reached by a
    valid request — because "the handler was never called" is also what a
    broken registry patch looks like.
    """
    reached: list[str] = []

    async def spy(_ctx: ServiceContext, request: SearchRequest) -> SearchResponse:
        reached.append(request.query)
        return SearchResponse(query=request.query, results=[], total_results=0)

    monkeypatch.setattr(api_module, "TOOLS", _with_handler("search", spy))
    app, _, _ = _make_app(tmp_path)

    with TestClient(app) as client:
        too_many = client.post("/v1/search", json={"query": "turbine", "top_k": MAX_TOP_K + 1})
        assert too_many.status_code == 422
        assert too_many.json()["kind"] == "validation_error"

        assert client.post("/v1/search", json={"query": ""}).status_code == 422
        assert (
            client.post("/v1/search", json={"query": "t" * (MAX_QUERY_LEN + 1)}).status_code == 422
        )
        assert reached == []

        ok = client.post("/v1/search", json={"query": "turbine", "top_k": MAX_TOP_K})
        assert ok.status_code == 200
        assert reached == ["turbine"]


def test_no_route_accepts_a_query_parameter(tmp_path: Path) -> None:
    """ADR-0014 decision 6, from the route side rather than the model side.

    ``test_service_tools.py`` asserts no request *model* carries ``base_url``,
    ``index_dir``, ``base_dir`` or an ``embed_*`` field. This asserts the
    complementary thing: the generated routes read nothing but path variables
    and a JSON body, so there is no query-string channel for such a value to
    arrive on even if a model were widened to accept one.
    """
    app = create_app(_empty_context(tmp_path / "index"))
    schema = app.openapi()
    offenders: list[str] = []
    for path, operations in schema["paths"].items():
        for method, operation in operations.items():
            offenders.extend(
                f"{method.upper()} {path} ?{parameter['name']}"
                for parameter in operation.get("parameters", [])
                if parameter.get("in") == "query"
            )
    assert not offenders, f"routes accept query parameters: {offenders}"


def test_no_generated_schema_carries_provider_or_filesystem_configuration(tmp_path: Path) -> None:
    """ADR-0014 decision 6's REST half, walked off the generated OpenAPI document.

    ``test_service_tools.py`` walks the registered request *models*; this walks
    what the app actually publishes, which is the artifact a client codegens
    against. The difference matters because a model reached only through
    nesting — a field whose type is another model — appears here and not there.

    Every component schema is checked, request and response alike. A response
    that disclosed ``base_url`` would be the same leak read the other way, and
    it is a leak ``IndexStatusResponse``'s docstring already promises not to
    make; this is what holds that promise to the generated document.

    The prefix rule is ``embed_`` here, one character stricter than the
    ``embed`` ``test_service_tools.py`` applies to request models, and the
    difference is deliberate rather than an oversight. ``IndexStatusResponse``
    carries an ``embedding`` field — the identity manifest ADR-0014 decision 8
    requires it to report — so a bare ``embed`` prefix applied to responses
    would forbid the one thing the ADR mandates. Request models stay under the
    broader prefix in that file, where nothing legitimately begins with
    ``embed``.
    """
    forbidden_exact = {"base_url", "index_dir", "base_dir", "api_key_env"}
    app = create_app(_empty_context(tmp_path / "index"))
    components: dict[str, Any] = app.openapi().get("components", {}).get("schemas", {})
    assert components, "the app published no component schemas to check"

    offenders = [
        f"{name}.{field}"
        for name, component in components.items()
        for field in component.get("properties", {})
        if field in forbidden_exact or field.startswith("embed_")
    ]
    assert not offenders, f"generated schemas expose server configuration: {offenders}"


def test_the_rest_surface_mounts_an_mcp_transport_without_importing_one(tmp_path: Path) -> None:
    """The inverted dependency in ``McpMount``, exercised rather than asserted.

    ``api.py`` does not import ``service/mcp_server.py``; the transport arrives
    as an ``McpMount`` the caller builds. The payoff is that this test mounts a
    stand-in ASGI app and a stand-in lifespan and gets the whole wiring —
    mounted path reachable, REST routes untouched, transport lifespan entered
    on startup and exited on shutdown — **with the MCP SDK's session lifecycle
    nowhere in the picture**. If the import ran the other way this test could
    not exist, and every REST test would depend on a session manager it never
    uses.

    The mount is a Starlette ``Mount`` rather than an ``APIRoute``, so it does
    not disturb registry parity; that is asserted here too, because "mounting
    MCP quietly added a route" is exactly the kind of thing the parity test
    would otherwise catch only once someone mounted one.
    """
    events: list[str] = []

    @asynccontextmanager
    async def transport_lifespan() -> AsyncIterator[None]:
        events.append("start")
        try:
            yield
        finally:
            events.append("stop")

    async def transport_app(scope: Scope, receive: Receive, send: Send) -> None:
        await PlainTextResponse("mcp-ok")(scope, receive, send)

    app = create_app(
        _empty_context(tmp_path / "index"),
        mcp_mount=McpMount(path="/mcp", app=transport_app, lifespan=transport_lifespan),
    )
    _assert_registry_route_parity(app)

    with TestClient(app) as client:
        assert events == ["start"]
        mounted = client.get("/mcp")
        assert mounted.status_code == 200
        assert mounted.text == "mcp-ok"
        assert client.get("/v1/collections").status_code == 200

    assert events == ["start", "stop"]

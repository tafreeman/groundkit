"""The REST half of Phase 4's read-only surface, generated from the registry (ADR-0014).

Every route this app carries is produced by iterating
:data:`~groundkit.service.tools.TOOLS`. Nothing is registered with a bare
decorator, and that is exactly what ADR-0014 decision 2 check 2 tests: a
hand-added route is a ``(path, method)`` pair the registry does not contain, so
comparing the two sets decides the question. Generation is therefore not a
tidiness preference — it is what turns "this surface exposes exactly the four
declared read-only operations" into a property a test can settle, rather than a
claim a reviewer has to re-derive by reading the file each time it changes.

**This module deliberately does not import**
:mod:`groundkit.service.mcp_server`. An MCP transport arrives as an
:class:`McpMount` the *caller* constructs, so the REST surface stays
constructible and testable without the MCP SDK's session lifecycle. Inverting
the dependency is what keeps that true: importing the server module for the
convenience of a default would couple every REST test to a session manager it
never exercises, and would make a broken MCP import break plain ``grk serve``.

## Method shapes are decided, not incidental

``search`` is a **POST** with a body even though it mutates nothing. Access logs
record the request line at INFO, so a query travelling in a query string writes
every query into an INFO log — directly against SPEC.md §3. That is also why
method alone cannot certify read-only-ness and the registry has to (ADR-0014
alternatives). The three reads that carry no free text are GETs whose parameters
live in the path; **no route accepts a query parameter**, and none may, because
a query parameter is the shape configuration would arrive in.

## Errors and logging split by audience, on purpose

A handler's exception is rendered for the caller by
:mod:`groundkit.service.errors` — which never reads ``__cause__``,
``__context__``, ``repr(exc)``, or a traceback — while the full, unscrubbed
exception goes to the server log at ERROR against the response's request id.
The two halves are joined only by that id, which every response carries in the
``X-Request-ID`` header. The header is the carrier rather than a body field
because ``search`` returns :class:`~groundkit.contracts.SearchResponse`
*unchanged* (ADR-0014 decision 8), so there is nowhere in a success body to put
one without inventing the parallel DTO that decision exists to refuse.

The INFO access log carries the request id, method, route, tool, status,
latency and result count, and **never the query text**; query text is DEBUG
only (SPEC.md §3, ADR-0014 decision 9).
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from inspect import Parameter, Signature
from typing import TYPE_CHECKING, Any, Final, get_type_hints

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from starlette.types import ASGIApp

from groundkit import __version__
from groundkit.contracts import SearchResponse
from groundkit.errors import ConfigurationError, GroundkitError
from groundkit.service.errors import (
    ErrorRendering,
    check_collection,
    map_exception,
    unexpected_error_rendering,
)
from groundkit.service.tools import TOOLS

if TYPE_CHECKING:
    from groundkit.service.tools import ServiceContext, ToolResult, ToolSpec

logger = logging.getLogger(__name__)

#: Header every response carries, success or failure. Named here rather than
#: written inline at three call sites so a client and a log line cannot drift
#: apart on its spelling (SPEC.md §5.2: thresholds and names are constants).
REQUEST_ID_HEADER: Final[str] = "X-Request-ID"

#: ``kind`` reported when a request is rejected by schema validation before any
#: handler runs. It is a *rendering label*, not an exception type —
#: ``service/errors.py`` stays closed, and pydantic's ``ValidationError`` is not
#: a :class:`~groundkit.errors.GroundkitError` for it to have an entry in. Kept
#: distinct from ``invalid_request`` (the ``ConfigurationError`` label) because
#: the two answer different questions: this one means "your request did not
#: parse", that one means "your request parsed and named something invalid".
VALIDATION_ERROR_KIND: Final[str] = "validation_error"

#: Names of the parameters the generated endpoints are given. Both are prefixed
#: so that a future ``rest_path`` template variable cannot collide with them —
#: a collision would silently shadow the injected response object, and a path
#: variable named ``_gk_response`` is not a thing anyone writes by accident.
_BODY_PARAM: Final[str] = "_gk_body"
_RESPONSE_PARAM: Final[str] = "_gk_response"

#: Matches ``{name}`` in a ``ToolSpec.rest_path``. The registry is the source of
#: the path, so the parameter names are read back off it rather than restated
#: here — a second list of ``("collection", "chunk_id")`` would be one more
#: thing to keep in step with ``tools.py``.
_PATH_PARAM_RE: Final[re.Pattern[str]] = re.compile(r"\{([^{}]+)\}")

#: FastAPI's own documentation routes. Not registered from the registry, so the
#: parity test excludes them by path; listed here so the test and the app agree
#: on the exclusion set instead of each carrying its own copy.
DOC_PATHS: Final[frozenset[str]] = frozenset(
    {"/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"}
)


@dataclass(frozen=True, slots=True)
class McpMount:
    """An MCP transport to mount, supplied by the caller.

    api.py deliberately does NOT import service/mcp_server.py: the REST surface
    must be constructible and testable without the MCP SDK's session lifecycle,
    and inverting the dependency here is what keeps that true.

    Attributes:
        path: Sub-path the transport is mounted at.
        app: The transport's ASGI application.
        lifespan: Zero-argument factory returning the transport's async context
            manager. A *factory* rather than an already-entered context manager
            because :func:`create_app` may be called before any event loop
            exists, and an ``AsyncContextManager`` created eagerly at that
            moment would bind to the wrong loop — or to none.
    """

    path: str
    app: ASGIApp
    lifespan: Callable[[], AbstractAsyncContextManager[None]]


def create_app(ctx: ServiceContext, *, mcp_mount: McpMount | None = None) -> FastAPI:
    """Build the read-only REST app, one route per registered tool.

    With ``mcp_mount`` omitted the app is pure REST and pulls in nothing from
    the MCP SDK. With it supplied, its ASGI app is mounted at its path and its
    lifespan is entered inside the app's own lifespan, which is the only place
    a streamable-HTTP session manager can be started and stopped in step with
    the server process.

    The context's registry is closed on shutdown. The app does not *own* the
    :class:`~groundkit.service.tools.ServiceContext` — it is assembled at serve
    time and handed in — but it is the only component that knows when serving
    has stopped, and ``CollectionRegistry.aclose`` is idempotent, so a caller
    that also closes it is correct rather than in conflict. The consequence
    worth naming: an app whose lifespan has completed cannot serve again,
    because its registry is closed.

    Args:
        ctx: Serve-time context. Everything a handler may reach lives here;
            no request model carries any of it (ADR-0014 decision 6).
        mcp_mount: Optional MCP transport to mount alongside the REST routes.

    Returns:
        A FastAPI application exposing exactly the operations in
        :data:`~groundkit.service.tools.TOOLS`.
    """

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            if mcp_mount is None:
                yield
            else:
                async with mcp_mount.lifespan():
                    yield
        finally:
            await ctx.registry.aclose()

    app = FastAPI(
        title="groundkit",
        version=__version__,
        summary=(
            "Read-only retrieval surface: search, fetch_chunk, list_collections, index_status."
        ),
        lifespan=lifespan,
    )
    app.add_exception_handler(RequestValidationError, _handle_validation_error)

    for spec in TOOLS:
        _register(app, spec, ctx)

    if mcp_mount is not None:
        # A Mount, not an APIRoute, so it does not participate in the
        # registry<->route parity comparison. That is correct rather than a
        # loophole: the MCP surface has its own parity test against the same
        # registry (ADR-0014 decision 2 check 3), and asserting the mount here
        # would be asserting the same thing twice from the weaker side.
        app.mount(mcp_mount.path, mcp_mount.app)

    return app


# -- Route generation ------------------------------------------------------


def _register(app: FastAPI, spec: ToolSpec, ctx: ServiceContext) -> None:
    """Add one route for ``spec``, deriving everything from the spec itself."""
    path_params = _path_parameters(spec.rest_path)
    if spec.rest_method == "POST":
        endpoint = _body_endpoint(spec, ctx, path_params)
    else:
        endpoint = _path_endpoint(spec, ctx, path_params)

    app.add_api_route(
        spec.rest_path,
        endpoint,
        methods=[spec.rest_method],
        name=spec.name,
        operation_id=spec.name,
        summary=spec.summary,
        response_model=_response_model(spec),
    )


def _path_parameters(path: str) -> tuple[str, ...]:
    """Return the ``{name}`` template variables in ``path``, in order."""
    return tuple(_PATH_PARAM_RE.findall(path))


def _response_model(spec: ToolSpec) -> Any:
    """Read the response model off the handler's own return annotation.

    Declared nowhere else on purpose. ``ToolSpec`` carries a ``request_model``
    because a transport must *parse* into one, but a second field naming the
    response type would be a restatement of what the handler already declares,
    and the two could disagree — which is the seam drift ADR-0001 hazard 4
    exists to catch, in the direction that would silently publish an OpenAPI
    schema no response matches.
    """
    return get_type_hints(spec.handler)["return"]


def _body_endpoint(
    spec: ToolSpec, ctx: ServiceContext, path_params: tuple[str, ...]
) -> Callable[..., Awaitable[Response | ToolResult]]:
    """Build the endpoint for a method whose request arrives as a JSON body.

    FastAPI parses and validates the body into ``spec.request_model`` before
    this coroutine runs, which is what makes ADR-0014 decision 9's
    ``RetrievalError`` → 409 mapping correct: the caller-error cases (empty
    query, out-of-range ``top_k``) are rejected as 422 by the schema and never
    reach a handler, so the only ``RetrievalError`` that can surface from one
    is a genuine index inconsistency.
    """
    if path_params:
        # No registered POST has path parameters, and one would be ambiguous:
        # the body model is authoritative here, so a path variable carrying the
        # same field would be read from neither place consistently. Refused at
        # construction rather than resolved by a rule nobody would remember.
        raise ConfigurationError(
            f"tool {spec.name!r} declares a body method with path parameters "
            f"{path_params!r}; the body model is the single source of a request"
        )

    async def endpoint(**kwargs: Any) -> Response | ToolResult:
        http_response: Response = kwargs[_RESPONSE_PARAM]
        return await _dispatch(spec, ctx, http_response, kwargs[_BODY_PARAM])

    _apply_signature(
        endpoint,
        [Parameter(_BODY_PARAM, Parameter.POSITIONAL_OR_KEYWORD, annotation=spec.request_model)],
    )
    return endpoint


def _path_endpoint(
    spec: ToolSpec, ctx: ServiceContext, path_params: tuple[str, ...]
) -> Callable[..., Awaitable[Response | ToolResult]]:
    """Build the endpoint for a method whose request arrives as path variables.

    The request model is constructed from the path variables alone. **No query
    parameter is accepted**, and that is a security property rather than a
    style: a query parameter is the shape provider or filesystem configuration
    would arrive in, and the surface that never reads one cannot be talked into
    honouring a ``?base_url=`` (ADR-0014 decision 6). A path variable that the
    model rejects is a 422, rendered here rather than by FastAPI because
    FastAPI validated it only as ``str``.
    """

    async def endpoint(**kwargs: Any) -> Response | ToolResult:
        http_response: Response = kwargs.pop(_RESPONSE_PARAM)
        return await _dispatch(spec, ctx, http_response, kwargs)

    _apply_signature(
        endpoint,
        [Parameter(name, Parameter.POSITIONAL_OR_KEYWORD, annotation=str) for name in path_params],
    )
    return endpoint


def _apply_signature(endpoint: Callable[..., Any], parameters: list[Parameter]) -> None:
    """Give a generated endpoint the signature FastAPI introspects.

    FastAPI decides what to parse — a body, a path variable, an injected
    response — from ``inspect.signature``, which honours ``__signature__``.
    Writing one is therefore how a generated route gets a *real* OpenAPI
    schema, and the reason this file can generate routes without giving up the
    schema walk ADR-0014 decision 6's test performs over the generated
    components.

    The signature is left with no return annotation deliberately: the endpoints
    return ``Response | ToolResult``, and FastAPI would try to build a response
    model out of that union if it were visible. ``response_model`` is passed to
    ``add_api_route`` explicitly instead, from :func:`_response_model`.
    """
    full = [
        *parameters,
        Parameter(_RESPONSE_PARAM, Parameter.POSITIONAL_OR_KEYWORD, annotation=Response),
    ]
    # ``Any`` because a function object has no typed ``__signature__``
    # attribute; the assignment is the documented FastAPI extension point.
    target: Any = endpoint
    target.__signature__ = Signature(full)


# -- Dispatch --------------------------------------------------------------


async def _dispatch(
    spec: ToolSpec,
    ctx: ServiceContext,
    http_response: Response,
    raw: BaseModel | Mapping[str, str],
) -> Response | ToolResult:
    """Run one tool, rendering every failure through the shared error mapping.

    ``raw`` is already a model when FastAPI parsed a body, and a mapping of
    path variables otherwise. Both paths converge on one validated request
    object before anything else happens, so the precondition and error rules
    below are stated once rather than once per method shape.
    """
    request_id = uuid.uuid4().hex
    http_response.headers[REQUEST_ID_HEADER] = request_id
    started = time.perf_counter()

    if isinstance(raw, BaseModel):
        request: BaseModel = raw
    else:
        try:
            request = spec.request_model.model_validate(dict(raw))
        except ValidationError as exc:
            _log_access(spec, request_id, 422, started, 0)
            return _json_error(
                status_code=422,
                kind=VALIDATION_ERROR_KIND,
                detail=_pydantic_detail(exc),
                request_id=request_id,
            )

    # Query text is DEBUG-only (SPEC.md §3). This is the one place a request's
    # full contents are written, and it sits below the INFO access log
    # deliberately so a default-configured server logs the fact of a search and
    # never its subject.
    logger.debug("request_id=%s tool=%s request=%r", request_id, spec.name, request)

    # ADR-0014 decision 3: name validation then existence, as a PRECONDITION
    # before any handler runs. Order is the security property — a traversal
    # attempt is reported as a validation failure, never as a not-found that
    # would have confirmed whether the traversed-to path exists. It also keeps
    # ``SQLiteMetadataStore.open`` off names that do not exist, since that
    # method *creates* the file it does not find.
    collection = getattr(request, "collection", None)
    if isinstance(collection, str):
        refusal = check_collection(ctx.index_dir, collection)
        if refusal is not None:
            _log_access(spec, request_id, refusal.status_code, started, 0)
            return _error_response(refusal, request_id)

    try:
        result = await spec.handler(ctx, request)
    except GroundkitError as exc:
        rendering = map_exception(exc)
        _log_failure(spec, request_id, rendering, exc)
        _log_access(spec, request_id, rendering.status_code, started, 0)
        return _error_response(rendering, request_id)
    except Exception as exc:
        # A non-GroundkitError escaping a handler is a bug, and its message is
        # whatever an arbitrary library chose to write, so it is logged in full
        # and never rendered. Caught rather than allowed to reach Starlette's
        # 500 handler because that path returns no request id, which would
        # leave the operator's log entry unjoinable to the caller's report.
        rendering = unexpected_error_rendering()
        _log_failure(spec, request_id, rendering, exc)
        _log_access(spec, request_id, rendering.status_code, started, 0)
        return _error_response(rendering, request_id)

    _log_access(spec, request_id, 200, started, _result_count(result))
    return result


def _result_count(result: ToolResult) -> int:
    """Number of items an access-log line reports for ``result``.

    A single-object read counts as one: the field exists so an operator can see
    "this search matched nothing" in the log without the query being in it, and
    reporting ``0`` for a successful ``fetch_chunk`` would make that reading
    wrong.
    """
    if isinstance(result, SearchResponse):
        return len(result.results)
    if isinstance(result, list):
        return len(result)
    return 1


# -- Rendering and logging -------------------------------------------------


def _json_error(*, status_code: int, kind: str, detail: str, request_id: str) -> JSONResponse:
    """Render an error body: caller-safe detail, stable kind, joinable id.

    ``kind`` is the field a client should branch on. ``detail`` is prose and
    may change; ``request_id`` is what an operator needs to find the full
    exception in the server log, and is the only thing tying the two halves of
    ADR-0014 decision 9 together.
    """
    return JSONResponse(
        status_code=status_code,
        content={"detail": detail, "kind": kind, "request_id": request_id},
        headers={REQUEST_ID_HEADER: request_id},
    )


def _error_response(rendering: ErrorRendering, request_id: str) -> JSONResponse:
    """Render an :class:`ErrorRendering` produced by ``service/errors.py``."""
    return _json_error(
        status_code=rendering.status_code,
        kind=rendering.kind,
        detail=rendering.detail,
        request_id=request_id,
    )


def _pydantic_detail(exc: RequestValidationError | ValidationError) -> str:
    """Summarize a schema rejection without echoing the rejected value.

    pydantic's own ``errors()`` payload carries ``input``, which for ``query``
    is the caller's search text — the one field SPEC.md §3 keeps out of logs
    and out of anything that might be logged downstream. Only the location and
    the message are returned; the caller already knows what they sent.

    The entries are read as plain mappings because the two exception types
    describe them differently (a ``TypedDict`` on one side, a bare ``dict`` on
    the other) while agreeing on the two keys used here.
    """
    entries: list[Any] = list(exc.errors())
    parts: list[str] = []
    for error in entries:
        location = ".".join(str(part) for part in error.get("loc", ()))
        message = str(error.get("msg", "invalid value"))
        parts.append(f"{location}: {message}" if location else message)
    return "; ".join(parts) or "the request could not be validated"


async def _handle_validation_error(request: Request, exc: Exception) -> Response:
    """Render FastAPI's own body/path rejection in this surface's error shape.

    Registered so a 422 carries the same ``detail``/``kind``/``request_id``
    triple every other failure does — a client that has to parse two error
    shapes will parse one of them wrong. Typed against ``Exception`` because
    that is the signature Starlette's handler registry declares; the narrowing
    happens in :func:`_pydantic_detail`.

    Only ``request.url.path`` is logged, never ``request.url``: the full URL
    would carry a query string, and this handler runs on requests that failed
    validation, which is precisely when a caller has sent something malformed
    into a place it did not belong.
    """
    request_id = uuid.uuid4().hex
    detail = (
        _pydantic_detail(exc)
        if isinstance(exc, RequestValidationError | ValidationError)
        else "the request could not be validated"
    )
    logger.info(
        "request_id=%s method=%s route=%s status=%d rejected=schema_validation",
        request_id,
        request.method,
        request.url.path,
        422,
        extra={
            "request_id": request_id,
            "method": request.method,
            "route": request.url.path,
            "status": 422,
            "rejected": "schema_validation",
        },
    )
    return _json_error(
        status_code=422, kind=VALIDATION_ERROR_KIND, detail=detail, request_id=request_id
    )


def _log_access(
    spec: ToolSpec, request_id: str, status_code: int, started: float, result_count: int
) -> None:
    """Write the INFO access line: id, route, status, latency, result count.

    **Never the query text.** SPEC.md §3 and ADR-0014 decision 9 make this the
    line a test asserts against, so the fields are spelled ``key=value`` and
    kept stable rather than prose that a reword would quietly break.
    """
    latency_ms = (time.perf_counter() - started) * 1000.0
    logger.info(
        "request_id=%s method=%s route=%s tool=%s status=%d latency_ms=%.2f results=%d",
        request_id,
        spec.rest_method,
        spec.rest_path,
        spec.name,
        status_code,
        latency_ms,
        result_count,
        extra={
            "request_id": request_id,
            "method": spec.rest_method,
            "route": spec.rest_path,
            "tool": spec.name,
            "status": status_code,
            "latency_ms": latency_ms,
            "results": result_count,
        },
    )


def _log_failure(
    spec: ToolSpec, request_id: str, rendering: ErrorRendering, exc: BaseException
) -> None:
    """Log the full, unscrubbed exception server-side against ``request_id``.

    ``exc_info`` is the whole point: it writes the traceback *and* the
    ``__cause__`` chain, which is exactly what the response body must not
    contain. Nothing is lost to an operator by the mapper's refusal to read
    that chain — it is written here instead, and the request id in the caller's
    response is what finds this line.
    """
    logger.error(
        "request_id=%s tool=%s failed kind=%s status=%d",
        request_id,
        spec.name,
        rendering.kind,
        rendering.status_code,
        exc_info=exc,
        extra={
            "request_id": request_id,
            "tool": spec.name,
            "failure_kind": rendering.kind,
            "status": rendering.status_code,
        },
    )

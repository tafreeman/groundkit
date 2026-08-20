"""Transport-agnostic operations and the registry that declares them (ADR-0014).

Both surfaces — FastAPI and MCP — dispatch through :data:`TOOLS`. Neither
defines an operation of its own, and that is the mechanism by which "Phase 4 is
read-only" is *enforced* rather than asserted: :attr:`ToolSpec.side_effect` is a
closed ``Literal`` with exactly one member, so adding a mutating operation
requires widening that literal, which ``mypy --strict`` blocks until someone
does it deliberately and visibly in the diff. Parity tests on each transport
then assert that its registered operations are exactly this tuple, so an
operation added directly to a router or an MCP server — bypassing the registry —
fails rather than shipping.

The handlers here reach collections only through
:class:`~groundkit.runtime.CollectionRuntime`, whose public surface is
search-by-acquire plus reads. There is no path from this module to
``upsert_document``, ``add_chunks``, ``replace_document``, ``delete_document``
or ``write_manifest``, and an import scan over this package asserts that no
service module reaches the ingest path at all.

SPEC.md §1.2 names the four operations. Ingest is not among them; that is the
cited authority, so a later reader finds a decision rather than an omission.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, Literal

from pydantic import BaseModel

from groundkit import snapshots
from groundkit.contracts import Citation, SearchResponse
from groundkit.errors import ConfigurationError, RerankerNotConfiguredError, RetrievalError
from groundkit.index.metadata import SCHEMA_VERSION, is_groundkit_store
from groundkit.retrieval.citations import resolve_citation
from groundkit.retrieval.search import MAX_TOP_K
from groundkit.service.schemas import (
    ChunkFetchResponse,
    FetchChunkRequest,
    IndexStatusRequest,
    IndexStatusResponse,
    ListCollectionsRequest,
    SearchRequest,
    VerificationVerdict,
)
from groundkit.utils.path_safety import is_within_base

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path
    from typing import TypeAlias

    from groundkit.retrieval.protocols import RerankerProtocol
    from groundkit.runtime import CollectionRegistry

logger = logging.getLogger(__name__)

#: Every value a tool's declared effect may take. A closed ``Literal`` with one
#: member, on purpose: Phase 4 exposes no mutating operation, and the type
#: system is what keeps that true. Widening this is the deliberate, reviewable
#: act that a future mutating operation must perform — and per SPEC.md §7 it
#: must arrive in the same change as the shared-secret header, the
#: constant-time compare, and the unset-secret disable.
SideEffect = Literal["read_only"]

#: What a handler may return. Not ``object``: the transports serialize these,
#: and an open return type would let a handler return something neither
#: surface knows how to render.
ToolResult = SearchResponse | ChunkFetchResponse | IndexStatusResponse | list[str]


def result_count(result: ToolResult) -> int:
    """Number of items an access log reports for ``result``.

    A single-object read counts as one: the field exists so an operator can
    see "this search matched nothing" in the log without the query being in
    it, and reporting ``0`` for a successful ``fetch_chunk`` would make that
    reading wrong.

    Lives here, beside :data:`ToolResult`, because **both** transports need
    it and neither may import the other. ``api.py`` deliberately does not
    import ``mcp_server``, and the reverse would pull FastAPI into the stdio
    MCP path for the sake of one ``isinstance`` chain. A shared seam with a
    shared type is the right home; a second copy in the other transport is
    how the two access logs would drift into disagreeing about what a
    "result" is.

    Args:
        result: Whatever a handler returned.

    Returns:
        The count an access log should report.
    """
    if isinstance(result, SearchResponse):
        return len(result.results)
    if isinstance(result, list):
        return len(result)
    return 1


if TYPE_CHECKING:
    #: The request parameter is ``Any`` because the registry is heterogeneous —
    #: each handler takes its own request model, and a homogeneous tuple cannot
    #: express that without erasing it. The narrowing is real rather than lost:
    #: each concrete handler below declares its exact model, and a transport
    #: builds its request from :attr:`ToolSpec.request_model`, so the object a
    #: handler receives is the one it declared. This is the one place the alias
    #: is wider than the functions it holds.
    #:
    #: Declared under ``TYPE_CHECKING`` because ``Callable`` and ``Awaitable``
    #: are imported there: a module-level alias is evaluated at import time and
    #: would raise ``NameError`` even though every use of it is an annotation.
    ToolHandler: TypeAlias = Callable[["ServiceContext", Any], Awaitable[ToolResult]]


#: How many corpus-scale operations may be in flight at once, across both
#: transports, in one server process.
#:
#: The operations this bounds — :func:`handle_search` and
#: :func:`handle_index_status` — are O(corpus) regardless of ``top_k``: BM25
#: scores every indexed chunk (ADR-0002's accepted trade), and status reads an
#: aggregate over every chunk row plus every document row. Each one in flight
#: holds its own corpus-scale working set, so without a bound peak memory is
#: decided by arrival rate, and the single replica ``infra/k8s/deployment.yaml``
#: describes is OOMKilled rather than slowed.
#:
#: **This is the bound on work. uvicorn's ``limit_concurrency`` is not**, and
#: the first version of this control used it as though it were. That setting
#: trips on ``len(connections) >= limit or len(tasks) >= limit`` — connections
#: server-wide, idle keep-alive included — and substitutes a 503 app *before*
#: routing, so it answers every route including the Kubernetes probes. `grk
#: serve` mounts a stateful MCP transport whose clients hold an SSE stream open
#: for up to :data:`~groundkit.service.mcp_server._SESSION_IDLE_TIMEOUT_SECONDS`,
#: so idle sessions alone could have tripped it, 503-ing a server doing no work
#: and restart-looping the pod through its liveness probe. A cap on connections
#: cannot express "bound the expensive work": only a bound at the operation can.
#:
#: Waiters queue rather than being shed. A waiter holds a connection but *not*
#: a corpus-scale working set, which is the distinction the connection cap
#: could not draw; connection exhaustion is bounded separately, by
#: ``cli.SERVE_MAX_CONNECTIONS``. Eight rather than the thread-pool width
#: because the binding resource is memory, not CPU: the scan itself already
#: runs on the shared ``asyncio.to_thread`` executor, so more concurrent scans
#: buy queued working sets rather than throughput.
MAX_CONCURRENT_CORPUS_SCANS: Final[int] = 8


def _new_scan_limiter() -> asyncio.Semaphore:
    """Build a fresh limiter for one :class:`ServiceContext`.

    Per context rather than module-global so tests are isolated from each
    other; safe to construct outside a running loop, since an
    ``asyncio.Semaphore`` binds no loop until it first has to wait.
    """
    return asyncio.Semaphore(MAX_CONCURRENT_CORPUS_SCANS)


@dataclass(frozen=True, slots=True)
class ServiceContext:
    """Everything a handler may reach. Assembled at serve time, never per request.

    This object *is* ADR-0014 decision 6's guarantee. Provider settings, the
    index directory and the containment root arrive here from serve-time
    resolution, so no request model needs to carry them and no handler has a
    path to them other than this. The constraint holds by construction; the
    schema test that asserts no request field is named ``base_url`` or
    ``embed_*`` exists because a structural argument decays as a surface grows.

    Attributes:
        registry: Collection registry; the only route to a retriever.
        index_dir: Directory holding collections, for enumeration.
        base_dir: Containment root every citation must resolve within.
        reranker: Cross-encoder, or ``None`` when the server was started
            without ``--rerank``. ``None`` is a refusal, never a passthrough.
        default_top_k: Applied when a request omits ``top_k``.
        scan_limiter: Bounds concurrent corpus-scale work to
            :data:`MAX_CONCURRENT_CORPUS_SCANS`. Shared by both transports
            because they share these handlers, which is the point: two
            transports over one runtime must contend for one budget, not two.
    """

    registry: CollectionRegistry
    index_dir: Path
    base_dir: Path
    reranker: RerankerProtocol | None = None
    default_top_k: int = 5
    scan_limiter: asyncio.Semaphore = field(default_factory=_new_scan_limiter, repr=False)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """One operation, declared once and rendered by both transports.

    Attributes:
        name: MCP tool name, and the identity both parity tests compare on.
        summary: One-line description, surfaced to MCP clients.
        request_model: Pydantic model the transport parses into. The MCP input
            schema is *generated* from this via ``model_json_schema()`` rather
            than hand-written a second time — a second schema is exactly the
            seam drift ADR-0001 hazard 4 exists to catch.
        handler: Coroutine performing the operation.
        side_effect: Always ``"read_only"`` in Phase 4; see :data:`SideEffect`.
        rest_path: Path the REST surface mounts this on.
        rest_method: HTTP method for the REST mirror.
    """

    name: str
    summary: str
    request_model: type[BaseModel]
    handler: ToolHandler
    side_effect: SideEffect
    rest_path: str
    rest_method: Literal["GET", "POST"]


async def handle_search(ctx: ServiceContext, request: SearchRequest) -> SearchResponse:
    """Search one collection, optionally reranking the result.

    Rerank is a per-request option, not a fourth ``SearchMode`` — ADR-0012
    decision 2 deferred exactly this question to the service boundary, and the
    answer keeps ``SearchMode`` the three-way literal ADR-0007 settled. It
    reorders what ``Retriever.search`` returned; ``retrieval/search.py`` is
    untouched.

    When ``rerank`` is requested on a server started without one, this raises
    :class:`~groundkit.errors.RerankerNotConfiguredError` rather than returning
    an unreranked list. That distinction is the whole reason the error type
    exists: a reranker that quietly returns what it was given is
    indistinguishable from one that worked, so a misconfigured deployment would
    report results a caller believes were reranked.

    Candidate depth is pinned to ``MAX_TOP_K`` and is not a request field, for
    ADR-0012 decision 1a's reason — depth decides whether the ranking could
    move at all, so a per-request knob makes two responses silently
    incomparable in the dimension a reader is least likely to check.
    """
    # Bound only when the request actually asked for it, so the `is None`
    # check below means "not reranking" rather than "no reranker configured" —
    # the two are different states and conflating them is how a server with a
    # reranker starts reranking requests that did not ask.
    reranker: RerankerProtocol | None = None
    if request.rerank:
        if ctx.reranker is None:
            raise RerankerNotConfiguredError(
                "rerank was requested but this server was started without a reranker. "
                "Restart with --rerank (and install the 'rerank' extra). Returning an "
                "unreranked result instead would be indistinguishable from a working one."
            )
        reranker = ctx.reranker

    top_k = request.top_k if request.top_k is not None else ctx.default_top_k
    fetch_k = MAX_TOP_K if request.rerank else top_k

    # Held across the retriever acquisition too, not just the search: a
    # rebuild is itself the O(corpus) BM25 reconstruction (ADR-0002), so
    # bounding the search alone would leave the more expensive half unbounded.
    async with ctx.scan_limiter, ctx.registry.acquire(request.collection) as runtime:
        acquired = await runtime.acquire()
        response = await acquired.retriever.search(request.query, top_k=fetch_k, mode=request.mode)

    if reranker is None:
        return response.model_copy(update={"metadata": {**response.metadata, "reranked": False}})

    reranked = await reranker.rerank(request.query, list(response.results), top_k=top_k)
    metadata = {
        **response.metadata,
        "reranked": True,
        "rerank_model": getattr(reranker, "model_name", type(reranker).__name__),
        # Recorded because a reranked hybrid result is NOT hybrid@top_k
        # reordered: RRF is not depth-invariant, so the set fused at this depth
        # differs in membership from the one at top_k (ADR-0012 decision 1).
        # Without this a client cannot tell which experiment it received.
        "rerank_input_depth": fetch_k,
    }
    return response.model_copy(
        update={"results": reranked, "total_results": len(reranked), "metadata": metadata}
    )


async def handle_fetch_chunk(ctx: ServiceContext, request: FetchChunkRequest) -> ChunkFetchResponse:
    """Return one chunk with its citation re-verified against the source.

    Verification is a byte comparison against text read back from disk, which
    makes it strictly stronger than ``resolve_citation``'s length-only drift
    check — it catches the same-length in-place edit ``KNOWN_LIMITATIONS.md``
    records as undetectable — but only at this one boundary. ``search`` does
    not verify per hit, because that is one file read per result per query;
    this is the step a client performs on a hit it intends to quote.
    """
    async with ctx.registry.acquire(request.collection) as runtime:
        chunk = await runtime.get_chunk(request.chunk_id)
        # Keyed, and conditional: this used to read every row of ``documents``
        # unconditionally — before even establishing the chunk exists — to
        # look up the one document that chunk belongs to (GK-019). A request
        # naming an unknown chunk now performs no document read at all.
        record = None if chunk is None else await runtime.get_document_record(chunk.document_id)

    if chunk is None:
        raise ConfigurationError(
            f"no chunk {request.chunk_id!r} in collection {request.collection!r}"
        )

    if record is None:
        # The dangling-document case Retriever.search already fails closed on.
        raise RetrievalError(
            f"chunk {chunk.chunk_id!r} belongs to document {chunk.document_id!r}, "
            "which has no stored source — the index is inconsistent"
        )

    # A ``DocumentRecord`` rather than ``get_document_sources``' bare string
    # (ADR-0016).
    # A bare source string would leave this Citation at its ``("text", None)``
    # field defaults no matter what the document was ingested as, and the
    # consequences are not cosmetic: ``search`` would report a chunk's citation
    # as ``extracted`` while ``fetch_chunk`` reported the same chunk as
    # ``text`` — two read-only tools disagreeing about one stored fact — and
    # ``resolve_citation`` would take the plain read-and-slice branch instead of
    # refusing. For an extracted source that can "verify" against raw bytes the
    # offsets were never measured against; for a snapshot source it hands a URL
    # to ``ensure_within_base``, which is exactly the relative-path confusion
    # ADR-0016 decision 4 closes by classifying before resolving.
    citation = Citation(
        document_id=chunk.document_id,
        chunk_id=chunk.chunk_id,
        source=record.source,
        source_class=record.source_class,
        extractor=record.extractor,
        start_offset=chunk.start_offset,
        end_offset=chunk.end_offset,
    )

    # The per-collection snapshot containment root (ADR-0016 decision 4,
    # spec §10.1) — computed here rather than recomputed by resolve_citation
    # itself, mirroring how ctx.base_dir is already resolved once by the
    # caller. Harmless to compute unconditionally even for a non-snapshot
    # citation: resolve_citation only ever reads it on the "snapshot" branch.
    snapshot_dir = snapshots.snapshot_dir_for(ctx.index_dir, request.collection)

    verification: VerificationVerdict
    content: str | None = None
    detail: str | None = None
    try:
        resolved = await resolve_citation(citation, ctx.base_dir, snapshot_dir=snapshot_dir)
    except RetrievalError as exc:
        # resolve_citation sets `verdict` explicitly at every one of its raise
        # sites (ADR-0016 decision 6), so this reads a typed attribute instead
        # of pattern-matching the message text the way an earlier version did.
        # The `is None` fallback is defensive, not load-bearing: every path
        # inside resolve_citation now sets a verdict, so it should never
        # trigger. It exists so a future RetrievalError raised there without
        # one fails toward the more conservative verdict — `unresolvable`
        # never claims a definite "the source changed", `drifted` would.
        verification = exc.verdict if exc.verdict is not None else "unresolvable"
        detail = str(exc)
    else:
        if resolved == chunk.content:
            verification = "verified"
            content = resolved
        else:
            verification = "drifted"
            detail = "source text at the cited offsets no longer matches the indexed chunk"

    return ChunkFetchResponse(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        chunk_index=chunk.chunk_index,
        citation=citation,
        verification=verification,
        content=content,
        detail=detail,
    )


async def handle_list_collections(
    ctx: ServiceContext, request: ListCollectionsRequest
) -> list[str]:
    """Return the names of collections in the index directory.

    Enumeration writes nothing. ``SQLiteMetadataStore.open`` applies the schema
    and stamps PRAGMAs, so confirming a candidate by *opening* it would make
    listing a write — which is why each candidate is checked with
    :func:`~groundkit.index.metadata.is_groundkit_store` instead, a read-only
    (``mode=ro``) identity probe.

    That check is not cosmetic. Advertising every ``*.sqlite3`` meant an
    unrelated database a user happened to leave here was reported as a
    collection, and a caller who then asked ``index_status`` for it had
    groundkit's four tables written into it. Filtering here and refusing in
    ``open`` close the two halves of that.

    Each candidate is checked for containment before its name is reported, so a
    planted symlink (``evil.sqlite3 -> /etc/passwd``) resolves outside the base
    and is skipped rather than advertised. A non-conforming entry is skipped
    with a debug log, not an error: a hand-placed file in the index directory
    is not a groundkit collection, and refusing to list anything because of one
    would be a denial of the whole operation.

    The whole scan runs in **one** :func:`asyncio.to_thread` hop. Every step of
    it blocks: ``glob`` and ``is_within_base`` hit the filesystem, and
    ``is_groundkit_store`` opens a real ``sqlite3`` connection and reads a
    pragma per candidate. Run inline, that stalled the single event loop for N
    sequential sqlite opens -- in the tool an MCP or REST client typically
    calls *first*, so the stall lands on every other in-flight request. One hop
    rather than one per candidate because the thread-switch cost would
    otherwise dominate a scan whose individual steps are each tiny.
    """
    del request  # the operation takes no parameters; the signature is uniform

    def _enumerate() -> list[str]:
        if not ctx.index_dir.is_dir():
            return []

        names: list[str] = []
        for path in sorted(ctx.index_dir.glob("*.sqlite3")):
            if not is_within_base(str(path), ctx.index_dir):
                logger.debug("Skipping %s: resolves outside the index directory", path.name)
                continue
            if not is_groundkit_store(path):
                logger.debug("Skipping %s: not a groundkit collection", path.name)
                continue
            names.append(path.stem)
        return names

    return await asyncio.to_thread(_enumerate)


async def handle_index_status(
    ctx: ServiceContext, request: IndexStatusRequest
) -> IndexStatusResponse:
    """Report one collection's size and embedding identity.

    Counts and identity only. No document sources, no chunk content, no
    directory paths: SPEC.md §7 records that SQLite here is content-bearing
    data, so enumerating sources would disclose corpus layout to an
    unauthenticated reader.

    Both counts are aggregates. Each was once a ``len()`` over a materialized
    table — the chunk half pulling every chunk's full text, the document half
    every source string — to produce two integers for the cheapest-*looking*
    call on this surface. Bounded by :data:`MAX_CONCURRENT_CORPUS_SCANS` even
    so: an aggregate over every row is still a full table scan, and this
    handler is reachable unauthenticated.

    It also reports the runtime's rebuild counters (ADR-0026). This is the one
    operation on the surface that already holds the runtime and already
    reports the staleness marker, so it is where the marker's *cost* belongs
    too. The read is free — four attributes off the object already checked
    out, no store round-trip — and it is deliberately **not** an acquire: this
    handler never builds a retriever, so reading the counters here cannot move
    them.
    """
    async with ctx.scan_limiter, ctx.registry.acquire(request.collection) as runtime:
        document_count = await runtime.document_count()
        chunk_count = await runtime.chunk_count()
        manifest = await runtime.get_manifest()
        generation = await runtime.get_generation()
        stats = runtime.rebuild_stats()

    return IndexStatusResponse(
        collection=request.collection,
        document_count=document_count,
        chunk_count=chunk_count,
        embedding=manifest,
        # Read from the same manifest the refusal in Retriever.search reads, so
        # a client cannot be told "available" by one path and refused by
        # another (ADR-0008).
        dense_search_available=manifest is not None,
        generation=generation,
        cache_enabled=generation is not None,
        schema_version=SCHEMA_VERSION,
        retriever_acquires=stats.acquires,
        retriever_rebuilds=stats.rebuilds,
        rebuild_seconds_total=stats.rebuild_seconds_total,
        last_rebuild_seconds=stats.last_rebuild_seconds,
    )


TOOLS: Final[tuple[ToolSpec, ...]] = (
    ToolSpec(
        name="search",
        summary="Hybrid retrieval over a collection, returning citation-bearing results.",
        request_model=SearchRequest,
        handler=handle_search,
        side_effect="read_only",
        rest_path="/v1/search",
        rest_method="POST",
    ),
    ToolSpec(
        name="fetch_chunk",
        summary="Fetch one chunk and re-verify its citation against the source file.",
        request_model=FetchChunkRequest,
        handler=handle_fetch_chunk,
        side_effect="read_only",
        rest_path="/v1/collections/{collection}/chunks/{chunk_id}",
        rest_method="GET",
    ),
    ToolSpec(
        name="list_collections",
        summary="List the collections available in this server's index directory.",
        request_model=ListCollectionsRequest,
        handler=handle_list_collections,
        side_effect="read_only",
        rest_path="/v1/collections",
        rest_method="GET",
    ),
    ToolSpec(
        name="index_status",
        summary="Report a collection's document and chunk counts and embedding identity.",
        request_model=IndexStatusRequest,
        handler=handle_index_status,
        side_effect="read_only",
        rest_path="/v1/collections/{collection}",
        rest_method="GET",
    ),
)

#: The four operations SPEC.md §1.2 names, as a set, for the parity tests each
#: transport runs against its own registrations.
TOOL_NAMES: Final[frozenset[str]] = frozenset(spec.name for spec in TOOLS)

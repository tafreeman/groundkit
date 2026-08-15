"""One exception-to-transport mapping, used identically by REST and MCP (ADR-0014 decision 9).

**No new exception type is defined here, and none may be.** ``errors.py``'s
hierarchy is sufficient; this module only decides how each existing type is
rendered at an egress boundary. That constraint is what keeps the two transports
honest: they cannot disagree about what a given failure *means*, because neither
one classifies anything — both call :func:`map_exception` and render what it
returns.

Three properties are load-bearing rather than stylistic.

**Matching is subclass-first, and the order is data rather than control flow.**
:data:`_MAPPINGS` is an ordered tuple walked front to back, with every subclass
ahead of its base. A mis-ordered chain is a real bug class — put ``RetrievalError``
before ``RerankerNotConfiguredError`` and a missing optional extra starts
reporting as an index inconsistency — so ``test_service_errors.py`` asserts the
ordering property directly against the tuple rather than trusting that the
entries were written in a sensible sequence.

**Only an allow-list of types has its own message returned to a caller.** Every
other type renders a fixed detail. The distinction is not severity, it is
provenance: a :class:`~groundkit.errors.ConfigurationError` names a value the
caller supplied, while an :class:`~groundkit.errors.EmbeddingError` may carry a
sanitized provider URL, an endpoint, or text a provider returned. Credential
safety here is by construction rather than by scrubbing at the edge — the class
that could carry a secret never has its message read at all.

**The mapper never reads ``__cause__``, ``__context__``, ``repr(exc)``, or a
traceback.** That is the ADR-0001 hazard 6 chain-severing discipline applied at
the egress boundary: a scrubbed message with an unscrubbed ``__cause__`` chained
behind it is the exact shape that leaked credentials in the ported code this repo
replaced. The full, unscrubbed exception is logged server-side against the
response's request id instead, so nothing is lost to an operator.

## The not-found question, settled

``CollectionRegistry`` raises :class:`~groundkit.errors.ConfigurationError` for a
collection that does not exist, while ADR-0014 decision 3 calls that a *not-found*
and decision 9 maps ``ConfigurationError`` to *bad request*. Read naively the two
conflict. They do not, because decision 3 does not describe an exception path at
all: the not-found is produced by :func:`check_collection` as a **precondition at
the boundary, before any handler runs**, so the registry's own refusal is
defense-in-depth that a well-formed request never reaches. No message is matched
and no type is distinguished — the two answers come from two different places in
the call, which is why this is robust rather than a string comparison waiting to
rot.

One residual asymmetry is recorded rather than left to omission: a **missing
``chunk_id``** surfaces as ``400``, not ``404``. ``handle_fetch_chunk`` raises
``ConfigurationError`` for it, and separating that from an invalid collection
name would require either a new exception type (forbidden by ADR-0014 decision 9)
or message matching (forbidden for being fragile). ``400`` is also the reading
decision 9's own rationale gives — "every reachable one in Phase 4 is caused by a
request field", and ``chunk_id`` is a request field. ADR-0014 carries a matching
amendment; this docstring and that amendment are the same decision written twice
on purpose, because a reader hitting a ``400`` on a ``GET`` of a missing chunk
will look here first.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from groundkit.errors import (
    ConfigurationError,
    EmbeddingError,
    GroundkitError,
    IndexIdentityError,
    ProviderNotConfiguredError,
    RerankerNotConfiguredError,
    RetrievalError,
    StorageError,
)
from groundkit.index.metadata import validate_collection_name

#: JSON-RPC's reserved implementation-defined server range is -32000..-32099.
#: These sit inside it rather than reusing -32603 for everything, so an MCP
#: client can branch on the kind of failure without parsing prose.
_RPC_INVALID_PARAMS: Final[int] = -32602
_RPC_INTERNAL: Final[int] = -32603
_RPC_NOT_FOUND: Final[int] = -32002
_RPC_CONFLICT: Final[int] = -32003
_RPC_NOT_IMPLEMENTED: Final[int] = -32004
_RPC_UPSTREAM: Final[int] = -32005

#: Returned in place of a real message for every type outside the allow-list.
#: Deliberately says nothing about what went wrong: the honest detail is in the
#: server log, keyed by the request id the response carries.
GENERIC_DETAIL: Final[str] = (
    "the server could not complete this request; see the server log for this request id"
)


@dataclass(frozen=True, slots=True)
class ErrorRendering:
    """How one failure is rendered on both transports.

    Attributes:
        status_code: HTTP status for the REST surface.
        rpc_code: JSON-RPC error code carried in the MCP error payload.
        detail: Text safe to return to a caller — either the exception's own
            message (allow-listed types only) or :data:`GENERIC_DETAIL`.
        kind: Stable machine-readable label. A client branching on this is not
            parsing prose, which is the failure mode ``detail`` invites.
    """

    status_code: int
    rpc_code: int
    detail: str
    kind: str


#: ``(type, status, rpc code, kind, expose the exception's own message)``.
#:
#: Ordered subclass-first and walked front to back. Every entry's position is a
#: decision:
#:
#: * ``RerankerNotConfiguredError`` precedes ``RetrievalError`` — a missing
#:   optional extra is not an index inconsistency.
#: * ``ProviderNotConfiguredError`` precedes ``EmbeddingError`` — an
#:   unconfigured provider is an operator fault, not an upstream failure.
#: * ``IndexIdentityError`` precedes ``StorageError`` — an identity mismatch is
#:   actionable and names no path.
#: * ``GroundkitError`` is last, as the catch-all root.
_MAPPINGS: Final[tuple[tuple[type[GroundkitError], int, int, str, bool], ...]] = (
    # 501, not 400 or 503: the request is well-formed and would be servable on
    # an install carrying the extra, and a missing optional dependency is not
    # transient, so an "unavailable" would invite a retry that can never work.
    (RerankerNotConfiguredError, 501, _RPC_NOT_IMPLEMENTED, "reranker_not_configured", True),
    # The identity triple is already disclosed by `index_status`, so echoing it
    # here leaks nothing new and tells an operator exactly which collection to
    # re-ingest.
    (IndexIdentityError, 409, _RPC_CONFLICT, "index_identity_mismatch", True),
    # Message withheld: an unconfigured provider's message names the env var and
    # the endpoint it expected.
    (ProviderNotConfiguredError, 503, _RPC_UPSTREAM, "provider_not_configured", False),
    # Message withheld, and this is the entry the credential rule exists for --
    # an embedding failure may carry a sanitized URL or provider response text.
    # ADR-0014 decision 9 routes the outbound-endpoint rejection here on purpose:
    # `_raise_embedding_error` converts it inside the embedding call path, so an
    # operator-fault ConfigurationError never reaches the 400 entry below.
    (EmbeddingError, 502, _RPC_UPSTREAM, "embedding_backend_error", False),
    # 409, not 400. Correct ONLY because SearchRequest bounds `query` and
    # `top_k`, so the caller-error RetrievalErrors are rejected by schema
    # validation and the only one that can reach here is a server-side index
    # inconsistency. `test_service_errors.py` pins that precondition; if it ever
    # stops holding, this mapping is wrong and that test is what says so.
    (RetrievalError, 409, _RPC_CONFLICT, "index_inconsistent", True),
    # 400: every reachable one in Phase 4 is caused by a request field -- an
    # invalid collection name, a dense mode against a collection with no
    # manifest (ADR-0008), or an unknown chunk id. Startup config faults exit
    # non-zero before the socket binds and never reach a response.
    (ConfigurationError, 400, _RPC_INVALID_PARAMS, "invalid_request", True),
    # Message withheld: StorageError messages carry absolute database paths.
    (StorageError, 500, _RPC_INTERNAL, "storage_error", False),
    (GroundkitError, 500, _RPC_INTERNAL, "internal_error", False),
)


def map_exception(exc: GroundkitError) -> ErrorRendering:
    """Render ``exc`` for a caller, subclass-first.

    Reads ``exc``'s own message only for allow-listed types, and reads nothing
    else about it — not its cause, its context, its ``repr``, or its traceback.

    Args:
        exc: The exception raised by a handler.

    Returns:
        The rendering both transports use.
    """
    for exc_type, status, rpc_code, kind, expose in _MAPPINGS:
        if isinstance(exc, exc_type):
            return ErrorRendering(
                status_code=status,
                rpc_code=rpc_code,
                detail=str(exc) if expose else GENERIC_DETAIL,
                kind=kind,
            )
    # Unreachable while GroundkitError anchors the tuple, and deliberately not
    # written as an assert: a stripped-optimized run must still fail closed
    # rather than fall off the end returning None.
    return ErrorRendering(500, _RPC_INTERNAL, GENERIC_DETAIL, "internal_error")


def unexpected_error_rendering() -> ErrorRendering:
    """Render a non-``GroundkitError`` escape.

    A bare ``Exception`` reaching a transport is a bug, and its message is
    whatever an arbitrary library chose to write — which is precisely why it is
    never returned. Both transports call this so an unexpected failure is
    rendered by the same rule as an expected one.
    """
    return ErrorRendering(500, _RPC_INTERNAL, GENERIC_DETAIL, "internal_error")


def check_collection(index_dir: Path, collection: str) -> ErrorRendering | None:
    """Validate a collection name, then confirm the collection exists.

    ``None`` means the request may proceed. This is ADR-0014 decision 3's
    precondition, and the **order of the two checks is the security property**:
    name validation runs first, so a traversal attempt is reported as a
    validation failure rather than as a not-found that would have confirmed
    whether the traversed-to path exists. A probe learns nothing either way.

    It also keeps ``SQLiteMetadataStore.open`` off non-existent collections
    entirely. That method *creates* the file when absent — correct for
    ``grk ingest``, and a disk-fill primitive at an unauthenticated read
    boundary, where ``index_status?collection=<anything>`` would otherwise write
    to disk on every request.

    Args:
        index_dir: Directory holding the collections.
        collection: Caller-supplied collection name.

    Returns:
        ``None`` when the request may proceed, otherwise the rendering to
        return without invoking any handler.
    """
    try:
        validate_collection_name(collection)
    except ConfigurationError as exc:
        return map_exception(exc)

    if not (index_dir / f"{collection}.sqlite3").is_file():
        return ErrorRendering(
            status_code=404,
            rpc_code=_RPC_NOT_FOUND,
            # Echoes the name back because it passed validation, so it is a
            # known-safe string, and a caller with a typo needs to see it. The
            # index directory is NOT named: that is server layout.
            detail=(
                f"collection {collection!r} does not exist. Collections are created by "
                "`grk ingest`, never by reading one."
            ),
            kind="collection_not_found",
        )
    return None

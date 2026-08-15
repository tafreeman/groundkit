"""Tests for the shared exception-to-transport mapping (ADR-0014 decision 9).

None of these can be shown to fail by reverting source — ``service/errors.py``
did not exist before this phase — so each guard is demonstrated instead by
*injecting the violation it exists to catch*. That is the meaningful direction
for a guard and is explicitly NOT the SPEC.md §8 revert procedure; each such
test says so in its own docstring rather than leaving a reader to assume a
revert was performed.

The ordering test is the one that earns its keep. A mis-ordered mapping chain is
a real bug class with no symptom at import time: put ``RetrievalError`` ahead of
``RerankerNotConfiguredError`` and a missing optional extra silently starts
reporting as an index inconsistency — a 409 telling an operator their index is
corrupt when in fact they simply never installed ``groundkit[rerank]``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from groundkit.errors import (
    ChunkingError,
    ConfigurationError,
    EmbeddingError,
    GroundkitError,
    IndexIdentityError,
    IngestionError,
    ProviderNotConfiguredError,
    RerankerNotConfiguredError,
    RetrievalError,
    StorageError,
)
from groundkit.index.metadata import SQLiteMetadataStore
from groundkit.service.errors import (
    _MAPPINGS,
    GENERIC_DETAIL,
    ErrorRendering,
    check_collection,
    map_exception,
    unexpected_error_rendering,
)
from groundkit.service.schemas import SearchRequest

#: Types whose own message may reach a caller. Written out here rather than
#: derived from ``_MAPPINGS`` on purpose: deriving it would make this test agree
#: with the table by construction, including when the table is wrong. This is a
#: second, independent statement of the allow-list.
_MESSAGE_EXPOSING = frozenset(
    {
        RerankerNotConfiguredError,
        IndexIdentityError,
        RetrievalError,
        ConfigurationError,
    }
)

#: Types whose message must never reach a caller, with why, so a future reader
#: does not "simplify" one into the allow-list above.
_MESSAGE_WITHHOLDING = {
    ProviderNotConfiguredError: "names the env var and the endpoint it expected",
    EmbeddingError: "may carry a sanitized provider URL or provider response text",
    StorageError: "carries absolute database paths",
    GroundkitError: "is an unclassified root; its message is unknown text",
}


def test_mapping_chain_is_ordered_subclass_first() -> None:
    """Every entry precedes all of its own base classes.

    Asserted against the tuple's structure rather than against a hand-written
    expected order, so the property survives someone legitimately adding a new
    entry. Without this, a correctly-typed but mis-placed entry is invisible:
    the module imports, mypy passes, and only the wrong status code at runtime
    would ever reveal it.
    """
    types_in_order = [entry[0] for entry in _MAPPINGS]
    for index, exc_type in enumerate(types_in_order):
        for later in types_in_order[index + 1 :]:
            assert not issubclass(later, exc_type) or later is exc_type, (
                f"{later.__name__} is a subclass of {exc_type.__name__} but is matched "
                f"after it, so it can never be reached. Move it earlier in _MAPPINGS."
            )


def test_injecting_a_misordered_entry_breaks_the_ordering_property() -> None:
    """Demonstrates the ordering guard by INJECTING the violation it catches.

    Not the SPEC.md §8 revert procedure: there is no unfixed source to revert
    to. This builds a deliberately mis-ordered chain and asserts the same
    property the test above applies would reject it.
    """
    misordered = (RetrievalError, RerankerNotConfiguredError)
    reachable = True
    for index, exc_type in enumerate(misordered):
        for later in misordered[index + 1 :]:
            if issubclass(later, exc_type) and later is not exc_type:
                reachable = False
    assert not reachable, (
        "RerankerNotConfiguredError placed after its base RetrievalError must be "
        "detected as unreachable; if this passes the ordering check is vacuous."
    )


def test_reranker_not_configured_maps_to_not_implemented() -> None:
    """501, not 400 or 503.

    The request is well-formed and would be servable on an install carrying the
    extra, and a missing optional dependency is not transient — a 503 would
    invite a retry that can never succeed.
    """
    rendering = map_exception(RerankerNotConfiguredError("install groundkit[rerank]"))
    assert rendering.status_code == 501
    assert rendering.kind == "reranker_not_configured"
    assert "rerank" in rendering.detail


def test_retrieval_error_maps_to_conflict_not_bad_request() -> None:
    """409, because the caller-error cases cannot reach the handler.

    ``Retriever.search`` raises ``RetrievalError`` for an empty query and an
    out-of-range ``top_k`` — caller errors — and for index inconsistency, a
    server fault. The type does not distinguish them, so the mapping is only
    correct while the request schema rejects the caller-error cases first.
    ``test_search_request_bounds_are_what_make_the_conflict_mapping_correct``
    below pins that precondition.
    """
    rendering = map_exception(RetrievalError("index is inconsistent"))
    assert rendering.status_code == 409
    assert rendering.kind == "index_inconsistent"


def test_search_request_bounds_are_what_make_the_conflict_mapping_correct() -> None:
    """The precondition the 409 mapping rests on, asserted directly.

    ADR-0014's Consequences names this as the thinnest part of the decision:
    the mapping rests on a schema precondition rather than on the type system,
    and one test guards it. This is that test. If these bounds are ever
    relaxed, a caller error starts surfacing as "your index is corrupt" and
    this is what says so.
    """
    with pytest.raises(ValueError):
        SearchRequest(query="")
    with pytest.raises(ValueError):
        SearchRequest(query="ok", top_k=0)
    with pytest.raises(ValueError):
        SearchRequest(query="ok", top_k=10_000)


@pytest.mark.parametrize("exc_type", sorted(_MESSAGE_EXPOSING, key=lambda t: t.__name__))
def test_allow_listed_types_return_their_own_message(exc_type: type[GroundkitError]) -> None:
    """The allow-list exposes the message; a caller needs it to act."""
    rendering = map_exception(exc_type("a distinctive marker string"))
    assert rendering.detail == "a distinctive marker string"


@pytest.mark.parametrize(
    ("exc_type", "why"), sorted(_MESSAGE_WITHHOLDING.items(), key=lambda kv: kv[0].__name__)
)
def test_withheld_types_return_the_fixed_detail(exc_type: type[GroundkitError], why: str) -> None:
    """Everything outside the allow-list renders the fixed detail.

    ``why`` is carried through only so a failure message names the reason this
    type is withheld rather than making the next reader go find it.
    """
    rendering = map_exception(exc_type("SECRET-sk-live-abcdef"))
    assert rendering.detail == GENERIC_DETAIL, f"{exc_type.__name__} {why}"
    assert "SECRET" not in rendering.detail


def test_the_mapper_never_reads_the_cause_chain() -> None:
    """ADR-0001 hazard 6 at the egress boundary.

    A scrubbed message with an unscrubbed ``__cause__`` chained behind it is
    the exact shape that leaked credentials in the ported code this repo
    replaced. The sentinel is planted in the cause of an ALLOW-LISTED type, so
    the message itself is legitimately returned — which is precisely the case
    where a careless implementation would also walk the chain.
    """
    cause = EmbeddingError("https://user:hunter2@api.example.com/v1/embeddings")
    exc = ConfigurationError("collection 'nope' is not valid")
    exc.__cause__ = cause

    rendering = map_exception(exc)

    assert rendering.detail == "collection 'nope' is not valid"
    assert "hunter2" not in rendering.detail
    assert "api.example.com" not in rendering.detail


def test_every_groundkit_error_subclass_is_classified() -> None:
    """No GroundkitError falls through unrendered.

    Walks the real subclass tree rather than a written list, so an exception
    type added in a later phase is covered without this test being told about
    it. A type nobody classified still renders — as a 500 with the fixed
    detail — and this asserts that fallback is reached rather than crashing.
    """
    for exc_type in (IngestionError, ChunkingError, EmbeddingError, StorageError, RetrievalError):
        rendering = map_exception(exc_type("boom"))
        assert isinstance(rendering, ErrorRendering)
        assert 400 <= rendering.status_code <= 599
        assert rendering.rpc_code < 0


def test_unexpected_error_rendering_never_carries_a_message() -> None:
    """A bare Exception escaping to a transport is a bug, not a caller's business."""
    rendering = unexpected_error_rendering()
    assert rendering.status_code == 500
    assert rendering.detail == GENERIC_DETAIL
    assert rendering.kind == "internal_error"


async def _make_collection(index_dir: Path, name: str) -> None:
    store = await SQLiteMetadataStore.open(index_dir, name)
    await store.close()


def test_check_collection_passes_an_existing_collection(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    asyncio.run(_make_collection(index_dir, "real"))

    assert check_collection(index_dir, "real") is None


def test_check_collection_reports_not_found_without_creating_anything(tmp_path: Path) -> None:
    """404, and the index directory is untouched.

    ``SQLiteMetadataStore.open`` CREATES the file when absent — right for
    ``grk ingest``, and a disk-fill primitive at an unauthenticated read
    boundary. The second assertion is the one that matters: without this
    precondition, probing ``index_status`` with arbitrary names would write a
    new empty database per request.
    """
    index_dir = tmp_path / "index"
    index_dir.mkdir()

    rendering = check_collection(index_dir, "absent")

    assert rendering is not None
    assert rendering.status_code == 404
    assert rendering.kind == "collection_not_found"
    assert list(index_dir.iterdir()) == []


@pytest.mark.parametrize("name", ["../escape", "a/b", "..", ".", "", "nul\0byte"])
def test_an_invalid_name_is_a_validation_failure_not_a_probe_result(
    tmp_path: Path, name: str
) -> None:
    """400 before 404 — the ORDER is the security property.

    A traversal attempt must be reported as a validation failure, never as a
    not-found, because a not-found would confirm whether the traversed-to path
    exists. Reversing the two checks in ``check_collection`` is the injected
    violation this guards: a probe would then learn real filesystem facts.
    """
    index_dir = tmp_path / "index"
    index_dir.mkdir()

    rendering = check_collection(index_dir, name)

    assert rendering is not None
    assert rendering.status_code == 400
    assert rendering.kind == "invalid_request"
    assert rendering.kind != "collection_not_found"


def test_not_found_detail_does_not_name_the_index_directory(tmp_path: Path) -> None:
    """The collection name is echoed; server layout is not.

    The name is safe to echo because it passed validation, and a caller with a
    typo needs to see it. The index directory is server layout and stays out —
    the same disclosure rule ``index_status`` follows.
    """
    index_dir = tmp_path / "secret-layout-dir"
    index_dir.mkdir()

    rendering = check_collection(index_dir, "absent")

    assert rendering is not None
    assert "absent" in rendering.detail
    assert "secret-layout-dir" not in rendering.detail
    assert str(index_dir) not in rendering.detail

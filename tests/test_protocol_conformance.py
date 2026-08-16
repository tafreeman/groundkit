"""Signature-parity conformance tests for component Protocol seams.

SPEC.md §5.1 claims each component Protocol seam is closed structurally by a
conformance test, and ``groundkit.ingestion.protocols``'s module docstring
claims those tests assert "exact signature parity". Reality (before this
file): every conformance test in the suite is only
``isinstance(impl, SomeProtocol)``. For a ``@runtime_checkable`` Protocol,
``isinstance`` only checks that members of the same NAME exist on the
implementation — it does not check parameter names, arity,
keyword-only-ness, defaults, or types. A rename like ADR-0001 hazard 4's
``query`` -> ``_query`` would pass every existing ``isinstance`` check
untouched.

:func:`assert_signature_parity` closes that gap: for every PUBLIC member a
Protocol declares in its own class body (not inherited from ``Protocol`` or
``object``), it asserts the implementation has a same-shaped member —
matching property-vs-method kind, sync-vs-async, parameter names, kinds,
order, and defaults (``self``/``cls`` ignored on both sides), plus resolved
type hints via :func:`typing.get_type_hints` whenever both sides can resolve
them. Extra members an implementation adds beyond the Protocol are never a
failure — implementations are always free to grow. That also makes this
helper robust to a concurrent change that adds a new method to
``SQLiteMetadataStore`` without (yet) adding it to ``MetadataStoreProtocol``:
only members the Protocol itself declares are ever checked.

``VectorStoreProtocol`` (``groundkit/index/protocols.py``) and
``RerankerProtocol`` (``groundkit/retrieval/protocols.py``) have no
implementations yet — Phase 3 stubs — so they are intentionally not
exercised here.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, get_type_hints, runtime_checkable

import pytest

from groundkit.extraction import ExtractorProtocol, HtmlExtractor, PdfExtractor
from groundkit.index.dense import InMemoryVectorStore, LanceDBVectorStore
from groundkit.index.metadata import SQLiteMetadataStore
from groundkit.index.protocols import MetadataStoreProtocol, VectorStoreProtocol
from groundkit.ingestion.chunking import RecursiveChunker
from groundkit.ingestion.loaders import FileLoader
from groundkit.ingestion.protocols import ChunkerProtocol, LoaderProtocol
from groundkit.providers.embeddings import (
    InMemoryEmbedder,
    OllamaEmbedder,
    OpenAICompatibleEmbedder,
)
from groundkit.providers.llm import (
    OllamaChat,
    OpenAICompatChat,
    RedactingChat,
    ScriptedChatProvider,
)
from groundkit.providers.protocols import ChatProtocol, EmbeddingProtocol
from groundkit.retrieval.protocols import RerankerProtocol
from groundkit.retrieval.rerank import CrossEncoderReranker

# ── The signature-parity helper ────────────────────────────────────────────


@dataclass(frozen=True)
class _ParamSpec:
    """Normalized, comparable shape of one parameter (``self``/``cls`` already stripped)."""

    name: str
    kind: str
    default_repr: str
    annotation: str


def _raw_annotation_text(annotation: object) -> str:
    """Render a signature annotation as comparable text.

    Every module compared by this helper uses ``from __future__ import
    annotations`` (PEP 563), so ``annotation`` is normally the literal
    source text as a string (e.g. ``"list[Document]"``), not a resolved
    type object. Comparing that text directly catches a parameter or return
    type rename without needing to resolve any forward reference — which
    matters because the Protocol files import their contract types
    (``Document``, ``Chunk``, ``RetrievalResult``) only under
    ``TYPE_CHECKING``, so those names do not exist in the Protocol module's
    runtime globals at all.
    """
    if annotation is inspect.Signature.empty:
        return "<empty>"
    return str(annotation)


def _param_specs(signature: inspect.Signature) -> list[_ParamSpec]:
    """Return every parameter of *signature* as a comparable spec, ``self``/``cls`` stripped."""
    params = list(signature.parameters.values())
    if params and params[0].name in ("self", "cls"):
        params = params[1:]
    return [
        _ParamSpec(
            name=p.name,
            kind=p.kind.name,
            default_repr=(
                "<no default>" if p.default is inspect.Parameter.empty else repr(p.default)
            ),
            annotation=_raw_annotation_text(p.annotation),
        )
        for p in params
    ]


def _callable_target(member: property | Callable[..., object]) -> Callable[..., object]:
    """Return the underlying callable for a plain method or a property's getter.

    Raises:
        AssertionError: If *member* is a write-only property (no getter) —
            unreachable for every Protocol in this codebase today, but a
            case worth failing loudly on rather than silently skipping.
    """
    if isinstance(member, property):
        if member.fget is None:
            raise AssertionError(f"property {member!r} has no getter")
        return member.fget
    return member


def _resolved_hints(target: Callable[..., object]) -> dict[str, Any] | None:
    """Best-effort :func:`typing.get_type_hints` for *target*.

    Returns ``None`` when a forward reference cannot be resolved in the
    defining module's namespace — expected for Protocol methods whose
    annotations name a ``TYPE_CHECKING``-only import, not a defect. Callers
    only compare resolved hints when both sides of a comparison resolve;
    otherwise they rely on :func:`_raw_annotation_text` instead, which never
    needs resolution.
    """
    try:
        return get_type_hints(target, include_extras=True)
    except NameError:
        return None


def _assert_member_parity(
    protocol: type,
    implementation: type,
    name: str,
    proto_raw: property | Callable[..., object],
) -> None:
    """Assert *implementation* has a same-shaped member *name* as *protocol* declares.

    Raises:
        AssertionError: On a missing member, a property/method kind
            mismatch, a sync/async mismatch, a parameter-shape mismatch, a
            return-annotation mismatch, or (when both sides resolve) a
            resolved-type-hints mismatch. Every message names the protocol,
            the implementation, the member, and the concrete diff.
    """
    impl_raw = inspect.getattr_static(implementation, name, None)
    if impl_raw is None:
        raise AssertionError(
            f"{implementation.__qualname__} is missing member {name!r} "
            f"declared by {protocol.__qualname__}"
        )

    proto_is_property = isinstance(proto_raw, property)
    impl_is_property = isinstance(impl_raw, property)
    if proto_is_property != impl_is_property:
        raise AssertionError(
            f"{protocol.__qualname__}.{name} is a "
            f"{'property' if proto_is_property else 'method'}, but "
            f"{implementation.__qualname__}.{name} is a "
            f"{'property' if impl_is_property else 'method'}"
        )

    proto_target = _callable_target(proto_raw)
    impl_target = _callable_target(impl_raw)

    proto_async = inspect.iscoroutinefunction(proto_target)
    impl_async = inspect.iscoroutinefunction(impl_target)
    if proto_async != impl_async:
        raise AssertionError(
            f"{protocol.__qualname__}.{name} is "
            f"{'async' if proto_async else 'sync'}, but "
            f"{implementation.__qualname__}.{name} is "
            f"{'async' if impl_async else 'sync'}"
        )

    proto_sig = inspect.signature(proto_target)
    impl_sig = inspect.signature(impl_target)

    proto_params = _param_specs(proto_sig)
    impl_params = _param_specs(impl_sig)
    if proto_params != impl_params:
        raise AssertionError(
            f"{protocol.__qualname__}.{name} parameters do not match "
            f"{implementation.__qualname__}.{name}:\n"
            f"  protocol:       {proto_params}\n"
            f"  implementation: {impl_params}"
        )

    proto_return = _raw_annotation_text(proto_sig.return_annotation)
    impl_return = _raw_annotation_text(impl_sig.return_annotation)
    if proto_return != impl_return:
        raise AssertionError(
            f"{protocol.__qualname__}.{name} return annotation {proto_return!r} does not "
            f"match {implementation.__qualname__}.{name} return annotation {impl_return!r}"
        )

    proto_hints = _resolved_hints(proto_target)
    impl_hints = _resolved_hints(impl_target)
    if proto_hints is not None and impl_hints is not None and proto_hints != impl_hints:
        raise AssertionError(
            f"{protocol.__qualname__}.{name} resolved type hints {proto_hints} do not "
            f"match {implementation.__qualname__}.{name} resolved type hints {impl_hints}"
        )


def assert_signature_parity(protocol: type, implementation: type) -> None:
    """Assert *implementation* has exact signature parity with *protocol*.

    For every public member declared directly in *protocol*'s own class
    body (its ``__dict__``, not anything inherited from ``Protocol`` or
    ``object``), assert *implementation* has a matching member: same
    property-vs-method kind, same sync-vs-async, same parameter names,
    kinds, order and defaults (``self``/``cls`` ignored), the same return
    annotation text, and — whenever both sides can resolve their forward
    references — the same resolved type hints.

    Members *implementation* defines beyond what *protocol* declares are
    never inspected and never fail this check; conformance is one-directional.

    Args:
        protocol: A ``@runtime_checkable`` Protocol class.
        implementation: A concrete class expected to satisfy *protocol*.

    Raises:
        AssertionError: On any missing member or signature mismatch.
    """
    for name, proto_member in vars(protocol).items():
        if name.startswith("_"):
            continue
        if not (isinstance(proto_member, property) or inspect.isfunction(proto_member)):
            continue
        _assert_member_parity(protocol, implementation, name, proto_member)


# ── Self-test: the helper must actually catch what isinstance() misses ────


@runtime_checkable
class _FakeProtocol(Protocol):
    """A tiny synthetic Protocol used only to sanity-check the helper above."""

    def do_thing(self, value: int, *, flag: bool = False) -> str: ...


class _FakeConformingImpl:
    def do_thing(self, value: int, *, flag: bool = False) -> str:
        return "ok"


class _FakeRenamedParamImpl:
    """isinstance(..., _FakeProtocol) would pass this — the member exists — but
    the parameter name has drifted, exactly the ADR-0001 hazard 4 shape."""

    def do_thing(self, renamed_value: int, *, flag: bool = False) -> str:
        return "ok"


class _FakeIncompleteImpl:
    """isinstance(..., _FakeProtocol) would already reject this one; kept as
    the baseline "missing member" case for the helper's own error message."""


class TestAssertSignatureParityHelper:
    def test_matching_signature_passes(self) -> None:
        assert_signature_parity(_FakeProtocol, _FakeConformingImpl)

    def test_renamed_parameter_fails(self) -> None:
        assert isinstance(_FakeRenamedParamImpl(), _FakeProtocol)  # isinstance is blind to this
        with pytest.raises(AssertionError, match="do_thing"):
            assert_signature_parity(_FakeProtocol, _FakeRenamedParamImpl)

    def test_missing_member_fails(self) -> None:
        with pytest.raises(AssertionError, match="missing member"):
            assert_signature_parity(_FakeProtocol, _FakeIncompleteImpl)


# ── LoaderProtocol <- FileLoader ───────────────────────────────────────────


class TestLoaderProtocolConformance:
    def test_file_loader_matches_loader_protocol(self) -> None:
        assert_signature_parity(LoaderProtocol, FileLoader)


# ── ChunkerProtocol <- RecursiveChunker ────────────────────────────────────


class TestChunkerProtocolConformance:
    def test_recursive_chunker_matches_chunker_protocol(self) -> None:
        assert_signature_parity(ChunkerProtocol, RecursiveChunker)


# ── EmbeddingProtocol <- InMemoryEmbedder, OllamaEmbedder, OpenAICompatibleEmbedder ──


class TestEmbeddingProtocolConformance:
    def test_in_memory_embedder_matches_embedding_protocol(self) -> None:
        assert_signature_parity(EmbeddingProtocol, InMemoryEmbedder)

    def test_ollama_embedder_matches_embedding_protocol(self) -> None:
        assert_signature_parity(EmbeddingProtocol, OllamaEmbedder)

    def test_openai_compatible_embedder_matches_embedding_protocol(self) -> None:
        assert_signature_parity(EmbeddingProtocol, OpenAICompatibleEmbedder)


# ── MetadataStoreProtocol <- SQLiteMetadataStore ───────────────────────────


class TestMetadataStoreProtocolConformance:
    def test_sqlite_metadata_store_matches_metadata_store_protocol(self) -> None:
        assert_signature_parity(MetadataStoreProtocol, SQLiteMetadataStore)


# ── VectorStoreProtocol <- InMemoryVectorStore, LanceDBVectorStore ─────────


class TestVectorStoreProtocolConformance:
    """Both dense stores, checked against the seam ADR-0001 hazard 3 shaped.

    The protocol's no-``**kwargs`` ``search`` signature is the half of that
    hazard fixed by declaration: a misspelled ``metadata_filter`` is a
    ``TypeError`` at the call site rather than a silently unfiltered result.
    Parity is what keeps it that way — an implementation free to re-add a
    catch-all would reopen it while still passing ``isinstance``.
    """

    def test_in_memory_vector_store_matches_vector_store_protocol(self) -> None:
        assert_signature_parity(VectorStoreProtocol, InMemoryVectorStore)

    def test_lancedb_vector_store_matches_vector_store_protocol(self) -> None:
        assert_signature_parity(VectorStoreProtocol, LanceDBVectorStore)


# ── RerankerProtocol <- CrossEncoderReranker ───────────────────────────────


class TestRerankerProtocolConformance:
    """The Wave D implementation against the seam ADR-0001 hazard 4 shaped.

    Importing :class:`CrossEncoderReranker` must not require the optional
    ``rerank`` extra — the heavy import is deferred to model load — so this
    test runs in the default suite, in a base install, exactly like every
    other conformance test here. If that laziness ever regressed, collection
    of this module would fail rather than the failure hiding until someone
    ran the gated suite.
    """

    def test_cross_encoder_reranker_matches_reranker_protocol(self) -> None:
        assert_signature_parity(RerankerProtocol, CrossEncoderReranker)


# ── ChatProtocol <- OllamaChat, OpenAICompatChat, ScriptedChatProvider, RedactingChat ──


class TestExtractorProtocolConformance:
    """``ExtractorProtocol`` <- ``PdfExtractor``, ``HtmlExtractor`` (Wave 3).

    Worth the parity check rather than ``isinstance`` for a reason specific to
    this seam: ADR-0016 decision 2 rests on the *same* extractor object being
    used at ingest and at citation-resolution time, so a drift between the two
    implementations -- one growing a keyword argument the other lacks, or
    ``identity`` becoming a method on one and staying a property on the other
    -- would break re-extraction in exactly the way ``isinstance`` cannot see.
    """

    def test_pdf_extractor_matches_extractor_protocol(self) -> None:
        assert_signature_parity(ExtractorProtocol, PdfExtractor)

    def test_html_extractor_matches_extractor_protocol(self) -> None:
        assert_signature_parity(ExtractorProtocol, HtmlExtractor)


# ── ChatProtocol <- OllamaChat, OpenAICompatChat, ScriptedChatProvider, RedactingChat ──


class TestChatProtocolConformance:
    def test_ollama_chat_matches_chat_protocol(self) -> None:
        assert_signature_parity(ChatProtocol, OllamaChat)

    def test_openai_compat_chat_matches_chat_protocol(self) -> None:
        assert_signature_parity(ChatProtocol, OpenAICompatChat)

    def test_scripted_chat_provider_matches_chat_protocol(self) -> None:
        assert_signature_parity(ChatProtocol, ScriptedChatProvider)

    def test_redacting_chat_matches_chat_protocol(self) -> None:
        """The decorator most of all (ADR-0017 decision 3 / SPEC.md's own
        argument): a wrapper that drifts from the seam it wraps is
        ADR-0001 hazard 4 with an extra frame, and plain ``isinstance``
        would not catch it."""
        assert_signature_parity(ChatProtocol, RedactingChat)

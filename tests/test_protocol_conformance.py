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
matching property-vs-method-vs-classmethod kind, sync-vs-async, parameter
names, kinds, order, and defaults (``self``/``cls`` ignored on both sides),
plus resolved type hints via :func:`typing.get_type_hints` whenever both
sides can resolve them. Extra members an implementation adds beyond the
Protocol are never a failure — implementations are always free to grow.
That also makes this helper robust to a concurrent change that adds a new
method to ``SQLiteMetadataStore`` without (yet) adding it to
``MetadataStoreProtocol``: only members the Protocol itself declares are
ever checked. A classmethod member is the one exception to "matches": its
return annotation is deliberately left uncompared (see
:func:`_assert_member_parity`), because a construction factory's whole
point is that each implementation returns its own concrete type.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, get_type_hints, runtime_checkable

import pytest

from groundkit.answer import SearchCallable
from groundkit.extraction import ExtractorProtocol, HtmlExtractor, PdfExtractor
from groundkit.index.bm25 import BM25Index
from groundkit.index.dense import InMemoryVectorStore, LanceDBVectorStore
from groundkit.index.metadata import SQLiteMetadataStore
from groundkit.index.protocols import (
    LexicalIndexProtocol,
    MetadataStoreProtocol,
    VectorStoreProtocol,
)
from groundkit.ingestion.chunking import RecursiveChunker
from groundkit.ingestion.loaders import FileLoader
from groundkit.ingestion.protocols import ChunkerProtocol, LoaderProtocol
from groundkit.ingestion.url_loader import UrlLoader
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
from groundkit.retrieval.search import Retriever

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


def _callable_target(
    member: property | classmethod[Any, ..., Any] | Callable[..., object],
) -> Callable[..., object]:
    """Return the underlying callable for a plain method, a classmethod, or a property's getter.

    Raises:
        AssertionError: If *member* is a write-only property (no getter) —
            unreachable for every Protocol in this codebase today, but a
            case worth failing loudly on rather than silently skipping.
    """
    if isinstance(member, property):
        if member.fget is None:
            raise AssertionError(f"property {member!r} has no getter")
        return member.fget
    if isinstance(member, classmethod):
        return member.__func__
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
    proto_raw: property | classmethod[Any, ..., Any] | Callable[..., object],
    impl_name: str | None = None,
) -> None:
    """Assert *implementation* has a same-shaped member as *protocol* declares.

    Args:
        protocol: The Protocol declaring *name*.
        implementation: The class expected to satisfy it.
        name: The member as the Protocol declares it.
        proto_raw: The Protocol's raw member object.
        impl_name: The member on *implementation* that satisfies *name*, when
            it is spelled differently — a ``__call__``-shaped Protocol is
            satisfied by a bound method, so the two names genuinely differ.
            Defaults to *name*.

    Raises:
        AssertionError: On a missing member, a property/method/classmethod
            kind mismatch, a sync/async mismatch, a parameter-shape
            mismatch, a return-annotation mismatch, or (when both sides
            resolve) a resolved-type-hints mismatch. Every message names
            the protocol, the implementation, the member, and the concrete
            diff.
    """
    impl_name = impl_name if impl_name is not None else name
    impl_raw = inspect.getattr_static(implementation, impl_name, None)
    if impl_raw is None:
        raise AssertionError(
            f"{implementation.__qualname__} is missing member {impl_name!r} "
            f"satisfying {protocol.__qualname__}.{name}"
        )

    proto_is_property = isinstance(proto_raw, property)
    impl_is_property = isinstance(impl_raw, property)
    if proto_is_property != impl_is_property:
        raise AssertionError(
            f"{protocol.__qualname__}.{name} is a "
            f"{'property' if proto_is_property else 'method'}, but "
            f"{implementation.__qualname__}.{impl_name} is a "
            f"{'property' if impl_is_property else 'method'}"
        )

    # A classmethod factory is the one member shape where the return
    # annotation is *expected* to differ: each implementation's factory
    # returns its own concrete type (BM25Index.from_store -> BM25Index), not
    # the Protocol's name, and there is no single spelling ("Self", the
    # Protocol's own name, ...) every implementation could be made to share
    # without lying about what it actually returns. Checked below: kind
    # (classmethod-ness), sync/async, and every parameter including their
    # resolved types. Not checked: the return annotation, in either its raw
    # or its resolved form.
    proto_is_classmethod = isinstance(proto_raw, classmethod)
    impl_is_classmethod = isinstance(impl_raw, classmethod)
    if proto_is_classmethod != impl_is_classmethod:
        raise AssertionError(
            f"{protocol.__qualname__}.{name} is a "
            f"{'classmethod' if proto_is_classmethod else 'plain method'}, but "
            f"{implementation.__qualname__}.{impl_name} is a "
            f"{'classmethod' if impl_is_classmethod else 'plain method'}"
        )

    proto_target = _callable_target(proto_raw)
    impl_target = _callable_target(impl_raw)

    proto_async = inspect.iscoroutinefunction(proto_target)
    impl_async = inspect.iscoroutinefunction(impl_target)
    if proto_async != impl_async:
        raise AssertionError(
            f"{protocol.__qualname__}.{name} is "
            f"{'async' if proto_async else 'sync'}, but "
            f"{implementation.__qualname__}.{impl_name} is "
            f"{'async' if impl_async else 'sync'}"
        )

    proto_sig = inspect.signature(proto_target)
    impl_sig = inspect.signature(impl_target)

    proto_params = _param_specs(proto_sig)
    impl_params = _param_specs(impl_sig)
    if proto_params != impl_params:
        raise AssertionError(
            f"{protocol.__qualname__}.{name} parameters do not match "
            f"{implementation.__qualname__}.{impl_name}:\n"
            f"  protocol:       {proto_params}\n"
            f"  implementation: {impl_params}"
        )

    if not proto_is_classmethod:
        proto_return = _raw_annotation_text(proto_sig.return_annotation)
        impl_return = _raw_annotation_text(impl_sig.return_annotation)
        if proto_return != impl_return:
            raise AssertionError(
                f"{protocol.__qualname__}.{name} return annotation {proto_return!r} does not "
                f"match {implementation.__qualname__}.{impl_name} return annotation "
                f"{impl_return!r}"
            )

    proto_hints = _resolved_hints(proto_target)
    impl_hints = _resolved_hints(impl_target)
    if proto_is_classmethod:
        # Same exemption as the raw-text check above, applied to the
        # resolved form: every OTHER key (each parameter's resolved type)
        # still has to agree.
        if proto_hints is not None:
            proto_hints = {k: v for k, v in proto_hints.items() if k != "return"}
        if impl_hints is not None:
            impl_hints = {k: v for k, v in impl_hints.items() if k != "return"}
    if proto_hints is not None and impl_hints is not None and proto_hints != impl_hints:
        raise AssertionError(
            f"{protocol.__qualname__}.{name} resolved type hints {proto_hints} do not "
            f"match {implementation.__qualname__}.{impl_name} resolved type hints {impl_hints}"
        )


#: Dunder members this helper checks despite the ``_``-prefix filter below.
#:
#: The filter exists to skip everything ``Protocol`` and ``object`` contribute
#: (``__init__``, ``__subclasshook__``, ``__protocol_attrs__``, ...), none of
#: which describes the seam. ``__call__`` is the one dunder that *is* the seam:
#: a callable-shaped Protocol declares its entire contract there. Before it was
#: allowlisted, such a Protocol yielded zero checked members and every call
#: against it passed vacuously — a mismatched implementation included.
_CHECKED_DUNDERS: frozenset[str] = frozenset({"__call__"})


def assert_signature_parity(
    protocol: type,
    implementation: type,
    *,
    member_map: dict[str, str] | None = None,
) -> None:
    """Assert *implementation* has exact signature parity with *protocol*.

    For every public member declared directly in *protocol*'s own class
    body (its ``__dict__``, not anything inherited from ``Protocol`` or
    ``object``), assert *implementation* has a matching member: same
    property-vs-method kind, same sync-vs-async, same parameter names,
    kinds, order and defaults (``self``/``cls`` ignored), the same return
    annotation text, and — whenever both sides can resolve their forward
    references — the same resolved type hints.

    ``__call__`` counts as public here (see :data:`_CHECKED_DUNDERS`), so a
    callable-shaped Protocol is checked rather than skipped.

    Members *implementation* defines beyond what *protocol* declares are
    never inspected and never fail this check; conformance is one-directional.

    Args:
        protocol: A ``@runtime_checkable`` Protocol class.
        implementation: A concrete class expected to satisfy *protocol*.
        member_map: Protocol-member name -> implementation-member name, for
            the case where the two are legitimately spelled differently. A
            ``__call__``-shaped Protocol is satisfied by a *bound method*
            (``retriever.search``), so its implementing member has that
            method's name, not ``__call__``.

    Raises:
        AssertionError: On any missing member, any signature mismatch, or a
            *protocol* that declares nothing this helper can check — the last
            because a check that inspects zero members reports success while
            proving nothing, which is worse than no check at all.
    """
    member_map = member_map or {}
    checked = 0
    for name, proto_member in vars(protocol).items():
        if name.startswith("_") and name not in _CHECKED_DUNDERS:
            continue
        if not (
            isinstance(proto_member, (property, classmethod)) or inspect.isfunction(proto_member)
        ):
            continue
        _assert_member_parity(protocol, implementation, name, proto_member, member_map.get(name))
        checked += 1

    if checked == 0:
        raise AssertionError(
            f"{protocol.__qualname__} declares no members this helper can check, so "
            f"parity against {implementation.__qualname__} would pass vacuously. If the "
            f"seam is shaped around a dunder, add it to _CHECKED_DUNDERS."
        )


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


@runtime_checkable
class _FakeCallableProtocol(Protocol):
    """A ``__call__``-shaped Protocol — the shape the helper could not see.

    Until ``__call__`` was allowlisted past the ``_``-prefix filter, iterating
    this class's ``vars()`` yielded no checkable member, so parity against
    *any* implementation passed without comparing anything."""

    async def __call__(self, query: str, top_k: int | None = None) -> str: ...


class _FakeConformingCallableImpl:
    async def __call__(self, query: str, top_k: int | None = None) -> str:
        return "ok"


class _FakeMismatchedCallableImpl:
    """Mismatched on parameter name, default, and return type at once — a
    single one of those is enough to fail, so a helper that reports parity
    here is not comparing signatures at all."""

    async def __call__(self, renamed_query: str, top_k: int = 5) -> bytes:
        return b"ok"


class _FakeMappedImpl:
    """Satisfies ``_FakeCallableProtocol`` through a differently-named member,
    the way ``Retriever.search`` satisfies ``SearchCallable.__call__``."""

    async def run(self, query: str, top_k: int | None = None) -> str:
        return "ok"


class _FakeMappedMismatchImpl:
    async def run(self, renamed_query: str, top_k: int | None = None) -> str:
        return "ok"


@runtime_checkable
class _FakeUncheckableProtocol(Protocol):
    """Declares only an attribute, so the helper can compare no member.

    Guards the general case behind the ``__call__`` bug: any future seam whose
    whole contract sits somewhere the filter skips must fail loudly rather than
    report a pass it never earned."""

    some_attribute: int


class _FakeIncompleteImpl:
    """isinstance(..., _FakeProtocol) would already reject this one; kept as
    the baseline "missing member" case for the helper's own error message."""


@runtime_checkable
class _FakeClassmethodProtocol(Protocol):
    """A classmethod-factory-shaped Protocol -- the shape
    ``LexicalIndexProtocol.from_store`` and ``BM25Index.from_store`` share
    (GK-016). Before ``_callable_target``/``_assert_member_parity`` learned
    to unwrap ``classmethod``, this shape was invisible to the helper the
    same way ``__call__`` once was: ``inspect.isfunction`` is ``False`` for a
    raw ``classmethod`` object, so the member-admission filter skipped it
    silently."""

    @classmethod
    async def build(cls, value: int, *, flag: bool = False) -> _FakeClassmethodProtocol: ...


class _FakeConformingClassmethodImpl:
    """Matches on kind, async-ness and parameters. Deliberately returns its
    OWN type, not the Protocol's -- the case the return-annotation exemption
    exists for: every real classmethod factory in this codebase does this,
    since returning literally ``_FakeClassmethodProtocol`` would defeat the
    point of a typed constructor."""

    @classmethod
    async def build(cls, value: int, *, flag: bool = False) -> _FakeConformingClassmethodImpl:
        return cls()


class _FakeMismatchedClassmethodParamImpl:
    """Same kind and same return-type shape as the conforming impl above,
    but a renamed parameter -- proves the return-annotation exemption does
    not also exempt parameters from being compared."""

    @classmethod
    async def build(
        cls, renamed_value: int, *, flag: bool = False
    ) -> _FakeMismatchedClassmethodParamImpl:
        return cls()


class _FakeNonClassmethodImpl:
    """Same name and parameter shape as the Protocol's ``build``, but a
    plain instance method rather than a classmethod. ``isinstance`` cannot
    see this distinction, and neither could this helper before the
    classmethod fix -- ``build`` would have failed ``inspect.isfunction``
    on the Protocol side and never been compared against anything."""

    async def build(self, value: int, *, flag: bool = False) -> _FakeNonClassmethodImpl:
        return self


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

    def test_matching_call_shaped_protocol_passes(self) -> None:
        assert_signature_parity(_FakeCallableProtocol, _FakeConformingCallableImpl)

    def test_mismatched_call_shaped_protocol_fails(self) -> None:
        """The H2 regression: this pair disagrees on parameter name, default and
        return type, and the helper reported parity anyway because ``__call__``
        starts with an underscore and was filtered out before comparison."""
        assert isinstance(_FakeMismatchedCallableImpl(), _FakeCallableProtocol)
        with pytest.raises(AssertionError, match="__call__"):
            assert_signature_parity(_FakeCallableProtocol, _FakeMismatchedCallableImpl)

    def test_member_map_compares_a_differently_named_member(self) -> None:
        """A ``__call__``-shaped Protocol is satisfied by a bound method, whose
        name is not ``__call__`` — so the mapping has to carry real comparison,
        not just suppress the missing-member error."""
        assert_signature_parity(
            _FakeCallableProtocol, _FakeMappedImpl, member_map={"__call__": "run"}
        )
        with pytest.raises(AssertionError, match="__call__"):
            assert_signature_parity(
                _FakeCallableProtocol, _FakeMappedMismatchImpl, member_map={"__call__": "run"}
            )

    def test_protocol_with_no_checkable_members_fails(self) -> None:
        with pytest.raises(AssertionError, match="vacuously"):
            assert_signature_parity(_FakeUncheckableProtocol, _FakeConformingImpl)

    def test_matching_classmethod_shaped_protocol_passes(self) -> None:
        """The return-annotation exemption is exercised for real here:
        ``_FakeConformingClassmethodImpl.build`` returns
        ``_FakeConformingClassmethodImpl``, not ``_FakeClassmethodProtocol``
        -- a raw-text return-annotation comparison would reject this pair,
        which is exactly why classmethod members skip that comparison."""
        assert_signature_parity(_FakeClassmethodProtocol, _FakeConformingClassmethodImpl)

    def test_classmethod_with_renamed_parameter_still_fails(self) -> None:
        """The return-annotation exemption does not widen into a parameter
        exemption: a renamed parameter is still caught."""
        with pytest.raises(AssertionError, match="build"):
            assert_signature_parity(_FakeClassmethodProtocol, _FakeMismatchedClassmethodParamImpl)

    def test_classmethod_satisfied_by_a_plain_method_fails(self) -> None:
        """Before ``classmethod`` unwrapping existed, this pair's ``build``
        would never have been compared at all: ``inspect.isfunction`` is
        ``False`` for a raw ``classmethod`` object, so the Protocol side's
        member-admission filter skipped it, and the only other member
        (there is none here) would have left ``checked == 0`` -- a vacuous
        pass hiding a real defect (a factory that must be callable on the
        class itself, not only on an instance)."""
        with pytest.raises(AssertionError, match="classmethod"):
            assert_signature_parity(_FakeClassmethodProtocol, _FakeNonClassmethodImpl)


# ── LoaderProtocol <- FileLoader ───────────────────────────────────────────


class TestLoaderProtocolConformance:
    def test_file_loader_matches_loader_protocol(self) -> None:
        assert_signature_parity(LoaderProtocol, FileLoader)

    def test_url_loader_matches_loader_protocol(self) -> None:
        """``UrlLoader`` (Wave 4) had only an ``isinstance`` check, in its own
        test file, until this entry — ``isinstance`` on a ``runtime_checkable``
        Protocol only confirms member names exist, exactly the gap CLAUDE.md
        calls out (it "would pass through the exact ``query`` -> ``_query``
        rename that caused ARP's signature drift"). ``assert_signature_parity``
        would catch, for example, ``UrlLoader.load`` losing its ``async``, or
        gaining/losing a parameter relative to ``LoaderProtocol.load``."""
        assert_signature_parity(LoaderProtocol, UrlLoader)


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


# ── LexicalIndexProtocol <- BM25Index ───────────────────────────────────────


class TestLexicalIndexProtocolConformance:
    def test_bm25_index_matches_lexical_index_protocol(self) -> None:
        assert_signature_parity(LexicalIndexProtocol, BM25Index)


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


# ── SearchCallable <- Retriever.search ─────────────────────────────────────


class TestSearchCallableConformance:
    """``SearchCallable`` (``answer.py``) against the method it is shaped after.

    ADR-0013 decision 10 and ADR-0014 decision 13 require a conformance entry
    for every Protocol; this one was missing, and could not have been written
    usefully before ``__call__`` was allowlisted — the helper would have
    inspected zero members and passed no matter what ``Retriever.search`` said.

    The mapping is the point rather than a workaround: ``AnswerPipeline`` is
    handed ``retriever.search``, a *bound method*, so what must match
    ``SearchCallable.__call__`` is ``Retriever.search``. The docstring on
    ``SearchCallable`` claims it "matches ``Retriever.search``'s exact
    positional/keyword shape"; this is the test that makes the claim hold —
    ``mode`` losing its keyword-only marker, ``top_k`` changing default, or
    ``search`` losing its ``async`` all fail here now.
    """

    def test_retriever_search_matches_search_callable(self) -> None:
        assert_signature_parity(SearchCallable, Retriever, member_map={"__call__": "search"})

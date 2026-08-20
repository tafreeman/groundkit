"""URL loader tests (ADR-0016 decision 4; Wave 4 of
``docs/specs/loaders-extracted-and-remote-sources.md``, §10).

No test here ever makes a real network call, and that now includes DNS.
Every ``UrlLoader`` under test is constructed with a client wired to
``httpx.MockTransport``, and the ``_no_real_dns`` autouse fixture below
replaces ``url_safety._default_resolver``, exactly the pattern
``tests/test_embeddings.py`` and ``tests/test_llm_provider.py`` already
establish for the same reason. pytest-asyncio is not part of this repo's
dependency set, so async code under test is driven with ``asyncio.run()``
inside plain ``def`` test functions.

The DNS half was missed when this file was first written, and the omission
was not merely cosmetic. ``ensure_safe_endpoint`` calls the system resolver
for any non-literal host, so every ``https://example.com/...`` case here was
issuing a real ``getaddrinfo`` — which makes the suite fail on an offline or
DNS-less runner, and, worse, hands the pass/fail decision to whatever
``example.com`` happens to resolve to rather than to the code under test.
It also left the guard's resolve-and-classify branch effectively unexercised:
``example.com`` resolves publicly, so the branch always said "safe", and no
test here could distinguish it running from it being skipped. See
:func:`test_a_hostname_resolving_to_an_unsafe_address_is_refused`, which is
only reachable now that the answer is under the test's control.

The SSRF tests are written so they would FAIL if
``groundkit.utils.url_safety.ensure_safe_endpoint`` were no longer called:
each one asserts the guard's own exception type and verdict substring *and*
that the mock transport's handler was never invoked (a handler that raises
``AssertionError`` when called) -- with ``MockTransport`` nothing really
connects, so a test that only checked "an error was raised" could pass for
the wrong reason (e.g. a malformed URL producing some other error).
"""

from __future__ import annotations

import asyncio
import builtins
import logging
import os
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path, PurePath
from typing import Any, Final

import httpx
import pytest

from groundkit import extraction
from groundkit.contracts import Document
from groundkit.errors import ConfigurationError, IngestionError
from groundkit.ingestion import url_loader
from groundkit.ingestion.protocols import LoaderProtocol
from groundkit.ingestion.url_loader import DEFAULT_MAX_BYTES, DEFAULT_TIMEOUT_SECONDS, UrlLoader
from groundkit.utils import url_safety
from groundkit.utils.path_safety import ensure_within_base

Handler = Callable[[httpx.Request], httpx.Response]

#: A real, globally-routable address (example.com's long-standing IP), used
#: only as a stand-in DNS answer and never actually contacted. Same constant
#: and same reasoning as ``tests/test_embeddings.py``.
_PUBLIC_ADDRESS: Final[str] = "93.184.216.34"


async def _resolve_to_public_address(host: str) -> Sequence[str]:
    """Fake resolver: answers every lookup with one public address."""
    del host
    return [_PUBLIC_ADDRESS]


@pytest.fixture(autouse=True)
def _no_real_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the system resolver for every test in this file.

    ``UrlLoader.load`` deliberately exposes no resolver parameter -- see
    :func:`test_load_accepts_no_resolver_or_allow_private_parameter`, which
    pins that narrow surface on purpose -- so the injection seam at this layer
    is ``url_safety._default_resolver``, whose own docstring names this exact
    case ("so a test can replace it wholesale ... when a call site has no
    resolver parameter of its own to inject through").

    This is deliberately *not* solved by giving ``UrlLoader`` a ``resolver``
    constructor argument. That would add a production-visible parameter whose
    only purpose is to tell the SSRF guard what a hostname resolves to, which
    is precisely the input the guard exists to determine for itself; the
    monkeypatch keeps the escape hatch inside the test process.

    Note this fixture cannot mask any SSRF case in this file: an address
    literal (``127.0.0.1``, ``[::1]``, ``10.0.0.5`` ...) takes
    ``ensure_safe_endpoint``'s literal fast path and never consults a resolver
    at all, so every guard test above still classifies the real address it was
    given.
    """
    monkeypatch.setattr(url_safety, "_default_resolver", _resolve_to_public_address)


def _client(handler: Handler) -> httpx.AsyncClient:
    """Build an httpx.AsyncClient wired to a MockTransport -- no network."""
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _load(loader: UrlLoader, source: str) -> list[Document]:
    return asyncio.run(loader.load(source))


@pytest.fixture(autouse=True)
def _clear_extractor_caches() -> Iterator[None]:
    """Reset every ``lru_cache`` in :mod:`groundkit.extraction` around a test.

    Mirrors ``tests/test_extraction.py``'s identical fixture: the accessors
    are memoized per process (spec §9.2), so a test in this file that blocks
    ``bs4`` would otherwise leak a "not installed" registry into unrelated
    tests, or silently inherit an already-populated one from a test that ran
    first.
    """
    caches = (extraction.pdf_extractor, extraction.html_extractor, extraction.active_extractors)
    for cached in caches:
        cached.cache_clear()
    yield
    for cached in caches:
        cached.cache_clear()


# -- protocol conformance ----------------------------------------------------


def test_conforms_to_loader_protocol(tmp_path: Path) -> None:
    assert isinstance(UrlLoader(tmp_path), LoaderProtocol)


def test_supported_extensions_is_always_empty(tmp_path: Path) -> None:
    """A URL is routed by shape, not extension -- see the module docstring."""
    assert UrlLoader(tmp_path).supported_extensions == []


def test_snapshot_dir_property_is_resolved(tmp_path: Path) -> None:
    relative_like = tmp_path / "col.snapshots"
    assert UrlLoader(relative_like).snapshot_dir == relative_like.resolve()
    assert UrlLoader(relative_like).snapshot_dir.is_absolute()


# -- SSRF guard: MUST fail if ensure_safe_endpoint were deleted --------------


class TestUrlLoaderSsrfGuard:
    """Each case asserts the guard's own refusal AND that the transport was
    never invoked -- both must hold for the test to actually pin the guard
    rather than merely observing *some* error.
    """

    @staticmethod
    def _boom(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("transport must not be invoked when the SSRF guard refuses")

    @pytest.mark.parametrize(
        ("url", "verdict"),
        [
            ("http://127.0.0.1/doc", "loopback"),
            ("http://[::1]/doc", "loopback"),
            ("http://[::ffff:127.0.0.1]/doc", "loopback"),  # IPv4-mapped IPv6 spelling
            ("http://10.0.0.5/doc", "private"),
            ("http://172.17.0.5/doc", "private"),
            ("http://169.254.169.254/doc", "link_local"),  # cloud metadata endpoint
        ],
    )
    def test_unsafe_endpoint_refused_before_any_request(
        self, tmp_path: Path, url: str, verdict: str
    ) -> None:
        loader = UrlLoader(tmp_path, client=_client(self._boom))
        with pytest.raises(ConfigurationError, match=verdict):
            _load(loader, url)

    def test_the_ollama_private_allowance_is_unreachable_from_url_ingestion(
        self, tmp_path: Path
    ) -> None:
        """``allow_private_endpoint=False`` is unconditional here -- there is
        no constructor or ``load()`` parameter that could re-open it. A
        loopback URL is refused exactly like any other unsafe endpoint,
        regardless of what an embedding provider is separately permitted."""
        loader = UrlLoader(tmp_path, client=_client(self._boom))
        with pytest.raises(ConfigurationError, match="loopback"):
            _load(loader, "http://127.0.0.1:11434/doc")

    @pytest.mark.parametrize(
        ("address", "verdict"),
        [
            ("127.0.0.1", "loopback"),
            ("10.0.0.5", "private"),
            ("169.254.169.254", "link_local"),
        ],
    )
    def test_a_hostname_resolving_to_an_unsafe_address_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, address: str, verdict: str
    ) -> None:
        """The guard's resolve-and-classify branch, which every other case in
        this class skips.

        The parametrized cases above all pass an address *literal*, which
        ``ensure_safe_endpoint`` classifies directly without ever calling a
        resolver. So none of them proves the DNS branch works -- and until the
        resolver was faked, nothing could: a real lookup of ``example.com``
        returns a public address, so the branch could only ever answer "safe",
        and deleting it entirely would not have failed a single test here.

        This is also the shape that matters in the wild. An attacker who
        controls a hostname does not need to put ``127.0.0.1`` in the URL; they
        point an ordinary-looking name at it. ``ensure_safe_endpoint``
        classifies what the name *resolves to*, and this asserts that -- with a
        public-looking hostname and an unsafe answer, so the refusal can only
        come from classifying the resolved address.
        """

        async def _resolve_to_unsafe_address(host: str) -> Sequence[str]:
            del host
            return [address]

        monkeypatch.setattr(url_safety, "_default_resolver", _resolve_to_unsafe_address)

        loader = UrlLoader(tmp_path, client=_client(self._boom))
        with pytest.raises(ConfigurationError, match=verdict):
            _load(loader, "https://entirely-ordinary-looking.example.com/doc")

    def test_every_resolved_address_is_classified_not_just_the_first(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A multi-homed name whose *second* answer is unsafe is still refused.

        A resolver may return several addresses and a caller that checked only
        ``[0]`` would be trivially defeated by ordering the safe one first.
        ``ensure_safe_endpoint`` documents that it classifies every address;
        this pins it from the loader's side, where the consequence lands.
        """

        async def _resolve_to_mixed_addresses(host: str) -> Sequence[str]:
            del host
            return [_PUBLIC_ADDRESS, "127.0.0.1"]

        monkeypatch.setattr(url_safety, "_default_resolver", _resolve_to_mixed_addresses)

        loader = UrlLoader(tmp_path, client=_client(self._boom))
        with pytest.raises(ConfigurationError, match="loopback"):
            _load(loader, "https://multi-homed.example.com/doc")

    def test_load_accepts_no_resolver_or_allow_private_parameter(self, tmp_path: Path) -> None:
        """MECHANISM PIN: asserting the narrow public surface itself, not
        just its effect -- a future change that adds an escape hatch
        parameter to ``load`` would need to touch this test."""
        import inspect

        signature = inspect.signature(UrlLoader.load)
        assert list(signature.parameters) == ["self", "source"]


# -- redirects: MUST fail if follow_redirects=False / the 3xx check were gone


class TestUrlLoaderRedirects:
    @pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
    def test_redirect_refused_and_target_never_requested(self, tmp_path: Path, status: int) -> None:
        requested_urls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requested_urls.append(str(request.url))
            return httpx.Response(
                status, headers={"location": "https://internal.example.com/secret"}
            )

        loader = UrlLoader(tmp_path, client=_client(handler))
        with pytest.raises(IngestionError, match="redirect"):
            _load(loader, "https://example.com/doc")

        # The destination named in Location must never have been fetched --
        # only the original URL may appear in what the transport saw.
        assert requested_urls == ["https://example.com/doc"]

    def test_redirect_refused_by_the_real_unwrapped_async_client(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exercises ``UrlLoader``'s PRODUCTION construction path.

        Every case above (and every other test in this file) injects a
        client via ``client=_client(handler)``. ``grk ingest <url>`` never
        does that -- it always takes the ``client is None`` branch in
        ``load()``, which builds a real ``httpx.AsyncClient(follow_redirects
        =False)``. Before this test, that construction call was evaluated by
        zero tests: the redundant-looking ``follow_redirects=False`` (httpx's
        own default is already ``False``) could be "cleaned up" to ``True``
        and every existing test would keep passing.

        ``httpx.AsyncClient.__init__`` is monkeypatched to (a) record the
        kwargs ``UrlLoader`` itself passes to the real class, and (b) mount a
        ``MockTransport`` in place of the real network transport so no
        request ever leaves the process. Crucially, ``follow_redirects`` is
        forwarded to the real ``__init__`` completely unmodified -- nothing
        here fakes ``.stream()`` or ``.send()``, so whether the 302 is
        followed is decided entirely by httpx's own client-level redirect
        logic, driven by whatever value ``UrlLoader`` actually passed.

        WOULD FAIL IF ``follow_redirects=False`` in ``url_loader.py`` were
        changed to ``True`` (or the argument were dropped in a way that let
        it default to ``True``): the client would silently follow the 302 to
        ``https://internal.example.com/secret`` through this same mounted
        transport, the handler would be invoked a second time and return a
        200, and ``client.stream()`` would hand that 200 back to ``_fetch``
        instead of the 302 -- so ``pytest.raises(IngestionError,
        match="redirect")`` below would fail with "DID NOT RAISE", and
        ``requested_urls`` would additionally contain the internal URL. The
        ``follow_redirects is False`` assertion on the captured kwargs is a
        second, independent pin on the same hazard: it fails immediately
        (with no redirect chain needed) the moment the literal argument
        changes, even before any request is made.
        """
        requested_urls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requested_urls.append(str(request.url))
            if str(request.url) == "https://example.com/doc":
                return httpx.Response(
                    302, headers={"location": "https://internal.example.com/secret"}
                )
            # Only reachable if the client followed the redirect itself --
            # i.e. only if follow_redirects were (incorrectly) True.
            return httpx.Response(200, content=b"should never be reached")

        captured_kwargs: list[dict[str, Any]] = []
        real_init = httpx.AsyncClient.__init__

        def tracking_init(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> None:
            captured_kwargs.append(dict(kwargs))
            kwargs["transport"] = httpx.MockTransport(handler)
            real_init(self, *args, **kwargs)

        monkeypatch.setattr(httpx.AsyncClient, "__init__", tracking_init)

        loader = UrlLoader(tmp_path)  # no client injected -- the production path
        with pytest.raises(IngestionError, match="redirect"):
            _load(loader, "https://example.com/doc")

        assert requested_urls == ["https://example.com/doc"]
        assert captured_kwargs
        assert captured_kwargs[-1].get("follow_redirects") is False


# -- error statuses -----------------------------------------------------------


class TestUrlLoaderErrorStatuses:
    @pytest.mark.parametrize("status", [404, 500, 503])
    def test_error_status_raises_ingestion_error(self, tmp_path: Path, status: int) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(status, content=b"error page body")

        loader = UrlLoader(tmp_path, client=_client(handler))
        with pytest.raises(IngestionError, match=str(status)):
            _load(loader, "https://example.com/doc")


# -- transport failure and client lifecycle -----------------------------------


class TestUrlLoaderTransportFailure:
    def test_transport_failure_is_wrapped_as_ingestion_error(self, tmp_path: Path) -> None:
        """A connection-level failure (DNS, refused connection, reset) is an
        ``httpx.HTTPError`` subclass raised by the transport itself, not a
        response the handler returns -- wrapped so callers never see a bare
        ``httpx`` exception type."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        loader = UrlLoader(tmp_path, client=_client(handler))
        with pytest.raises(IngestionError, match="failed to fetch"):
            _load(loader, "https://example.com/doc")

    def test_loader_closes_a_client_it_constructed_itself(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When no client is injected, ``UrlLoader`` builds and must close its
        own -- verified without any real network I/O by replacing
        ``httpx.AsyncClient.stream`` with a fake that yields a canned
        response, so ``ensure_safe_endpoint``'s literal-address fast path
        (a public IP, no DNS) is the only other thing exercised."""
        import contextlib

        closed = {"value": False}
        real_aclose = httpx.AsyncClient.aclose

        async def tracking_aclose(self: httpx.AsyncClient) -> None:
            closed["value"] = True
            await real_aclose(self)

        @contextlib.asynccontextmanager
        async def fake_stream(
            _self: httpx.AsyncClient, _method: str, _url: str, **_kwargs: object
        ) -> Any:
            yield httpx.Response(200, content=b"hello from a self-constructed client")

        monkeypatch.setattr(httpx.AsyncClient, "aclose", tracking_aclose)
        monkeypatch.setattr(httpx.AsyncClient, "stream", fake_stream)

        loader = UrlLoader(tmp_path)  # no client injected
        docs = _load(loader, "http://93.184.216.34/doc")  # a public literal address, no DNS

        assert docs[0].content == "hello from a self-constructed client"
        assert closed["value"] is True


# -- successful fetch and snapshot write --------------------------------------


class TestUrlLoaderFetchAndSnapshot:
    def test_plain_text_is_stored_verbatim_and_returned(self, tmp_path: Path) -> None:
        snapshot_dir = tmp_path / "default.snapshots"

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, headers={"content-type": "text/plain; charset=utf-8"}, content=b"Hello, world."
            )

        loader = UrlLoader(snapshot_dir, client=_client(handler))
        docs = _load(loader, "https://example.com/doc.txt")

        assert len(docs) == 1
        doc = docs[0]
        assert doc.content == "Hello, world."
        assert doc.source == "https://example.com/doc.txt"
        assert doc.source_class == "snapshot"
        assert doc.extractor is None

        snapshot_path = snapshot_dir / doc.document_id
        assert snapshot_path.read_text(encoding="utf-8") == "Hello, world."

    def test_missing_content_type_defaults_to_utf8(self, tmp_path: Path) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content="café".encode())

        loader = UrlLoader(tmp_path, client=_client(handler))
        docs = _load(loader, "https://example.com/doc")
        assert docs[0].content == "café"

    def test_document_id_is_the_default_uuid_generation(self, tmp_path: Path) -> None:
        """``document_id`` is never derived from the URL (spec §10.1) --
        pinned by checking it round-trips through Document's own default
        factory shape (32 lowercase hex characters)."""

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"body text")

        loader = UrlLoader(tmp_path, client=_client(handler))
        doc = _load(loader, "https://example.com/../../weird?a=1")[0]
        assert len(doc.document_id) == 32
        assert all(c in "0123456789abcdef" for c in doc.document_id)

    def test_empty_body_returns_empty_list_and_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"")

        loader = UrlLoader(tmp_path, client=_client(handler))
        with caplog.at_level(logging.WARNING):
            docs = _load(loader, "https://example.com/empty")
        assert docs == []
        assert any("Empty or whitespace-only" in rec.message for rec in caplog.records)

    def test_whitespace_only_body_returns_empty_list(self, tmp_path: Path) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"   \n\t  ")

        loader = UrlLoader(tmp_path, client=_client(handler))
        assert _load(loader, "https://example.com/blank") == []

    def test_injected_client_is_never_closed_by_the_loader(self, tmp_path: Path) -> None:
        """The caller owns an injected client's lifecycle -- this loader must
        not close a client it did not construct."""

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"body")

        client = _client(handler)
        loader = UrlLoader(tmp_path, client=client)
        _load(loader, "https://example.com/doc")
        assert not client.is_closed


# -- HTML-shaped extraction ----------------------------------------------------


class TestUrlLoaderHtmlExtraction:
    def test_html_content_type_strips_tags(self, tmp_path: Path) -> None:
        html = "<html><body><h1>Title</h1><p>Body text.</p></body></html>"

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, headers={"content-type": "text/html; charset=utf-8"}, content=html.encode()
            )

        loader = UrlLoader(tmp_path, client=_client(handler))
        doc = _load(loader, "https://example.com/page.html")[0]

        assert "<h1>" not in doc.content
        assert "<p>" not in doc.content
        assert "Title" in doc.content
        assert "Body text." in doc.content

        snapshot_path = tmp_path / doc.document_id
        assert snapshot_path.read_text(encoding="utf-8") == doc.content

    def test_xhtml_content_type_also_strips_tags(self, tmp_path: Path) -> None:
        xhtml = "<html><body><p>XHTML body</p></body></html>"

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"content-type": "application/xhtml+xml; charset=utf-8"},
                content=xhtml.encode(),
            )

        loader = UrlLoader(tmp_path, client=_client(handler))
        doc = _load(loader, "https://example.com/page.xhtml")[0]
        assert "<p>" not in doc.content
        assert "XHTML body" in doc.content

    def test_non_html_content_type_keeps_markup_literal(self, tmp_path: Path) -> None:
        raw = "<div>not actually stripped</div>"

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, headers={"content-type": "text/plain; charset=utf-8"}, content=raw.encode()
            )

        loader = UrlLoader(tmp_path, client=_client(handler))
        doc = _load(loader, "https://example.com/page.txt")[0]
        assert doc.content == raw

    def test_html_shaped_response_stays_source_class_snapshot_not_extracted(
        self, tmp_path: Path
    ) -> None:
        """Spec §10.2's explicit call: source_class stays "snapshot" and
        extractor stays None even when HTML extraction ran -- source_class
        encodes *where verification looks* (the local snapshot), not whether
        extraction happened somewhere upstream in the pipeline."""

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, headers={"content-type": "text/html"}, content=b"<p>hi</p>")

        loader = UrlLoader(tmp_path, client=_client(handler))
        doc = _load(loader, "https://example.com/page.html")[0]
        assert doc.source_class == "snapshot"
        assert doc.extractor is None

    def test_missing_html_extra_raises_configuration_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_import = builtins.__import__

        def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "bs4":
                raise ImportError("No module named 'bs4'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, headers={"content-type": "text/html"}, content=b"<p>hi</p>")

        loader = UrlLoader(tmp_path, client=_client(handler))
        with pytest.raises(ConfigurationError, match=r"groundkit\[html\]"):
            _load(loader, "https://example.com/page.html")


# -- size cap: refused, never truncated ---------------------------------------


class TestUrlLoaderSizeCap:
    def test_oversized_response_is_refused_not_truncated(self, tmp_path: Path) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"a" * 100)

        loader = UrlLoader(tmp_path, max_bytes=50, client=_client(handler))
        with pytest.raises(IngestionError, match="exceeds"):
            _load(loader, "https://example.com/big")

        # Refused, not truncated: nothing must land on disk under snapshot_dir.
        assert list(tmp_path.rglob("*")) == []

    def test_response_at_exact_cap_is_accepted(self, tmp_path: Path) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"a" * 50)

        loader = UrlLoader(tmp_path, max_bytes=50, client=_client(handler))
        doc = _load(loader, "https://example.com/exact")[0]
        assert doc.content == "a" * 50

    def test_default_max_bytes_matches_file_loaders_default(self) -> None:
        from groundkit.ingestion.loaders import DEFAULT_MAX_BYTES as FILE_LOADER_DEFAULT

        assert DEFAULT_MAX_BYTES == FILE_LOADER_DEFAULT


# -- decode: fail closed, never substitute -------------------------------------


class TestUrlLoaderDecoding:
    def test_undecodable_bytes_under_declared_charset_raise(self, tmp_path: Path) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"content-type": "text/plain; charset=utf-8"},
                content=b"\xff\xfe not valid utf-8",
            )

        loader = UrlLoader(tmp_path, client=_client(handler))
        with pytest.raises(IngestionError, match="not valid"):
            _load(loader, "https://example.com/bad")

    def test_no_replacement_character_is_ever_substituted(self, tmp_path: Path) -> None:
        """The failure mode this whole module exists to avoid: httpx's own
        ``Response.text`` decodes with ``errors="replace"`` and would return
        U+FFFD in place of the bad byte instead of raising, silently
        corrupting the offset space. This loader must never do that."""

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"before\xffafter")

        loader = UrlLoader(tmp_path, client=_client(handler))
        with pytest.raises(IngestionError) as excinfo:
            _load(loader, "https://example.com/bad")
        assert "\ufffd" not in str(excinfo.value)

    def test_unknown_declared_charset_raises_rather_than_guessing(self, tmp_path: Path) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"content-type": "text/plain; charset=totally-bogus-encoding"},
                content=b"hello",
            )

        loader = UrlLoader(tmp_path, client=_client(handler))
        with pytest.raises(IngestionError, match="not valid"):
            _load(loader, "https://example.com/bad-charset")

    def test_declared_non_utf8_charset_is_honored(self, tmp_path: Path) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"content-type": "text/plain; charset=latin-1"},
                content="café".encode("latin-1"),
            )

        loader = UrlLoader(tmp_path, client=_client(handler))
        doc = _load(loader, "https://example.com/latin1")[0]
        assert doc.content == "café"


# -- snapshot path containment --------------------------------------------------


class TestUrlLoaderSnapshotContainment:
    """``document_id`` is always Document's own safe ``uuid4().hex`` default
    in production (spec §10.1), but the write path re-checks containment
    unconditionally as defense-in-depth -- exercised directly against the
    private write helper the same way ``tests/test_embeddings.py`` tests
    ``_sanitize_url`` directly, since ``UrlLoader.load`` itself never accepts
    a caller-supplied document_id to inject a hostile one through.
    """

    @pytest.mark.parametrize(
        "malicious_document_id",
        [
            "../../etc/passwd",
            "../escape.txt",
            "/etc/passwd",
        ],
    )
    def test_traversal_and_absolute_shapes_are_refused(
        self, tmp_path: Path, malicious_document_id: str
    ) -> None:
        snapshot_dir = tmp_path / "col.snapshots"
        loader = UrlLoader(snapshot_dir)
        with pytest.raises(IngestionError, match="escapes"):
            loader._write_snapshot(malicious_document_id, "content")

        # Nothing must have been written anywhere outside (or inside) the
        # intended root as a side effect of the refused attempt.
        assert not (snapshot_dir.parent / "etc").exists()
        assert not Path("/etc/passwd_should_never_exist_from_this_test").exists()

    def test_windows_drive_shape_cannot_escape_on_either_platform(self, tmp_path: Path) -> None:
        """A Windows drive-absolute ``document_id`` is refused on Windows and
        inert on POSIX -- and the property under test is containment, not the
        refusal it happens to take on one of them.

        Split out of the parametrized case above, which asserted the refusal
        unconditionally and so passed on Windows and failed on Linux CI:
        ``C:\\Windows\\...`` is only an absolute path where the drive-letter
        syntax means something. On POSIX a backslash is an ordinary filename
        character, so the whole string is a single legal component under
        ``snapshot_dir`` -- ``ensure_within_base`` correctly does not raise,
        because nothing escaped.

        Branching on ``PurePath(...).is_absolute()`` rather than ``os.name``
        states that reasoning as the condition itself: the guard refuses
        exactly what the running platform considers absolute. Neither branch
        is a weakened assertion -- both end at "no byte was written outside
        the containment root", which is the guarantee spec §10.1 actually
        makes.
        """
        document_id = "C:\\Windows\\System32\\drivers\\etc\\hosts"
        snapshot_dir = tmp_path / "col.snapshots"
        loader = UrlLoader(snapshot_dir)

        if PurePath(document_id).is_absolute():
            with pytest.raises(IngestionError, match="escapes"):
                loader._write_snapshot(document_id, "content")
            # A refused write leaves nothing behind, not even the root it would
            # have written into. (`_write_snapshot` mkdirs only after the
            # containment check passes.)
            assert not snapshot_dir.exists()
        else:
            written = loader._write_snapshot(document_id, "content")
            assert written.is_relative_to(snapshot_dir.resolve())
            assert written.parent == snapshot_dir.resolve()
            assert written.read_text(encoding="utf-8") == "content"
            # The backslashes bought no extra directory levels: one file, one
            # level down, which is what "inert" means here.
            assert [entry.name for entry in snapshot_dir.iterdir()] == [document_id]

        # Whatever the platform decided above, nothing escaped `tmp_path`.
        # Deliberately not phrased as `not Path("/Windows/...").exists()`: on
        # Windows that resolves against the current drive to the REAL system
        # hosts file, so the assertion passes or fails on what was already on
        # the machine rather than on anything this test did.
        assert [entry.name for entry in tmp_path.iterdir()] in ([], ["col.snapshots"])

    def test_encoded_separator_is_inert_not_a_traversal(self, tmp_path: Path) -> None:
        """A document_id containing a percent-encoded separator is never
        URL-decoded anywhere on this write path, so it is just an unusual
        (but fully contained) filename -- pinned explicitly so a future
        "helpful" decode step would be caught by this test turning into an
        escape."""
        snapshot_dir = tmp_path / "col.snapshots"
        loader = UrlLoader(snapshot_dir)
        written = loader._write_snapshot("..%2f..%2fescape.txt", "content")
        assert written.is_relative_to(snapshot_dir.resolve())
        assert written.read_text(encoding="utf-8") == "content"

    def test_safe_document_id_is_written_under_snapshot_dir(self, tmp_path: Path) -> None:
        snapshot_dir = tmp_path / "col.snapshots"
        loader = UrlLoader(snapshot_dir)
        written = loader._write_snapshot("abc123", "hello")
        assert written == snapshot_dir.resolve() / "abc123"
        assert written.read_text(encoding="utf-8") == "hello"

    def test_snapshot_dir_is_created_if_missing(self, tmp_path: Path) -> None:
        snapshot_dir = tmp_path / "does" / "not" / "exist" / "yet.snapshots"
        loader = UrlLoader(snapshot_dir)
        assert not snapshot_dir.exists()
        loader._write_snapshot("doc-1", "hello")
        assert (snapshot_dir / "doc-1").read_text(encoding="utf-8") == "hello"


# -- no fetched body content ever reaches an exception message ----------------


class TestUrlLoaderNeverLeaksFetchedBodyIntoMessages:
    """SPEC.md §7 / ADR-0001 hazard 6, extended to fetched bodies: nothing
    read off the wire may appear in a raised exception's message. Each case
    below plants a sentinel in the response body on a path that raises, and
    asserts the sentinel never surfaces.
    """

    _SENTINEL = "UNLEAKED-SENTINEL-4f19c2"

    def test_error_status_body_not_leaked(self, tmp_path: Path) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, content=self._SENTINEL.encode())

        loader = UrlLoader(tmp_path, client=_client(handler))
        with pytest.raises(IngestionError) as excinfo:
            _load(loader, "https://example.com/doc")
        assert self._SENTINEL not in str(excinfo.value)

    def test_redirect_body_not_leaked(self, tmp_path: Path) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                302,
                headers={"location": "https://internal.example.com/secret"},
                content=self._SENTINEL.encode(),
            )

        loader = UrlLoader(tmp_path, client=_client(handler))
        with pytest.raises(IngestionError) as excinfo:
            _load(loader, "https://example.com/doc")
        assert self._SENTINEL not in str(excinfo.value)

    def test_oversize_body_not_leaked(self, tmp_path: Path) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=(self._SENTINEL + "x" * 100).encode())

        loader = UrlLoader(tmp_path, max_bytes=10, client=_client(handler))
        with pytest.raises(IngestionError) as excinfo:
            _load(loader, "https://example.com/doc")
        assert self._SENTINEL not in str(excinfo.value)

    def test_decode_failure_body_not_leaked(self, tmp_path: Path) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"\xff\xfe" + self._SENTINEL.encode())

        loader = UrlLoader(tmp_path, client=_client(handler))
        with pytest.raises(IngestionError) as excinfo:
            _load(loader, "https://example.com/doc")
        assert self._SENTINEL not in str(excinfo.value)

    def test_ssrf_refusal_carries_no_url_or_body_detail(self, tmp_path: Path) -> None:
        def boom(_request: httpx.Request) -> httpx.Response:
            raise AssertionError("transport must not be invoked")

        loader = UrlLoader(tmp_path, client=_client(boom))
        with pytest.raises(ConfigurationError) as excinfo:
            _load(loader, "http://127.0.0.1/doc")
        assert self._SENTINEL not in str(excinfo.value)


class TestUrlLoaderRefusesQueryStringCredentials:
    """The persisted half of the credential leak `tests/test_url_redaction.py`
    documents.

    `_reject_unsafe_url_shape` refused *userinfo* from the start, and its
    docstring gave the reason: this loader records the URL verbatim as
    `Document.source`, so a credential there is persisted to the index and
    returned in every citation built from it. That reasoning applies unchanged
    to `?token=...`, which the same function permitted -- so the pre-fix loader
    fetched these URLs and returned a Document carrying the secret in `source`.

    Both assertions matter, per this file's established discipline: the
    refusal's own message, and that the transport was never invoked. A test
    asserting only "an error was raised" could pass for the wrong reason.
    """

    @staticmethod
    def _boom(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("transport must not be invoked when the URL carries a credential")

    @pytest.mark.parametrize(
        "url",
        [
            "https://internal.example.com/export?token=sk-live-1",
            "https://internal.example.com/export?api-key=sk-live-1",
            "https://internal.example.com/export?id=42&access_token=sk-live-1",
            "https://internal.example.com/x?password=hunter2",
        ],
    )
    def test_a_credential_bearing_url_is_refused(self, tmp_path: Path, url: str) -> None:
        loader = UrlLoader(tmp_path, client=_client(self._boom))
        with pytest.raises(IngestionError, match="credential in its query string"):
            _load(loader, url)

    def test_the_refusal_does_not_echo_the_credential(self, tmp_path: Path) -> None:
        """The message names the offending *parameter*, never its value -- a
        refusal that quoted the secret back would move the leak from the index
        into the operator's terminal rather than closing it."""
        secret = "sk-live-must-not-appear"  # noqa: S105  # gitleaks:allow
        loader = UrlLoader(tmp_path, client=_client(self._boom))
        with pytest.raises(IngestionError) as excinfo:
            _load(loader, f"https://internal.example.com/export?token={secret}")
        rendered = str(excinfo.value)
        assert secret not in rendered
        assert "'token'" in rendered
        assert "internal.example.com" in rendered

    def test_no_snapshot_is_written_for_a_refused_url(self, tmp_path: Path) -> None:
        """The refusal happens before any fetch, so it must also happen before
        any disk write -- otherwise the URL is refused while its content is
        left on the volume."""
        loader = UrlLoader(tmp_path, client=_client(self._boom))
        with pytest.raises(IngestionError):
            _load(loader, "https://internal.example.com/export?token=sk-live-1")
        assert list(tmp_path.rglob("*")) == []


# -- content-sized writes never run on the event loop --------------------------

#: The scratch filename ``UrlLoader._extract_html`` writes the decoded body to
#: before handing it to the extractor. Named here rather than inlined because
#: the write-off-the-loop tests below identify that write by filename; if the
#: production name changes, ``assert scratch_writes`` fails loudly rather than
#: the test quietly asserting nothing.
_SCRATCH_FILENAME: Final[str] = "fetched.html"

_WriteRecord = tuple[Path, int]


def _record_write_text(monkeypatch: pytest.MonkeyPatch) -> list[_WriteRecord]:
    """Record ``(path, thread ident)`` for every ``Path.write_text`` call.

    Hooks the innermost blocking primitive, deliberately, rather than any
    helper a fix might introduce: ``Path.write_text`` exists identically
    before and after the change, so a test written against it exercises the
    defect itself. A test that patched a post-fix ``_write_sync`` would also
    fail against the unfixed code -- but with ``AttributeError``, which is a
    failure about the helper's existence, not about which thread the write
    ran on.

    Used by the HTML scratch write only. The snapshot write stopped being a
    ``Path.write_text`` call when it took ``O_NOFOLLOW`` (GK-028), and there
    is no lower-level primitive to hook in its place that is worth patching
    process-wide, so that test records ``_write_snapshot`` itself -- an
    established method whose absence would be just as loud.
    """
    records: list[_WriteRecord] = []
    original = Path.write_text

    def _recording_write_text(
        self: Path,
        data: str,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> int:
        records.append((Path(self), threading.get_ident()))
        return original(self, data, encoding=encoding, errors=errors, newline=newline)

    monkeypatch.setattr(Path, "write_text", _recording_write_text)
    return records


class TestUrlLoaderWritesOffTheEventLoop:
    """Neither content-sized write may run on the loop thread.

    ``UrlLoader`` had the only content-sized filesystem I/O in the package
    not dispatched through ``asyncio.to_thread``: the snapshot write at the
    end of ``load`` and the scratch write inside ``_extract_html``, each
    handling up to ``max_bytes`` of fetched body. Bounded today only because
    URL ingestion is CLI-only and nothing else is scheduled on that loop --
    which is a property of the current caller, not of this code, and this is
    exactly the module a service-side ingest tool would reuse verbatim.

    Asserts thread identity, not timing, the same way
    ``tests/test_retrieval.py::test_bm25_scoring_does_not_run_on_the_event_loop``
    and ``tests/test_dense.py::test_in_memory_search_does_not_score_on_the_event_loop``
    do: ``asyncio.to_thread`` always dispatches to an executor worker and an
    inline call always reports the loop's own thread, so there is no sleep,
    no timeout, and nothing to make flaky.

    The two writes are asserted in separate tests, each recording only its
    own write, so neither fix can mask the other: reverting one leaves
    exactly one of these red. They hook different things -- the snapshot
    write is no longer a ``Path.write_text`` call at all (GK-028) -- which is
    why the recording helper below serves only the scratch-write test.
    """

    @staticmethod
    def _text_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/plain; charset=utf-8"},
            content=b"snapshot body written to disk",
        )

    @staticmethod
    def _html_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            content=b"<html><body><p>scratch body written to disk</p></body></html>",
        )

    def test_snapshot_write_does_not_run_on_the_event_loop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Recorded at `_write_snapshot` rather than at `Path.write_text` (which
        # the scratch-write test below still hooks): the snapshot write is an
        # `os.open`/`os.fdopen` pair now, so it can carry `O_NOFOLLOW`
        # (GK-028), and no `Path.write_text` call is left on this path to
        # record. The property under test is unchanged and just as reverting:
        # the whole helper -- containment check and write together -- must run
        # off the loop thread, so an inline call here still turns this red.
        idents: list[int] = []
        original_write_snapshot = UrlLoader._write_snapshot

        def _recording_write_snapshot(loader: UrlLoader, document_id: str, content: str) -> Path:
            idents.append(threading.get_ident())
            return original_write_snapshot(loader, document_id, content)

        monkeypatch.setattr(UrlLoader, "_write_snapshot", _recording_write_snapshot)
        loader = UrlLoader(tmp_path, client=_client(self._text_handler))

        async def run() -> None:
            loop_thread = threading.get_ident()
            docs = await loader.load("https://example.com/doc.txt")

            assert len(docs) == 1
            assert idents, "the snapshot was never written"
            offenders = [ident for ident in idents if ident == loop_thread]
            assert not offenders, (
                f"the snapshot write ran on the event loop thread ({loop_thread}) "
                f"for {len(offenders)} of {len(idents)} write(s)"
            )

        asyncio.run(run())

    def test_html_scratch_write_does_not_run_on_the_event_loop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        writes = _record_write_text(monkeypatch)
        loader = UrlLoader(tmp_path, client=_client(self._html_handler))

        async def run() -> None:
            loop_thread = threading.get_ident()
            docs = await loader.load("https://example.com/page.html")

            assert len(docs) == 1
            scratch_writes = [
                (path, ident) for path, ident in writes if path.name == _SCRATCH_FILENAME
            ]
            assert scratch_writes, (
                f"no {_SCRATCH_FILENAME!r} write was recorded; the extraction path did not "
                "run, or the scratch filename changed. Recorded writes: "
                f"{[str(path) for path, _ident in writes]}"
            )
            offenders = [ident for _path, ident in scratch_writes if ident == loop_thread]
            assert not offenders, (
                f"the HTML scratch write ran on the event loop thread ({loop_thread}) "
                f"for {len(offenders)} of {len(scratch_writes)} write(s)"
            )

        asyncio.run(run())


# -- total request bound and the snapshot write's O_NOFOLLOW (GK-028) --------

#: How long the deliberately slow handler below stalls before answering. Far
#: longer than the bound the loader is given, and finite on purpose: against
#: code with no total bound this test must FAIL, not hang, so the handler has
#: to return eventually (a `threading.Event` handshake would deadlock instead).
_SLOW_HANDLER_SECONDS: Final[float] = 5.0

#: The bound the loader is given. Small enough that the difference between
#: "abandoned at the bound" and "waited for the server" is unmistakable.
_TINY_TIMEOUT_SECONDS: Final[float] = 0.05


class TestUrlLoaderTotalRequestBound:
    """One fetch is bounded in wall-clock, not merely per operation.

    httpx's ``timeout=`` is per operation, so a server that answers just
    inside the read timeout, forever, never trips it: the connection is held
    for as long as it likes while the caller believes it set a bound. The
    loader wraps the whole exchange in ``asyncio.timeout`` instead, which is
    what these tests pin.
    """

    def test_a_slow_response_is_abandoned_at_the_total_bound(self, tmp_path: Path) -> None:
        async def _slow_handler(_request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(_SLOW_HANDLER_SECONDS)
            return httpx.Response(
                200, headers={"content-type": "text/plain; charset=utf-8"}, content=b"too late"
            )

        loader = UrlLoader(
            tmp_path,
            timeout_seconds=_TINY_TIMEOUT_SECONDS,
            client=httpx.AsyncClient(transport=httpx.MockTransport(_slow_handler)),
        )

        started = time.monotonic()
        with pytest.raises(IngestionError, match="exceeded the total"):
            _load(loader, "https://example.com/slow.txt")
        elapsed = time.monotonic() - started

        # Stated twice on purpose: the refusal proves the bound is enforced,
        # the clock proves it is enforced *at the bound* rather than after the
        # server finally answered.
        assert elapsed < _SLOW_HANDLER_SECONDS / 2, (
            f"the fetch was bounded at {_TINY_TIMEOUT_SECONDS}s but took {elapsed:.2f}s, "
            f"which is the server's {_SLOW_HANDLER_SECONDS}s pace, not the caller's bound"
        )
        assert list(tmp_path.iterdir()) == [], "an abandoned fetch must snapshot nothing"

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_a_non_positive_bound_is_refused_at_construction(
        self, tmp_path: Path, bad: float
    ) -> None:
        """Zero or negative is refused rather than silently meaning
        "immediately" (``asyncio.timeout(0)``) or "never" -- mirroring
        ``EmbeddingConfig.timeout_seconds``' ``gt=0``, which is a startup
        failure for the same reason."""
        with pytest.raises(ValueError, match="timeout_seconds"):
            UrlLoader(tmp_path, timeout_seconds=bad)

    def test_default_bound_matches_the_embedding_config_default(self) -> None:
        from groundkit.config import EmbeddingConfig

        assert EmbeddingConfig().timeout_seconds == DEFAULT_TIMEOUT_SECONDS


class TestUrlLoaderSnapshotWriteDoesNotFollowASymlink:
    """The containment check and the open are two syscalls; the gap is closed.

    ``ensure_within_base`` resolves symlinks, so a snapshot path that is
    *already* a link out of the root is refused by the check itself (the
    traversal class above covers that). What the check cannot see is a link
    created after it returned and before the file is opened. The window is
    narrow -- ``document_id`` is an unguessable ``uuid4`` and the read side
    returns nothing unless the bytes still match -- but ``O_NOFOLLOW`` costs
    nothing, so it is closed rather than argued about (GK-028).

    The race is driven from the code path itself rather than from a second
    thread: the containment check is wrapped so that planting the symlink is
    the last thing it does before returning. That makes the interleaving
    deterministic instead of timing-dependent, and it is the only way to
    reach the window at all -- a symlink planted before the call is caught by
    the check, and one planted after the write is too late to matter.
    """

    def test_a_symlink_planted_after_the_containment_check_is_not_followed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if not hasattr(os, "O_NOFOLLOW"):
            pytest.skip("O_NOFOLLOW does not exist on this platform (Windows)")

        outside = tmp_path / "outside.txt"
        outside.write_text("original", encoding="utf-8")
        probe = tmp_path / "symlink-probe"
        try:
            probe.symlink_to(outside)
        except OSError:
            pytest.skip("this platform or user cannot create symlinks")
        probe.unlink()

        snapshot_dir = tmp_path / "col.snapshots"
        loader = UrlLoader(snapshot_dir)
        # Bind the real function from its defining module, not from
        # ``url_loader``'s re-export: the monkeypatch below replaces the
        # latter, and mypy --strict rejects reading a name a module does not
        # explicitly export.
        real_ensure_within_base = ensure_within_base

        def _plant_symlink_after_checking(path: str | Path, base_dir: str | Path) -> Path:
            resolved = real_ensure_within_base(path, base_dir)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.symlink_to(outside)
            return resolved

        monkeypatch.setattr(url_loader, "ensure_within_base", _plant_symlink_after_checking)

        with pytest.raises(IngestionError, match="symbolic link"):
            loader._write_snapshot("doc-1", "hijacked")

        assert outside.read_text(encoding="utf-8") == "original", (
            "the snapshot write followed a symlink out of the containment root "
            "and overwrote the file it pointed at"
        )

    def test_a_write_failure_that_is_not_a_symlink_is_not_reinterpreted(
        self, tmp_path: Path
    ) -> None:
        """Only the symlink errno becomes an ``IngestionError``; every other
        ``OSError`` propagates unchanged, so an unrelated I/O failure is never
        reported as a containment refusal. A directory where the snapshot file
        belongs is the portable way to provoke one (``EISDIR`` on POSIX,
        ``EACCES`` on Windows)."""
        snapshot_dir = tmp_path / "col.snapshots"
        (snapshot_dir / "doc-1").mkdir(parents=True)
        loader = UrlLoader(snapshot_dir)

        # `IngestionError` is not an `OSError`, so demanding an `OSError` here
        # is the whole assertion: had the handler swallowed every errno into a
        # containment refusal, this would not match.
        with pytest.raises(OSError):
            loader._write_snapshot("doc-1", "content")

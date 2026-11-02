"""Outbound-endpoint safety tests (ADR-0014 decision 10).

``url_safety.py`` is new, so most of what it owes is either a MUTATION CHECK
(the address-form logic has no unfixed source to revert — force is
demonstrated by weakening the implementation, running the affected tests,
and observing them fail) or a NEW SURFACE test (no prior baseline at all).
Both kinds are marked in each test's docstring. The one GENUINE REVERT this
module owns lives in ``tests/test_embeddings.py`` instead, at the call site
that invokes :func:`~groundkit.utils.url_safety.ensure_safe_endpoint`.

No test here ever touches the network: literal-address cases exercise
:func:`~groundkit.utils.url_safety.classify_address` directly (pure,
synchronous), and every hostname-resolution case injects a fake resolver via
:func:`~groundkit.utils.url_safety.ensure_safe_endpoint`'s ``resolver``
parameter. pytest-asyncio is not part of this repo's dependency set, so
async code under test is driven with ``asyncio.run()`` inside plain ``def``
test functions, matching ``tests/test_embeddings.py``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest

from groundkit.errors import ConfigurationError
from groundkit.utils.url_safety import (
    DEFAULT_ALLOWED_SCHEMES,
    classify_address,
    credential_query_params,
    ensure_safe_endpoint,
    sanitize_url,
    validate_endpoint_shape,
)

# ── validate_endpoint_shape ─────────────────────────────────────────────


class TestValidateEndpointShape:
    def test_default_schemes_are_http_and_https(self) -> None:
        assert frozenset({"http", "https"}) == DEFAULT_ALLOWED_SCHEMES

    @pytest.mark.parametrize("scheme", ["http", "https"])
    def test_accepts_default_allowed_schemes(self, scheme: str) -> None:
        validate_endpoint_shape(f"{scheme}://api.example.com/v1")

    @pytest.mark.parametrize("bad_url", ["ftp://api.example.com/", "file:///etc/passwd"])
    def test_rejects_scheme_outside_allow_list(self, bad_url: str) -> None:
        with pytest.raises(ConfigurationError, match="scheme"):
            validate_endpoint_shape(bad_url)

    def test_allowed_schemes_is_overridable(self) -> None:
        validate_endpoint_shape("ws://api.example.com/", allowed_schemes=frozenset({"ws"}))
        with pytest.raises(ConfigurationError, match="scheme"):
            validate_endpoint_shape("http://api.example.com/", allowed_schemes=frozenset({"ws"}))

    def test_rejects_empty_host(self) -> None:
        with pytest.raises(ConfigurationError, match="host"):
            validate_endpoint_shape("http:///v1/embeddings")

    def test_rejects_username_only(self) -> None:
        with pytest.raises(ConfigurationError, match="userinfo"):
            validate_endpoint_shape("http://user@api.example.com/")

    def test_rejects_password_only(self) -> None:
        with pytest.raises(ConfigurationError, match="userinfo"):
            validate_endpoint_shape("http://:hunter2@api.example.com/")

    def test_rejects_username_and_password(self) -> None:
        with pytest.raises(ConfigurationError, match="userinfo"):
            validate_endpoint_shape("http://user:hunter2@api.example.com/")

    def test_userinfo_rejection_message_never_echoes_the_url(self) -> None:
        """The credential being rejected lives in the URL this call was given
        — the whole point of rejecting it is that it must never reach a log
        line or exception message (ADR-0014 decision 10)."""
        secret_url = "http://admin:sk-super-secret@api.example.com/"  # noqa: S105 - test fixture
        with pytest.raises(ConfigurationError) as excinfo:
            validate_endpoint_shape(secret_url)
        assert "sk-super-secret" not in str(excinfo.value)
        assert secret_url not in str(excinfo.value)

    def test_rejects_query_string(self) -> None:
        with pytest.raises(ConfigurationError, match="query"):
            validate_endpoint_shape("http://api.example.com/v1?api-key=sk-live-123")

    def test_query_rejection_message_never_echoes_the_url(self) -> None:
        secret_url = "http://api.example.com/v1?api-key=sk-live-123"  # noqa: S105 - test fixture
        with pytest.raises(ConfigurationError) as excinfo:
            validate_endpoint_shape(secret_url)
        assert "sk-live-123" not in str(excinfo.value)

    def test_rejects_fragment(self) -> None:
        with pytest.raises(ConfigurationError, match="fragment"):
            validate_endpoint_shape("http://api.example.com/v1#section")

    def test_accepts_a_path_with_no_query_or_fragment(self) -> None:
        """Only query/fragment are forbidden — base_url legitimately carries a
        path prefix for proxied deployments (several existing tests use
        one)."""
        validate_endpoint_shape("https://embed-proxy.example.com/openai-proxy")

    @pytest.mark.parametrize(
        "bad_host_url",
        [
            "http://999.1.1.1/",  # octet out of range
            "http://0177.0.0.1/",  # leading zero — ipaddress rejects as ambiguous
        ],
    )
    def test_rejects_host_that_looks_like_ipv4_but_does_not_parse(self, bad_host_url: str) -> None:
        with pytest.raises(ConfigurationError, match="IPv4"):
            validate_endpoint_shape(bad_host_url)

    def test_five_dot_separated_groups_does_not_match_the_dotted_quad_shape(self) -> None:
        """Five groups is not "shaped like IPv4" under this module's narrow
        heuristic (exactly four dot-separated all-digit runs) — it is
        treated as an ordinary hostname, same as any other non-matching
        string. This pins the boundary of the heuristic rather than
        asserting a rejection."""
        validate_endpoint_shape("http://1.2.3.4.5/")

    def test_accepts_a_valid_ipv4_host(self) -> None:
        validate_endpoint_shape("http://127.0.0.1:11434")

    def test_accepts_an_ordinary_hostname(self) -> None:
        """A hostname with digits and dots in it must not be mistaken for a
        malformed IPv4 literal — the dotted-quad check only fires when every
        dot-separated component is entirely digits."""
        validate_endpoint_shape("http://api2.example.com/")

    def test_does_not_reject_a_bare_decimal_hostname(self) -> None:
        """MECHANISM CHECK, not a safety claim: ``2130706433`` has no dots,
        so it does not match the dotted-quad shape and is treated as an
        ordinary (if unusual) hostname here. It is unsafe as an *address* —
        see TestEnsureSafeEndpoint's decimal-IPv4 case — but that is a
        resolver-time concern, not a shape concern; ``validate_endpoint_shape``
        does no DNS and cannot know what a resolver would later return."""
        validate_endpoint_shape("http://2130706433/")


# ── classify_address ─────────────────────────────────────────────────────


#: NEW SURFACE: every literal the task specification requires covered that
#: parses directly via ``ipaddress`` (the resolver-mechanism cases —
#: ``2130706433`` and ``0177.0.0.1`` — live in TestEnsureSafeEndpoint
#: instead, since ``ipaddress`` refuses to parse them as literals at all).
_UNSAFE_LITERALS: Sequence[tuple[str, str]] = [
    ("::ffff:127.0.0.1", "IPv4-mapped loopback"),
    ("::ffff:10.0.0.1", "IPv4-mapped RFC1918 private"),
    ("fc00::1", "unique local address (IPv6 private)"),
    ("fe80::1", "IPv6 link-local"),
    ("169.254.169.254", "cloud metadata endpoint (link-local)"),
    ("100.64.0.1", "RFC6598 shared/carrier-grade-NAT address space"),
    ("0.0.0.0", "unspecified IPv4"),  # noqa: S104 - address-classification fixture, not a bind
    ("::", "unspecified IPv6"),
    ("127.0.0.1", "IPv4 loopback"),
    ("::1", "IPv6 loopback"),
    ("224.0.0.1", "IPv4 multicast"),
    ("240.0.0.1", "IPv4 reserved (Class E)"),
]


class TestClassifyAddressRejectsUnsafeLiterals:
    """NEW SURFACE for the verdict-vs-safe distinction; MUTATION CHECK for the
    unmap-before-classify mechanism specifically (see the mapped/6to4/Teredo
    cases below, and the module docstring's account of the weakened-version
    run against them)."""

    @pytest.mark.parametrize(("literal", "_description"), _UNSAFE_LITERALS)
    def test_rejects(self, literal: str, _description: str) -> None:
        assert classify_address(literal) is not None

    def test_ipv4_mapped_loopback_is_named_loopback(self) -> None:
        """Precise-verdict check, not just non-None: this is the exact
        spelling ADR-0014 decision 10 calls out by name —
        ``IPv6Address("::ffff:127.0.0.1").is_loopback`` disagrees with itself
        across CPython point releases, so unmapping before classifying (not
        relying on the property alone) is what makes this assertion hold
        version-independently."""
        assert classify_address("::ffff:127.0.0.1") == "loopback"

    def test_link_local_is_not_swallowed_by_the_broader_private_check(self) -> None:
        """Verdict-ordering check: ``is_private`` is true for the entire
        169.254.0.0/16 block in Python's ipaddress, so if it were checked
        before ``is_link_local`` this cloud-metadata address would be
        misclassified "private" — and then wrongly *allowed* by
        ``ensure_safe_endpoint``'s ``allow_private_endpoint=True`` allowance,
        which permits "private" but must never permit "link_local"."""
        assert classify_address("169.254.169.254") == "link_local"

    def test_shared_address_space_is_not_swallowed_by_reserved_or_private(self) -> None:
        assert classify_address("100.64.0.1") == "shared_address_space"

    def test_sixtofour_embedding_a_private_address_is_rejected(self) -> None:
        """NEW SURFACE, unmap mechanism: a 6to4 address (2002::/16) embeds an
        IPv4 address in its next 32 bits. Unlike IPv4-mapped addresses, no
        released CPython auto-consults this embedding for ``is_private`` —
        the explicit unmap in ``classify_address`` is the only thing that
        makes this reachable at all, mapped or not."""
        # 2002:c000:0204:: embeds 192.0.2.4 (TEST-NET-1, in ipaddress's
        # private-network list).
        assert classify_address("2002:c000:0204::") is not None

    def test_teredo_embedding_a_private_address_is_rejected(self) -> None:
        """NEW SURFACE, unmap mechanism, third fallback (teredo[1])."""
        # Client address 192.0.2.45 (TEST-NET-1) obfuscated per RFC 4380.
        assert classify_address("2001:0000:4136:e378:8000:63bf:3fff:fdd2") is not None

    def test_ipv4_mapped_shared_address_space_requires_the_unmap(self) -> None:
        """MUTATION CHECK, the sharpest case in this module: without the
        explicit unmap, this literal is a bare ``IPv6Address`` when it
        reaches the ``isinstance(addr, ipaddress.IPv4Address)`` membership
        test that backs the shared-address-space check, so that check can
        never fire — and none of the six standard predicates fire for it
        either (100.64.0.0/10's ``is_private`` and ``is_global`` are both
        False, the documented ipaddress gap this module's constant exists
        to close). Weakening ``classify_address`` by deleting its unmap
        block and rerunning this test flips it from a rejection to a false
        "safe": observed directly (see accompanying report), not asserted
        here — this test only pins the correct (unmapped) behavior."""
        assert classify_address("::ffff:100.64.0.1") == "shared_address_space"


class TestClassifyAddressIsInterpreterIndependent:
    """GK-3 F-GK3-5: on CPython 3.11, ``is_private`` covers only part of
    192.0.0.0/24 and none of 3fff::/20, so these addresses classified as safe
    on the supported floor and as private on 3.12+. 192.0.0.192 is a cloud
    instance-metadata address."""

    @pytest.mark.parametrize(
        "literal", ["192.0.0.1", "192.0.0.8", "192.0.0.192", "192.0.0.255", "::ffff:192.0.0.192"]
    )
    def test_protocol_assignments_are_refused_on_every_supported_version(
        self, literal: str
    ) -> None:
        assert classify_address(literal) == "protocol_assignment"

    def test_the_ipv6_documentation_range_is_private_on_every_supported_version(self) -> None:
        assert classify_address("3fff::1") == "private"

    def test_the_private_endpoint_allowance_does_not_admit_a_metadata_address(self) -> None:
        """The allowance admits "private" for a compose-network Ollama. A
        metadata address must stay refused even then, like 169.254.169.254."""

        async def boom(_host: str) -> Sequence[str]:
            raise AssertionError("resolver must not be called for a literal host")

        async def run() -> None:
            with pytest.raises(ConfigurationError, match="protocol_assignment"):
                await ensure_safe_endpoint(
                    "http://192.0.0.192/api/embed", allow_private_endpoint=True, resolver=boom
                )

        _run(run())


class TestClassifyAddressAllowsSafeLiterals:
    @pytest.mark.parametrize(
        "literal",
        ["8.8.8.8", "1.1.1.1", "93.184.216.34", "2001:4860:4860::8888"],
    )
    def test_allows(self, literal: str) -> None:
        assert classify_address(literal) is None

    @pytest.mark.parametrize("literal", ["192.0.0.9", "192.0.0.10"])
    def test_the_two_globally_reachable_protocol_assignments_stay_allowed(
        self, literal: str
    ) -> None:
        """IANA marks these two addresses in 192.0.0.0/24 globally reachable,
        and CPython 3.12+ excludes them from ``is_private`` too."""
        assert classify_address(literal) is None

    def test_rfc1918_private_is_named_private_not_something_narrower(self) -> None:
        """Verdict-ordering complement to the link-local test above: an
        ordinary RFC1918 address must still fall through every narrower
        predicate to reach — and be correctly caught by — the private
        catch-all, which is what ``allow_private_endpoint=True`` relies on
        to admit a compose bridge-network Ollama address."""
        assert classify_address("172.17.0.5") == "private"


# ── ensure_safe_endpoint ─────────────────────────────────────────────────


def _run(coro: object) -> None:
    asyncio.run(coro)  # type: ignore[arg-type]


class TestEnsureSafeEndpointLiteralFastPath:
    def test_safe_literal_never_invokes_the_resolver(self) -> None:
        """A host that already parses as a literal must skip DNS entirely —
        the default Ollama endpoint (127.0.0.1) takes exactly this path on
        every request, at zero resolution cost."""

        async def boom(_host: str) -> Sequence[str]:
            raise AssertionError("resolver must not be called for a literal host")

        async def run() -> None:
            await ensure_safe_endpoint("http://8.8.8.8/api/embed", resolver=boom)

        _run(run())

    def test_unsafe_literal_is_rejected_without_a_resolver(self) -> None:
        async def boom(_host: str) -> Sequence[str]:
            raise AssertionError("resolver must not be called for a literal host")

        async def run() -> None:
            with pytest.raises(ConfigurationError, match="loopback"):
                await ensure_safe_endpoint("http://127.0.0.1:11434/api/embed", resolver=boom)

        _run(run())


class TestEnsureSafeEndpointResolvedHosts:
    def test_hostname_resolving_to_a_public_address_is_allowed(self) -> None:
        async def resolver(host: str) -> Sequence[str]:
            assert host == "embed.example.com"
            return ["93.184.216.34"]

        async def run() -> None:
            await ensure_safe_endpoint("https://embed.example.com/v1/embeddings", resolver=resolver)

        _run(run())

    def test_hostname_resolving_to_a_private_address_is_rejected_by_default(self) -> None:
        async def resolver(_host: str) -> Sequence[str]:
            return ["10.0.0.5"]

        async def run() -> None:
            with pytest.raises(ConfigurationError, match="private"):
                await ensure_safe_endpoint("https://internal.example.com/v1", resolver=resolver)

        _run(run())

    def test_unresolvable_host_is_rejected(self) -> None:
        async def resolver(_host: str) -> Sequence[str]:
            return []

        async def run() -> None:
            with pytest.raises(ConfigurationError, match="resolve"):
                await ensure_safe_endpoint("https://nowhere.example.com/v1", resolver=resolver)

        _run(run())

    def test_mixed_safe_and_unsafe_resolved_addresses_is_rejected(self) -> None:
        """MUTATION CHECK: every resolved address must be classified, not
        just the first. A resolver answering with a safe address first and
        an unsafe one second is exactly what an implementation that stops at
        ``addresses[0]`` would wrongly let through — weakening
        ``ensure_safe_endpoint`` to check only the first address and
        rerunning this test flips it from a rejection to a false "safe"
        (observed directly; see the accompanying report)."""

        async def resolver(_host: str) -> Sequence[str]:
            return ["8.8.8.8", "127.0.0.1"]

        async def run() -> None:
            with pytest.raises(ConfigurationError, match="loopback"):
                await ensure_safe_endpoint("https://mixed.example.com/v1", resolver=resolver)

        _run(run())

    def test_decimal_ipv4_hostname_is_rejected_via_the_resolver_not_the_literal_parser(
        self,
    ) -> None:
        """MUTATION CHECK / mechanism test: ``ipaddress.ip_address`` refuses
        ``"2130706433"`` (it is not dotted-quad or colon-separated), so this
        host falls through to the "treat as hostname, resolve it" branch —
        exactly the case a naive guard that only classified addresses which
        parsed as literals, and treated everything else as automatically
        safe, would miss. glibc's own ``getaddrinfo`` accepts this spelling
        and returns 127.0.0.1; this test's fake resolver stands in for that
        behavior so no real DNS is touched. Deleting the resolve step (or
        replacing it with an unconditional "safe") and rerunning this test
        flips it from a rejection to a false "safe" (observed directly; see
        the accompanying report)."""

        async def resolver(host: str) -> Sequence[str]:
            assert host == "2130706433"
            return ["127.0.0.1"]

        async def run() -> None:
            with pytest.raises(ConfigurationError, match="loopback"):
                await ensure_safe_endpoint("http://2130706433/api/embed", resolver=resolver)

        _run(run())

    def test_leading_zero_ipv4_hostname_is_refused_before_resolution(self) -> None:
        """A different ambiguous spelling from the decimal case, and a different
        path: the guard reads the host the way httpx does (F-GK3-4), and httpx
        refuses ``0177.0.0.1`` as an invalid IPv4 address. No request could be
        made to it, so the guard refuses it without consulting DNS."""

        async def boom(_host: str) -> Sequence[str]:
            raise AssertionError("resolver must not be called for an unparseable host")

        async def run() -> None:
            with pytest.raises(ConfigurationError, match="cannot parse"):
                await ensure_safe_endpoint("http://0177.0.0.1/api/embed", resolver=boom)

        _run(run())


class TestEnsureSafeEndpointResolvesWhatHttpxConnectsTo:
    """GK-3 F-GK3-4: the guard must classify the DNS name the client will
    query. For ``faß.example`` the stdlib's IDNA2003 codec (what getaddrinfo
    applies to ``urlsplit().hostname``) gives ``fass.example``, while httpx
    connects to the IDNA2008 ``xn--fa-hia.example`` -- a different name, and
    an attacker's zone can answer the two differently."""

    def test_a_non_ascii_host_is_resolved_in_httpx_encoding(self) -> None:
        seen: list[str] = []

        async def resolver(host: str) -> Sequence[str]:
            seen.append(host)
            return ["127.0.0.1"] if host == "xn--fa-hia.example" else ["93.184.216.34"]

        async def run() -> None:
            with pytest.raises(ConfigurationError, match="loopback"):
                await ensure_safe_endpoint("https://faß.example/doc", resolver=resolver)

        _run(run())
        assert seen == ["xn--fa-hia.example"]

    @pytest.mark.parametrize("url", ["http://a\tb/x", "http://[::1/x"])
    def test_a_host_httpx_cannot_parse_is_refused_without_dns(self, url: str) -> None:
        async def boom(_host: str) -> Sequence[str]:
            raise AssertionError("resolver must not be called for an unparseable host")

        async def run() -> None:
            with pytest.raises(ConfigurationError):
                await ensure_safe_endpoint(url, resolver=boom)

        _run(run())


class TestCredentialParamsAndSanitize:
    """GK-3 F-GK3-1 and F-GK3-3, at the helper level."""

    @pytest.mark.parametrize(
        ("url", "names"),
        [
            ("https://h.example/doc#access_token=FRAG", ("access_token",)),
            ("https://h.example/doc#state=x&id_token=FRAG", ("id_token",)),
            ("https://h.example/doc;key=PATH", ("key",)),
            ("https://h.example/a;v=1/b;sig=PATH/c", ("sig",)),
            ("https://h.example/doc?x=1;token=SEMI", ("token",)),
            ("https://h.example/doc?token=Q", ("token",)),
            ("https://h.example/app#/callback?access_token=FRAG", ("access_token",)),
            ("https://h.example/app#?access_token=FRAG", ("access_token",)),
            ("https://h.example/doc%3Btoken=PATH", ("token",)),
            ("https://h.example/doc%3btoken%3DPATH", ("token",)),
        ],
    )
    def test_credentials_are_found_outside_the_ampersand_query(
        self, url: str, names: tuple[str, ...]
    ) -> None:
        assert credential_query_params(url) == names

    @pytest.mark.parametrize(
        "url",
        [
            "https://h.example/doc#signature",
            "https://h.example/doc#key",
            "https://h.example/doc;jsessionid",
            "https://h.example/doc?id=42#section-2",
        ],
    )
    def test_ordinary_anchors_and_params_are_not_credentials(self, url: str) -> None:
        assert credential_query_params(url) == ()

    def test_sanitize_survives_a_malformed_port_and_drops_userinfo(self) -> None:
        rendered = sanitize_url("https://user:hunter2@example.com:abc/x")
        assert "hunter2" not in rendered
        assert rendered == "https://***@example.com/x"

    def test_sanitize_redacts_path_parameter_values(self) -> None:
        rendered = sanitize_url("https://h.example/doc;key=PATHSECRET/x")
        assert "PATHSECRET" not in rendered
        assert rendered == "https://h.example/doc;key=***/x"

    def test_sanitize_redacts_an_encoded_path_parameter(self) -> None:
        rendered = sanitize_url("https://h.example/doc%3Btoken%3DPATHSECRET/x")
        assert "PATHSECRET" not in rendered
        assert rendered == "https://h.example/doc%3Btoken%3D***/x"

    @pytest.mark.parametrize("url", ["https://user:hunter2@[invalid]/x", "http://[::1/x"])
    def test_sanitize_never_raises_and_echoes_nothing_it_cannot_parse(self, url: str) -> None:
        """It runs on error paths: raising there replaced the error being
        reported, and a URL it cannot take apart cannot be echoed safely."""
        assert sanitize_url(url) == "<unparseable URL>"

    def test_sanitize_leaves_a_filesystem_path_alone(self) -> None:
        assert sanitize_url("docs/sub;v=1/a.md") == "docs/sub;v=1/a.md"


class TestEnsureSafeEndpointAllowPrivateEndpoint:
    """The Ollama allowance: admits loopback and RFC1918/ULA "private", never
    anything narrower and never anything the shape check already blocks."""

    def test_loopback_literal_is_allowed(self) -> None:
        async def boom(_host: str) -> Sequence[str]:
            raise AssertionError("resolver must not be called for a literal host")

        async def run() -> None:
            await ensure_safe_endpoint(
                "http://127.0.0.1:11434/api/embed", allow_private_endpoint=True, resolver=boom
            )

        _run(run())

    def test_rfc1918_literal_is_allowed(self) -> None:
        """SPEC.md §9's Phase 6 compose topology reaches Ollama at a
        bridge-network address — the reason the allowance is named for
        private endpoints generally, not loopback specifically."""

        async def boom(_host: str) -> Sequence[str]:
            raise AssertionError("resolver must not be called for a literal host")

        async def run() -> None:
            await ensure_safe_endpoint(
                "http://172.17.0.5:11434/api/embed", allow_private_endpoint=True, resolver=boom
            )

        _run(run())

    def test_link_local_resolved_address_is_still_rejected(self) -> None:
        """The allowance is scoped to loopback+private, not "everything
        non-global" — 169.254.169.254 is a common cloud metadata endpoint
        and must stay refused even for a provider that permits private
        endpoints."""

        async def resolver(_host: str) -> Sequence[str]:
            return ["169.254.169.254"]

        async def run() -> None:
            with pytest.raises(ConfigurationError, match="link_local"):
                await ensure_safe_endpoint(
                    "http://ollama-host.internal.example:11434/api/embed",
                    allow_private_endpoint=True,
                    resolver=resolver,
                )

        _run(run())

    def test_multicast_resolved_address_is_still_rejected(self) -> None:
        async def resolver(_host: str) -> Sequence[str]:
            return ["224.0.0.1"]

        async def run() -> None:
            with pytest.raises(ConfigurationError, match="multicast"):
                await ensure_safe_endpoint(
                    "http://ollama-host.internal.example:11434/api/embed",
                    allow_private_endpoint=True,
                    resolver=resolver,
                )

        _run(run())

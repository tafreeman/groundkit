"""URL safety helpers for validating outbound embedding-provider endpoints.

New for ADR-0014 decision 10, shaped as ``path_safety.py``'s peer: the same
``ensure_*`` naming for the raising entry point, the same
validate-at-the-boundary / raise-typed / never-coerce contract, and the same
docstring discipline of stating *why* a check is spelled as it is. Two
divergences from that peer are deliberate, not oversights:

- ``path_safety`` inlines its containment check into both public helpers
  because that duplication is the barrier pattern CodeQL recognizes for
  ``py/path-injection``. There is no comparably recognized sanitizer for
  ``py/request-forgery``, so this module's checks each live in one place
  instead of being duplicated for a static-analysis tool's benefit.
- No non-raising ``is_safe_*`` peer ships, because nothing in this codebase
  needs one — unlike ``is_within_base``, which ``path_safety`` needs for a
  boolean call site.

Everything here raises :class:`~groundkit.errors.ConfigurationError`, never a
new exception type: an outbound endpoint is operator configuration, which is
exactly what that type already means (ADR-0014 decision 10, "Alternatives
considered").

The check runs in two independent parts, at two different times:

- :func:`validate_endpoint_shape` — once, at embedder construction. No DNS.
- :func:`ensure_safe_endpoint` — per request, immediately before the POST.
  Construction-time-only checking was rejected: a service binds once and
  serves for days, which is the widest possible window between check and
  connect (SSRF via a DNS answer that changes after startup).
"""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
from collections.abc import Awaitable, Callable, Sequence
from typing import Final
from urllib.parse import urlsplit

from groundkit.errors import ConfigurationError

#: Schemes :func:`validate_endpoint_shape` accepts when the caller does not
#: override ``allowed_schemes``. Both are plain HTTP: nothing in this module
#: implements TLS-only enforcement, and file/data/other schemes have no
#: business appearing in an embedding provider's ``base_url``.
DEFAULT_ALLOWED_SCHEMES: Final[frozenset[str]] = frozenset({"http", "https"})

#: Matches a host that is shaped like dotted-quad IPv4 — four dot-separated
#: runs of digits — without regard to whether each run is a legal octet or
#: leading-zero-free. Used only to decide whether a parse failure means
#: "reject as malformed" (this pattern matched) or "treat as an ordinary
#: hostname" (it didn't). Never used to accept or classify an address; that
#: is exclusively :mod:`ipaddress`'s job (ADR-0014 decision 10: "never a
#: regex").
_LOOKS_LIKE_DOTTED_QUAD: Final[re.Pattern[str]] = re.compile(r"^\d+(\.\d+){3}$")

#: RFC 6598 "Shared Address Space" (100.64.0.0/10), carved out for
#: carrier-grade NAT. Needs its own explicit check because it is a
#: documented gap in :mod:`ipaddress`: ``IPv4Address("100.64.0.1").is_private``
#: and ``.is_global`` are **both** ``False`` for this range — the one case
#: where those two properties are not opposites, per ``IPv4Address.is_global``'s
#: own docstring. None of the six standard predicates this module otherwise
#: relies on (loopback, private, link-local, multicast, reserved,
#: unspecified) fire for it, so without this check a shared-address-space
#: target would silently classify as safe.
_SHARED_ADDRESS_SPACE: Final[ipaddress.IPv4Network] = ipaddress.IPv4Network("100.64.0.0/10")

#: The only :func:`classify_address` verdicts :func:`ensure_safe_endpoint`
#: will accept when the caller has set ``allow_private_endpoint=True``.
#: Deliberately excludes ``"link_local"``, ``"multicast"``, ``"reserved"``,
#: ``"unspecified"`` and ``"shared_address_space"`` — the allowance exists so
#: a compose bridge-network address (RFC1918, "private") and the loopback
#: default keep working, not so *every* non-global address becomes
#: reachable. 169.254.169.254 (a common cloud metadata endpoint) is
#: link-local and stays refused even for an allowed provider.
_ALLOWED_WHEN_PRIVATE_ENDPOINT_PERMITTED: Final[frozenset[str]] = frozenset({"loopback", "private"})

#: An injectable DNS resolver: given a hostname, return every address it
#: resolves to (as address-literal strings, IPv4 or IPv6). The seam
#: :func:`ensure_safe_endpoint` accepts so no unit test touches the network —
#: mirrors how ``httpx.AsyncClient`` is already injectable in
#: ``providers/embeddings.py``.
Resolver = Callable[[str], Awaitable[Sequence[str]]]


async def _default_resolver(host: str) -> Sequence[str]:
    """Resolve *host* via the system resolver, off the event-loop thread.

    ``socket.getaddrinfo`` blocks on network I/O; calling it directly from a
    coroutine would stall every other in-flight request on this process's
    single event loop. ``asyncio.to_thread`` hands it to a worker thread
    instead. Module-level (not a closure) so a test can replace it wholesale
    with ``monkeypatch.setattr`` when a call site has no resolver parameter
    of its own to inject through.

    Returns:
        Every address ``getaddrinfo`` returned, as literal strings — IPv4 and
        IPv6 both, in whatever order the resolver produced them. Duplicates
        are not collapsed; the caller classifies each one regardless.
    """
    infos = await asyncio.to_thread(socket.getaddrinfo, host, None)
    # sockaddr is tuple[str, int] for AF_INET or tuple[str, int, int, int] for
    # AF_INET6; the address is always element 0 of either shape, but mypy
    # widens indexing a tuple-shaped union, hence the explicit str().
    return [str(info[4][0]) for info in infos]


def validate_endpoint_shape(
    url: str,
    *,
    allowed_schemes: frozenset[str] = DEFAULT_ALLOWED_SCHEMES,
) -> None:
    """Validate *url*'s shape as an embedding-provider ``base_url``. No DNS.

    Meant to run once, at embedder construction — cheap, synchronous,
    entirely local. It cannot tell whether the *address* behind the host is
    safe (that needs a resolver and belongs to :func:`ensure_safe_endpoint`);
    it only rejects shapes that are wrong regardless of what the host
    resolves to.

    Args:
        url: The configured ``base_url`` to validate.
        allowed_schemes: Schemes permitted for *url*'s scheme component.

    Raises:
        ConfigurationError: If the scheme is not in *allowed_schemes*; the
            host is empty; the URL carries userinfo (``user:pass@``) of any
            kind; the URL carries a query string or fragment (this module's
            callers concatenate request paths onto ``base_url``, so a query
            or fragment there would attach to every request, not just this
            one); or the host is shaped like a dotted-quad IPv4 address but
            does not parse as one (e.g. an out-of-range octet, or a leading
            zero — ``ipaddress`` rejects those as ambiguous, so passing one
            through as an ordinary hostname would hand it to whatever
            resolver eventually sees it, with no guarantee of the same
            rejection).

            The message never echoes *url* or any component derived from
            it — the very shape being rejected (userinfo) may be carrying
            the credential this check exists to keep out of a log line
            (ADR-0001 hazard 6).
    """
    parsed = urlsplit(url)

    if parsed.scheme not in allowed_schemes:
        raise ConfigurationError(
            f"embedding endpoint scheme must be one of {sorted(allowed_schemes)}"
        )

    host = parsed.hostname
    if not host:
        raise ConfigurationError("embedding endpoint URL has no host")

    if parsed.username is not None or parsed.password is not None:
        raise ConfigurationError(
            "embedding endpoint URL must not carry userinfo — a credential "
            "belongs in api_key_env, never embedded in base_url"
        )

    if parsed.query or parsed.fragment:
        raise ConfigurationError(
            "embedding endpoint URL must not carry a query string or "
            "fragment — request paths are concatenated onto base_url, so "
            "one there would attach to every request"
        )

    if _LOOKS_LIKE_DOTTED_QUAD.match(host):
        try:
            ipaddress.ip_address(host)
        except ValueError:
            raise ConfigurationError(
                "embedding endpoint host looks like an IPv4 address but is not a valid one"
            ) from None


def classify_address(literal: str) -> str | None:
    """Classify an address literal as safe or name the predicate that rejects it.

    Uses only :mod:`ipaddress`, never a regex — address classification is
    exactly the kind of parsing regexes get subtly wrong (leading zeros,
    embedded whitespace, alternate bases), and ``ipaddress`` is the stdlib's
    own answer to that.

    IPv6 forms that embed an IPv4 address are unmapped *before*
    classification, not after: ``IPv6Address("::ffff:127.0.0.1").is_loopback``
    has been observed both ``True`` and ``False`` across CPython point
    releases (the property gained ``ipv4_mapped`` consultation at some point
    without a corresponding language-version gate), and the same is true of
    ``.is_private``. Relying on either without unmapping first would make
    this function's correctness a function of the interpreter patch version
    rather than of the address. Explicit unmapping — mapped, 6to4, then
    Teredo, in that order, first match wins — keeps the result
    version-independent and is what actually matters for the Teredo/6to4
    cases: those forms are not auto-consulted by any released CPython.

    Args:
        literal: An address that has already been confirmed to parse via
            ``ipaddress.ip_address`` (:func:`ensure_safe_endpoint` only calls
            this after that check succeeds).

    Returns:
        ``None`` if the address is safe to connect to. Otherwise the name of
        the predicate that rejected it: one of ``"loopback"``,
        ``"link_local"``, ``"multicast"``, ``"reserved"``, ``"unspecified"``,
        ``"shared_address_space"``, or ``"private"``.

        The order these are checked in is load-bearing, not cosmetic:
        ``is_private`` is checked *last* because it is a broad catch-all —
        for IPv4 it is true for the entire 169.254.0.0/16 link-local block,
        for example — and checking it first would relabel a link-local
        address as merely "private". That distinction matters because
        ``ensure_safe_endpoint``'s ``allow_private_endpoint`` allowance
        permits ``"private"`` (for RFC1918 compose-network addresses) but
        must never permit ``"link_local"`` (169.254.169.254 is a common
        cloud metadata endpoint and must stay refused regardless).
    """
    addr: ipaddress.IPv4Address | ipaddress.IPv6Address = ipaddress.ip_address(literal)

    if isinstance(addr, ipaddress.IPv6Address):
        if addr.ipv4_mapped is not None:
            addr = addr.ipv4_mapped
        elif addr.sixtofour is not None:
            addr = addr.sixtofour
        elif addr.teredo is not None:
            addr = addr.teredo[1]

    if addr.is_loopback:
        return "loopback"
    if addr.is_link_local:
        return "link_local"
    if addr.is_multicast:
        return "multicast"
    if addr.is_reserved:
        return "reserved"
    if addr.is_unspecified:
        return "unspecified"
    if isinstance(addr, ipaddress.IPv4Address) and addr in _SHARED_ADDRESS_SPACE:
        return "shared_address_space"
    if addr.is_private:
        return "private"
    return None


async def ensure_safe_endpoint(
    url: str,
    *,
    allow_private_endpoint: bool = False,
    resolver: Resolver | None = None,
) -> None:
    """Raise unless every address *url*'s host reaches is safe to connect to.

    Meant to run per request, immediately before the POST. If the host
    already parses as an address literal, it is classified directly and no
    DNS lookup happens at all — the default Ollama endpoint
    (``http://127.0.0.1:11434``) takes this path on every call, at zero
    resolution cost. Otherwise the host is resolved and **every** address
    the resolver returns is classified; any single unsafe one refuses the
    whole call. Checking only the first address would miss a resolver that
    answers with a safe address first and an unsafe one second — nothing
    stops a DNS response from doing exactly that, deliberately or not.

    Python's ``ipaddress`` deliberately rejects some spellings a C resolver
    accepts: ``ipaddress.ip_address("0177.0.0.1")`` and
    ``ipaddress.ip_address("2130706433")`` both raise (ambiguous-leading-zero
    and non-dotted-decimal hardening), so those hosts fall through to the
    "resolve via DNS" branch below as ordinary hostnames — while glibc's
    ``getaddrinfo`` accepts both and returns ``127.0.0.1``. The resolver
    step, not the literal parser, is what catches that class of address; a
    guard that only classified literals and skipped resolution for anything
    that failed to parse would let both straight through.

    Args:
        url: The full request URL (or ``base_url``) about to be connected to.
        allow_private_endpoint: When ``True``, permits ``classify_address``
            verdicts of ``"loopback"`` and ``"private"`` — the allowance
            :class:`~groundkit.providers.embeddings.OllamaEmbedder` sets via
            its ``_allow_private_endpoint`` class attribute so a local or
            compose-bridge-network Ollama instance keeps working. Every other
            verdict (link-local, multicast, reserved, unspecified, shared
            address space) is refused regardless of this flag.
        resolver: Injectable DNS resolver; defaults to :func:`_default_resolver`
            (real ``getaddrinfo``, off the event-loop thread). Tests inject a
            fake here, or monkeypatch the module-level default, so no test
            ever touches the network.

    Raises:
        ConfigurationError: If the host cannot be resolved to any address, or
            if any resolved (or literal) address is unsafe per
            :func:`classify_address` and not covered by
            *allow_private_endpoint*. The message never echoes *url*, for the
            same reason :func:`validate_endpoint_shape`'s does not.
    """
    host = urlsplit(url).hostname
    if not host:
        raise ConfigurationError("embedding endpoint URL has no host to validate")

    try:
        literal_addr = ipaddress.ip_address(host)
    except ValueError:
        literal_addr = None

    if literal_addr is not None:
        addresses: Sequence[str] = [str(literal_addr)]
    else:
        resolve = resolver if resolver is not None else _default_resolver
        addresses = await resolve(host)
        if not addresses:
            raise ConfigurationError("embedding endpoint host did not resolve to any address")

    for address in addresses:
        verdict = classify_address(address)
        if verdict is None:
            continue
        if allow_private_endpoint and verdict in _ALLOWED_WHEN_PRIVATE_ENDPOINT_PERMITTED:
            continue
        raise ConfigurationError(
            f"embedding endpoint resolves to a {verdict} address, which is not permitted"
        )

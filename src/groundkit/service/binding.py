"""Inbound bind-host classification for ``grk serve`` (ADR-0014 decision 7).

**This is the service's only access control.** ADR-0014 decision 1 makes every
Phase 4 operation read-only, which satisfies SPEC.md §7's shared-secret clause
vacuously — the set of mutating operations is empty, so the set requiring the
header is empty — and therefore **no authentication of any kind ships in Phase
4**. SPEC.md §7 also records that the SQLite index is content-bearing: a
``search`` response carries document text and absolute source paths, and
``index_status``/``list_collections`` carry collection topology. With no
credential to present and nothing to present it to, the address the process
binds is the entire boundary between that corpus and everyone else. It is
load-bearing, not a default, and this module exists so that it is enforced in
one named place rather than spelled as an argparse default that a later edit
could quietly widen.

**Polarity note — the two address classifiers in this repo disagree on purpose,
and must not be "unified".** :mod:`groundkit.utils.url_safety` **rejects**
loopback (and every other non-global address) because it guards an *outbound*
provider endpoint, where reaching loopback is SSRF. This module **requires**
loopback because it guards an *inbound* bind, where reaching anything else
publishes the corpus. Same :mod:`ipaddress` classification, opposite verdict,
because the direction of the connection is what makes an address dangerous.
Merging them would have to pick one verdict and would silently invert the other
guard.

**The bind alone was not enough, and ADR-0024 records why.** A loopback socket
is unreachable from off-box at the IP layer, but a browser can be walked onto
it: a page on any site the victim visits re-points its *own* hostname at
``127.0.0.1`` with a short-TTL answer, and every subsequent request the browser
makes to that name is same-origin — no CORS preflight, response fully readable.
The packet still comes from loopback; the *attacker* is not on the box. The only
thing that distinguishes such a request from a legitimate one is the ``Host``
header, which still carries the attacker's name, so this module also derives the
``Host`` allow-list both transports enforce (:func:`derive_host_allow_list`). It
lives here for the same reason the bind classification does — one named place,
where widening it is a visible edit rather than a default nobody reads.

**Two questions about the same address, and only one of them resolves a name.**
Whether a *bind* may proceed is decided without a resolver
(:func:`_is_loopback_literal`), because a wrong answer there hands the operator
a routable socket they were told was loopback. Whether ``Host`` validation is
*enforced* is decided with one (:func:`derive_host_allow_list`), because a wrong
answer there costs at most a refused client. That is not an inconsistency: the
two errors are not the same size, and the direction each one fails in is what
picks the rule. :func:`derive_host_allow_list` argues it in full, because the
version of this module that used one predicate for both questions turned
``Host`` validation off for ``--host localhost --allow-remote-access`` — a
non-routable socket, which is the only kind rebinding can reach.

Classification is done with :mod:`ipaddress`, never a regex: leading zeros,
alternate bases, embedded scope ids and IPv4-in-IPv6 spellings are exactly what
a regex gets subtly wrong, and the stdlib already answers this question.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
import string
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final, TypeAlias

from groundkit.errors import ConfigurationError

logger = logging.getLogger(__name__)

#: What :mod:`ipaddress` hands back, spelled once. Named only so the resolver
#: helper's return type fits on a line; it carries no meaning of its own.
_IpAddress: TypeAlias = ipaddress.IPv4Address | ipaddress.IPv6Address

#: Address ``grk serve`` binds when ``--host`` is not given. Named rather than
#: inlined into the parser because ADR-0014 decision 7 makes the value itself a
#: decision, and a named constant is what a test can pin.
DEFAULT_SERVE_HOST: Final[str] = "127.0.0.1"

#: Port ``grk serve`` binds when ``--port`` is not given. High, unprivileged,
#: and not a common default for anything else this repo's users are likely to
#: be running.
DEFAULT_SERVE_PORT: Final[int] = 8765

#: What a published bind costs, in one sentence, shared by both warnings in
#: :func:`ensure_bindable_host` so the two cannot drift into describing
#: different exposures. ASCII only, for the reason ``cli._print_eval_summary``
#: gives about the section sign: this reaches a console that may be cp1252, and
#: it is the line in this module that most needs to be read literally.
#: ``tests/test_cli_serve.py`` pins its content, not merely its presence.
_EXPOSURE: Final[str] = (
    "This server has NO AUTHENTICATION OF ANY KIND. Anyone who can reach this port can read "
    "the full document content of every indexed collection and the absolute filesystem paths "
    "those documents were ingested from."
)

#: Shared tail of both "validation is off" warnings in
#: :func:`derive_host_allow_list`, for the same anti-drift reason as
#: :data:`_EXPOSURE`.
_UNRESTRICTED_TAIL: Final[str] = (
    "Any Host value is accepted on both the REST and the MCP transport, which is correct for "
    "a bind the operator published deliberately and is no protection at all against anyone "
    "who can already route to this port."
)

#: Characters an operator-supplied *name* may contain and still be added to the
#: derived allow-list. This is not a hostname grammar and does not try to be
#: one; it is a rejection filter with a single job. :func:`_trusted_host_pattern`
#: reduces a bare ``*`` to ``*``, which Starlette reads as its allow-any
#: wildcard, and any other ``*`` makes its constructor assert — so ``--host '*'``
#: would otherwise produce an allow-list that allows everything on the REST
#: surface. A name is added even when the resolver refuses it (see
#: :func:`derive_host_allow_list`), so this filter is the only thing standing
#: between that argument and a silently disabled check.
_NAME_CHARACTERS: Final[frozenset[str]] = frozenset(string.ascii_letters + string.digits + "-._")


def _is_loopback_address(addr: _IpAddress) -> bool:
    """True when *addr* names the loopback interface.

    The one classification both callers share: :func:`_is_loopback_literal`, for
    an address the operator typed, and :func:`_resolved_addresses`' output, for
    the answers a resolver gave. Keeping it in one function is what stops the
    typed and the resolved path from disagreeing about what "loopback" means.

    Only the **IPv4-mapped** IPv6 form is unmapped before classification, and
    the asymmetry against :func:`groundkit.utils.url_safety.classify_address`
    — which also unmaps 6to4 and Teredo — follows from the polarity inversion
    described in this module's docstring. There, unmapping *widens rejection*,
    so unmapping more forms is strictly safer. Here it *widens acceptance*, so
    each unmapped form must genuinely be the same interface:
    ``::ffff:127.0.0.1`` really is a loopback bind, while a 6to4 or Teredo
    address embedding ``127.0.0.1`` is a globally-routable address that merely
    encodes one, and treating it as loopback would hand out exactly the bind
    this module exists to refuse.

    ``IPv6Address("::ffff:127.0.0.1").is_loopback`` is **False** on some CPython
    patch releases (the IPv6 loopback test is equality with ``::1``) and
    ``True`` on others, so reading ``.is_loopback`` on the un-unmapped address
    makes the verdict a function of the interpreter build. Unmapping first is
    what closes that, and ``tests/test_cli_serve.py`` pins the case.
    """
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped

    return addr.is_loopback


def _is_address_literal(host: str) -> bool:
    """True when *host* is an address literal rather than a name.

    Split out from :func:`_is_loopback_literal` because two call sites need the
    halves of that question separately: whether the value can be classified at
    all without a resolver, and how it then classifies. A literal that is *not*
    loopback and a name that is not loopback are different situations —
    :func:`ensure_bindable_host` can say what the first one publishes and cannot
    yet say it of the second — and collapsing them is what produced a warning
    that stated the opposite of what happened.
    """
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def _is_loopback_literal(host: str) -> bool:
    """True when *host* is an address literal that names the loopback interface.

    A **literal only**. A hostname — ``"localhost"`` most of all — returns
    ``False`` here, and that is a decision rather than an omission:

    - Resolving the name would make this guard's verdict depend on a mutable
      answer. ``localhost`` is loopback by convention, not by rule: a
      ``hosts`` entry or a DNS search domain can point it anywhere, and the
      operator reading "bound to localhost" would be told loopback while the
      socket sat on a routable interface.
    - Even a correct resolution would be a different fact from the one that
      matters. The bind happens later, in the ASGI server, which resolves the
      name *again* — so a name that classified as loopback here can be bound
      to another address moments later. The literal is the only value that is
      both what this function classifies and what actually gets bound.

    The refusal is cheap to work around correctly (type ``127.0.0.1`` or
    ``::1``), and the error message says so.

    This is the *bind* question. :func:`derive_host_allow_list` asks a
    differently-shaped one and does resolve; see its docstring for why the same
    reasoning does not carry over. The classification itself is
    :func:`_is_loopback_address`.

    Args:
        host: The raw ``--host`` value, unmodified.

    Returns:
        ``True`` only for an address literal whose (unmapped) form is
        loopback — ``127.0.0.0/8``, ``::1``, or ``::ffff:127.0.0.0/104``.
    """
    try:
        addr: _IpAddress = ipaddress.ip_address(host)
    except ValueError:
        return False

    return _is_loopback_address(addr)


def _resolved_addresses(host: str) -> tuple[_IpAddress, ...] | None:
    """Every address *host* resolves to right now, or ``None`` if it does not.

    ``None`` means *the resolver did not answer*, which is a different fact from
    an answer containing nothing, and :func:`derive_host_allow_list` says out
    loud which one it got. Entries that do not parse as an address are dropped
    rather than raising: the caller needs positive evidence that something is
    routable before it widens anything, so a dropped entry can only make the
    result narrower, never wider.

    An IPv6 scope id (``fe80::1%eth0``) is stripped before parsing. It names the
    interface, not the address, and no address this function classifies as
    loopback is scoped.

    Args:
        host: A hostname. Callers must have excluded address literals first —
            resolving one would only add a way to be wrong about a value that
            already answers the question exactly.
    """
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError):
        return None

    addresses: list[_IpAddress] = []
    for info in infos:
        # A `sockaddr` whose first element is not a string belongs to an address
        # family this function does not classify (typeshed types the union to
        # cover them), and one that does not parse is a spelling `ipaddress`
        # does not know. Both are skipped rather than raised on: the caller
        # widens only on positive evidence of routability, so a skipped entry
        # can make the verdict narrower and never wider.
        literal = info[4][0]
        if not isinstance(literal, str):
            continue
        try:
            addresses.append(ipaddress.ip_address(literal.partition("%")[0]))
        except ValueError:
            continue
    return tuple(addresses)


def ensure_bindable_host(host: str, *, allow_remote_access: bool) -> None:
    """Refuse a non-loopback bind unless the operator acknowledged the exposure.

    Called before anything else in ``grk serve`` — before the index directory
    is opened and before a registry exists — so a refused host costs nothing
    and touches nothing.

    The refusal is literal-only and stays that way; see
    :func:`_is_loopback_literal`. What the acknowledgement *means* is not the
    same in both cases, though, so it is not reported as though it were: a
    routable literal publishes the corpus, while a hostname defers the question
    to :func:`derive_host_allow_list`, which resolves it and logs the verdict.

    Args:
        host: The address to bind, as typed. See :func:`_is_loopback_literal`
            for why a hostname is not resolved.
        allow_remote_access: The operator's explicit acknowledgement
            (``--allow-remote-access``). It does not make the exposure safe;
            it makes it deliberate, and is answered with a warning naming
            exactly what is published — or, for a name, what would be.

    Raises:
        ConfigurationError: *host* is not a loopback literal and
            *allow_remote_access* is ``False``. No new exception type is
            introduced: a bind address is operator configuration, which is
            what :class:`~groundkit.errors.ConfigurationError` already means
            (ADR-0014 decision 9's "no new exception types").
    """
    if _is_loopback_literal(host):
        return

    if not allow_remote_access:
        # ASCII only, for the reason `_EXPOSURE` gives.
        raise ConfigurationError(
            f"refusing to bind {host!r}: it is not a loopback address literal, and this "
            "service ships no authentication of any kind, so the bind is its only access "
            "control (ADR-0014 decision 7). Bind 127.0.0.1 or ::1 instead. A hostname such as "
            "'localhost' is refused here because only an address literal can be classified "
            "without a resolver whose answer can change, and that answer decides whether a "
            "corpus is published. --allow-remote-access accepts either kind, and they are not "
            "equivalent: a routable address publishes the corpus, while a hostname is resolved "
            "at serve time and keeps Host validation enforced if every answer is loopback "
            "(ADR-0024)."
        )

    if _is_address_literal(host):
        logger.warning(
            "Binding %s, which is not a loopback address, because --allow-remote-access was "
            "passed. Passing this flag publishes the corpus. %s",
            host,
            _EXPOSURE,
        )
        return

    logger.warning(
        "Binding the hostname %s because --allow-remote-access was passed. Whether that "
        "publishes anything depends on what the name resolves to, which this guard does not "
        "ask; the serve-time Host allow-list derivation resolves it and logs the verdict "
        "(ADR-0024). If it resolves anywhere outside loopback, passing this flag publishes the "
        "corpus. %s",
        host,
        _EXPOSURE,
    )


#: ``Host`` values that name *this* machine's loopback interface, in the MCP
#: SDK's dialect: an exact string, or a ``base:*`` pattern matching any port.
#: Both spellings of each address are listed because the SDK's matcher treats
#: them as unrelated — ``base:*`` is implemented as
#: ``host.startswith(base + ":")``, so it matches ``127.0.0.1:8765`` and does
#: **not** match a bare ``127.0.0.1`` (a request to port 80, where the port is
#: omitted from ``Host``). Listing only the wildcard would refuse that request,
#: which is the failure mode where a security control gets switched off in
#: frustration rather than narrowed.
#:
#: ``localhost`` is included even though :func:`_is_loopback_literal` refuses it
#: as a *bind* address, and the two are not in conflict: a bind is a value this
#: process chooses and must be able to classify without a resolver, while a
#: ``Host`` is a value a client sends and cannot forge past a browser. A page
#: cannot make a browser send ``Host: localhost`` without the browser having
#: resolved ``localhost``, which no attacker-controlled DNS zone answers.
#:
#: These three canonical spellings are allowed regardless of which one was
#: bound. The ones that were not bound are unreachable — a request with
#: ``Host: [::1]`` cannot arrive at a socket bound to ``127.0.0.1`` — so
#: narrowing the list by bind address would add a branch that removes nothing an
#: attacker could have used.
#:
#: **They are this list's floor, not its contents, and an earlier version of
#: this comment got that wrong.** It asserted that the bound address was always
#: one of the three; :func:`_is_loopback_literal` accepts all of
#: ``127.0.0.0/8`` and ``::ffff:127.0.0.0/104``, so ``--host 127.0.0.2`` is a
#: legal bind that none of the three names. The result was a server that started
#: and then refused every legitimate client on both transports.
#: :func:`_restricted_allow_list` appends the bound address's own spellings, so
#: the tuple below is a starting point that the derivation adds to.
_LOOPBACK_HOST_PATTERNS: Final[tuple[str, ...]] = (
    "127.0.0.1",
    "127.0.0.1:*",
    "localhost",
    "localhost:*",
    "[::1]",
    "[::1]:*",
)

#: ``Origin`` values the MCP transport accepts. Same three addresses, ``http``
#: only: this server speaks plain HTTP, and a page served over ``https`` cannot
#: fetch an ``http`` endpoint at all (mixed content), so an ``https`` loopback
#: origin describes a request no browser will make.
#:
#: The SDK treats an **absent** ``Origin`` as allowed and every *present* value
#: not on this list as forbidden, so this list is what decides whether a browser
#: page can talk to the transport at all. A rebinding page's origin is its own
#: name and port, never a loopback literal, so such a request is refused here as
#: well as by the ``Host`` check — two independent refusals of one request,
#: which is why the list is populated rather than left empty to mean "no
#: browser, ever".
#:
#: A floor rather than the contents, for the same reason as
#: :data:`_LOOPBACK_HOST_PATTERNS`: a browser pointed at ``http://127.0.0.2:8765``
#: sends that as its ``Origin``, and refusing it would be the same defect one
#: header along.
_LOOPBACK_ORIGIN_PATTERNS: Final[tuple[str, ...]] = (
    "http://127.0.0.1",
    "http://127.0.0.1:*",
    "http://localhost",
    "http://localhost:*",
    "http://[::1]",
    "http://[::1]:*",
)


def _trusted_host_pattern(host_pattern: str) -> str:
    """Reduce one MCP host pattern to the value Starlette will compare against.

    The two matchers this module feeds do **not** share a dialect, and papering
    over that would produce an allow-list that is silently permissive in one of
    them:

    - The MCP SDK compares the whole ``Host`` header, port included, and
      understands a trailing ``:*``.
    - :class:`starlette.middleware.trustedhost.TrustedHostMiddleware` compares
      ``headers["host"].split(":")[0]`` — the port is stripped before the
      comparison, and no port wildcard exists or is needed.

    So the Starlette list is *derived* from the MCP list by applying Starlette's
    own reduction, rather than hand-written beside it. That is what keeps the
    two surfaces answering the same question: a spelling added to one is added
    to the other by construction, including the per-bind spellings
    :func:`_restricted_allow_list` appends.

    **The IPv6 artifact, named rather than hidden.** Starlette splits on the
    *first* colon, so ``[::1]:8765`` and a bare ``[::1]`` both reduce to
    ``"["``, and ``"["`` is therefore what its allow-list has to contain for an
    IPv6 loopback client to be served at all. The consequence is that Starlette
    also admits any other ``[::…]``-shaped ``Host``. That is a real widening,
    and it is *not* reachable by the attack this guard exists to stop: a
    bracketed address literal is never the product of a DNS answer, so a
    rebinding page cannot cause a browser to send one. It is recorded as a
    residual in ADR-0024 and in SECURITY.md rather than left for a reader to
    rediscover. It also means an IPv6 bind's appended spellings reduce onto an
    entry that is already present, so the MCP list is the surface where those
    additions do real work.

    Args:
        host_pattern: An entry of :data:`_LOOPBACK_HOST_PATTERNS`, or a
            per-bind spelling derived by :func:`_host_header_bases`.

    Returns:
        The exact string Starlette's matcher will find equal to a matching
        ``Host`` header.
    """
    return host_pattern.split(":")[0]


def _deduplicated(values: Iterable[str]) -> tuple[str, ...]:
    """Collapse repeats while keeping first-seen order.

    The reduction is many-to-one — three of the six host patterns collapse onto
    ``"["`` — and a list carrying one pattern three times would still match
    correctly while reading as though three distinct things were allowed. It is
    also what makes the per-bind spellings free when the bind *is* one of the
    canonical three: ``--host 127.0.0.1`` appends nothing new.
    """
    unique: dict[str, None] = {}
    for value in values:
        unique[value] = None
    return tuple(unique)


def _host_header_bases(host: str) -> tuple[str, ...]:
    """The portless ``Host`` values by which a client reaches *this* bind.

    A bind is not obliged to be one of the three canonical loopback spellings —
    ``127.0.0.2`` and ``::ffff:127.0.0.1`` are both loopback and neither is
    named by them — so the allow-list has to be able to name the address it was
    actually given.

    Adding the operator's own bind address is not a widening in the sense that
    matters. The value comes from this process's command line, never from a
    request, so nothing an attacker controls reaches this list; and a browser
    can only be made to send a ``Host`` it resolved, which is the property the
    whole check rests on.

    - An **address literal** is offered as typed *and* in :mod:`ipaddress`'s
      canonical rendering, bracketed for IPv6. The two differ more often than
      one expects, and the typed form is the one that matters:
      ``::ffff:127.0.0.1`` canonicalises to ``::ffff:7f00:1``, while a client
      pointed at ``http://[::ffff:127.0.0.1]:8765/`` sends the dotted form it
      was given. Listing only the canonical spelling would leave the MCP matcher
      refusing a request the REST matcher serves — which is the cross-surface
      inconsistency this function exists to remove.
    - A **name** is offered as typed *and* lowercased. DNS names are
      case-insensitive and the matchers are not (see ADR-0024's residuals), so
      canonicalising what this process emits is free; enumerating what a client
      might send is not, and is not attempted.
    - A name containing anything outside :data:`_NAME_CHARACTERS` contributes
      nothing. That is the guard against ``--host '*'``, whose reduction is
      Starlette's allow-any wildcard.

    Args:
        host: The address or name ``grk serve`` will bind, as typed.

    Returns:
        Zero or more portless ``Host`` bases; :func:`_restricted_allow_list`
        renders each into both of the SDK's spellings.
    """
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        if host and _NAME_CHARACTERS.issuperset(host):
            return _deduplicated((host, host.lower()))
        return ()

    if isinstance(addr, ipaddress.IPv6Address):
        return _deduplicated((f"[{host}]", f"[{addr}]"))
    return _deduplicated((host, str(addr)))


@dataclass(frozen=True, slots=True)
class HostAllowList:
    """One ``Host`` decision, rendered into each matcher's own dialect.

    Both transports enforce the same decision and neither re-derives it:
    :func:`groundkit.service.mcp_server.create_session_manager` reads the MCP
    fields, :func:`groundkit.service.api.create_app` reads
    :attr:`trusted_hosts`, and both read one object built in one place. An edit
    that widens one surface and forgets the other has to widen this object,
    where both readers see it.

    Attributes:
        enforced: Whether ``Host`` is checked at all. ``False`` is the
            deliberately-published mode; see :func:`derive_host_allow_list`.
        mcp_allowed_hosts: ``Host`` patterns in the MCP SDK's dialect.
        mcp_allowed_origins: ``Origin`` patterns in the MCP SDK's dialect.
        trusted_hosts: ``allowed_hosts`` for Starlette's
            ``TrustedHostMiddleware``, derived from *mcp_allowed_hosts* by
            :func:`_trusted_host_pattern`.
    """

    enforced: bool
    mcp_allowed_hosts: tuple[str, ...]
    mcp_allowed_origins: tuple[str, ...]
    trusted_hosts: tuple[str, ...]


def _restricted_allow_list(host: str) -> HostAllowList:
    """The enforced allow-list for a bind only this machine can reach.

    The canonical loopback spellings, plus the spellings of the address actually
    bound (:func:`_host_header_bases`). Both dialects are built from the one
    list, Starlette's by reduction, so a per-bind addition cannot land on one
    surface and be forgotten on the other — which is precisely how
    ``--host ::ffff:127.0.0.1`` came to be served by the REST matcher (via the
    ``"["`` residual) and refused by the MCP one.

    Args:
        host: The address or name ``grk serve`` will bind, as typed.
    """
    bases = _host_header_bases(host)
    hosts = _deduplicated(
        (
            *_LOOPBACK_HOST_PATTERNS,
            *(pattern for base in bases for pattern in (base, f"{base}:*")),
        )
    )
    origins = _deduplicated(
        (
            *_LOOPBACK_ORIGIN_PATTERNS,
            *(pattern for base in bases for pattern in (f"http://{base}", f"http://{base}:*")),
        )
    )
    return HostAllowList(
        enforced=True,
        mcp_allowed_hosts=hosts,
        mcp_allowed_origins=origins,
        trusted_hosts=_deduplicated(_trusted_host_pattern(pattern) for pattern in hosts),
    )


#: The allow-list for the **default** loopback bind, named because
#: ``cli._build_mcp_mount`` defaults to it and tests pin it. It is exactly what
#: :func:`_restricted_allow_list` derives for :data:`DEFAULT_SERVE_HOST` rather
#: than a second hand-written copy — a bind to any other loopback address
#: derives a superset of it, and equality against this constant is therefore a
#: statement about the default bind, not about every restricted bind.
LOOPBACK_HOST_ALLOW_LIST: Final[HostAllowList] = _restricted_allow_list(DEFAULT_SERVE_HOST)

#: The allow-list for a bind the operator has explicitly published: every
#: ``Host`` is accepted. Named, so that every call site says which mode it is
#: in, rather than a bare ``None`` leaving a reader to work out what "no
#: allow-list" was meant to mean.
UNRESTRICTED_HOST_ALLOW_LIST: Final[HostAllowList] = HostAllowList(
    enforced=False,
    mcp_allowed_hosts=(),
    mcp_allowed_origins=(),
    trusted_hosts=("*",),
)


def derive_host_allow_list(host: str) -> HostAllowList:
    """Derive the ``Host`` allow-list that goes with a bind address.

    Called *after* :func:`ensure_bindable_host`, which has already refused every
    non-loopback-literal address the operator did not acknowledge with
    ``--allow-remote-access``. The bind address therefore already carries the
    flag's answer, which is why one parameter is sufficient.

    **This function resolves a hostname and :func:`_is_loopback_literal`
    refuses to, and that asymmetry is the design rather than a contradiction.**
    Both are asking whether the socket is reachable only from this machine, and
    a resolver's answer to that is mutable either way — the ASGI server resolves
    the name again moments later, so the address seen here need not be the
    address bound. What differs is the cost of being wrong, and it is not
    symmetric:

    - For the **bind**, being wrong publishes a corpus. Resolution says
      loopback, the socket lands on a routable interface, and the operator was
      told the opposite by the guard that exists to tell them. So the bind
      classifies literals only.
    - For the **allow-list**, the two errors are different sizes. Resolution
      says loopback while the socket is really routable: the list is merely too
      *narrow*, legitimate clients get a 400 or a 421, and the operator sees it
      on their first request. Resolution says routable while the socket is
      really loopback: ``Host`` validation is switched off on the one kind of
      bind DNS rebinding can reach, which is the CRITICAL this module exists to
      close.

    Resolution errs closed here and would err open there. That is the whole
    reason it is used here and refused there, and it also settles what an
    **unresolvable** name does: it stays restricted. A name nobody can resolve
    cannot be shown to be routable, and going unrestricted requires positive
    evidence of routability, not merely the absence of evidence against it. No
    second error message is invented for that case either — the bind that
    follows resolves the same name through the same resolver and fails there,
    with the ASGI server's own words.

    **Address literals are never resolved.** A literal is both what this
    function classifies and what actually gets bound, so a resolver could only
    add a way to be wrong.

    **A routable bind is unrestricted, and that is the decision rather than an
    omission** (ADR-0024 decision 3). Once the socket is genuinely routable —
    every routable literal, ``0.0.0.0`` and ``::`` included, and every name that
    resolves onto one — ``Host`` validation protects nothing: rebinding exists
    to bridge a browser onto a service the attacker cannot route to, and an
    attacker who *can* route to it simply connects, to a service that ships no
    authentication either way. A restrictive list would also be wrong in every
    deployment that has one, since the service is reached through whatever name
    the operator's clients, reverse proxy or overlay network use and this
    process cannot know it. Refusing those requests would not add a boundary; it
    would make the flag unusable and push operators onto a fork with no check at
    all.

    The tighter alternative — accept the bound address plus loopback — is
    recorded in ADR-0024 and rejected there: it breaks the reverse-proxy
    deployment, which is the one way an operator can put authentication in front
    of this service.

    Args:
        host: The address ``grk serve`` will bind, as typed, already accepted by
            :func:`ensure_bindable_host`.

    Returns:
        An enforced :func:`_restricted_allow_list` when the bind is loopback, or
        cannot be shown not to be; :data:`UNRESTRICTED_HOST_ALLOW_LIST` when it
        is routable.
    """
    if _is_address_literal(host):
        if _is_loopback_literal(host):
            return _restricted_allow_list(host)
        logger.warning(
            "Host header validation is DISABLED because %s is a routable address. %s",
            host,
            _UNRESTRICTED_TAIL,
        )
        return UNRESTRICTED_HOST_ALLOW_LIST

    resolved = _resolved_addresses(host)
    if resolved is None:
        logger.warning(
            "Host header validation stays ENFORCED for the hostname %s: it does not resolve "
            "here, so it cannot be shown to reach anything outside loopback, and this "
            "derivation fails closed (ADR-0024). If the bind succeeds anyway and the address "
            "is in fact routable, legitimate clients will be refused with 400/421 -- bind an "
            "address literal instead.",
            host,
        )
        return _restricted_allow_list(host)

    answers = ", ".join(str(address) for address in resolved)
    if any(not _is_loopback_address(address) for address in resolved):
        logger.warning(
            "Host header validation is DISABLED because the hostname %s resolves to %s, which "
            "is not confined to loopback. %s",
            host,
            answers,
            _UNRESTRICTED_TAIL,
        )
        return UNRESTRICTED_HOST_ALLOW_LIST

    allow_list = _restricted_allow_list(host)
    logger.warning(
        "Host header validation stays ENFORCED: the hostname %s resolves only to %s, so "
        "--allow-remote-access widened the bind guard but published nothing. This socket is "
        "still reachable only from this machine, which is exactly the case DNS rebinding "
        "attacks, so the check is kept on and only this machine's own names are accepted: %s. "
        "Bind a routable address if you meant to publish the corpus.",
        host,
        answers,
        ", ".join(allow_list.mcp_allowed_hosts),
    )
    return allow_list

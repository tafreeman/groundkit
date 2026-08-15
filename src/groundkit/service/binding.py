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

Classification is done with :mod:`ipaddress`, never a regex: leading zeros,
alternate bases, embedded scope ids and IPv4-in-IPv6 spellings are exactly what
a regex gets subtly wrong, and the stdlib already answers this question.
"""

from __future__ import annotations

import ipaddress
import logging
from typing import Final

from groundkit.errors import ConfigurationError

logger = logging.getLogger(__name__)

#: Address ``grk serve`` binds when ``--host`` is not given. Named rather than
#: inlined into the parser because ADR-0014 decision 7 makes the value itself a
#: decision, and a named constant is what a test can pin.
DEFAULT_SERVE_HOST: Final[str] = "127.0.0.1"

#: Port ``grk serve`` binds when ``--port`` is not given. High, unprivileged,
#: and not a common default for anything else this repo's users are likely to
#: be running.
DEFAULT_SERVE_PORT: Final[int] = 8765


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

    ``IPv6Address("::ffff:127.0.0.1").is_loopback`` is **False** (the IPv6
    loopback test is equality with ``::1``), so reading ``.is_loopback`` on the
    un-unmapped address admits that exact spelling. Unmapping first is what
    closes it, and ``tests/test_cli_serve.py`` pins the case.

    Args:
        host: The raw ``--host`` value, unmodified.

    Returns:
        ``True`` only for an address literal whose (unmapped) form is
        loopback — ``127.0.0.0/8``, ``::1``, or ``::ffff:127.0.0.0/104``.
    """
    try:
        addr: ipaddress.IPv4Address | ipaddress.IPv6Address = ipaddress.ip_address(host)
    except ValueError:
        return False

    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped

    return addr.is_loopback


def ensure_bindable_host(host: str, *, allow_remote_access: bool) -> None:
    """Refuse a non-loopback bind unless the operator acknowledged the exposure.

    Called before anything else in ``grk serve`` — before the index directory
    is opened and before a registry exists — so a refused host costs nothing
    and touches nothing.

    Args:
        host: The address to bind, as typed. See :func:`_is_loopback_literal`
            for why a hostname is not resolved.
        allow_remote_access: The operator's explicit acknowledgement
            (``--allow-remote-access``). It does not make the exposure safe;
            it makes it deliberate, and is answered with a warning naming
            exactly what is published.

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
        # ASCII only, for the reason `cli._print_eval_summary` gives about the
        # section sign: this string is printed to a console that may be cp1252,
        # and it is the one line in this module that most needs to be read
        # literally. The same applies to the warning below.
        raise ConfigurationError(
            f"refusing to bind {host!r}: it is not a loopback address, and this service "
            "ships no authentication of any kind, so the bind is its only access control "
            "(ADR-0014 decision 7). Bind 127.0.0.1 or ::1 instead. A hostname such as "
            "'localhost' is refused here because only an address literal can be "
            "classified without a resolver whose answer can change. To publish this "
            "corpus deliberately, pass --allow-remote-access."
        )

    logger.warning(
        "Binding %s, which is not a loopback address, because --allow-remote-access was "
        "passed. This server has NO AUTHENTICATION OF ANY KIND. Anyone who can reach this "
        "port can read the full document content of every indexed collection and the "
        "absolute filesystem paths those documents were ingested from. Passing this flag "
        "publishes the corpus.",
        host,
    )

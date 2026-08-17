"""Credential redaction on any URL that reaches a message, log or client field.

An automated review of ADR-0016 Wave 4 found the raw ingested URL interpolated
verbatim into six exception/log sites in ``ingestion/url_loader.py`` and three in
``retrieval/citations.py``. Both halves matter and they leak to different
audiences: the loader's messages reach operator logs, while the citation ones are
forwarded by ``handle_fetch_chunk`` as ``ChunkFetchResponse.detail`` — **a
client-visible field on the MCP/REST surface.**

``grk ingest "https://svc:hunter2@internal.example.com/export?token=sk-live-1"`` is
an ordinary way to reach an authenticated internal resource, and every one of a
404, a size-cap overflow, a refused redirect, a non-UTF-8 body or a connection
error would have echoed those credentials.

These tests pin the redaction rather than the prose: they assert the *secret* is
absent and the *host* is still present, so a message may be reworded freely but
cannot start leaking again.
"""

from __future__ import annotations

import pytest

from groundkit.utils.url_safety import REDACTED_PLACEHOLDER, sanitize_url

#: Credentials planted in a URL. Neither may appear in any rendered output.
#: The S105 suppressions below are the opposite of what that rule guards: these
#: are sentinels asserted ABSENT from every rendered message, so a real secret
#: here would defeat the test rather than constitute one.
_PASSWORD = "hunter2-must-never-be-logged"  # noqa: S105
_TOKEN = "sk-live-must-never-be-logged"  # noqa: S105

_CREDENTIALED_URL = f"https://svc:{_PASSWORD}@internal.example.com/export?token={_TOKEN}"


def test_userinfo_password_is_redacted() -> None:
    """The netloc rebuild is what closes this.

    Reverting ``sanitize_url`` to the obvious
    ``urlunsplit((scheme, parsed.netloc, ...))`` carries ``user:password@``
    through untouched — ``netloc`` is the raw authority, and redacting the query
    does nothing to it. This test fails against that version.
    """
    sanitized = sanitize_url(_CREDENTIALED_URL)
    assert _PASSWORD not in sanitized
    assert REDACTED_PLACEHOLDER in sanitized


def test_query_parameter_values_are_redacted_unconditionally() -> None:
    """Not only when they match a known secret.

    The function is never told that ``token`` is sensitive; it redacts every
    query value regardless, because the next credential-bearing parameter name
    is one nobody added to a list.
    """
    sanitized = sanitize_url(_CREDENTIALED_URL)
    assert _TOKEN not in sanitized


def test_the_endpoint_is_still_identifiable() -> None:
    """Redaction that removed the host would defeat the message's purpose.

    Scheme, host and path survive so an operator can still tell *which*
    endpoint failed — the reason the URL is in the message at all.
    """
    sanitized = sanitize_url(_CREDENTIALED_URL)
    assert "internal.example.com" in sanitized
    assert sanitized.startswith("https://")
    assert "/export" in sanitized


def test_an_ipv6_literal_stays_a_valid_authority() -> None:
    """``urlsplit().hostname`` strips the brackets; the rebuild must restore them.

    Without that, an IPv6 URL sanitizes into something that is no longer a
    parseable authority.
    """
    sanitized = sanitize_url(f"https://user:{_PASSWORD}@[2001:db8::1]:8443/x")
    assert "[2001:db8::1]:8443" in sanitized
    assert _PASSWORD not in sanitized


def test_an_explicit_secret_is_scrubbed_from_what_remains() -> None:
    """The belt-and-braces pass for a credential that is not in the userinfo
    or a query value — e.g. one embedded in the path."""
    sanitized = sanitize_url(f"https://api.example.com/v1/{_TOKEN}/x", _TOKEN)
    assert _TOKEN not in sanitized


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/doc",
        "http://example.com:8080/a/b?x=1",
        "https://example.com",
    ],
)
def test_a_credential_free_url_survives_recognisably(url: str) -> None:
    """Redaction must not mangle the ordinary case into something unreadable."""
    sanitized = sanitize_url(url)
    assert "example.com" in sanitized
    assert sanitized.startswith(("http://", "https://"))

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

from groundkit.utils.url_safety import (
    REDACTED_PLACEHOLDER,
    credential_query_params,
    sanitize_url,
)

#: Credentials planted in a URL. Neither may appear in any rendered output.
#: The S105 suppressions below are the opposite of what that rule guards: these
#: are sentinels asserted ABSENT from every rendered message, so a real secret
#: here would defeat the test rather than constitute one.
#:
#: `gitleaks:allow` for the same reason, one scanner over: gitleaks' default
#: `generic-api-key` rule matched `_PASSWORD` on shape alone and failed the
#: `secrets` job. Both lines carry the marker rather than only the one that
#: tripped, because `sk-live-` is a live-key prefix by construction and a
#: future ruleset flagging it would fail CI on a value that is, by this file's
#: whole design, not a secret. A repo-level `.gitleaks.toml` allowlist was the
#: alternative; an inline marker keeps the exemption next to the reason for it
#: and cannot silently widen to cover a real leak elsewhere in the tree.
_PASSWORD = "hunter2-must-never-be-logged"  # noqa: S105  # gitleaks:allow
_TOKEN = "sk-live-must-never-be-logged"  # noqa: S105  # gitleaks:allow

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


# ── The persisted half: a credential in the query string reaches storage ────
#
# Everything above pins redaction in *transient* output — a message, a log
# line, a client-visible `detail`. An audit of this file found it proved only
# that half of its own stated threat model. The URL it names in its module
# docstring carries the credential twice: `svc:hunter2@` in the userinfo, and
# `?token=sk-live-1` in the query. `_reject_unsafe_url_shape` refused the
# first before the fetch; the second was permitted, and `UrlLoader.load()`
# then wrote it verbatim into `Document.source` — which is durable
# (`documents.source`), re-served in every `RetrievalResult` and `Citation`,
# and logged by the indexer on later runs. Redacting messages cannot reach it,
# because the leak is the stored value.
#
# These tests pin the refusal. They would FAIL against the pre-fix loader,
# which returned a Document for every URL below.


def test_a_query_string_credential_is_refused_before_any_fetch() -> None:
    """The whole finding, in one assertion.

    The pre-fix loader fetched this URL and returned a Document whose
    ``source`` was the string below, token included.
    """
    assert credential_query_params(_CREDENTIALED_URL) == ("token",)


@pytest.mark.parametrize(
    "name",
    [
        "token",
        "api_key",
        "api-key",
        "apikey",
        "APIKey",
        "key",
        "access_token",
        "secret",
        "client_secret",
        "password",
        "sig",
        "signature",
        "Authorization",
    ],
)
def test_every_credential_spelling_is_caught(name: str) -> None:
    """Case and separator folding is load-bearing, not cosmetic: Azure spells
    it ``api-key``, a Python client ``api_key``, and a URL builder may upper
    the first letter. One of those slipping through leaks as badly as no
    check at all."""
    assert credential_query_params(f"https://h/x?{name}={_TOKEN}") == (name,)


def test_the_offending_name_is_reported_as_the_caller_spelled_it() -> None:
    """The refusal message quotes these back, so normalizing them would show
    the caller a parameter they did not write."""
    assert credential_query_params(f"https://h/x?API-Key={_TOKEN}&id=42") == ("API-Key",)


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/doc?id=42",
        "https://example.com/doc?page=2&format=raw",
        "https://example.com/search?q=hello+world",
        "https://example.com/doc",
        "https://example.com/a?code=US&state=CA",
    ],
)
def test_an_ordinary_document_url_is_still_accepted(url: str) -> None:
    """Over-refusal is the failure mode the permit-query-strings decision
    exists to avoid, so it gets a test of its own.

    ``?code=`` and ``?state=`` are the sharp case: both are OAuth credentials
    in one context and a country/US state in another. They are deliberately
    absent from the denylist, and this pins that so a later "be thorough"
    edit has to argue with a red test rather than quietly reject real pages.
    """
    assert credential_query_params(url) == ()


def test_refusal_rather_than_redaction_is_what_keeps_documents_distinct() -> None:
    """The reason this is a refusal and not a `sanitize_url` call on the way
    into storage.

    ``documents.source`` is ``TEXT UNIQUE NOT NULL``. ``sanitize_url`` redacts
    every query value unconditionally — so sanitizing before storage maps two
    genuinely different documents onto one identity, and the second ingest
    overwrites the first. This asserts the collision that fix would have
    caused, so nobody re-proposes it without meeting it.
    """
    assert sanitize_url("https://h/doc?id=42") == sanitize_url("https://h/doc?id=43")

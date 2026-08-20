"""URL loader: fetches a remote resource and stores a local snapshot of it
(ADR-0016 decision 4; Wave 4 of
``docs/specs/loaders-extracted-and-remote-sources.md``, §10).

A ``snapshot`` document exists because a citation into a remote resource
cannot be verified against a re-fetch -- the fetch is a different
observation at a different time, so a mismatch could not distinguish "the
index is stale" from "the server changed" from "the network lied" (ADR-0016
decision 4). So this loader stores the fetched, decoded (and, for
HTML-shaped responses, tag-stripped -- ADR-0016 decision 3, extended to
snapshots by spec §10.2) text under a per-collection snapshot directory, and
returns a :class:`~groundkit.contracts.Document` whose ``source`` is the URL
(provenance only) and whose ``content`` is the exact string also written to
disk. ``retrieval.citations.resolve_citation``'s ``snapshot`` branch (Wave
4's other half, not owned by this module) reads that same file back to
verify a citation later -- never by re-fetching.

Three non-negotiable constraints, ADR-0016 decision 4 and spec §10.3:

- :func:`~groundkit.utils.url_safety.ensure_safe_endpoint` guards every
  fetch, immediately before the request (not once at construction -- DNS can
  change in between), with ``allow_private_endpoint=False`` unconditionally.
  The Ollama private-endpoint allowance
  (``_HttpEmbedder._allow_private_endpoint``) is a provider-side ``ClassVar``
  this module never reads and has no way to reach.
- Redirects are refused, not silently followed:
  ``httpx.AsyncClient(follow_redirects=False)``, and a 3xx status is an
  :class:`~groundkit.errors.IngestionError` -- the destination is never
  requested.
- The response is bounded by :data:`DEFAULT_MAX_BYTES` and refused, never
  truncated, past it. Truncating would silently produce offsets into a
  partial document, corrupting every citation into it -- the same reasoning
  :class:`~groundkit.ingestion.loaders.FileLoader` already applies to an
  oversized local file.

Not this module's job (owned by whichever caller wires this loader into
``grk ingest`` -- spec §9.6 and §10.3 both name these explicitly
undecided): how a URL reaches this loader from the CLI, and where
``snapshot_dir`` -- the per-collection containment root this loader writes
under -- is computed from
(``snapshots.snapshot_dir_for(index_dir, collection)`` once that shared
module exists; spec §10.1).
"""

from __future__ import annotations

import asyncio
import errno
import logging
import os
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from groundkit import extraction
from groundkit.contracts import Document
from groundkit.errors import IngestionError
from groundkit.utils.path_safety import ensure_within_base
from groundkit.utils.url_safety import (
    credential_query_params,
    ensure_safe_endpoint,
    sanitize_url,
)

logger = logging.getLogger(__name__)

#: Response byte cap. Matches ``FileLoader.DEFAULT_MAX_BYTES`` so a remote
#: resource and a local file are bounded by the same default -- refused past
#: it, never truncated.
DEFAULT_MAX_BYTES: int = 10 * 1024 * 1024

#: Total wall-clock bound on one fetch, in seconds -- connect, headers, and
#: every body read together, not per operation. Mirrors
#: ``EmbeddingConfig.timeout_seconds`` (``config.py``): same default, same
#: strictly-positive invariant, spelled as a constructor argument because this
#: loader takes no config object of its own (neither does ``FileLoader``).
#:
#: httpx's own ``timeout=`` is per operation, so a server that dribbles one
#: byte just inside the read timeout holds the connection indefinitely without
#: ever tripping it -- the bound the caller thought it set is not the bound it
#: got. Only a wall-clock bound around the whole exchange closes that, which is
#: why this is enforced with ``asyncio.timeout`` rather than handed to httpx.
DEFAULT_TIMEOUT_SECONDS: float = 30.0

#: ``os.O_NOFOLLOW`` where the platform defines it, ``0`` (a no-op in a flag
#: mask) where it does not. Windows has no such flag, and CI runs Linux while
#: this repo is developed on Windows, so the guard must degrade rather than
#: raise ``AttributeError`` at import on the maintainer's own machine. The
#: consequence is stated plainly rather than hidden: on Windows the snapshot
#: write is exactly as racy as it was before, and the refusal below can only
#: fire where the flag exists.
_O_NOFOLLOW: int = getattr(os, "O_NOFOLLOW", 0)

#: ``os.O_BINARY`` on Windows, ``0`` elsewhere. Required because ``os.open``
#: leaves the descriptor in the C runtime's default (text) mode on Windows,
#: which would translate ``\n`` on write *underneath* the text wrapper doing
#: the same -- turning one newline into ``\r\r\n``. With this flag and
#: ``newline=""`` the file is byte-for-byte ``content.encode("utf-8")`` on
#: every platform, which is what this module's docstring already promises
#: ("``content`` is the exact string also written to disk") and what the
#: citation offsets are measured against.
_O_BINARY: int = getattr(os, "O_BINARY", 0)

#: Errnos a POSIX ``open(..., O_NOFOLLOW)`` reports when the final path
#: component is a symbolic link. POSIX specifies ``ELOOP``; some BSDs have
#: historically used ``EMLINK`` for this case alone, so both are treated as
#: the refusal rather than as an unrelated I/O failure.
_SYMLINK_ERRNOS: frozenset[int] = frozenset({errno.ELOOP, errno.EMLINK})

#: Content-Type media types (the part before any ";" parameter) treated as
#: HTML-shaped, triggering the identical tag-stripping step Wave 3's
#: ``HtmlExtractor`` already applies to a local ``.html`` file (ADR-0016
#: decision 3; spec §10.2's "regardless of where the HTML came from" --
#: decision 3's "BM25 scores <div> as a term" problem is exactly as real for
#: a fetched page as for a local file). Anything else is stored as decoded
#: plain text with no extraction step.
_HTML_MEDIA_TYPES: frozenset[str] = frozenset({"text/html", "application/xhtml+xml"})


def _is_html_media_type(content_type: str | None) -> bool:
    """True when *content_type* names an HTML-shaped media type.

    Only the media type (before any ``;`` parameter, e.g. ``charset=...``) is
    compared, case-insensitively.
    """
    if not content_type:
        return False
    media_type = content_type.split(";", 1)[0].strip().lower()
    return media_type in _HTML_MEDIA_TYPES


def _decode(body: bytes, charset: str | None, *, source: str) -> str:
    """Decode *body* under *charset* (or UTF-8), never substituting.

    ``httpx.Response.text`` is deliberately not used anywhere in this module:
    it decodes with ``errors="replace"`` (``httpx._decoders.TextDecoder``),
    which silently substitutes U+FFFD for any byte sequence that does not fit
    the detected encoding rather than raising -- exactly the "corrupt the
    offset space" failure mode this loader must fail closed on instead of
    (mirrors ``FileLoader``'s and ``HtmlExtractor``'s own UTF-8-strict reads).

    Raises:
        IngestionError: *body* cannot be decoded under the resolved encoding,
            or the declared charset name is not one Python knows. The message
            never includes *body* or any substring of it, only the exception
            type name -- ``UnicodeDecodeError``'s own ``str()`` can echo a
            byte value and position, which is diagnostic metadata, not
            fetched content, but this module treats fetched bytes as
            categorically unloggable rather than drawing that line itself.
    """
    encoding = charset or "utf-8"
    try:
        return body.decode(encoding, errors="strict")
    except (UnicodeDecodeError, LookupError) as exc:
        raise IngestionError(
            f"{sanitize_url(source)!r} is not valid {encoding!r} text ({type(exc).__name__})"
        ) from exc


def _reject_unsafe_url_shape(source: str) -> None:
    """Refuse a URL whose *shape* is unsafe, before any DNS work happens.

    Deliberately **not** :func:`~groundkit.utils.url_safety.validate_endpoint_shape`,
    and the difference matters. That helper also rejects any query string or
    fragment, which is right for a provider's fixed ``base_url`` (request paths
    get concatenated onto it) and wrong here: ``?id=42`` is completely ordinary
    in a document URL, and refusing it would reject a large share of real pages.
    This checks only what is genuinely unsafe for an ingest source.

    **Userinfo is refused rather than stripped.** ``https://user:pass@host/x``
    is a credential the caller put in a locator, and this loader records
    ``source`` on the ``Document`` verbatim for provenance (ADR-0016 decision 4)
    — so accepting it writes the password into SQLite, into every ``Citation``
    built from that document, and into any client that fetches one. Redacting
    it in *messages* (which this module does) does not help, because the leak is
    the persisted value, not the log line. Stripping it silently would instead
    send an unauthenticated request the caller believed was authenticated, and
    then store a URL that never worked. Refusing names the problem at the point
    the caller can still fix it.

    **A credential-shaped query parameter is refused for the same reason.**
    Permitting query strings in general is still right — ``?id=42`` is ordinary
    in a document URL — but ``?token=sk-live-…`` is the userinfo leak wearing a
    different hat, and it reached storage for exactly as long as this function
    checked only the authority. Which names count is
    :data:`~groundkit.utils.url_safety.CREDENTIAL_QUERY_PARAMS`; why this is a
    refusal rather than a redaction (``Document.source`` is a UNIQUE identity,
    and ``sanitize_url`` would collapse ``?id=42`` with ``?id=43``) is argued on
    :func:`~groundkit.utils.url_safety.credential_query_params`.

    Args:
        source: The URL to check.

    Raises:
        IngestionError: The scheme is not http/https, the host is empty, the
            URL carries userinfo, or a query parameter names a credential.
    """
    parsed = urlsplit(source)
    if parsed.scheme not in ("http", "https"):
        raise IngestionError(
            f"refusing to fetch {sanitize_url(source)!r}: scheme must be http or https, "
            f"got {parsed.scheme!r}"
        )
    if not parsed.hostname:
        raise IngestionError(f"refusing to fetch {sanitize_url(source)!r}: no host")
    if parsed.username is not None or parsed.password is not None:
        raise IngestionError(
            "refusing to fetch a URL carrying userinfo credentials "
            f"({sanitize_url(source)!r}): this loader records the URL verbatim as "
            "Document.source for provenance, so the credential would be persisted "
            "to the index and returned in every citation built from it. Supply "
            "credentials out of band rather than in the locator."
        )
    if offenders := credential_query_params(source):
        named = ", ".join(repr(name) for name in offenders)
        raise IngestionError(
            f"refusing to fetch a URL carrying a credential in its query string "
            f"({sanitize_url(source)!r}): the parameter(s) {named} name a secret, "
            "and this loader records the URL verbatim as Document.source for "
            "provenance — so the value would be persisted to the index, returned "
            "in every citation built from it, and written to the ingest log. "
            "Supply credentials out of band rather than in the locator."
        )


class UrlLoader:
    """Fetches one URL and returns it as a ``source_class="snapshot"`` Document.

    Satisfies :class:`~groundkit.ingestion.protocols.LoaderProtocol`
    structurally. ``supported_extensions`` is always ``[]`` -- a URL is
    routed to this loader by shape (scheme), not by file extension, and how
    that routing decision is made at the CLI layer is explicitly out of this
    spec's scope (§9.6/§10.3's "not decided here").

    Args:
        snapshot_dir: The per-collection containment root fetched content is
            written under (spec §10.1 -- the caller computes this, mirroring
            how :class:`~groundkit.ingestion.loaders.FileLoader` takes an
            already-resolved ``allowed_base_dir`` rather than deriving one
            internally). Its parent directories are created on first write if
            they do not already exist.
        max_bytes: Maximum response body size in bytes; a larger response is
            refused rather than read partway and truncated.
        timeout_seconds: Total wall-clock bound on one fetch (see
            :data:`DEFAULT_TIMEOUT_SECONDS`). Must be greater than zero;
            zero or negative is a :class:`ValueError` at construction rather
            than a bound that silently means "immediately" or "never".
        client: Optional pre-built ``httpx.AsyncClient`` -- the seam tests
            replace with one wired to ``httpx.MockTransport``, the same
            pattern ``providers/embeddings.py`` already uses. When omitted, a
            client is built per call with ``follow_redirects=False`` and
            closed after use; when supplied, the caller owns its lifecycle
            (this loader never closes an injected client).
    """

    def __init__(
        self,
        snapshot_dir: Path,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {timeout_seconds!r}")
        self._snapshot_dir = snapshot_dir.resolve()
        self._max_bytes = max_bytes
        self._timeout_seconds = timeout_seconds
        self._client = client

    @property
    def supported_extensions(self) -> list[str]:
        """Always empty -- see the class docstring."""
        return []

    @property
    def snapshot_dir(self) -> Path:
        """Resolved containment root snapshots are written under."""
        return self._snapshot_dir

    async def load(self, source: str) -> list[Document]:
        """Fetch *source*, snapshot it, and return zero or one Document.

        An empty or whitespace-only fetched body (before or after HTML
        extraction) returns ``[]`` (logged as a warning), matching
        :class:`~groundkit.ingestion.loaders.FileLoader`'s treatment of an
        empty file rather than letting ``Document``'s ``min_length=1``
        content validator raise a bare :class:`~pydantic.ValidationError`.

        Args:
            source: The URL to fetch. Its resolved address (after DNS, if it
                is not already a literal) must be publicly routable -- see
                :func:`~groundkit.utils.url_safety.ensure_safe_endpoint`.

        Returns:
            A list containing zero or one :class:`Document` with
            ``source_class="snapshot"``, ``source=source``, ``extractor=None``,
            and ``content`` equal, character for character, to what was
            written to the local snapshot file.

        Raises:
            ConfigurationError: *source*'s resolved address is not safe to
                connect to (:func:`~groundkit.utils.url_safety.ensure_safe_endpoint`),
                or the response is HTML-shaped and the ``html`` extra is not
                installed (:func:`~groundkit.extraction.html_extractor`).
            IngestionError: The response is a redirect (3xx), an error status
                (4xx/5xx), exceeds ``max_bytes``, took longer than
                ``timeout_seconds`` in total, cannot be decoded under its
                declared (or default UTF-8) charset, the transport itself
                failed, or the resulting snapshot path would escape
                ``snapshot_dir`` (or turned into a symlink after that check).
        """
        # Guard immediately before the request, not at construction -- this
        # loader has no fixed endpoint the way an embedder's base_url is
        # fixed, and every call targets a different host. allow_private_endpoint
        # is unconditionally False: the Ollama allowance is unreachable from
        # here by construction, not merely by convention.
        _reject_unsafe_url_shape(source)
        await ensure_safe_endpoint(source, allow_private_endpoint=False)

        owns_client = self._client is None
        client = (
            self._client if self._client is not None else httpx.AsyncClient(follow_redirects=False)
        )
        try:
            # A *total* bound, deliberately not httpx's per-operation
            # `timeout=`: see DEFAULT_TIMEOUT_SECONDS. It covers the whole
            # exchange -- connect, status, headers, and every body read --
            # because the failure mode being closed is a server that stays
            # just inside each per-operation deadline forever.
            async with asyncio.timeout(self._timeout_seconds):
                body, content_type, charset = await self._fetch(client, source)
        except TimeoutError as exc:
            raise IngestionError(
                f"fetching {sanitize_url(source)!r} exceeded the total "
                f"{self._timeout_seconds}-second request bound; abandoned rather "
                "than left holding the connection"
            ) from exc
        finally:
            if owns_client:
                await client.aclose()

        text = _decode(body, charset, source=source)

        if _is_html_media_type(content_type):
            text = await self._extract_html(text)

        if not text.strip():
            logger.warning("Empty or whitespace-only URL source: %s", sanitize_url(source))
            return []

        document = Document(
            source=source,
            content=text,
            source_class="snapshot",
            metadata={"content_type": content_type} if content_type else {},
        )
        # Dispatched rather than called inline: the snapshot is content-sized
        # (up to ``max_bytes``), and every other content-sized read or write in
        # this package already runs off the loop. Nothing else is scheduled on
        # this loop today -- URL ingestion is CLI-only -- but this is exactly
        # the code a service-side ingest tool would reuse verbatim.
        await asyncio.to_thread(self._write_snapshot, document.document_id, text)
        return [document]

    async def _fetch(
        self, client: httpx.AsyncClient, source: str
    ) -> tuple[bytes, str | None, str | None]:
        """Stream *source*, refusing a redirect, an error status, or an
        oversize body before any of it is assembled into a Document.

        Returns:
            ``(body, content_type, charset)`` -- ``content_type`` and
            ``charset`` are read straight off the response headers, before
            any body decoding is attempted.

        Raises:
            IngestionError: A 3xx status (refused, never followed
                automatically -- ADR-0016 decision 4), a 4xx/5xx status, more
                than ``self._max_bytes`` read (refused, not truncated), or
                the transport itself failed. No message here ever includes
                response body content -- only status codes, byte counts, and
                exception type names.
        """
        try:
            # `follow_redirects=False` per REQUEST, not merely on the client.
            # Setting it at construction is not a guard at all: `client` may be
            # caller-supplied (the injection seam these tests use), and a client
            # built with `follow_redirects=True` follows the chain inside
            # `stream()` before this function ever sees a status code — so the
            # 3xx check below never fires and the body returned is whatever the
            # final hop served. Reproduced: a 302 to `http://127.0.0.1/private`
            # was fetched and ingested, reaching a loopback address
            # `ensure_safe_endpoint` never classified. The per-request argument
            # wins over the client's setting, so the refusal holds regardless of
            # how the client was configured (ADR-0016 decision 4).
            async with client.stream("GET", source, follow_redirects=False) as response:
                if 300 <= response.status_code < 400:
                    raise IngestionError(
                        f"refusing a redirect ({response.status_code}) fetching "
                        f"{sanitize_url(source)!r} (ADR-0016 decision 4) -- the destination is "
                        "never followed automatically"
                    )
                if response.is_error:
                    raise IngestionError(
                        f"fetching {sanitize_url(source)!r} returned HTTP {response.status_code} "
                        f"{response.reason_phrase}"
                    )

                content_type = response.headers.get("content-type")
                charset = response.charset_encoding

                chunks: list[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > self._max_bytes:
                        raise IngestionError(
                            f"{sanitize_url(source)!r} exceeds the {self._max_bytes}-byte limit; "
                            "refused rather than truncated -- a truncated read "
                            "would silently produce offsets into a partial document"
                        )
                    chunks.append(chunk)
                return b"".join(chunks), content_type, charset
        except httpx.HTTPError as exc:
            raise IngestionError(
                f"failed to fetch {sanitize_url(source)!r} ({type(exc).__name__})"
            ) from exc

    async def _extract_html(self, text: str) -> str:
        """Strip tags from HTML-shaped fetched text via the shared extractor
        (ADR-0016 decision 3, applied to snapshots per spec §10.2).

        ``ExtractorProtocol.extract`` reads from a path, not a string, so the
        already-decoded text is written to a private scratch file (never
        under ``snapshot_dir`` -- it can never collide with or be mistaken
        for a real snapshot), extracted, and the scratch file is always
        removed regardless of outcome.

        That scratch write is content-sized and so is dispatched to a worker
        thread, for the same reason the snapshot write is: it is the whole
        fetched body, and it would otherwise be the one blocking write left
        on the loop between two awaits that already yield
        (``ExtractorProtocol.extract`` itself dispatches).

        Raises:
            ConfigurationError: The ``html`` extra is not installed.
        """
        extractor = extraction.html_extractor()
        with tempfile.TemporaryDirectory() as scratch:
            scratch_path = Path(scratch) / "fetched.html"
            await asyncio.to_thread(scratch_path.write_text, text, encoding="utf-8")
            return await extractor.extract(scratch_path)

    def _write_snapshot(self, document_id: str, content: str) -> Path:
        """Write *content* to ``snapshot_dir/document_id``, contained.

        Blocking, and deliberately left synchronous: callers on the event
        loop run it via ``asyncio.to_thread`` (see :meth:`load`), matching
        ``FileLoader._read_text`` on the read side. Tests call it directly to
        exercise the containment check without a fetch.

        In practice ``document_id`` is always ``Document``'s own
        ``uuid.uuid4().hex`` default (never derived from ``source`` -- spec
        §10.1), but the containment check runs unconditionally regardless,
        matching ``resolve_citation``'s read-side defense-in-depth for this
        exact path: a hand-constructed ``document_id`` must not be trusted
        implicitly (spec §10.1).

        The open is ``O_NOFOLLOW`` where the platform has it, which closes the
        window between the containment check and the write. ``ensure_within_base``
        resolves symlinks, so a snapshot path that is *already* a link out of
        the root is refused above -- but the check and the open are two
        syscalls, and anything that can create a file in the snapshot
        directory can win the gap between them and have the write land
        wherever it points. The window is narrow (``document_id`` is an
        unguessable ``uuid4``, and the read side returns nothing unless the
        bytes still match) and the flag costs nothing, so it is closed rather
        than argued about. See :data:`_O_NOFOLLOW` for what happens on
        Windows, which has no such flag.

        Raises:
            IngestionError: The resulting path would escape ``snapshot_dir``,
                or is a symbolic link at the moment of the write.
            OSError: The write failed for any other reason (propagated
                unchanged -- only the symlink refusal is reinterpreted).
        """
        candidate = self._snapshot_dir / document_id
        try:
            path = ensure_within_base(candidate, self._snapshot_dir)
        except ValueError as exc:
            raise IngestionError(
                f"snapshot path for document {document_id!r} escapes the snapshot "
                "containment root; refused rather than written"
            ) from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | _O_NOFOLLOW | _O_BINARY
        try:
            # 0o600: a snapshot is the local copy citation verification trusts,
            # written and read back by the same process identity, so there is
            # no reader for it to be group- or world-readable for.
            descriptor = os.open(path, flags, 0o600)
        except OSError as exc:
            if exc.errno in _SYMLINK_ERRNOS:
                raise IngestionError(
                    f"snapshot path for document {document_id!r} is a symbolic link; "
                    "refused rather than followed -- it was a regular path when it "
                    "was containment-checked, so something replaced it in between"
                ) from exc
            raise
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        return path

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

import logging
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from groundkit import extraction
from groundkit.contracts import Document
from groundkit.errors import IngestionError
from groundkit.utils.path_safety import ensure_within_base
from groundkit.utils.url_safety import ensure_safe_endpoint, sanitize_url

logger = logging.getLogger(__name__)

#: Response byte cap. Matches ``FileLoader.DEFAULT_MAX_BYTES`` so a remote
#: resource and a local file are bounded by the same default -- refused past
#: it, never truncated.
DEFAULT_MAX_BYTES: int = 10 * 1024 * 1024

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

    Args:
        source: The URL to check.

    Raises:
        IngestionError: The scheme is not http/https, the host is empty, or the
            URL carries userinfo.
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
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._snapshot_dir = snapshot_dir.resolve()
        self._max_bytes = max_bytes
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
                (4xx/5xx), exceeds ``max_bytes``, cannot be decoded under its
                declared (or default UTF-8) charset, the transport itself
                failed, or the resulting snapshot path would escape
                ``snapshot_dir``.
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
            body, content_type, charset = await self._fetch(client, source)
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
        self._write_snapshot(document.document_id, text)
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

        Raises:
            ConfigurationError: The ``html`` extra is not installed.
        """
        extractor = extraction.html_extractor()
        with tempfile.TemporaryDirectory() as scratch:
            scratch_path = Path(scratch) / "fetched.html"
            scratch_path.write_text(text, encoding="utf-8")
            return await extractor.extract(scratch_path)

    def _write_snapshot(self, document_id: str, content: str) -> Path:
        """Write *content* to ``snapshot_dir/document_id``, contained.

        In practice ``document_id`` is always ``Document``'s own
        ``uuid.uuid4().hex`` default (never derived from ``source`` -- spec
        §10.1), but the containment check runs unconditionally regardless,
        matching ``resolve_citation``'s read-side defense-in-depth for this
        exact path: a hand-constructed ``document_id`` must not be trusted
        implicitly (spec §10.1).

        Raises:
            IngestionError: The resulting path would escape ``snapshot_dir``.
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
        path.write_text(content, encoding="utf-8")
        return path

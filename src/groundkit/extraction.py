"""Deterministic content extraction shared between ingest and citation
resolution (ADR-0016 decisions 2, 3; ``docs/specs/loaders-extracted-and-remote-sources.md`` §9).

:class:`PdfExtractor` and :class:`HtmlExtractor` are imported by exactly two
kinds of caller: the Wave 3 PDF/HTML loaders (to build ``Document.content``
at ingest time) and :func:`groundkit.retrieval.citations.resolve_citation`
(to re-derive that same text at citation-resolution time). One class per
format, used from both sides, is what makes "the same extractor that
produced it" (ADR-0016 decision 2) hold *by construction* rather than by two
independent implementations staying in sync by hand.

Placement follows :mod:`groundkit.identity`'s precedent, not
``*/protocols.py``'s: a module outside whichever caller happened to need it
first, so sharing it creates no dependency between ingest and retrieval, and
nothing that imports it is imported back. This is a deliberate departure from
this repo's other four protocol modules, each of which holds a protocol
whose implementations live in the same package as the code that needs them —
``ExtractorProtocol`` has no such single owning package.

Neither ``pypdf`` nor ``beautifulsoup4`` is imported at module level.
Importing this module must never require either optional extra, matching the
established pattern in :mod:`groundkit.index.dense` (``_import_lancedb``)
and :mod:`groundkit.retrieval.rerank` (``_import_cross_encoder``).
"""

from __future__ import annotations

import asyncio
import importlib.metadata
from collections.abc import Callable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

from groundkit.errors import ConfigurationError, IngestionError

#: Page separator for joined PDF text. Named and pinned rather than an inline
#: literal -- the join behavior is part of what "deterministic" and "the same
#: extractor" (ADR-0016 decision 2) must mean identically at ingest and at
#: resolve time.
_PDF_PAGE_SEPARATOR: Final[str] = "\n\n"

#: pypdf extraction mode pinned explicitly, never left at whatever pypdf
#: currently defaults to -- "layout" mode is documented to depend on system
#: font metrics, which would make extraction a function of the machine it
#: runs on rather than of the file's bytes alone.
_PDF_EXTRACTION_MODE: Final[str] = "plain"

#: The stdlib parser, pinned explicitly. Omitting a parser lets
#: BeautifulSoup auto-select the "best available" one, which depends on what
#: else happens to be installed (lxml, html5lib) -- the same bytes could then
#: parse differently in CI than on an operator's machine. "html.parser" is
#: always present (stdlib) and never a second dependency.
_HTML_PARSER: Final[str] = "html.parser"

#: Tags whose *text content* must not reach the extracted output. BeautifulSoup's
#: get_text() walks every NavigableString in the tree regardless of which tag
#: contains it, so script/style bodies are ordinary text nodes to it and would
#: otherwise be indexed as prose -- exactly the "BM25 scores <div> as a term"
#: problem ADR-0016 decision 3 exists to avoid, one level down. Removed (not
#: merely skipped) before get_text() ever walks the tree.
_HTML_STRIPPED_TAGS: Final[tuple[str, ...]] = ("script", "style")

#: Separator BeautifulSoup.get_text() inserts between what were separate
#: elements, and whether surrounding whitespace on each text node is
#: stripped. Named for the same reason as _PDF_PAGE_SEPARATOR above: this
#: choice IS the offset space, forever, for every extracted-class citation.
_HTML_TEXT_SEPARATOR: Final[str] = " "
_HTML_TEXT_STRIP: Final[bool] = True


@runtime_checkable
class ExtractorProtocol(Protocol):
    """Deterministically converts a file's bytes into extracted text.

    Satisfied by every extractor this build can run -- both to produce a
    freshly-ingested ``Document.content`` and to re-derive that same text at
    citation-resolution time (ADR-0016 decision 2). Same bytes in, same text
    out: no extractor implementation may consult wall-clock time, randomness,
    environment-dependent library auto-selection, or anything outside the
    file's own bytes and this object's own pinned configuration.
    """

    @property
    def identity(self) -> str:
        """This extractor's identity string: ``"<distribution>/<version>"``.

        Derived at runtime from the installed package's own distribution
        metadata via ``importlib.metadata.version`` -- never hardcoded -- so
        a library upgrade changes this string automatically and a mismatch
        against a citation's recorded identity (ADR-0016 decision 2) is
        detected without a separate manual version-tracking step.
        """
        ...

    async def extract(self, path: Path) -> str:
        """Return the deterministic extracted text for the file at ``path``.

        ``path`` is already containment-checked by the caller (the loader's
        own ``ensure_within_base``, or ``resolve_citation``'s) -- this method
        performs no path-safety check of its own.

        Raises:
            IngestionError: The file cannot be read, or its bytes cannot be
                parsed as this extractor's format.
        """
        ...


def _import_pypdf() -> Any:
    """Import ``pypdf`` on demand; never at module import time.

    Raises:
        ConfigurationError: ``pypdf`` is not installed (ADR-0016 decision 5).
    """
    try:
        import pypdf
    except ImportError as exc:
        raise ConfigurationError(
            "PDF extraction requires the optional 'pdf' extra: install with "
            "`pip install groundkit[pdf]` (provides pypdf)"
        ) from exc
    return pypdf


def _import_bs4() -> Any:
    """Import ``bs4`` on demand; never at module import time.

    Raises:
        ConfigurationError: ``beautifulsoup4`` is not installed (ADR-0016
            decision 5).
    """
    try:
        import bs4
    except ImportError as exc:
        raise ConfigurationError(
            "HTML extraction requires the optional 'html' extra: install with "
            "`pip install groundkit[html]` (provides beautifulsoup4)"
        ) from exc
    return bs4


class PdfExtractor:
    """Deterministic PDF text extraction via pypdf (ADR-0016 decisions 2, 3).

    Construct via :func:`pdf_extractor`, not directly -- that accessor is
    what guarantees ``pypdf`` already imported successfully before
    ``__init__`` reads its version, and is what makes every caller share one
    instance (and therefore one ``identity`` computed once, not recomputed
    per call).
    """

    def __init__(self) -> None:
        self._identity = f"pypdf/{importlib.metadata.version('pypdf')}"

    @property
    def identity(self) -> str:
        return self._identity

    async def extract(self, path: Path) -> str:
        pypdf = _import_pypdf()
        return await asyncio.to_thread(self._extract_sync, pypdf, path)

    def _extract_sync(self, pypdf: Any, path: Path) -> str:
        """Runs off the event loop.

        Raises:
            IngestionError: ``path`` cannot be opened or parsed as a PDF.
        """
        try:
            reader = pypdf.PdfReader(str(path))
            pages = [
                page.extract_text(extraction_mode=_PDF_EXTRACTION_MODE) or ""
                for page in reader.pages
            ]
        except Exception as exc:
            # pypdf's own exception hierarchy (PdfReadError and friends) is
            # not part of this repo's typed error surface; wrap unconditionally
            # rather than let a third-party exception type escape this
            # module's documented contract.
            raise IngestionError(f"Failed to extract PDF text from {path.name!r}: {exc}") from exc
        return _PDF_PAGE_SEPARATOR.join(pages)


class HtmlExtractor:
    """Deterministic HTML tag-stripping via BeautifulSoup + html.parser
    (ADR-0016 decisions 2, 3). Construct via :func:`html_extractor`, not
    directly -- see :class:`PdfExtractor`'s docstring for why.
    """

    def __init__(self) -> None:
        self._identity = f"beautifulsoup4/{importlib.metadata.version('beautifulsoup4')}"

    @property
    def identity(self) -> str:
        return self._identity

    async def extract(self, path: Path) -> str:
        bs4 = _import_bs4()
        return await asyncio.to_thread(self._extract_sync, bs4, path)

    def _extract_sync(self, bs4: Any, path: Path) -> str:
        """Runs off the event loop.

        Raises:
            IngestionError: ``path`` cannot be read as UTF-8 text.
        """
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise IngestionError(f"Failed to read {path.name!r}: {exc}") from exc
        except UnicodeDecodeError as exc:
            raise IngestionError(f"{path.name!r} is not valid UTF-8: {exc}") from exc
        # "html.parser" pinned explicitly -- see the module-level constants'
        # docstrings for the determinism argument.
        soup = bs4.BeautifulSoup(raw, _HTML_PARSER)
        for tag in soup(_HTML_STRIPPED_TAGS):
            tag.decompose()
        text: str = soup.get_text(separator=_HTML_TEXT_SEPARATOR, strip=_HTML_TEXT_STRIP)
        return text


@lru_cache(maxsize=1)
def pdf_extractor() -> PdfExtractor:
    """The process's one :class:`PdfExtractor`, built on first use and memoized.

    Raises:
        ConfigurationError: The 'pdf' extra is not installed.
    """
    _import_pypdf()
    return PdfExtractor()


@lru_cache(maxsize=1)
def html_extractor() -> HtmlExtractor:
    """The process's one :class:`HtmlExtractor`, built on first use and memoized.

    Raises:
        ConfigurationError: The 'html' extra is not installed.
    """
    _import_bs4()
    return HtmlExtractor()


#: Every accessor `active_extractors()` probes, typed as returning the
#: protocol rather than each accessor's own concrete class -- Callable's
#: covariant return lets `pdf_extractor`/`html_extractor` satisfy this
#: directly, since PdfExtractor/HtmlExtractor each structurally satisfy
#: ExtractorProtocol.
_EXTRACTOR_ACCESSORS: Final[tuple[Callable[[], ExtractorProtocol], ...]] = (
    pdf_extractor,
    html_extractor,
)


@lru_cache(maxsize=1)
def active_extractors() -> Mapping[str, ExtractorProtocol]:
    """Every extractor this process can actually run, keyed by identity string.

    Replaces ``retrieval.citations._ACTIVE_EXTRACTOR_IDENTITIES`` (a plain
    frozenset constant, Waves 1-2) with a lazily-memoized accessor. Called on
    the first ``extracted``-class citation resolved in this process -- never
    at module import time -- so ``import groundkit.extraction`` (and
    therefore ``import groundkit.retrieval.citations``, which imports it)
    never requires either extra.

    Each candidate is probed independently via :func:`pdf_extractor` /
    :func:`html_extractor`, catching only :class:`ConfigurationError` (never
    a broader except): one missing extra must never blank out the other, and
    a genuine bug inside an installed extractor's construction must not be
    silently swallowed and reported as "not registered."
    """
    registry: dict[str, ExtractorProtocol] = {}
    for accessor in _EXTRACTOR_ACCESSORS:
        try:
            extractor = accessor()
        except ConfigurationError:
            continue
        registry[extractor.identity] = extractor
    return registry

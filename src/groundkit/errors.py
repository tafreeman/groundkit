"""Typed exception hierarchy with a local root (ADR-0001: no ported ancestry).

Unconfigured provider or malformed output is a typed error, never a silent
fallback or coercion (SPEC.md §2, fail closed).
"""

from __future__ import annotations

from typing import Literal


class GroundkitError(Exception):
    """Base exception for all groundkit errors."""


#: The two failure verdicts a citation-resolution failure maps to. Distinct
#: from ``service.schemas.VerificationVerdict`` (which also has ``"verified"``
#: — a success, never carried by an exception) — ``CitationVerdict`` values
#: are a literal subtype of it, so assigning one to a ``VerificationVerdict``-
#: typed variable type-checks under ``mypy --strict`` with no cast.
CitationVerdict = Literal["drifted", "unresolvable"]


class ConfigurationError(GroundkitError):
    """Invalid, incomplete, or unknown configuration. Raised at startup."""


class IngestionError(GroundkitError):
    """Error while loading or ingesting a source."""


class ChunkingError(GroundkitError):
    """Error while chunking a document."""


class EmbeddingError(GroundkitError):
    """Error while generating embeddings."""


class ProviderNotConfiguredError(EmbeddingError):
    """A provider was requested but is not configured. Never falls back."""


class StorageError(GroundkitError):
    """Error in the persisted index (metadata store, BM25 store, vector store)."""


class IndexIdentityError(StorageError):
    """A collection was opened against an embedding identity it was not built with.

    Raised when the persisted collection manifest's
    ``(provider, model_name, dimensions)`` triple does not match the active
    :class:`~groundkit.config.EmbeddingConfig`, or when a store predates the
    manifest entirely (ADR-0004).

    Never a re-embed and never a fallback: mixing semantic spaces in one
    index corrupts it silently (SPEC.md §2), and vector width alone cannot
    detect the swap — distinct models share widths, so identity is the whole
    triple.
    """


class RetrievalError(GroundkitError):
    """Error during retrieval or search.

    Attributes:
        verdict: For a citation-resolution failure raised by
            ``retrieval.citations.resolve_citation``, which of
            ``fetch_chunk``'s two failure verdicts this maps to (ADR-0016
            decision 6). ``None`` for every other ``RetrievalError`` (an
            index inconsistency, an empty query, an out-of-range ``top_k``)
            — those have no ``fetch_chunk`` verdict to carry, and leaving
            this unset for them is the point: nothing downstream can read a
            guessed value for an error this attribute was never meant to
            describe.
    """

    def __init__(self, message: str, *, verdict: CitationVerdict | None = None) -> None:
        super().__init__(message)
        self.verdict = verdict


class RerankerNotConfiguredError(RetrievalError):
    """A reranker was requested but its backend is unavailable. Never falls back.

    Raised when the optional ``rerank`` extra is not installed, or when the
    configured cross-encoder model cannot be loaded. Deliberately **not** a
    silent passthrough of the input ordering: a reranker that quietly returns
    what it was given is indistinguishable from one that worked, so a
    misconfigured deployment would report rerank-stage numbers that are really
    the upstream stage's (SPEC.md §2, fail closed).
    """


class EvalError(GroundkitError):
    """Error loading, validating, or resolving the golden eval corpus."""


class ChatError(GroundkitError):
    """Error while calling a chat/completion provider (Phase 5 boundary)."""


class ChatProviderNotConfiguredError(ChatError):
    """A chat provider was requested but is not configured. Never falls back."""


class QueryRewriteError(GroundkitError):
    """An enabled query rewrite failed.

    Deliberately **not** a silent fallback to the original query: a rewrite
    that quietly passes its input through is indistinguishable from one that
    worked, so retrieval quality attributed to "rewrite on" would really be
    the un-rewritten path's (SPEC.md §2, fail closed).
    """


class SynthesisError(GroundkitError):
    """Synthesis failed or its output violated the citation contract.

    An answer citing a span that was not among the retrieved results is
    rejected, never repaired: synthesis may cite only retrieved spans
    (SPEC.md §2), and coercing an out-of-set marker would assert a
    verifiable citation that verifies nothing.
    """


class JudgeError(GroundkitError):
    """The faithfulness judge could not produce a schema-valid verdict.

    Malformed model output is a rejection, never a coercion (SPEC.md §2).
    Advisory semantics — the judge gating nothing — live at the harness
    surface, not here: a broken judge is still a typed failure.
    """


class RedactionError(GroundkitError):
    """Base error for the redaction pass at the LLM boundary (ADR-0017)."""


class UnknownRedactionTokenError(RedactionError):
    """``restore()`` saw a token shaped like a category this instance knows,
    but whose specific counter this instance never issued.

    Fail closed rather than leaving the bracketed text in place: a silent
    pass-through would hide exactly the failure mode this exists to catch —
    restoring text produced by a different ``Redactor`` instance (a
    different config, a different run), or a token mangled in transit
    through the LLM boundary the redaction module guards.
    """

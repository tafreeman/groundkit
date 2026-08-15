"""Optional cross-encoder rerank using a local, non-LLM model (Phase 3, Wave D).

A cross-encoder scores each ``(query, passage)`` pair jointly rather than
comparing two independently-computed vectors, which is why it is worth a second
pass over an already-retrieved candidate list. It is **not** an LLM and not a
generation step: SPEC.md §2's "deterministic core, LLM at the boundary" holds —
this runs a small local sequence-classification model, in the retrieval path, on
a fixed candidate set, and returns a permutation of what it was given.

**Nothing here imports torch or sentence-transformers at module scope.** The
import is deferred to :func:`_import_cross_encoder`, called only when a
:class:`CrossEncoderReranker` actually loads its model, so
``import groundkit.retrieval.rerank`` — and therefore every pure function below,
and the whole default test suite — works in a base install. That matters twice
over: ``retrieval/*`` is a glob in the core coverage subset
(``pyproject.toml``), so this module is held to the core gate, and it could not
be if exercising it required a multi-gigabyte optional dependency.

## Hazard 2 is closed here, structurally

ADR-0001 hazard 2: ARP fed raw cross-encoder logits straight into a
``score >= 0.0`` field and crashed, because MS MARCO-style rerankers emit
*unbounded* logits and routinely emit negative ones. ADR-0005 decision 4 fixes
the shape of the fix — a sigmoid, never min-max, never a clamp:

- **Total.** Sigmoid maps every real number into ``(0, 1)``, so no logit,
  however negative, can violate ``ge=0.0``. The contract is satisfied by
  construction rather than by defending against the value after the fact.
- **Monotonic.** It cannot change the ranking, which is the only property a
  reranker owes. A clamp would satisfy the contract while collapsing every
  negatively-scored document into one tie — contract-legal and corrupt, the
  worst combination (ADR-0005, alternatives).
- **Batch-independent.** Min-max would make the same document score differently
  depending on what it was ranked alongside, and degenerates to a zero range on
  a single-result batch.

The squashing is done **by** :func:`sigmoid` **in this module**, over raw
logits, rather than by asking sentence-transformers for pre-activated scores.
ADR-0005 decision 4 requires the activation to be set explicitly and never
inherited from a library default that could change; this goes one step further
in the same direction, so that the total-function guarantee lives in groundkit's
own code where :func:`rerank_by_logits`'s regression test can exercise the exact
path production uses. :func:`_import_cross_encoder` still sets the model's
activation explicitly — to *identity* — so what arrives here is logits by
declaration rather than by assumption, and a library default flipping cannot
silently double-squash the scores.
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import TYPE_CHECKING, Any

from groundkit.contracts import RetrievalResult
from groundkit.errors import RerankerNotConfiguredError, RetrievalError

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

#: Default cross-encoder. An MS MARCO-trained model, which is precisely the
#: family that emits unbounded logits (ADR-0001 hazard 2) — chosen rather than
#: avoided, so the default configuration is the one the hazard applies to.
DEFAULT_RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

#: Upper bound on ``|logit|`` before :func:`sigmoid` saturates to exactly 0.0 or
#: 1.0 in float64. Not a clamp and not enforced anywhere — recorded so the
#: saturation documented on :func:`sigmoid` is a known quantity rather than a
#: surprise, and so a test can pin it.
SIGMOID_SATURATION_LOGIT: float = 745.0


def sigmoid(logit: float) -> float:
    """Map an unbounded logit into ``[0.0, 1.0]``, monotonically and totally.

    Computed in two branches so neither ``exp`` call can overflow: for a large
    positive logit ``exp(-x)`` underflows harmlessly toward zero, and for a
    large negative one the algebraically-equal ``exp(x) / (1 + exp(x))`` form
    does the same. The naive single-branch spelling raises ``OverflowError``
    somewhere past ``x ≈ -745``, which would turn "a very confident negative
    score" — an ordinary reranker output — into a crash.

    Mathematically the range is the open interval ``(0, 1)``. In float64 it is
    the closed one: past :data:`SIGMOID_SATURATION_LOGIT` the result rounds to
    exactly ``0.0`` or ``1.0``. Both still satisfy
    :class:`~groundkit.contracts.RetrievalResult`'s ``ge=0.0`` bound, so
    saturation costs ordering resolution between two equally-hopeless
    candidates and never contract validity.

    Args:
        logit: Any finite float. Non-finite input is a caller bug and is
            rejected rather than propagated into a score.

    Returns:
        A float in ``[0.0, 1.0]``.

    Raises:
        RetrievalError: ``logit`` is NaN or infinite. A NaN score compares
            false against everything and would silently scramble the sort;
            infinity is a model that has gone wrong, not a confident result.
    """
    if not math.isfinite(logit):
        raise RetrievalError(
            f"Reranker produced a non-finite score ({logit!r}). A NaN compares false "
            "against every value and would silently scramble the ranking; an infinity "
            "is a broken model, not a confident one. Refusing to build a result from it."
        )
    if logit >= 0.0:
        return 1.0 / (1.0 + math.exp(-logit))
    exp_logit = math.exp(logit)
    return exp_logit / (1.0 + exp_logit)


def rerank_by_logits(
    results: Sequence[RetrievalResult],
    logits: Sequence[float],
    *,
    top_k: int,
) -> list[RetrievalResult]:
    """Reorder ``results`` by ``logits``, sigmoid-normalized, truncated to ``top_k``.

    The pure half of reranking: no model, no I/O, no optional dependency. Every
    ordering and contract guarantee this module makes is made here, which is
    what lets the hazard-2 regression test drive the production path directly
    with hand-written negative logits.

    Results are rebuilt through :class:`~groundkit.contracts.RetrievalResult`'s
    constructor rather than copied with ``model_copy(update=...)``. That is
    deliberate and load-bearing: ``model_copy`` skips validation, so a score
    that violated ``ge=0.0`` would sail through it and the hazard-2 test would
    be asserting nothing. Going through the constructor means the contract is
    genuinely re-checked on every reranked result.

    Ties break on ``(source, start_offset)`` — a chunk's position in its
    document, which is content-derived and survives re-ingest. ``chunk_id``
    would be the obvious key and is deliberately not used, for the reason
    ADR-0005 decision 3 gives for the fusion tie-break: it does not survive
    re-ingestion of the same corpus, so a tie would resolve differently between
    two runs over identical content.

    Args:
        results: Candidates to reorder. Not mutated; a new list of new objects
            is returned.
        logits: Raw model scores, positionally aligned with ``results``.
        top_k: Maximum number of results to return. Must be > 0.

    Returns:
        A new list, best-first, of at most ``top_k`` new
        :class:`~groundkit.contracts.RetrievalResult` objects carrying
        sigmoid-normalized scores.

    Raises:
        RetrievalError: ``logits`` and ``results`` differ in length (a
            misaligned batch would attach every score to the wrong passage),
            ``top_k`` is not positive, or a logit is non-finite.
    """
    if top_k <= 0:
        raise RetrievalError(f"rerank top_k must be > 0, got {top_k}")
    if len(logits) != len(results):
        raise RetrievalError(
            f"Reranker returned {len(logits)} scores for {len(results)} results. A "
            "misaligned batch attaches every score to the wrong passage, which "
            "reorders the list confidently and wrongly rather than failing."
        )

    rescored = [
        RetrievalResult(
            content=result.content,
            score=sigmoid(logit),
            document_id=result.document_id,
            chunk_id=result.chunk_id,
            source=result.source,
            start_offset=result.start_offset,
            end_offset=result.end_offset,
            metadata=result.metadata,
        )
        for result, logit in zip(results, logits, strict=True)
    ]
    rescored.sort(key=lambda r: (-r.score, r.source, r.start_offset))
    return rescored[:top_k]


def _import_cross_encoder() -> tuple[Any, Any]:
    """Import ``sentence_transformers.CrossEncoder`` and ``torch`` on demand.

    Never called at module import time — only when a
    :class:`CrossEncoderReranker` actually loads its model — so importing this
    module never requires the optional ``rerank`` extra.

    Returns:
        The ``CrossEncoder`` class and an identity activation module instance,
        the latter passed to the model so that :meth:`CrossEncoderReranker.rerank`
        receives raw logits by declaration rather than by assuming a library
        default (see the module docstring).

    Raises:
        RerankerNotConfiguredError: The extra is not installed.
    """
    try:
        # Neither ships a py.typed marker reachable in a base install, and the
        # extra is deliberately absent from the dev group, so mypy cannot see
        # them in the checked environment. The ignore is scoped to these two
        # lines rather than relaxing the strict settings repo-wide.
        import torch  # type: ignore[import-not-found]
        from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RerankerNotConfiguredError(
            "CrossEncoderReranker requires the optional 'rerank' extra: install with "
            "`pip install groundkit[rerank]` (provides sentence-transformers and torch). "
            "No fallback is attempted — an unreranked list returned from a reranker "
            "would be indistinguishable from a reranked one."
        ) from exc
    return CrossEncoder, torch.nn.Identity()


class CrossEncoderReranker:
    """Rerank retrieval results with a local cross-encoder.

    Satisfies :class:`~groundkit.retrieval.protocols.RerankerProtocol`, whose
    signature is held by a conformance test (ADR-0001 hazard 4) — the parameter
    is ``query``, and renaming it is the exact drift that test exists to catch.

    The model is loaded lazily on first :meth:`rerank` and cached for the
    instance's lifetime. Construction is therefore cheap and total: it never
    touches the filesystem, the network, or the optional extra, so a
    misconfigured install fails at the first real call with
    :class:`~groundkit.errors.RerankerNotConfiguredError` rather than at import.

    Usage::

        reranker = CrossEncoderReranker()
        top = await reranker.rerank("why is the sky blue", candidates, top_k=5)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANK_MODEL,
        *,
        max_length: int | None = None,
    ) -> None:
        """Record the model identity; load nothing.

        Args:
            model_name: Cross-encoder identifier passed to sentence-transformers.
            max_length: Optional token truncation length for the pair encoder.
                ``None`` leaves the model's own configured length in place.
        """
        self._model_name = model_name
        self._max_length = max_length
        self._model: Any | None = None

    @property
    def model_name(self) -> str:
        """The configured cross-encoder identifier."""
        return self._model_name

    def _load_model(self) -> Any:
        """Load and cache the cross-encoder, with an explicit identity activation.

        Raises:
            RerankerNotConfiguredError: The extra is missing, or the model
                could not be loaded (bad identifier, no local copy and no
                network, corrupt cache).
        """
        if self._model is not None:
            return self._model
        cross_encoder_cls, identity_activation = _import_cross_encoder()
        kwargs: dict[str, Any] = {"activation_fn": identity_activation}
        if self._max_length is not None:
            kwargs["max_length"] = self._max_length
        try:
            self._model = cross_encoder_cls(self._model_name, **kwargs)
        except TypeError:
            # sentence-transformers renamed this parameter across major
            # versions (`default_activation_function` before v5,
            # `activation_fn` after). Both spellings are tried rather than a
            # version being pinned, but neither is *omitted* — inheriting the
            # library default is the thing ADR-0005 decision 4 forbids, since
            # a default of sigmoid would double-squash against our own.
            kwargs.pop("activation_fn")
            kwargs["default_activation_function"] = identity_activation
            try:
                self._model = cross_encoder_cls(self._model_name, **kwargs)
            except Exception as exc:
                raise RerankerNotConfiguredError(
                    f"Could not load cross-encoder {self._model_name!r} under either "
                    "activation-parameter spelling. Refusing to fall back to the "
                    "library's default activation, which may already apply a sigmoid "
                    "and would silently double-squash every score."
                ) from exc
        except Exception as exc:
            raise RerankerNotConfiguredError(
                f"Could not load cross-encoder {self._model_name!r}: {exc}"
            ) from exc
        return self._model

    async def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        *,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Return ``results`` reordered by cross-encoder relevance, capped at ``top_k``.

        An empty candidate list short-circuits before the model is loaded:
        there is nothing to reorder, and loading a multi-gigabyte model to
        return ``[]`` would make an empty upstream stage pay the reranker's
        entire cost.

        Scoring runs in a worker thread. The underlying ``predict`` is
        synchronous, CPU-bound, and can take seconds on a real batch; calling
        it directly on the event loop would stall every other coroutine —
        including, in Phase 4, every other in-flight request.

        Args:
            query: The user query, scored jointly against each passage.
            results: Candidates to reorder. Not mutated.
            top_k: Maximum number of results to return.

        Returns:
            A new list of at most ``top_k`` results, best-first, with
            sigmoid-normalized scores.

        Raises:
            RerankerNotConfiguredError: The optional extra is missing or the
                model could not be loaded. Never a silent passthrough.
            RetrievalError: ``top_k`` is not positive, or the model returned a
                misaligned or non-finite batch.
        """
        if top_k <= 0:
            raise RetrievalError(f"rerank top_k must be > 0, got {top_k}")
        if not results:
            return []

        model = self._load_model()
        pairs = [[query, result.content] for result in results]
        raw = await asyncio.to_thread(model.predict, pairs)
        logits = [float(score) for score in raw]
        logger.debug(
            "Reranked %d candidates to %d with %s",
            len(results),
            min(top_k, len(results)),
            self._model_name,
        )
        return rerank_by_logits(results, logits, top_k=top_k)

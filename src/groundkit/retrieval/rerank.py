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
from groundkit.errors import GroundkitError, RerankerNotConfiguredError, RetrievalError

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
            source_class=result.source_class,
            extractor=result.extractor,
            start_offset=result.start_offset,
            end_offset=result.end_offset,
            metadata=result.metadata,
        )
        for result, logit in zip(results, logits, strict=True)
    ]
    rescored.sort(key=lambda r: (-r.score, r.source, r.start_offset))
    return rescored[:top_k]


def _coerce_logits(raw: Any, *, model_name: str) -> list[float]:
    """Convert a backend's prediction output into a flat list of floats.

    The conversion is a seam, not a formality: ``predict`` returns whatever the
    configured model returns, and ``model_name`` is caller-supplied, so the
    shape is not something this repo controls. A multi-label
    sequence-classification model emits one row per label — an ``(n, labels)``
    array — and ``float()`` on a row raises ``TypeError`` rather than producing
    a score. Left bare, that escaped as an arbitrary exception from a method
    documented to raise :class:`~groundkit.errors.RetrievalError`.

    Deliberately **not** an ``isinstance`` check. ``numpy.float32`` is not a
    subclass of Python's ``float`` (``numpy.float64`` is), so a type test would
    reject the very values a real cross-encoder produces while every stub in
    the default suite, returning Python floats, kept passing — a failure only
    the gated run could find. Coercing and translating the failure tests what
    actually matters: whether the value converts to one number.

    Args:
        raw: Whatever ``predict`` returned.
        model_name: Named in errors so a misconfigured model identifies itself.

    Returns:
        One float per element of ``raw``.

    Raises:
        RetrievalError: ``raw`` is not iterable, or an element is not a scalar.
    """
    # Both boundaries below catch `Exception`, not a named tuple, and that is
    # the fix for a *class* of defect rather than the two instances of it.
    # Enumerating the exceptions a conversion can raise has now failed twice
    # here: `list(raw)` was guarded for `TypeError` only, and `float(score)`
    # for `(TypeError, ValueError)` only — which misses `OverflowError` from
    # an int too large to represent, an ordinary outcome for a backend this
    # repo does not control. Both operands come from a caller-named model
    # (`--rerank-model`), so their `__iter__`/`__float__` are third-party
    # code that may raise anything at all; any named list is a guess at what
    # someone else's library does. An untyped escape here crosses the seam as
    # an arbitrary exception and `cli.py`'s `except GroundkitError` misses it,
    # which is the exact failure this function exists to prevent. Matches how
    # `rerank()` already wraps `model.predict` below.
    try:
        values = list(raw)
    except Exception as exc:
        raise RetrievalError(
            f"Cross-encoder {model_name!r} returned a non-iterable prediction "
            f"({type(raw).__name__}, which raised {type(exc).__name__}); one score "
            "per query/passage pair is required"
        ) from exc

    logits: list[float] = []
    for position, score in enumerate(values):
        try:
            logits.append(float(score))
        except Exception as exc:
            raise RetrievalError(
                f"Cross-encoder {model_name!r} returned a non-scalar score at position "
                f"{position} ({type(score).__name__}, which raised "
                f"{type(exc).__name__}). A multi-label model emits one row per label "
                "rather than a single relevance score; this reranker requires a "
                "single-label cross-encoder."
            ) from exc
    return logits


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
        #
        # `unused-ignore` is listed alongside because whether the first code
        # fires depends on the environment, not on this file: with the extra
        # absent (CI's default job, a base install) the import is unresolved
        # and the ignore is required; with it present (`uv sync --extra
        # rerank`, which the gated workflow and anyone reproducing a real
        # rerank measurement must run) the import resolves and mypy reports
        # the ignore itself as unused. Naming only the first code made
        # `uv run mypy` fail for exactly the people following the documented
        # gated-run instructions, and no CI job catches it because the gated
        # workflow runs pytest only.
        import torch  # type: ignore[import-not-found,unused-ignore]
        from sentence_transformers import (  # type: ignore[import-not-found,unused-ignore]
            CrossEncoder,
        )

        identity_activation = torch.nn.Identity()
    except ImportError as exc:
        raise RerankerNotConfiguredError(
            "CrossEncoderReranker requires the optional 'rerank' extra: install with "
            "`pip install groundkit[rerank]` (provides sentence-transformers and torch). "
            "No fallback is attempted — an unreranked list returned from a reranker "
            "would be indistinguishable from a reranked one."
        ) from exc
    except Exception as exc:
        # "Installed" and "usable" are different states, and only the first is
        # an ImportError. torch raises OSError when a native library is missing
        # — the WinError 126 / missing-CUDA-.so family — and both packages can
        # raise at import time over a version-incompatible transitive
        # dependency. Those escaped as arbitrary backend exceptions while this
        # function advertised a typed failure, so the distinction is reported
        # rather than the case being folded into "not installed", which would
        # send someone to reinstall a package they already have.
        raise RerankerNotConfiguredError(
            f"The 'rerank' extra is installed but failed to initialize "
            f"({type(exc).__name__}). This is an environment fault rather than a "
            "missing package — a missing native library or an incompatible "
            "transitive dependency — so reinstalling the extra alone may not fix "
            "it; see the chained cause."
        ) from exc
    return CrossEncoder, identity_activation


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
        self._lock = asyncio.Lock()
        self._load_task: asyncio.Task[Any] | None = None

    @property
    def model_name(self) -> str:
        """The configured cross-encoder identifier."""
        return self._model_name

    async def _ensure_model(self) -> Any:
        """Load the model once, off the event loop, surviving cancellation.

        :meth:`_load_model` is the expensive call in this class by a wide
        margin — importing torch alone costs seconds, and a first load also
        downloads and deserializes weights, which is a network round trip
        measured in minutes. Running that directly in an ``async`` method
        blocks the event loop for its whole duration: no other coroutine
        progresses, and no cancellation or timeout can fire, which in Phase 4
        means one cold rerank stalls every other in-flight request. So it goes
        to a worker thread for exactly the reason :meth:`rerank` already sends
        ``predict`` to one.

        **The lock guards task creation, not the load itself**, and that
        distinction is the whole design. Holding a lock across the awaited
        ``to_thread`` looks equivalent and is not: a worker thread cannot be
        cancelled, but the ``await`` in front of it can. A cold call cancelled
        by a timeout unwinds ``async with`` and frees the lock *while the
        worker keeps building*, so the next caller — still seeing ``_model is
        None``, because the abandoned worker has not finished — acquires the
        free lock and starts a second load. Under a Phase 4 request timeout
        that is not exotic, and the failure mode is several multi-gigabyte
        allocations and downloads running at once.

        So the in-flight load is a *shared task*. Every caller awaits the same
        one, and it is awaited through :func:`asyncio.shield` so that one
        caller's cancellation cannot cancel the load every other caller is
        waiting on. The lock is now held only long enough to create or adopt
        that task — microseconds, never the length of the load.

        A task that finished by failing is replaced rather than reused: a
        transient fault mid-download would otherwise poison the instance
        permanently, every later call re-raising a stale exception. A task that
        finished by succeeding cannot reach that branch, since
        :meth:`_load_model` sets ``_model`` before returning.

        Returns:
            The loaded cross-encoder.

        Raises:
            RerankerNotConfiguredError: Propagated from :meth:`_load_model`.
            asyncio.CancelledError: This caller was cancelled. The shared load
                continues for whoever else is waiting on it.
        """
        if self._model is not None:
            return self._model
        async with self._lock:
            if self._model is not None:
                return self._model
            if self._load_task is None or self._load_task.done():
                self._load_task = asyncio.create_task(asyncio.to_thread(self._load_model))
            load_task = self._load_task
        return await asyncio.shield(load_task)

    def _load_model(self) -> Any:
        """Load and cache the cross-encoder, with an explicit identity activation.

        Synchronous and blocking by design — :meth:`_ensure_model` is the only
        caller that should reach it from async code, and it does so in a worker
        thread. The gated suite calls it directly to inspect raw model output.

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

        **Both** blocking steps run in a worker thread: the first-call model
        load (:meth:`_ensure_model`, also serialized so concurrent cold calls
        build one model rather than several) and then ``predict`` itself. Each
        is synchronous and long enough to matter — the load costs a torch
        import and possibly a weight download, ``predict`` costs seconds on a
        real batch — and either one left on the event loop would stall every
        other coroutine, including, in Phase 4, every other in-flight request,
        along with the cancellations and timeouts meant to bound it.

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
            RetrievalError: ``top_k`` is not positive, the model returned a
                misaligned or non-finite batch, or inference itself failed —
                every backend exception is translated, so a caller handling
                this repo's error root is covered on the scoring path too.
        """
        if top_k <= 0:
            raise RetrievalError(f"rerank top_k must be > 0, got {top_k}")
        if not results:
            return []

        model = await self._ensure_model()
        pairs = [[query, result.content] for result in results]
        try:
            raw = await asyncio.to_thread(model.predict, pairs)
        except GroundkitError:
            raise
        except Exception as exc:
            # Inference fails in ways the backend owns and this repo does not
            # model — CUDA OOM, a tokenizer rejecting an input, a device that
            # went away. Left unwrapped they cross the seam as arbitrary
            # third-party exceptions, so a caller handling RetrievalError (in
            # Phase 4, the request boundary) is unprotected on the one path
            # most likely to fail under load, while every other failure in
            # this module already arrives typed.
            #
            # The exception's message is deliberately NOT interpolated, unlike
            # `_load_model`'s. A load failure cannot carry corpus text; an
            # inference failure can — a tokenizer error routinely quotes the
            # input it choked on, which here is the query and the passages.
            # The cause is chained instead, so a traceback still carries the
            # detail while the message this raises cannot itself leak content.
            raise RetrievalError(
                f"Cross-encoder {self._model_name!r} failed while scoring "
                f"{len(pairs)} query/passage pairs ({type(exc).__name__}); "
                "see the chained cause for the backend's own message"
            ) from exc
        logits = _coerce_logits(raw, model_name=self._model_name)
        logger.debug(
            "Reranked %d candidates to %d with %s",
            len(results),
            min(top_k, len(results)),
            self._model_name,
        )
        return rerank_by_logits(results, logits, top_k=top_k)

"""Dense-wiring invariants shared by ingest, retrieval, and evaluation.

Both helpers here enforce an ADR-0004 obligation at a seam, and all three
callers — :class:`~groundkit.indexer.Indexer`,
:class:`~groundkit.retrieval.search.Retriever` and
:func:`~groundkit.evals.runner.run_eval` — need them identically. They live in
a leaf module rather than in whichever caller happened to need them first,
precisely so that sharing them creates no dependency between ingest and
retrieval: this module imports only ``contracts``, ``errors`` and the two
protocol modules, and nothing it imports imports it back.

That neutrality is the point. Each helper previously existed as a per-caller
copy, and the copies had already begun to diverge in their prose while staying
identical in their logic — the shape that lets a later fix land in two of three
places.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from groundkit.contracts import EmbeddingIdentity
from groundkit.errors import ConfigurationError

if TYPE_CHECKING:
    from groundkit.index.protocols import VectorStoreProtocol
    from groundkit.providers.protocols import EmbeddingProtocol


def identity_of(embedder: EmbeddingProtocol) -> EmbeddingIdentity:
    """Read the ADR-0004 identity triple off the embedder itself.

    Never derived from a config object: sourcing all three fields from the
    object that actually produces the vectors makes "the manifest describes a
    different model than the one that embedded" unrepresentable, rather than a
    divergence a caller has to be trusted not to introduce. The same property
    is what stops an eval artifact from naming a model other than the one that
    produced its vectors.

    Args:
        embedder: The embedder whose identity a collection is bound to.

    Returns:
        The ``(provider, model_name, dimensions)`` triple, which is checked as
        a triple and never by dimensions alone.
    """
    return EmbeddingIdentity(
        provider=embedder.provider,
        model_name=embedder.model_name,
        dimensions=embedder.dimensions,
    )


def validate_dense_pair(
    embedder: EmbeddingProtocol | None,
    vector_store: VectorStoreProtocol | None,
    *,
    subject: str,
    without_store: str,
    without_embedder: str,
) -> None:
    """Reject half a dense pair before it can reach any dense work.

    The pair is inseparable in both directions, but *why* differs by caller —
    a write path silently discards the vectors it produced, a read path has no
    index to search — so each caller supplies its own consequence clause
    instead of sharing one generic sentence that would say less than any of
    them. What is shared is the branch structure, which is the part that can
    drift: a caller checking only one direction admits exactly the
    half-configured state this exists to reject.

    Args:
        embedder: The embedder half of the pair, or ``None``.
        vector_store: The vector-store half of the pair, or ``None``.
        subject: Caller named at the head of the message (e.g. ``"Retriever"``).
        without_store: Consequence clause for an embedder supplied with no
            store. Follows ``"The pair is inseparable: "`` and precedes
            ``"Pass both or neither."``, so it reads as a complete sentence.
        without_embedder: Consequence clause for a store supplied with no
            embedder, positioned identically.

    Raises:
        ConfigurationError: Exactly one of ``embedder`` / ``vector_store`` was
            supplied.
    """
    if embedder is not None and vector_store is None:
        raise ConfigurationError(
            f"{subject} was given an embedder but no vector_store. The pair is "
            f"inseparable: {without_store} Pass both or neither."
        )
    if vector_store is not None and embedder is None:
        raise ConfigurationError(
            f"{subject} was given a vector_store but no embedder. The pair is "
            f"inseparable: {without_embedder} Pass both or neither."
        )

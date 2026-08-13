"""Dense vector index behind :class:`~groundkit.index.protocols.VectorStoreProtocol`
(Phase 3, ADR-0001, ADR-0004).

Two implementations land together, never separately (SPEC.md §5.3: "Metadata
filtering is implemented on both in-memory and LanceDB paths from the first
dense-store commit, with regression tests on both paths"):

- :class:`InMemoryVectorStore` — pure Python, no ``lancedb`` import. The
  dev/test double for offline work and CI where the optional ``dense``
  extra may be absent.
- :class:`LanceDBVectorStore` — persisted, backed by LanceDB. The
  ``lancedb``/``pyarrow`` import is lazy (see :func:`_import_lancedb`): a
  BM25-only install can still ``import groundkit.index.dense`` freely, and
  only *constructing* a :class:`LanceDBVectorStore` without the ``dense``
  extra installed raises a clear :class:`~groundkit.errors.StorageError`
  instead of a bare ``ImportError`` surfacing from deep inside this module.

Both implementations close two ported ARP defects (ADR-0001):

- **Hazard 3 — silent filter drop.** ARP's ``search`` accepted a filter
  argument via ``**kwargs`` and never read it; any call spelling, keyword or
  positional, silently returned unfiltered results. ``VectorStoreProtocol``
  already closes the spelling half of this (no ``**kwargs``, so a misspelled
  keyword is a ``TypeError`` at the call site) — the implementation half is
  this module's job: :func:`_matches_filter` is the single place a
  ``metadata_filter`` is ever applied, on both stores, with identical
  semantics (subset match: every ``(key, value)`` pair in the filter must be
  present and equal in the chunk's metadata).
- **Hazard 5 — unescaped delete predicate.** LanceDB deletes through a SQL
  expression string built by the caller. ADR-0004 decision 6: a document ID
  is validated against a strict identifier pattern (:func:`_validate_document_id`,
  mirroring ``index/metadata.py``'s ``_validate_collection`` in spirit) before
  it can reach :func:`_build_delete_predicate` — the single function that
  builds that string. Escaping is defence in depth, applied after
  validation, never the primary defence. Verified live against this repo's
  pinned LanceDB 0.37.1: ``t.delete("document_id = 'nope\\' OR \\'1\\'=\\'1'")``
  wipes every row in the table; the regression test in ``tests/test_dense.py``
  asserts that input cannot reach a predicate at all.

**Filter-then-truncate.** Filtering a vector search result *after* asking
the backend for ``top_k`` nearest neighbors returns fewer than ``top_k``
whenever some of those neighbors don't match the filter — a silently short
result set, which SPEC.md §5.3 and ADR-0001 hazard 3 both call a defect, not
a quirk. Both stores over-fetch instead: when ``metadata_filter`` is given,
every stored vector's similarity is scored (:class:`InMemoryVectorStore`
already does this — it is a linear scan by construction) and the filter is
applied *before* truncating to ``top_k``, so a ``top_k`` slice is only ever
short when fewer than ``top_k`` chunks actually match. This is the same
"correctness over cleverness, revisit if it's ever measured to matter"
tradeoff ``index/bm25.py`` already accepts for its O(corpus) rebuild
(ADR-0002) — filtered dense search costs O(corpus) instead of O(log corpus)
per query; unfiltered search stays a cheap top-k vector query.

**Score mapping.** LanceDB reports a *distance*, not a similarity, and
``RetrievalResult.score`` is ``Field(ge=0.0)`` (contracts.py). Both stores
compute cosine similarity and then floor it at zero via :func:`_clamp_score`
— similarity is queried from LanceDB as ``1 - cosine_distance`` (verified
live: an identical vector pair reports ``_distance == 0.0``, orthogonal
reports ``1.0``, opposite reports ``2.0``, matching ``distance = 1 -
similarity``), or computed directly in :func:`_cosine_similarity` for the
in-memory path. A negative similarity (near-opposite vectors) floors to
``0.0`` rather than crashing the contract or being silently coerced —
ranking order among genuinely relevant, positive-similarity results is
unaffected; only the below-zero tail collapses to a shared floor.

**Determinism.** Ties are broken by ascending ``Chunk.content_hash``, the
exact convention ``index/bm25.py`` already pins (its module docstring: tied
scores fall back to insertion order only for two byte-identical chunks,
which share a hash and can't be told apart by content alone). Both stores
reconstruct full :class:`~groundkit.contracts.Chunk` objects internally
(from stored rows, for LanceDB) specifically so this tie-break key and
Pydantic's own offset/length re-validation come "for free" at read time.

**No metadata-store join.** Unlike ``BM25Index.search`` (returns ``(Chunk,
score)`` pairs and leaves the store join to ``retrieval/search.py``,
because ``Chunk`` carries no ``source`` field), ``VectorStoreProtocol.search``
must return citation-bearing :class:`~groundkit.contracts.RetrievalResult`
objects directly — its signature has no store handle to join against. This
works because ``ingestion/chunking.py`` seeds every chunk's metadata with
its document's source (``metadata={"source": document.source, **document.metadata}``),
so :func:`_to_retrieval_result` reads ``source`` straight out of
``chunk.metadata`` instead. A chunk stored without a ``"source"`` key is an
index inconsistency and fails closed with :class:`~groundkit.errors.StorageError`,
mirroring ``Retriever.search``'s own fail-closed behavior for the BM25 path.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from pathlib import Path
from typing import Any, Final

from groundkit.contracts import Chunk, RetrievalResult
from groundkit.errors import StorageError

#: Score floor: cosine similarity is clamped to this before it ever reaches
#: RetrievalResult, which requires score >= 0.0 (contracts.py).
_MIN_SCORE: Final[float] = 0.0

#: LanceDB distance metric. Cosine, not the client default (L2) — see the
#: module docstring's "Score mapping" section for the verified distance
#: formula this choice implies.
_DISTANCE_METRIC: Final[str] = "cosine"

#: Fixed vector column name across the schema, row construction, and query.
_VECTOR_COLUMN: Final[str] = "vector"

#: Default LanceDB table name when a caller doesn't pick one.
_DEFAULT_TABLE_NAME: Final[str] = "chunks"

#: Allowed characters for a document ID reaching a LanceDB delete predicate:
#: ASCII letters, digits, and the conservative separators ``-``, ``_``, ``.``
#: — the same character class ``index/metadata.py``'s ``_COLLECTION_NAME_PATTERN``
#: uses for collection names, applied here in the same spirit (ADR-0004
#: decision 6). The default ``Chunk.document_id`` (``uuid.uuid4().hex``)
#: always satisfies this; a caller-supplied custom document ID outside this
#: set is rejected before it can reach a predicate, in either store — see
#: the module docstring's "Hazard 5" section.
_DOCUMENT_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9._-]+$")


def _validate_add_shapes(chunks: list[Chunk], embeddings: list[list[float]]) -> int | None:
    """Validate one ``add()`` call's shape; shared by both store implementations.

    Args:
        chunks: Chunks passed to ``add()``.
        embeddings: Embeddings passed to ``add()``, expected one per chunk.

    Returns:
        The uniform embedding width for this call, or ``None`` if both lists
        are empty (a legal no-op ``add()``).

    Raises:
        StorageError: ``chunks`` and ``embeddings`` differ in length, or the
            embeddings in this call are not all the same width.
    """
    if len(chunks) != len(embeddings):
        raise StorageError(
            f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must have the same length"
        )
    if not embeddings:
        return None
    width = len(embeddings[0])
    for vector in embeddings:
        if len(vector) != width:
            raise StorageError(
                "all embeddings in one add() call must share one width; got widths "
                f"{width} and {len(vector)}"
            )
    return width


def _dimension_mismatch_error(what: str, got: int, expected: int) -> StorageError:
    """Build the (identically worded, across both stores) dimension-mismatch error."""
    return StorageError(
        f"{what} width {got} does not match this store's established width {expected}"
    )


def _clamp_score(similarity: float) -> float:
    """Floor a raw cosine similarity at 0.0 to satisfy ``RetrievalResult.score >= 0.0``.

    Args:
        similarity: Cosine similarity, mathematically in ``[-1.0, 1.0]``.

    Returns:
        ``similarity`` unchanged if non-negative, else ``0.0``.
    """
    return max(_MIN_SCORE, similarity)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors.

    Args:
        a: First vector.
        b: Second vector, same length as ``a``.

    Returns:
        The cosine similarity, or ``0.0`` (never a ``ZeroDivisionError``) if
        either vector has zero magnitude — an all-zero embedding carries no
        directional information, so "no similarity" is the only defensible
        answer.

    Note:
        Documented divergence from :class:`LanceDBVectorStore` (verified
        live against the pinned 0.37.1 client): LanceDB's cosine-metric
        search silently *excludes* a zero-magnitude stored vector from
        results entirely, rather than reporting it as a 0.0-similarity
        match the way this function's guard does. A genuinely all-zero
        embedding essentially never occurs from a real embedding model, so
        this is left as a documented backend-specific edge case rather than
        worked around by computing norms outside LanceDB to force parity.
    """
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _matches_filter(metadata: dict[str, Any], metadata_filter: dict[str, Any] | None) -> bool:
    """True if ``metadata`` contains every ``(key, value)`` pair in ``metadata_filter``.

    The single filtering rule shared by both stores (ADR-0001 hazard 3): a
    ``None`` or empty filter matches everything (no-op), and a filter key
    absent from ``metadata`` never matches — there is no partial credit.

    Args:
        metadata: A chunk's metadata dict.
        metadata_filter: Required key/value pairs, or ``None`` for no filter.

    Returns:
        Whether ``metadata`` satisfies ``metadata_filter``.
    """
    if not metadata_filter:
        return True
    return all(key in metadata and metadata[key] == value for key, value in metadata_filter.items())


def _to_retrieval_result(chunk: Chunk, score: float) -> RetrievalResult:
    """Build a citation-bearing :class:`RetrievalResult` from a chunk and a clamped score.

    See the module docstring's "No metadata-store join" section for why this
    reads ``source`` out of ``chunk.metadata`` instead of joining a store.

    Args:
        chunk: The matched chunk (its metadata must carry a string ``"source"``).
        score: Already-clamped score (``>= 0.0``).

    Returns:
        A fully-populated :class:`RetrievalResult`.

    Raises:
        StorageError: ``chunk.metadata`` has no string ``"source"`` key —
            an index inconsistency; fail closed rather than construct an
            unverifiable citation.
    """
    source = chunk.metadata.get("source")
    if not isinstance(source, str):
        raise StorageError(
            f"chunk {chunk.chunk_id!r} has no string 'source' in its metadata; cannot "
            "construct a citation-bearing RetrievalResult"
        )
    return RetrievalResult(
        content=chunk.content,
        score=score,
        document_id=chunk.document_id,
        chunk_id=chunk.chunk_id,
        source=source,
        start_offset=chunk.start_offset,
        end_offset=chunk.end_offset,
        metadata=chunk.metadata,
    )


def _sort_by_score(scored: list[tuple[Chunk, float]]) -> list[tuple[Chunk, float]]:
    """Sort ``(chunk, score)`` pairs by descending score, ties broken ascending by content hash.

    Mirrors ``index/bm25.py``'s tie-break convention exactly, so repeated
    searches against unchanged state return byte-identical ordering across
    runs and processes, not merely identical scores.
    """
    return sorted(scored, key=lambda pair: (-pair[1], pair[0].content_hash))


def _validate_document_id(document_id: str) -> None:
    """Validate that ``document_id`` is safe to interpolate into a delete predicate.

    Applied identically by both stores (ADR-0004 decision 6): even though
    :class:`InMemoryVectorStore` deletes by plain equality and builds no
    predicate at all, using the same strict validation on both paths keeps
    their behavior identical for the same input, and rejecting a malformed
    ID loudly surfaces a caller bug (this repo's own ID generation,
    ``uuid.uuid4().hex``, always satisfies this pattern) rather than
    silently no-op deleting.

    Args:
        document_id: Candidate document ID.

    Raises:
        StorageError: ``document_id`` is empty or whitespace-only, contains
            a null byte, or contains characters outside
            ``_DOCUMENT_ID_PATTERN`` (letters, digits, ``-``, ``_``, ``.``) —
            which, in particular, excludes quote characters, whitespace, and
            SQL operators, closing ADR-0001 hazard 5.
    """
    if "\0" in document_id:
        raise StorageError("document_id must not contain a null byte")
    if not document_id.strip():
        raise StorageError("document_id must not be empty or whitespace-only")
    if not _DOCUMENT_ID_PATTERN.fullmatch(document_id):
        raise StorageError(
            f"document_id {document_id!r} contains characters outside the allowed set "
            "(letters, digits, '-', '_', '.'); refusing to build a delete predicate from it"
        )


def _build_delete_predicate(document_id: str) -> str:
    """Build a LanceDB delete predicate for ``document_id`` — the ONE place this happens.

    Validation (:func:`_validate_document_id`) is the primary defence:
    called first, so a hostile ID never reaches the f-string below at all.
    The subsequent quote-doubling is defence in depth only, per ADR-0004's
    explicit rejection of "escape rather than validate" as the primary
    strategy — it is a no-op given the validated character set, which
    already excludes ``'``, but keeps this function safe even if the
    validation pattern is ever loosened without re-reading this docstring.

    Args:
        document_id: The document ID to delete.

    Returns:
        A SQL-expression predicate string, e.g. ``"document_id = 'abc123'"``.

    Raises:
        StorageError: Via :func:`_validate_document_id`.
    """
    _validate_document_id(document_id)
    escaped = document_id.replace("'", "''")
    return f"document_id = '{escaped}'"


class InMemoryVectorStore:
    """Pure-Python in-memory vector store — the dev/test double (ADR-0001 promote decision).

    No ``lancedb`` import anywhere in this class; safe to use in a BM25-only
    install. Every ``search`` is an exhaustive linear scan over all stored
    vectors — by construction, not an optimization to revisit, since that
    is what makes its filter-then-truncate behavior trivially correct (see
    the module docstring). Appropriate for the corpus sizes this repo
    targets pre-Phase-6 scale work, matching ``index/bm25.py``'s identical
    O(corpus) tradeoff.

    Satisfies :class:`~groundkit.index.protocols.VectorStoreProtocol`.
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []
        self._dimensions: int | None = None

    async def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Store chunks with their embeddings.

        Args:
            chunks: Chunks to store.
            embeddings: One embedding per chunk, same order.

        Raises:
            StorageError: ``chunks``/``embeddings`` length mismatch, embeddings
                in this call are not all the same width, or that width
                disagrees with a width already established by an earlier
                ``add()`` call on this store.
        """
        width = _validate_add_shapes(chunks, embeddings)
        if width is None:
            return
        if self._dimensions is not None and width != self._dimensions:
            raise _dimension_mismatch_error("embedding", width, self._dimensions)
        self._dimensions = width
        self._chunks.extend(chunks)
        self._vectors.extend(embeddings)

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Return the most similar chunks, most similar first.

        Args:
            query_embedding: The query vector.
            top_k: Maximum number of results.
            metadata_filter: Keep only chunks whose metadata contains every
                given key/value pair; ``None`` means no filter. Applied
                before truncating to ``top_k`` (see the module docstring's
                "Filter-then-truncate" section) so a filtered search still
                returns ``top_k`` results whenever that many chunks match.

        Returns:
            Up to ``top_k`` :class:`~groundkit.contracts.RetrievalResult`,
            ranked by descending score, ties broken by ascending
            ``Chunk.content_hash``. ``top_k <= 0`` or an empty store
            returns ``[]``.

        Raises:
            StorageError: ``query_embedding``'s width disagrees with this
                store's established width, or a matched chunk's metadata
                has no string ``"source"`` (see :func:`_to_retrieval_result`).
        """
        if top_k <= 0 or not self._chunks:
            return []
        if self._dimensions is not None and len(query_embedding) != self._dimensions:
            raise _dimension_mismatch_error(
                "query embedding", len(query_embedding), self._dimensions
            )

        scored: list[tuple[Chunk, float]] = []
        for chunk, vector in zip(self._chunks, self._vectors, strict=True):
            if not _matches_filter(chunk.metadata, metadata_filter):
                continue
            scored.append((chunk, _clamp_score(_cosine_similarity(query_embedding, vector))))

        ranked = _sort_by_score(scored)
        return [_to_retrieval_result(chunk, score) for chunk, score in ranked[:top_k]]

    async def delete(self, document_id: str) -> int:
        """Delete all vectors for ``document_id``.

        Args:
            document_id: The document whose chunks should be removed.

        Returns:
            The number of vectors actually removed (counted, not assumed).

        Raises:
            StorageError: ``document_id`` fails :func:`_validate_document_id`.
        """
        _validate_document_id(document_id)
        kept_chunks: list[Chunk] = []
        kept_vectors: list[list[float]] = []
        removed = 0
        for chunk, vector in zip(self._chunks, self._vectors, strict=True):
            if chunk.document_id == document_id:
                removed += 1
            else:
                kept_chunks.append(chunk)
                kept_vectors.append(vector)
        self._chunks = kept_chunks
        self._vectors = kept_vectors
        return removed


def _import_lancedb() -> tuple[Any, Any]:
    """Import ``lancedb`` and ``pyarrow`` on demand.

    Never called at module import time — only when a :class:`LanceDBVectorStore`
    is actually opened — so ``import groundkit.index.dense`` never requires
    the optional ``dense`` extra.

    Returns:
        The ``lancedb`` and ``pyarrow`` modules.

    Raises:
        StorageError: Either dependency is not importable.
    """
    try:
        import lancedb
        import pyarrow as pa  # type: ignore[import-untyped]  # pyarrow ships no py.typed marker
    except ImportError as exc:
        raise StorageError(
            "LanceDBVectorStore requires the optional 'dense' extra: install with "
            "`pip install groundkit[dense]` (provides lancedb and pyarrow)"
        ) from exc
    return lancedb, pa


def _build_schema(pa_module: Any, dimensions: int) -> Any:
    """Build the fixed-width pyarrow schema for a fresh LanceDB table.

    Stores the full chunk shape, not just the vector — ``content``,
    offsets, and JSON-encoded ``metadata`` all round-trip so
    :func:`_chunk_from_row` can reconstruct a complete, re-validated
    :class:`~groundkit.contracts.Chunk` without a metadata-store join (see
    the module docstring's "No metadata-store join" section).
    """
    return pa_module.schema(
        [
            pa_module.field(_VECTOR_COLUMN, pa_module.list_(pa_module.float32(), dimensions)),
            pa_module.field("chunk_id", pa_module.string()),
            pa_module.field("document_id", pa_module.string()),
            pa_module.field("chunk_index", pa_module.int64()),
            pa_module.field("content", pa_module.string()),
            pa_module.field("start_offset", pa_module.int64()),
            pa_module.field("end_offset", pa_module.int64()),
            pa_module.field("metadata", pa_module.string()),
        ]
    )


def _infer_dimensions(table: Any) -> int:
    """Read the vector column's fixed width from an already-open table's schema."""
    field = table.schema.field(_VECTOR_COLUMN)
    return int(field.type.list_size)


def _row_from_chunk(chunk: Chunk, vector: list[float]) -> dict[str, Any]:
    """Build one LanceDB row dict from a chunk and its embedding."""
    return {
        _VECTOR_COLUMN: vector,
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "chunk_index": chunk.chunk_index,
        "content": chunk.content,
        "start_offset": chunk.start_offset,
        "end_offset": chunk.end_offset,
        "metadata": json.dumps(chunk.metadata),
    }


def _chunk_from_row(row: dict[str, Any]) -> Chunk:
    """Reconstruct a :class:`Chunk` from a persisted LanceDB row.

    Every field :func:`_row_from_chunk` writes round-trips exactly, so
    ``Chunk``'s own offset/length validators re-verify integrity on every
    read instead of trusting the stored bytes blindly (the same
    "verifiable, not asserted" spirit ``retrieval/citations.py`` applies to
    citations).
    """
    return Chunk(
        chunk_id=row["chunk_id"],
        document_id=row["document_id"],
        chunk_index=row["chunk_index"],
        content=row["content"],
        start_offset=row["start_offset"],
        end_offset=row["end_offset"],
        metadata=json.loads(row["metadata"]),
    )


class LanceDBVectorStore:
    """Persisted vector store backed by LanceDB (ADR-0001 adapt decision).

    Construct via :meth:`open`, not directly — mirrors
    ``index/metadata.py``'s ``SQLiteMetadataStore.open()`` convention. All
    LanceDB calls are synchronous; every one is run via ``asyncio.to_thread``
    and serialized through one ``asyncio.Lock`` held across the awaited
    call, for the same reason ``SQLiteMetadataStore`` does this: concurrent
    coroutines can genuinely run these calls on different threads
    simultaneously, and the underlying table object is not documented as
    safe for that.

    Satisfies :class:`~groundkit.index.protocols.VectorStoreProtocol`.
    """

    def __init__(
        self,
        db: Any,
        pa_module: Any,
        table_name: str,
        table: Any,
        dimensions: int | None,
    ) -> None:
        self._db = db
        self._pa = pa_module
        self._table_name = table_name
        self._table = table
        self._dimensions = dimensions
        self._lock = asyncio.Lock()

    @classmethod
    async def open(
        cls, db_path: str | Path, table_name: str = _DEFAULT_TABLE_NAME
    ) -> LanceDBVectorStore:
        """Open (creating if absent) the LanceDB-backed store at ``db_path``.

        Args:
            db_path: Directory LanceDB should use for this collection's
                table(s). Created on first write if it does not exist.
            table_name: Table name within that directory.

        Returns:
            A ready-to-use store. If a table named ``table_name`` already
            exists at ``db_path`` (a prior session's data), it is opened
            and this store's dimensions are inferred from its schema;
            otherwise the table is created lazily on the first :meth:`add`.

        Raises:
            StorageError: The optional ``dense`` extra (``lancedb``,
                ``pyarrow``) is not installed, or the LanceDB connection
                could not be opened.
        """
        lancedb, pa = _import_lancedb()

        def _open_sync() -> tuple[Any, Any, int | None]:
            db = lancedb.connect(str(db_path))
            try:
                table = db.open_table(table_name)
            except ValueError:
                # LanceDB raises ValueError, not a missing-file OSError, for
                # "no table with this name yet" — verified live against the
                # pinned 0.37.1 client. Treated as "create lazily on add()",
                # not an error.
                table = None
            dimensions = _infer_dimensions(table) if table is not None else None
            return db, table, dimensions

        try:
            db, table, dimensions = await asyncio.to_thread(_open_sync)
        except OSError as exc:
            raise StorageError(f"failed to open LanceDB store at {db_path}") from exc
        return cls(db, pa, table_name, table, dimensions)

    def _create_table_sync(self, dimensions: int) -> Any:
        """Blocking table creation; run only via ``asyncio.to_thread``."""
        schema = _build_schema(self._pa, dimensions)
        return self._db.create_table(self._table_name, schema=schema)

    async def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Store chunks with their embeddings.

        Creates the backing table on the first call, with a schema fixed to
        this call's embedding width — LanceDB tables are fixed-width, so
        every later ``add()`` must agree with it.

        Args:
            chunks: Chunks to store.
            embeddings: One embedding per chunk, same order.

        Raises:
            StorageError: ``chunks``/``embeddings`` length mismatch, embeddings
                in this call are not all the same width, or that width
                disagrees with this store's already-established width.
        """
        width = _validate_add_shapes(chunks, embeddings)
        if width is None:
            return

        rows = [
            _row_from_chunk(chunk, vector) for chunk, vector in zip(chunks, embeddings, strict=True)
        ]
        async with self._lock:
            if self._table is None:
                self._table = await asyncio.to_thread(self._create_table_sync, width)
                self._dimensions = width
            elif self._dimensions is not None and width != self._dimensions:
                raise _dimension_mismatch_error("embedding", width, self._dimensions)
            await asyncio.to_thread(self._table.add, rows)

    def _search_sync(self, query_embedding: list[float], limit: int) -> list[dict[str, Any]]:
        """Blocking LanceDB vector search; run only via ``asyncio.to_thread``."""
        query = (
            self._table.search(query_embedding, vector_column_name=_VECTOR_COLUMN)
            .metric(_DISTANCE_METRIC)
            .limit(limit)
        )
        rows: list[dict[str, Any]] = query.to_list()
        return rows

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Return the most similar chunks, most similar first.

        Args:
            query_embedding: The query vector.
            top_k: Maximum number of results.
            metadata_filter: Keep only chunks whose metadata contains every
                given key/value pair; ``None`` means no filter. When set,
                every stored row is fetched and filtered in Python before
                truncating to ``top_k`` (see the module docstring's
                "Filter-then-truncate" section) — metadata is an opaque
                JSON blob column here, not structured columns, so pushing
                an arbitrary filter into a LanceDB ``WHERE`` predicate would
                mean building that predicate from caller-controlled values,
                the exact class of hazard ADR-0004 decision 6 exists to
                close for document IDs. Over-fetching and filtering in
                Python avoids ever constructing one.

        Returns:
            Up to ``top_k`` :class:`~groundkit.contracts.RetrievalResult`,
            ranked by descending score, ties broken by ascending
            ``Chunk.content_hash``. ``top_k <= 0`` or an empty/absent table
            returns ``[]``.

        Raises:
            StorageError: ``query_embedding``'s width disagrees with this
                store's established width, or a matched chunk's metadata
                has no string ``"source"`` (see :func:`_to_retrieval_result`).
        """
        if top_k <= 0:
            return []

        async with self._lock:
            if self._table is None:
                return []
            if self._dimensions is not None and len(query_embedding) != self._dimensions:
                raise _dimension_mismatch_error(
                    "query embedding", len(query_embedding), self._dimensions
                )
            if metadata_filter is None:
                fetch_limit = top_k
            else:
                fetch_limit = await asyncio.to_thread(self._table.count_rows)
            if fetch_limit <= 0:
                return []
            rows = await asyncio.to_thread(self._search_sync, query_embedding, fetch_limit)

        scored: list[tuple[Chunk, float]] = []
        for row in rows:
            chunk = _chunk_from_row(row)
            if not _matches_filter(chunk.metadata, metadata_filter):
                continue
            scored.append((chunk, _clamp_score(1.0 - float(row["_distance"]))))

        ranked = _sort_by_score(scored)
        return [_to_retrieval_result(chunk, score) for chunk, score in ranked[:top_k]]

    async def delete(self, document_id: str) -> int:
        """Delete all vectors for ``document_id``.

        Args:
            document_id: The document whose chunks should be removed.

        Returns:
            The number of vectors actually removed, computed by counting
            rows before and after the delete (counted, not assumed) — never
            read off LanceDB's own delete-result object, which is not
            guaranteed present across this repo's supported ``lancedb``
            version range (``>=0.13,<1``).

        Raises:
            StorageError: ``document_id`` fails :func:`_validate_document_id`
                (ADR-0004 decision 6 — validated before it can reach
                :func:`_build_delete_predicate`).
        """
        predicate = _build_delete_predicate(document_id)
        async with self._lock:
            if self._table is None:
                return 0
            before = await asyncio.to_thread(self._table.count_rows)
            await asyncio.to_thread(self._table.delete, predicate)
            after = await asyncio.to_thread(self._table.count_rows)
        return int(before) - int(after)

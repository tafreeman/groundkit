"""Fail loudly if a collection's dense search would brute-force scan instead of using its index.

Runs :func:`groundkit.index.dense.assert_search_uses_vector_index` against a
real collection's LanceDB store — the same ``<index_dir>/<collection>.lance``
path ``grk`` opens — and checks the plan of the exact query
``LanceDBVectorStore.search`` runs for an unfiltered ``--mode dense`` or
``--mode hybrid`` search.

The plan does not depend on the query vector's values, only its width, so a
unit probe vector is used and no embedding provider is needed. Offline and
credential-free.

Exit codes: 0 the index serves the query; 1 it would fall back to scanning
(the full plan is printed); 2 the collection has no dense store to check.

Usage::

    uv run python scripts/check_dense_index.py --collection dense
    uv run python scripts/check_dense_index.py --collection dense --max-unindexed-rows 500

Note: groundkit does not build an ANN index on any collection today, so this
exits 1 on every existing dense collection. That is the finding it exists to
surface, not a defect in the check.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from groundkit.errors import StorageError
from groundkit.index.dense import (
    LanceDBVectorStore,
    VectorIndexFallbackError,
    assert_search_uses_vector_index,
)
from groundkit.index.metadata import validate_collection_name


async def _check(index_dir: Path, collection: str, top_k: int, allowance: int) -> int:
    lance_path = index_dir / f"{collection}.lance"
    if not lance_path.exists():
        print(f"no dense store at {lance_path}; was the collection ingested with --dense?")
        return 2
    store = await LanceDBVectorStore.open(lance_path)
    width = store._dimensions  # read-only; the store has no public width accessor
    if width is None:
        print(f"{lance_path} has no vector table yet")
        return 2
    probe = [1.0] + [0.0] * (width - 1)
    try:
        plan = await assert_search_uses_vector_index(
            store, probe, top_k=top_k, max_unindexed_rows=allowance
        )
    except VectorIndexFallbackError as exc:
        print(f"FAIL [{collection}]: {exc}", file=sys.stderr)
        return 1
    print(f"OK [{collection}]: unfiltered dense search is served by the vector index.\n{plan}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--index-dir", default=".groundkit", type=Path)
    parser.add_argument("--collection", default="default")
    parser.add_argument("--top-k", type=int, default=5, help="the top_k the app searches with")
    parser.add_argument(
        "--max-unindexed-rows",
        type=int,
        default=0,
        help="rows allowed outside the index before failing (default: 0)",
    )
    args = parser.parse_args()
    try:
        validate_collection_name(args.collection)
        return asyncio.run(
            _check(args.index_dir, args.collection, args.top_k, args.max_unindexed_rows)
        )
    except StorageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

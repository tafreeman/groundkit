"""Measure ``Retriever.open()`` cost against ADR-0002's revisit trigger.

ADR-0002 accepts the O(corpus) BM25 rebuild-at-open with an explicit revisit
trigger: persisted postings are reconsidered only "if rebuild-at-open time is
*measured* to be a problem for a real corpus size — that measurement is the
trigger, not a guess made now." Phase 3 Wave C makes ``open()`` heavier (a
manifest verification, a LanceDB connection, and the open()-time document
snapshot join the BM25 rebuild — phase-3 spec R4), so this script is the
measurement method, kept in-repo so the numbers it produces are always
regenerable rather than quoted. Per repo policy ("no number in any doc that
wasn't generated"), docs reference *this script*, never its output.

Method: for each corpus size, synthesize documents of fixed shape into a temp
directory, ingest them once (BM25-only, and again dense-enabled with the
in-memory hash embedder — semantic signal is irrelevant here; only vector
plumbing cost is timed), then time ``Retriever.open()`` over ``--repeats``
fresh opens and report min/median/mean wall-clock per configuration. The
dense timing includes ``LanceDBVectorStore.open()`` because Phase 4's
per-request service would pay exactly that composite cost.

Usage::

    uv run python scripts/measure_retriever_open.py
    uv run python scripts/measure_retriever_open.py --sizes 100 1000 5000 --repeats 20
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import tempfile
import time
from pathlib import Path

from groundkit.index.dense import LanceDBVectorStore
from groundkit.index.metadata import SQLiteMetadataStore
from groundkit.indexer import Indexer
from groundkit.ingestion.loaders import FileLoader
from groundkit.providers.embeddings import InMemoryEmbedder
from groundkit.retrieval.search import Retriever

_NS_PER_MS = 1_000_000

#: Characters per synthetic document — a few chunks each at the default 512.
_DOC_CHARS = 1_600

#: Distinct filler words cycled through documents so BM25 postings are not
#: degenerate (a single repeated token would make the rebuild unrealistically
#: cheap).
_WORDS = [
    "retrieval",
    "index",
    "chunk",
    "citation",
    "manifest",
    "vector",
    "lexical",
    "dense",
    "fusion",
    "snapshot",
    "collection",
    "offset",
]


def _write_corpus(root: Path, doc_count: int) -> None:
    for i in range(doc_count):
        words = [_WORDS[(i + j) % len(_WORDS)] + str((i * 7 + j) % 97) for j in range(260)]
        text = " ".join(words)[:_DOC_CHARS]
        (root / f"doc-{i:05d}.md").write_text(text, encoding="utf-8")


async def _ingest(index_dir: Path, collection: str, corpus: Path, *, dense: bool) -> None:
    store = await SQLiteMetadataStore.open(index_dir, collection)
    try:
        if dense:
            vector_store = await LanceDBVectorStore.open(index_dir / f"{collection}.lance")
            indexer = Indexer(
                store,
                FileLoader(allowed_base_dir=corpus),
                embedder=InMemoryEmbedder(),
                vector_store=vector_store,
            )
        else:
            indexer = Indexer(store, FileLoader(allowed_base_dir=corpus))
        await indexer.index_directory(str(corpus))
    finally:
        await store.close()


async def _time_open(index_dir: Path, collection: str, *, dense: bool, repeats: int) -> list[float]:
    """Time ``repeats`` fresh opens (store + optional LanceDB + Retriever), in ms."""
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        store = await SQLiteMetadataStore.open(index_dir, collection)
        try:
            if dense:
                vector_store = await LanceDBVectorStore.open(index_dir / f"{collection}.lance")
                await Retriever.open(store, embedder=InMemoryEmbedder(), vector_store=vector_store)
            else:
                await Retriever.open(store)
        finally:
            await store.close()
        samples.append((time.perf_counter_ns() - started) / _NS_PER_MS)
    return samples


def _report(label: str, samples: list[float]) -> None:
    print(
        f"{label}: min={min(samples):.1f}ms "
        f"median={statistics.median(samples):.1f}ms "
        f"mean={statistics.fmean(samples):.1f}ms (n={len(samples)})"
    )


async def _run(sizes: list[int], repeats: int) -> None:
    for size in sizes:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus"
            corpus.mkdir()
            _write_corpus(corpus, size)

            bm25_dir = root / "bm25-index"
            await _ingest(bm25_dir, "bench", corpus, dense=False)
            store = await SQLiteMetadataStore.open(bm25_dir, "bench")
            try:
                chunk_count = len(await store.get_chunks())
            finally:
                await store.close()

            dense_dir = root / "dense-index"
            await _ingest(dense_dir, "bench", corpus, dense=True)

            print(f"\ncorpus: {size} documents, {chunk_count} chunks")
            bm25_samples = await _time_open(bm25_dir, "bench", dense=False, repeats=repeats)
            dense_samples = await _time_open(dense_dir, "bench", dense=True, repeats=repeats)
            _report("  bm25-only open", bm25_samples)
            _report("  dense open    ", dense_samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 100, 1000])
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    asyncio.run(_run(args.sizes, args.repeats))


if __name__ == "__main__":
    main()

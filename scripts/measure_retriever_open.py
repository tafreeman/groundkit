"""Measure retrieval-path cost against ADR-0002's and ADR-0013's revisit triggers.

ADR-0002 accepts the O(corpus) BM25 rebuild-at-open with an explicit revisit
trigger: persisted postings are reconsidered only "if rebuild-at-open time is
*measured* to be a problem for a real corpus size — that measurement is the
trigger, not a guess made now." ADR-0013 leaves this script owed "a mode that
times a warm acquire against a rebuild so this claim is measured rather than
asserted." Both are the same obligation: decide the scale work by measurement,
not by argument. Per repo policy ("no number in any doc that wasn't
generated"), docs reference *this script*, never its output.

Four sections, selectable with ``--sections``. Each states the cost it prices
and what would bound a fix for it; none states a verdict, because the verdict
is whatever a run prints on the machine and corpus it was run against.

``open``
    ``Retriever.open()`` as the corpus grows, BM25-only and dense. Phase 3
    Wave C made ``open()`` heavier (a manifest verification, a LanceDB
    connection, and the open()-time document snapshot join the BM25 rebuild —
    phase-3 spec R4), and the dense timing includes
    ``LanceDBVectorStore.open()`` because a per-request service would pay
    exactly that composite cost. This is also the height of the cliff the
    ``acquire`` section's rebuild path falls off.

``search``
    ``BM25Index.search`` (the scoring loop alone) and ``Retriever.search``
    (the same loop plus the store join), per query, over queries of *measured*
    selectivity. Today the loop visits every chunk regardless of how few
    contain a query term, so the reported candidate fraction — chunks holding
    at least one query term, over all chunks — is the share of that loop a
    postings list could skip, and therefore the ceiling on what one could win.
    A query whose terms match everything has no headroom at all; reporting the
    fraction beside the time is what keeps that visible.

``records``
    the full-table ``get_document_records()`` read ``Retriever.search``
    performs on **every** query, against a keyed single-row read of the same
    table, plus the amplification: rows materialized per query against the at
    most ``top_k`` rows the join actually consumes. The keyed number is a
    lower bound, not an equivalent — it is an existing indexed single-row
    SELECT returning one column and building no model, where a keyed record
    read would validate one ``DocumentRecord``.

``acquire``
    ``CollectionRuntime.acquire()`` warm (generation unchanged, cache hit)
    against rebuilding (generation bumped first), then an ingest window: the
    same number of commits timed with and without a concurrent acquire loop,
    reporting how many acquires that loop completed and how many hit the
    cache. A commit here is one ``upsert_document`` against a single reserved
    source — that is what an ingest commits per document and what bumps the
    generation, and it is *cheaper* than a real one (no chunk rows), so the
    contention it shows is a floor.

Method: for each corpus size, synthesize documents of fixed shape into a temp
directory, ingest them BM25-only — and, when the ``open`` section is selected,
a second time dense-enabled with the in-memory hash embedder, since semantic
signal is irrelevant here and only vector plumbing cost is timed — then run the
selected sections over that collection and report min/median/mean wall-clock
per configuration. Every other section reads the BM25-only collection, so the
cost they price is the one a default install pays. Queries are derived
from the corpus's own measured term frequencies rather than written in, so no
query string here can quietly stop being rare or common when the generator
changes. Everything runs offline, with no credentials and no network.

Sections run in the order listed, which matters in one place: ``acquire``
commits rows to the collection, so it runs last and never perturbs the
``records`` counts.

Usage::

    uv run python scripts/measure_retriever_open.py
    uv run python scripts/measure_retriever_open.py --sizes 100 1000 5000 --repeats 20
    uv run python scripts/measure_retriever_open.py --sections search records
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import tempfile
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from functools import partial
from pathlib import Path

from groundkit.contracts import Chunk
from groundkit.index.bm25 import BM25Index, _tokenize
from groundkit.index.dense import LanceDBVectorStore
from groundkit.index.metadata import SQLiteMetadataStore
from groundkit.indexer import Indexer
from groundkit.ingestion.loaders import FileLoader
from groundkit.providers.embeddings import InMemoryEmbedder
from groundkit.retrieval.search import Retriever
from groundkit.runtime import AcquiredRetriever, CollectionRuntime

_NS_PER_MS = 1_000_000

#: Characters per synthetic document — a few chunks each at the default 512.
_DOC_CHARS = 1_600

#: Words per synthetic document before truncation to ``_DOC_CHARS``.
_DOC_WORDS = 260

#: Collection every section measures against.
_COLLECTION = "bench"

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

#: Prefix of a per-document unique token. The filler vocabulary above is
#: bounded and near-uniform, so without this the corpus has no genuinely
#: selective term and the ``search`` section would measure only the
#: no-headroom end of the range — understating a postings list by an artifact
#: of the generator rather than a property of retrieval.
_MARKER_PREFIX = "zmarker"

#: Source the ``acquire`` section commits under. One reserved source, re-upserted,
#: so the generation advances exactly as an ingest advances it while the corpus
#: the rebuild reads stays the size the other sections measured.
_BUMP_SOURCE = "measure://generation-bump"

#: Result cap for every timed query, matching ``RetrievalConfig.top_k``'s default.
_DEFAULT_TOP_K = 5

#: Commits in the simulated ingest window of the ``acquire`` section.
_DEFAULT_INGEST_COMMITS = 20

#: Repeats for the sub-millisecond operations (queries, keyed reads, warm
#: acquires), which need more samples than an O(corpus) open to separate the
#: signal from scheduler noise.
_DEFAULT_QUERY_REPEATS = 50

_SECTIONS = ("open", "search", "records", "acquire")

_PERCENT = 100.0

#: Threshold at which ``_ms`` switches from one decimal to three.
_FINE_MS = 1.0


def _marker(doc_index: int) -> str:
    """Return the unique token planted at the head of one synthetic document."""
    return f"{_MARKER_PREFIX}{doc_index:05d}"


def _write_corpus(root: Path, doc_count: int) -> None:
    for i in range(doc_count):
        words = [_WORDS[(i + j) % len(_WORDS)] + str((i * 7 + j) % 97) for j in range(_DOC_WORDS)]
        text = f"{_marker(i)} " + " ".join(words)
        (root / f"doc-{i:05d}.md").write_text(text[:_DOC_CHARS], encoding="utf-8")


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


def _sync_samples(operation: Callable[[], object], repeats: int) -> list[float]:
    """Time ``repeats`` calls of a synchronous operation, in ms."""
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        operation()
        samples.append((time.perf_counter_ns() - started) / _NS_PER_MS)
    return samples


async def _async_samples(operation: Callable[[], Awaitable[object]], repeats: int) -> list[float]:
    """Time ``repeats`` calls of an awaitable operation, in ms."""
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        await operation()
        samples.append((time.perf_counter_ns() - started) / _NS_PER_MS)
    return samples


def _ms(value: float) -> str:
    """Format a millisecond figure, keeping resolution below the millisecond.

    One decimal at or above ``_FINE_MS`` — the format the ``open`` section has
    always printed, so its output stays comparable with earlier runs — and
    three below it, because a query or a warm acquire rounds to ``0.0ms`` at
    one decimal and a measurement that prints as zero is not a measurement.
    """
    return f"{value:.1f}ms" if value >= _FINE_MS else f"{value:.3f}ms"


def _report(label: str, samples: list[float]) -> None:
    print(
        f"{label}: min={_ms(min(samples))} "
        f"median={_ms(statistics.median(samples))} "
        f"mean={_ms(statistics.fmean(samples))} (n={len(samples)})"
    )


def _chunk_terms(chunks: list[Chunk]) -> list[set[str]]:
    """Return each chunk's term set, tokenized exactly as the index tokenizes.

    The index's own private ``_tokenize`` is imported rather than
    reimplemented: the candidate fraction reported below is only meaningful
    computed with the tokenizer the scoring loop actually uses, and a local
    copy would drift silently — turning the one number that bounds the
    postings-list win into a guess.
    """
    return [set(_tokenize(chunk.content)) for chunk in chunks]


def _candidate_count(chunk_terms: list[set[str]], query: str) -> int:
    """Return how many chunks hold at least one of ``query``'s terms.

    This is the union of the query terms' postings — the set a postings list
    would restrict the scoring loop to, and the only chunks that can score
    above zero. Every other chunk is scored to ``0.0`` today and then
    discarded, so the difference between this and the corpus size is the
    wasted half of the loop, measured rather than assumed.
    """
    terms = set(_tokenize(query))
    return sum(1 for chunk in chunk_terms if chunk & terms)


def _query_plan(chunk_terms: list[set[str]]) -> list[tuple[str, str]]:
    """Derive ``(label, query)`` pairs from the corpus's measured term frequencies.

    Ranked by (frequency, term) so the choice is deterministic under ties and
    reproducible across runs of the same generator. The two-term query is
    included because the union is only as selective as its *least* selective
    term: a rare term paired with a common one recovers nothing, which is the
    case a single-term measurement would hide.
    """
    freqs: Counter[str] = Counter()
    for chunk in chunk_terms:
        freqs.update(chunk)
    ranked = sorted(freqs, key=lambda term: (freqs[term], term))
    rarest = ranked[0]
    median = ranked[len(ranked) // 2]
    commonest = ranked[-1]
    return [
        ("rarest term", rarest),
        ("median term", median),
        ("commonest term", commonest),
        ("rarest + commonest", f"{rarest} {commonest}"),
    ]


async def _measure_open(bm25_dir: Path, dense_dir: Path, repeats: int) -> None:
    """Section ``open``: ADR-0002's rebuild-at-open cost, BM25-only and dense."""
    print("  open (ADR-0002 rebuild-at-open; also the acquire section's rebuild cost)")
    bm25_samples = await _time_open(bm25_dir, _COLLECTION, dense=False, repeats=repeats)
    dense_samples = await _time_open(dense_dir, _COLLECTION, dense=True, repeats=repeats)
    _report("    bm25-only open", bm25_samples)
    _report("    dense open    ", dense_samples)


async def _measure_search(
    store: SQLiteMetadataStore,
    chunk_terms: list[set[str]],
    *,
    top_k: int,
    repeats: int,
) -> None:
    """Section ``search``: per-query cost against measured query selectivity."""
    print("  search (every query scores every chunk; candidates bound what postings could skip)")
    total = len(chunk_terms)
    if not total:
        print("    empty collection; nothing to query")
        return
    bm25 = await BM25Index.from_store(store)
    retriever = await Retriever.open(store)
    for label, query in _query_plan(chunk_terms):
        candidates = _candidate_count(chunk_terms, query)
        share = _PERCENT * candidates / total
        print(f"    {label} {query!r}: candidates={candidates}/{total} ({share:.2f}%)")
        index_samples = _sync_samples(partial(bm25.search, query, top_k=top_k), repeats)
        retriever_samples = await _async_samples(
            partial(retriever.search, query, top_k=top_k, mode="bm25"), repeats
        )
        # The gap between these two is the store join: the full-table read the
        # records section prices, plus the to_thread hop and citation resolve.
        _report("      BM25Index.search ", index_samples)
        _report("      Retriever.search ", retriever_samples)


async def _measure_records(store: SQLiteMetadataStore, *, top_k: int, repeats: int) -> None:
    """Section ``records``: the full-table read every query pays, against a keyed read."""
    print("  document records (read in full on every Retriever.search, to use at most top_k)")
    records = await store.get_document_records()
    if not records:
        print("    empty collection; nothing to read")
        return
    probe = sorted(record.source for record in records.values())[len(records) // 2]
    full = await _async_samples(store.get_document_records, repeats)
    keyed = await _async_samples(partial(store.get_document_hash, probe), repeats)
    _report("    full-table get_document_records()", full)
    _report("    keyed single-row read (lower bound)", keyed)

    keyed_budget = top_k * statistics.median(keyed)
    ratio = f"{statistics.median(full) / keyed_budget:.1f}x" if keyed_budget > 0 else "n/a"
    print(
        f"    rows materialized per query: {len(records)}; "
        f"rows the join consumes: <= {top_k}; "
        f"full-table median vs {top_k} keyed reads: {ratio}"
    )


async def _measure_queries(
    index_dir: Path,
    chunk_terms: list[set[str]],
    *,
    sections: list[str],
    top_k: int,
    repeats: int,
) -> None:
    """Run the two sections that read the same collection through one open store."""
    store = await SQLiteMetadataStore.open(index_dir, _COLLECTION)
    try:
        if "search" in sections:
            await _measure_search(store, chunk_terms, top_k=top_k, repeats=repeats)
        if "records" in sections:
            await _measure_records(store, top_k=top_k, repeats=repeats)
    finally:
        await store.close()


async def _bump(store: SQLiteMetadataStore, seq: int) -> None:
    """Commit one document row, advancing the generation exactly as an ingest does."""
    await store.upsert_document(
        source=_BUMP_SOURCE,
        document_id=f"bump-{seq:08d}",
        content_hash=f"{seq:064x}",
    )


async def _commit_window(store: SQLiteMetadataStore, commits: int, offset: int) -> float:
    """Commit ``commits`` document rows back to back; return the wall time in ms."""
    started = time.perf_counter_ns()
    for i in range(commits):
        await _bump(store, offset + i)
    return (time.perf_counter_ns() - started) / _NS_PER_MS


async def _acquire_until(runtime: CollectionRuntime, stop: asyncio.Event) -> tuple[int, int]:
    """Acquire in a loop until ``stop``; return ``(acquires, cache hits)``.

    A hit is identity of the returned :class:`AcquiredRetriever` with the
    previous one — the cache hands back the same object, a rebuild cannot.
    """
    acquires = 0
    hits = 0
    previous: AcquiredRetriever | None = None
    while not stop.is_set():
        acquired = await runtime.acquire()
        acquires += 1
        if acquired is previous:
            hits += 1
        previous = acquired
    return acquires, hits


async def _measure_acquire(index_dir: Path, *, repeats: int, commits: int) -> None:
    """Section ``acquire``: ADR-0013's warm-vs-rebuild mode, plus an ingest window."""
    print("  acquire (ADR-0013 cached retriever; a generation bump costs a full rebuild)")
    store = await SQLiteMetadataStore.open(index_dir, _COLLECTION)
    runtime = CollectionRuntime(store, collection=_COLLECTION)
    try:
        await runtime.acquire()  # Prime the cache; the warm path is what follows.
        warm = await _async_samples(runtime.acquire, repeats)
        _report("    warm acquire (cache hit)", warm)

        rebuild: list[float] = []
        for seq in range(repeats):
            await _bump(store, seq)
            started = time.perf_counter_ns()
            await runtime.acquire()
            rebuild.append((time.perf_counter_ns() - started) / _NS_PER_MS)
        _report("    rebuild acquire         ", rebuild)

        quiet = await _commit_window(store, commits, repeats)
        stop = asyncio.Event()
        reader = asyncio.ensure_future(_acquire_until(runtime, stop))
        contended = await _commit_window(store, commits, repeats + commits)
        stop.set()
        acquires, hits = await reader
        print(f"    ingest window of {commits} commits:")
        print(f"      commits alone         : {_ms(quiet)}")
        print(
            f"      commits + acquire loop: {_ms(contended)} "
            f"(acquires={acquires}, cache hits={hits})"
        )
    finally:
        await runtime.aclose()


async def _run(
    sizes: list[int],
    repeats: int,
    *,
    query_repeats: int,
    top_k: int,
    commits: int,
    sections: list[str],
) -> None:
    for size in sizes:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = root / "corpus"
            corpus.mkdir()
            _write_corpus(corpus, size)

            bm25_dir = root / "bm25-index"
            await _ingest(bm25_dir, _COLLECTION, corpus, dense=False)
            store = await SQLiteMetadataStore.open(bm25_dir, _COLLECTION)
            try:
                chunk_terms = _chunk_terms(await store.get_chunks())
            finally:
                # Closed before the open section so its timings are measured
                # against the same one-connection state a fresh process sees.
                await store.close()
            print(f"\ncorpus: {size} documents, {len(chunk_terms)} chunks")

            if "open" in sections:
                dense_dir = root / "dense-index"
                await _ingest(dense_dir, _COLLECTION, corpus, dense=True)
                await _measure_open(bm25_dir, dense_dir, repeats)

            if "search" in sections or "records" in sections:
                await _measure_queries(
                    bm25_dir, chunk_terms, sections=sections, top_k=top_k, repeats=query_repeats
                )

            # Last: this section commits rows, so running it earlier would
            # change the document count the records section reports.
            if "acquire" in sections:
                await _measure_acquire(bm25_dir, repeats=repeats, commits=commits)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", type=int, nargs="+", default=[10, 100, 1000])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--query-repeats", type=int, default=_DEFAULT_QUERY_REPEATS)
    parser.add_argument("--top-k", type=int, default=_DEFAULT_TOP_K)
    parser.add_argument("--ingest-commits", type=int, default=_DEFAULT_INGEST_COMMITS)
    parser.add_argument("--sections", nargs="+", choices=_SECTIONS, default=list(_SECTIONS))
    args = parser.parse_args()
    asyncio.run(
        _run(
            args.sizes,
            args.repeats,
            query_repeats=args.query_repeats,
            top_k=args.top_k,
            commits=args.ingest_commits,
            sections=args.sections,
        )
    )


if __name__ == "__main__":
    main()

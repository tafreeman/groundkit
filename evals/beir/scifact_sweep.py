"""Isolate the sources of the SciFact gap: chunking vs BM25 parameters.

At chunk_size 12000 every SciFact document (max 10,126 chars) becomes ONE chunk,
so chunk-level ranking IS document-level ranking and the chunking confound is
removed entirely. Whatever gap survives that is tokenization or k1/b.
"""

import asyncio
import json
import math
import sys
import tempfile
from pathlib import Path

from groundkit.config import ChunkingConfig, RetrievalConfig
from groundkit.index.metadata import SQLiteMetadataStore
from groundkit.indexer import Indexer
from groundkit.ingestion.loaders import FileLoader
from groundkit.retrieval.search import Retriever

BASE = Path(sys.argv[1])
CORPUS = BASE / "corpus"
CAND = 50
SEPS = ["\n\n", "\n", ". ", " ", ""]


def score(order, gold, k=10):
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(order[:k]) if d in gold)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(gold), k)))
    rr = next((1 / i for i, d in enumerate(order, 1) if d in gold), 0.0)
    return (
        dcg / idcg if idcg else 0.0,
        rr,
        len(set(order[:10]) & gold) / len(gold),
        len(set(order[:1]) & gold) / len(gold),
    )


async def run(judgments, gold, chunk_size, overlap, k1, b, store_cache, root):
    key = (chunk_size, overlap)
    if key not in store_cache:
        # One subdirectory per chunking config under the single root `main`
        # owns. Three stores stay open concurrently via `store_cache`, so the
        # lifetime that matters is the whole sweep's, not this call's -- which
        # is why the root is passed in rather than created here.
        tmp = root / f"{chunk_size}-{overlap}"
        tmp.mkdir(parents=True, exist_ok=True)
        st = await SQLiteMetadataStore.open(index_dir=tmp, collection="s")
        idx = Indexer(
            store=st,
            loader=FileLoader(allowed_base_dir=CORPUS),
            chunking_config=ChunkingConfig(
                chunk_size=chunk_size, chunk_overlap=overlap, separators=SEPS
            ),
        )
        rep = await idx.index_directory(str(CORPUS))
        store_cache[key] = (st, rep.chunks_written)
    st, nchunks = store_cache[key]
    r = await Retriever.open(store=st, config=RetrievalConfig(bm25_k1=k1, bm25_b=b))
    tot = [0.0] * 4
    for j in judgments:
        resp = await r.search(j["query"], top_k=CAND, mode="bm25")
        seen, order = set(), []
        for x in resp.results:
            d = Path(x.source).name
            if d not in seen:
                seen.add(d)
                order.append(d)
        s = score(order, gold[j["query_id"]])
        tot = [t + v for t, v in zip(tot, s, strict=True)]
    n = len(judgments)
    return nchunks, [t / n for t in tot]


async def main():
    judgments = [
        json.loads(line)
        for line in (BASE / "judgments.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    gold = {j["query_id"]: {g["doc"] for g in j["gold"]} for j in judgments}
    cache = {}
    print(
        f"{'chunking':>14s} {'chunks':>7s} {'k1':>5s} {'b':>5s} | "
        f"{'nDCG@10':>8s} {'MRR':>7s} {'R@10':>7s} {'R@1':>7s}"
    )
    print("-" * 70)
    trials = [
        (512, 64, 1.5, 0.75, "groundkit default"),
        (12000, 0, 1.5, 0.75, "whole document"),
        (1500, 0, 1.5, 0.75, "chunk ~= median doc"),
    ]
    # One root for every store the sweep opens, removed when it finishes.
    # `store_cache` keeps all three open until then, so cleanup can only
    # happen out here.
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            for cs, ov, k1, b, label in trials:
                nch, m = await run(judgments, gold, cs, ov, k1, b, cache, Path(tmpdir))
                tag = f"{cs}/{ov}"
                print(
                    f"{tag:>14s} {nch:7,d} {k1:5.1f} {b:5.2f} | {m[0]:8.4f} {m[1]:7.3f} "
                    f"{m[2]:7.3f} {m[3]:7.3f}   {label}"
                )
        finally:
            # On Windows an open SQLite handle blocks the tempdir removal.
            for store, _ in cache.values():
                await store.close()
    print()
    print("BEIR published BM25 (Elasticsearch, English analyzer): nDCG@10 = 0.665")
    for st, _ in cache.values():
        await st.close()


asyncio.run(main())

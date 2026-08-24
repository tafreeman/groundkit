"""Score groundkit's SciFact retrieval at DOCUMENT level, the unit BEIR publishes.

BEIR's nDCG@10 ranks documents. groundkit ranks chunks. Collapsing the chunk
ranking to its first-seen distinct documents and scoring the top 10 of those is
the only way the two numbers mean the same thing.
"""

import asyncio
import json
import math
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from groundkit.config import ChunkingConfig
from groundkit.index.metadata import SQLiteMetadataStore
from groundkit.indexer import Indexer
from groundkit.ingestion.loaders import FileLoader
from groundkit.retrieval.search import Retriever

BASE = Path(sys.argv[1])
CORPUS = BASE / "corpus"
CFG = ChunkingConfig(chunk_size=512, chunk_overlap=64, separators=["\n\n", "\n", ". ", " ", ""])
CAND = 50  # groundkit caps top_k at MAX_TOP_K=50; this is the deepest pool available


def ndcg_at_k(ranked_docs, gold, k=10):
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked_docs[:k]) if d in gold)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / idcg if idcg else 0.0


def recall_at_k(ranked_docs, gold, k):
    return len(set(ranked_docs[:k]) & gold) / len(gold)


def rr(ranked_docs, gold):
    for i, d in enumerate(ranked_docs, 1):
        if d in gold:
            return 1.0 / i
    return 0.0


async def main():
    judgments = [
        json.loads(line)
        for line in (BASE / "judgments.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    gold = {j["query_id"]: {g["doc"] for g in j["gold"]} for j in judgments}

    # TemporaryDirectory rather than mkdtemp: the index here is scratch, and a
    # bare mkdtemp left one SQLite store per run behind in %TEMP% forever.
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        store = await SQLiteMetadataStore.open(index_dir=tmp, collection="scifact")
        try:
            idx = Indexer(
                store=store, loader=FileLoader(allowed_base_dir=CORPUS), chunking_config=CFG
            )
            t0 = time.perf_counter()
            rep = await idx.index_directory(str(CORPUS))
            ingest = time.perf_counter() - t0
            t0 = time.perf_counter()
            r = await Retriever.open(store=store)
            opened = (time.perf_counter() - t0) * 1000
            print(f"ingest {rep.chunks_written:,} chunks in {ingest:.1f}s | open {opened:.0f} ms")

            agg = defaultdict(float)
            lat = []
            ndocs = []
            for j in judgments:
                t0 = time.perf_counter()
                resp = await r.search(j["query"], top_k=CAND, mode="bm25")
                lat.append((time.perf_counter() - t0) * 1000)
                seen, order = set(), []
                for x in resp.results:
                    d = Path(x.source).name
                    if d not in seen:
                        seen.add(d)
                        order.append(d)
                ndocs.append(len(order))
                g = gold[j["query_id"]]
                agg["ndcg@10"] += ndcg_at_k(order, g, 10)
                agg["recall@10"] += recall_at_k(order, g, 10)
                agg["recall@5"] += recall_at_k(order, g, 5)
                agg["recall@1"] += recall_at_k(order, g, 1)
                agg["mrr"] += rr(order, g)
        finally:
            # Before the tempdir is removed: on Windows an open SQLite handle
            # blocks the cleanup outright.
            await store.close()
    n = len(judgments)
    lat.sort()
    print(
        f"\nDOCUMENT-LEVEL, {n} queries, {CAND} chunks/query -> "
        f"{sum(ndocs) / n:.1f} distinct docs avg"
    )
    for k in ("recall@1", "recall@5", "recall@10", "mrr", "ndcg@10"):
        print(f"  {k:10s} {agg[k] / n:.3f}")
    print(f"  latency p50 {lat[len(lat) // 2]:.0f} ms  p95 {lat[int(len(lat) * 0.95)]:.0f} ms")
    print("\n  BEIR published BM25 nDCG@10 on SciFact: 0.665")
    await store.close()


asyncio.run(main())

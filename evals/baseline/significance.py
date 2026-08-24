"""Paired bootstrap over per-query deltas: is groundkit vs generic distinguishable at n=36?

Standard IR practice (Smucker/Allan/Carterette CIKM 2007): resample query indices with
replacement, recompute the mean per-query difference, take the 2.5/97.5 percentiles.
A CI straddling zero means the systems are not separated by this test collection.
"""

import asyncio
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import generic_rag as gr
import numpy as np
from generic_rag import (
    RRF_K,
    TOP_K,
    bm25_search,
    dense_search,
    embed,
    load_corpus,
    resolve_gold,
    rrf,
    score,
)

from groundkit.config import ChunkingConfig, EmbeddingConfig
from groundkit.index.dense import LanceDBVectorStore
from groundkit.index.metadata import SQLiteMetadataStore
from groundkit.indexer import Indexer
from groundkit.ingestion.loaders import FileLoader
from groundkit.providers.embeddings import build_embedder
from groundkit.retrieval.search import Retriever

# evals/baseline/significance.py -> parents[0]=evals/baseline, [1]=evals, [2]=repo root.
ROOT = Path(__file__).resolve().parents[2]
RNG = np.random.default_rng(7)
corpus = load_corpus(ROOT / "evals/corpus")
judgments = [
    json.loads(line)
    for line in (ROOT / "evals/judgments.jsonl").open(encoding="utf-8")
    if line.strip()
]
gold = resolve_gold(corpus, judgments)
answerable = [j for j in judgments if j["gold"]]


async def groundkit_per_query():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        store = await SQLiteMetadataStore.open(index_dir=tmp, collection="gk")
        emb = build_embedder(
            EmbeddingConfig(provider="ollama", model_name="nomic-embed-text", dimensions=768)
        )
        vs = await LanceDBVectorStore.open(db_path=tmp / "lance")
        idx = Indexer(
            store=store,
            loader=FileLoader(allowed_base_dir=ROOT),
            chunking_config=ChunkingConfig(
                chunk_size=512, chunk_overlap=64, separators=["\n\n", "\n", ". ", " ", ""]
            ),
            embedder=emb,
            vector_store=vs,
            collection="gk",
        )
        await idx.index_directory(str(ROOT / "evals/corpus"))
        r = await Retriever.open(store=store, embedder=emb, vector_store=vs, collection="gk")
        out = {}
        for j in answerable:
            resp = await r.search(j["query"], top_k=TOP_K, mode="hybrid")
            ranked = [
                {"doc": Path(x.source).name, "start": x.start_offset, "end": x.end_offset}
                for x in resp.results
            ]
            out[j["query_id"]] = score(ranked, gold[j["query_id"]], TOP_K)
        await store.close()
        return out


def generic_per_query(size, overlap):
    gr.CHUNK_SIZE, gr.CHUNK_OVERLAP = size, overlap
    chunks = gr.build_chunks(corpus)
    by_id = {c["id"]: c for c in chunks}
    ids = [c["id"] for c in chunks]
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        con = sqlite3.connect(tmp / "g.sqlite3")
        con.execute("CREATE VIRTUAL TABLE chunks USING fts5(chunk_id UNINDEXED, text)")
        con.executemany(
            "INSERT INTO chunks(chunk_id,text) VALUES (?,?)",
            ((c["id"], c["text"]) for c in chunks),
        )
        con.commit()
        mat = embed([c["text"] for c in chunks])
        qv = embed([j["query"] for j in answerable])
        out = {}
        for i, j in enumerate(answerable):
            b = bm25_search(con, j["query"], TOP_K)
            d = dense_search(mat, ids, qv[i], TOP_K)
            f = rrf([b, d], RRF_K)[:TOP_K]
            out[j["query_id"]] = score([by_id[c] for c, _ in f], gold[j["query_id"]], TOP_K)
    return out, len(chunks)


def bootstrap(a: dict, b: dict, metric: str, n=10000):
    qids = list(a.keys())
    d = np.array([a[q][metric] - b[q][metric] for q in qids])
    means = np.array([RNG.choice(d, size=len(d), replace=True).mean() for _ in range(n)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    non_positive = int((means <= 0).sum())
    non_negative = int((means >= 0).sum())
    # (r+1)/(B+1): a Monte Carlo estimate over B resamples cannot resolve a
    # probability below 1/B, so a zero tail count must not print as p = 0.0.
    # This is a floor on the true p-value, not proof that it is nonzero.
    p = min(1.0, 2.0 * (min(non_positive, non_negative) + 1) / (n + 1))
    return d.mean(), lo, hi, p


gk = asyncio.run(groundkit_per_query())
print(f"groundkit fusion scored on {len(gk)} answerable queries\n")
for size, overlap in [(512, 64), (320, 40)]:
    gen, nch = generic_per_query(size, overlap)
    print(f"=== generic @ chunk {size}/{overlap} -> {nch} chunks  (groundkit: 84) ===")
    print(
        f"{'metric':12s} {'gk':>7s} {'generic':>8s} {'delta':>8s} "
        f"{'95% CI':>20s} {'p':>7s}  verdict"
    )
    for m in ("recall_at_1", "recall_at_5", "recall_at_10", "mrr", "ndcg_at_10"):
        mean, lo, hi, p = bootstrap(gk, gen, m)
        gkv = np.mean([gk[q][m] for q in gk])
        gnv = np.mean([gen[q][m] for q in gen])
        verdict = "SEPARATED" if (lo > 0 or hi < 0) else "indistinguishable"
        print(
            f"{m:12s} {gkv:7.3f} {gnv:8.3f} {mean:+8.3f} [{lo:+.3f}, {hi:+.3f}] {p:7.3f}  {verdict}"
        )
    print()

"""Run the generic pipeline over groundkit's golden corpus and report quality + cost."""

import json
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from generic_rag import (
    RRF_K,
    TOP_K,
    aggregate,
    bm25_search,
    build_chunks,
    dense_search,
    embed,
    load_corpus,
    resolve_gold,
    rrf,
    score,
)

# evals/baseline/run_generic.py -> parents[0]=evals/baseline, [1]=evals, [2]=repo root.
ROOT = Path(__file__).resolve().parents[2]
corpus = load_corpus(ROOT / "evals/corpus")
judgments = [
    json.loads(line)
    for line in (ROOT / "evals/judgments.jsonl").open(encoding="utf-8")
    if line.strip()
]
gold = resolve_gold(corpus, judgments)
answerable = [j for j in judgments if j["gold"]]
noanswer = [j for j in judgments if not j["gold"]]
print(
    f"corpus: {len(corpus)} docs  |  judgments: {len(judgments)} "
    f"({len(answerable)} answerable, {len(noanswer)} no-answer)"
)


def main() -> None:
    t0 = time.perf_counter()
    chunks = build_chunks(corpus)
    by_id = {c["id"]: c for c in chunks}
    ids = [c["id"] for c in chunks]
    t_chunk = time.perf_counter() - t0
    print(f"chunks: {len(chunks)}  (chunking {t_chunk * 1000:.0f} ms)")

    # Scoped to a context manager so the index directory is always cleaned up,
    # including on an exception raised anywhere in the block below.
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        t0 = time.perf_counter()
        con = sqlite3.connect(tmp / "g.sqlite3")
        con.execute("CREATE VIRTUAL TABLE chunks USING fts5(chunk_id UNINDEXED, text)")
        con.executemany(
            "INSERT INTO chunks(chunk_id, text) VALUES (?,?)",
            ((c["id"], c["text"]) for c in chunks),
        )
        con.commit()
        t_fts = time.perf_counter() - t0

        t0 = time.perf_counter()
        mat = embed([c["text"] for c in chunks])
        t_embed = time.perf_counter() - t0
        print(
            f"index: FTS5 {t_fts * 1000:.0f} ms  |  embed {len(chunks)} chunks {t_embed:.1f}s "
            f"({len(chunks) / t_embed:.0f} chunks/s)"
        )

        t0 = time.perf_counter()
        con2 = sqlite3.connect(tmp / "g.sqlite3")
        con2.execute("SELECT count(*) FROM chunks").fetchone()
        mat2 = np.array(mat)
        print(f"open: {(time.perf_counter() - t0) * 1000:.1f} ms (FTS5 + vectors already built)")

        qvecs = embed([j["query"] for j in judgments])
        qv = {j["query_id"]: qvecs[i] for i, j in enumerate(judgments)}

        results = {"bm25": [], "dense": [], "fusion": []}
        lat = {"bm25": [], "dense": [], "fusion": []}
        abstain = {"bm25": 0, "dense": 0, "fusion": 0}

        for j in judgments:
            qid = j["query_id"]
            runs = {}
            for stage in ("bm25", "dense", "fusion"):
                t0 = time.perf_counter()
                if stage == "bm25":
                    r = bm25_search(con2, j["query"], TOP_K)
                elif stage == "dense":
                    r = dense_search(mat2, ids, qv[qid], TOP_K)
                else:
                    r = rrf([runs["bm25"], runs["dense"]], RRF_K)[:TOP_K]
                lat[stage].append((time.perf_counter() - t0) * 1000)
                runs[stage] = r
                if j["gold"]:
                    results[stage].append(score([by_id[c] for c, _ in r], gold[qid], TOP_K))
                elif not r:
                    abstain[stage] += 1

    print()
    print(
        f"{'stage':8s} {'r@1':>6s} {'r@5':>6s} {'r@10':>6s} {'MRR':>6s} {'nDCG@10':>8s} "
        f"{'abstain':>8s} {'p50 ms':>8s}"
    )
    print("-" * 64)
    out = {}
    for stage in ("bm25", "dense", "fusion"):
        a = aggregate(results[stage])
        out[stage] = a
        ls = sorted(lat[stage])
        print(
            f"{stage:8s} {a['recall_at_1']:6.3f} {a['recall_at_5']:6.3f} {a['recall_at_10']:6.3f} "
            f"{a['mrr']:6.3f} {a['ndcg_at_10']:8.3f} {abstain[stage]:>5d}/{len(noanswer)} "
            f"{ls[len(ls) // 2]:8.2f}"
        )

    with (Path(__file__).parent / "generic_results.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "quality": out,
                "abstain": abstain,
                "n_chunks": len(chunks),
                "cost": {"chunk_ms": t_chunk * 1000, "fts_ms": t_fts * 1000, "embed_s": t_embed},
            },
            fh,
            indent=2,
        )


if __name__ == "__main__":
    main()

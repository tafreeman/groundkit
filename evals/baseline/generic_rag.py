"""A deliberately GENERIC RAG pipeline, for head-to-head comparison with groundkit.

What "generic" means here, concretely -- the choices a competent engineer makes
in an afternoon with standard tools, and the ones groundkit deliberately does not:

* plain dicts for chunks, no validation, no frozen models, no offset invariant
* fixed-size character chunking with overlap (no separator cascade)
* SQLite FTS5 for lexical -- persisted, opened in milliseconds, no rebuild
* a numpy matrix for dense -- the smallest thing that works
* RRF fusion at the same k=60 groundkit uses
* citations are (doc, text) with no way to verify them against the source

Same corpus, same judgments, same embedding model, same chunk size/overlap as
groundkit's EVAL_CHUNKING_CONFIG, so the only differences are architectural.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import httpx
import numpy as np

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
RRF_K = 60
TOP_K = 10
OLLAMA = "http://127.0.0.1:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"


# ---------------------------------------------------------------- ingestion


def load_corpus(corpus_dir: Path) -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(corpus_dir.glob("*.md"))}


def chunk_text(text: str) -> list[tuple[int, int]]:
    """Fixed-size character windows. Offsets kept ONLY so the scorer can be
    chunking-independent -- a generic pipeline would not persist or verify them."""
    step = CHUNK_SIZE - CHUNK_OVERLAP
    spans = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        spans.append((start, end))
        if end == len(text):
            break
        start += step
    return spans


def build_chunks(corpus: dict[str, str]) -> list[dict]:
    chunks = []
    for doc, text in corpus.items():
        for i, (start, end) in enumerate(chunk_text(text)):
            chunks.append(
                {
                    "id": f"{doc}#{i}",
                    "doc": doc,
                    "text": text[start:end],
                    "start": start,
                    "end": end,
                }
            )
    return chunks


# ---------------------------------------------------------------- embedding


def embed(texts: list[str], batch: int = 32) -> np.ndarray:
    out = []
    with httpx.Client(timeout=120) as client:
        for i in range(0, len(texts), batch):
            r = client.post(OLLAMA, json={"model": EMBED_MODEL, "input": texts[i : i + batch]})
            r.raise_for_status()
            out.extend(r.json()["embeddings"])
    m = np.asarray(out, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


# ---------------------------------------------------------------- retrieval

_FTS_SAFE = re.compile(r"[^\w\s]+")


def fts_query(q: str) -> str:
    terms = [t for t in _FTS_SAFE.sub(" ", q).split() if t]
    return " OR ".join(terms)


def bm25_search(con: sqlite3.Connection, q: str, k: int) -> list[tuple[str, float]]:
    expr = fts_query(q)
    if not expr:
        return []
    rows = con.execute(
        "SELECT chunk_id, bm25(chunks) FROM chunks WHERE chunks MATCH ? "
        "ORDER BY bm25(chunks) LIMIT ?",
        (expr, k),
    ).fetchall()
    # FTS5 bm25() is negative-better; flip so higher is better.
    return [(cid, -score) for cid, score in rows]


def dense_search(mat: np.ndarray, ids: list[str], qv: np.ndarray, k: int):
    sims = mat @ qv
    top = np.argsort(-sims)[:k]
    return [(ids[i], float(sims[i])) for i in top]


def rrf(runs: list[list[tuple[str, float]]], k: int) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for run in runs:
        for rank, (cid, _) in enumerate(run, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


# ---------------------------------------------------------------- scoring


def resolve_gold(
    corpus: dict[str, str], judgments: list[dict]
) -> dict[str, list[tuple[str, int, int]]]:
    """Resolve each gold quote to (doc, start, end). Fails closed, like groundkit's."""
    gold: dict[str, list[tuple[str, int, int]]] = {}
    for j in judgments:
        spans = []
        for g in j["gold"]:
            text = corpus[g["doc"]]
            idx = text.find(g["quote"])
            if idx < 0:
                raise SystemExit(f"gold quote not found in {g['doc']}: {g['quote'][:60]!r}")
            if text.find(g["quote"], idx + 1) >= 0:
                raise SystemExit(f"gold quote ambiguous in {g['doc']}: {g['quote'][:60]!r}")
            spans.append((g["doc"], idx, idx + len(g["quote"])))
        gold[j["query_id"]] = spans
    return gold


def is_hit(chunk: dict, spans: list[tuple[str, int, int]]) -> bool:
    """Chunking-independent relevance: same doc AND character spans overlap.

    This is what makes groundkit and the generic pipeline comparable at all --
    they chunk differently, so gold *chunk ids* cannot be shared, but the gold
    *character span* is a property of the corpus, not of either chunker.
    """
    for doc, gs, ge in spans:
        if chunk["doc"] == doc and chunk["start"] < ge and gs < chunk["end"]:
            return True
    return False


def score(ranked: list[dict], spans: list[tuple[str, int, int]], k: int = 10) -> dict:
    rel = [1 if is_hit(c, spans) else 0 for c in ranked[:k]]
    first = next((i + 1 for i, r in enumerate(rel) if r), None)
    dcg = sum(r / np.log2(i + 2) for i, r in enumerate(rel))
    ideal = [1] * min(len(spans), k)
    idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal)) or 1.0
    return {
        "recall_at_1": float(any(rel[:1])),
        "recall_at_5": float(any(rel[:5])),
        "recall_at_10": float(any(rel[:10])),
        "mrr": 1.0 / first if first else 0.0,
        "ndcg_at_10": dcg / idcg,
    }


def aggregate(per_query: list[dict]) -> dict:
    keys = ["recall_at_1", "recall_at_5", "recall_at_10", "mrr", "ndcg_at_10"]
    return {k: round(sum(q[k] for q in per_query) / len(per_query), 4) for k in keys}

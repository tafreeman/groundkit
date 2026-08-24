"""Compare embedding models on an adapted BEIR set — one persisted index per model.

UNTRACKED, local-only, alongside the rest of ``evals/beir/``.

    uv run --no-sync python evals/beir/embed_bakeoff.py --data <dir>/scifact-gk --work <dir>/bakeoff

Each model gets **its own collection and its own LanceDB directory**. That is not
tidiness — ADR-0004 binds a collection to the embedding identity triple
``(provider, model_name, dimensions)`` and refuses to mix, so one store per model
is the only shape the library permits. It also makes the run resumable: an index
that already exists is reused rather than re-embedded, which matters when a single
7.6B model takes ~40 minutes over 20k chunks.

Per-query scores are cached to ``scores/<model>.json`` as well, so re-running the
comparison — adding a metric, changing the bootstrap, plotting something — costs
nothing. Deleting a model's score file forces just that model to be re-scored;
deleting its index directory forces a re-embed.

Scoring is **document-level**: the chunk ranking is collapsed to first-seen distinct
documents before scoring, because BEIR's qrels and published numbers are per
document. Scoring chunks directly against document-level gold understates every
system (see RESULTS-scifact-2026-08-21.md).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from groundkit.config import ChunkingConfig, EmbeddingConfig
from groundkit.index.dense import LanceDBVectorStore
from groundkit.index.metadata import SQLiteMetadataStore
from groundkit.indexer import Indexer
from groundkit.ingestion.loaders import FileLoader
from groundkit.providers.embeddings import build_embedder
from groundkit.retrieval.search import Retriever

#: Chunking is held fixed across every model so the only variable is the embedder.
#: These are ``EVAL_CHUNKING_CONFIG``'s values, pinned here rather than imported so
#: a library-default change cannot silently move every boundary mid-comparison.
CFG = ChunkingConfig(chunk_size=512, chunk_overlap=64, separators=["\n\n", "\n", ". ", " ", ""])

#: ``top_k`` is capped at ``MAX_TOP_K`` (50). A chunk ranking needs a deeper pool
#: than a document ranking to yield the same top-10 documents, so take the cap.
CANDIDATES = 50

#: Bumped by hand whenever `doc_level_scores` changes how a number is computed.
#: It is part of the score cache's identity, so bumping it invalidates every
#: cached score rather than letting old and new scoring share one leaderboard.
SCORING_VERSION = 1

#: Written into a model's index directory only after ``index_directory`` returns
#: successfully. Its presence alone is not enough to trust a cached index -- see
#: ``_sentinel_valid`` -- because a bare directory-presence check treats a
#: partial index left by an interrupted run as complete, and a partial index
#: scores all-zeros, which then gets cached to ``scores/<model>.json`` as if it
#: were a real measurement.
SENTINEL_NAME = "index_complete.json"

#: Models grouped into **waves**; a wave runs concurrently, waves run in order.
#: ``(ollama tag, embedding width)``. Width is not guessed — it is what the model
#: actually returns, verified by a one-off ``/api/embed`` call; a mismatch here is
#: refused by ``EmbeddingConfig``/the manifest rather than silently truncating.
#:
#: The grouping is a measurement, not a preference. This machine's iGPU shares
#: 89 GB/s of DDR5 with the CPU, and a model's floor is the time to stream its
#: weights once per forward pass. Measured against that roofline: qwen3 (11 GB)
#: runs at ~100% of it — 8.5 chunks/s against a theoretical 8.1 — so anything
#: co-scheduled with it can only take bandwidth away. The others sit at 9-16%
#: of their own rooflines, bound by per-request overhead rather than bandwidth,
#: which is why running them together measured 1.26x faster than in sequence.
WAVES: list[list[tuple[str, int]]] = [
    [
        ("nomic-embed-text", 768),
        ("bge-m3", 1024),
        ("mxbai-embed-large", 1024),
        ("snowflake-arctic-embed2", 1024),
    ],
    [("qwen3-embedding", 4096)],
]

METRICS = ("ndcg_at_10", "mrr", "recall_at_1", "recall_at_10")


def score_ranking(order: list[str], gold: set[str], k: int = 10) -> dict[str, float]:
    """Binary-relevance IR metrics over a document ranking.

    Binary is correct here rather than a simplification: SciFact's qrels score
    every judged pair ``1``, so graded gain would have nothing to grade.
    """
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(order[:k]) if d in gold)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(gold), k)))
    rank = next((i for i, d in enumerate(order, 1) if d in gold), None)
    return {
        "ndcg_at_10": dcg / idcg if idcg else 0.0,
        "mrr": 1.0 / rank if rank else 0.0,
        "recall_at_1": len(set(order[:1]) & gold) / len(gold),
        "recall_at_10": len(set(order[:10]) & gold) / len(gold),
    }


def corpus_fingerprint(corpus: Path) -> str:
    """A content hash over every file in the adapted corpus directory.

    Names *and* bytes, in sorted order, so a rename, an edit, an addition and
    a deletion are all visible. A path plus a file count is not enough: a
    corpus edited or regenerated in place keeps both, and every downstream
    number would then be scored against an index of the previous contents
    with nothing to indicate it. The corpus is a flat directory of small
    per-document text files (see ``adapt_beir.py``), so reading it once per
    run costs a second or so against index builds measured in tens of
    minutes.
    """
    digest = hashlib.sha256()
    for path in sorted(p for p in corpus.rglob("*") if p.is_file()):
        digest.update(path.relative_to(corpus).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _sentinel_inputs(corpus: Path, model: str, dims: int) -> dict:
    """Inputs that must stay identical for a cached index to still be valid.

    ``corpus`` is resolved so a run from a different working directory or a
    relative-vs-absolute ``--data`` doesn't look like a different corpus, and
    fingerprinted by content so an in-place edit invalidates the cache too.
    The chunking config is included verbatim because it is the other input
    that determines every chunk boundary in the index.

    ``model`` and ``dims`` are included even though ``index_dir`` is already
    keyed by model slug, because the slug does not carry the dimension. If a
    dimension is corrected for a model that already has an index, the score
    cache invalidates (it records ``dims``) but the index would not, so the
    stored ADR-0004 manifest would then disagree with the new embedder and
    ``Retriever.open`` would raise. ``one_model`` catches broadly, so the
    model would be dropped from the leaderboard with only a FAILED line --
    a silently missing row rather than a rebuilt index.
    """
    resolved = corpus.resolve()
    return {
        "corpus_path": str(resolved),
        "corpus_fingerprint": corpus_fingerprint(resolved),
        "chunking_config": CFG.model_dump(),
        "model": model,
        "dims": dims,
    }


def _sentinel_valid(index_dir: Path, expected: dict) -> bool:
    """True only if a completion sentinel exists and matches the current inputs."""
    sentinel = index_dir / SENTINEL_NAME
    if not sentinel.exists():
        return False
    try:
        recorded = json.loads(sentinel.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return recorded == expected


def _write_sentinel(index_dir: Path, inputs: dict) -> None:
    """Write the completion sentinel atomically.

    A plain ``write_text`` can be interrupted mid-write (process kill, power
    loss), leaving a truncated or corrupt sentinel that ``_sentinel_valid``
    would then either wrongly trust (rare) or wrongly reject (harmless, just
    forces a re-index) -- but the truncated-then-trusted case is the one worth
    closing. Writing to a ``.tmp`` file and ``os.replace``-ing it over the real
    name means the sentinel is only ever fully-written or absent, never partial.
    """
    sentinel = index_dir / SENTINEL_NAME
    tmp_path = sentinel.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(inputs, indent=2), encoding="utf-8")
    os.replace(tmp_path, sentinel)


async def index_and_score(
    model: str, dims: int, corpus: Path, work: Path, judgments: list[dict], gold: dict
) -> dict:
    """Build (or reuse) this model's index, then score every query against it."""
    slug = model.replace(":", "-").replace("/", "-")
    index_dir = work / "index" / slug
    lance_dir = index_dir / "lance"

    sentinel_inputs = _sentinel_inputs(corpus, model, dims)
    fresh = not _sentinel_valid(index_dir, sentinel_inputs)
    if fresh:
        # Discard the whole directory, not just the sentinel. Two reasons, and
        # deleting the sentinel alone fixes only the second:
        #
        # 1. `Indexer` cannot re-embed into a collection whose ADR-0004
        #    manifest names a different embedding identity -- `_verify_identity`
        #    refuses it before any work, by design. So invalidating the
        #    sentinel on a dims change makes the run *try* to rebuild and then
        #    raise `IndexIdentityError`, which `one_model` catches and turns
        #    into a FAILED line: the model drops out of the leaderboard instead
        #    of being rebuilt. Delete-and-re-ingest is ADR-0004 decision 5's
        #    stated remedy for exactly this, and it is what the directory
        #    removal performs.
        # 2. Indexing mutates an existing store, so a rebuild that dies partway
        #    leaves a mix of the old corpus and the new one. Removing the
        #    directory first makes an interrupted rebuild fail closed; the
        #    worst case is a re-index nobody needed.
        shutil.rmtree(index_dir, ignore_errors=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    store = await SQLiteMetadataStore.open(index_dir=index_dir, collection="beir")
    embedder = build_embedder(EmbeddingConfig(provider="ollama", model_name=model, dimensions=dims))
    vectors = await LanceDBVectorStore.open(db_path=lance_dir)

    embed_seconds = 0.0
    chunks = 0
    try:
        if fresh:
            indexer = Indexer(
                store=store,
                loader=FileLoader(allowed_base_dir=corpus),
                chunking_config=CFG,
                embedder=embedder,
                vector_store=vectors,
                collection="beir",
            )
            started = time.perf_counter()
            report = await indexer.index_directory(str(corpus))
            embed_seconds = time.perf_counter() - started
            chunks = report.chunks_written
            # Only recorded once indexing has actually finished -- an
            # interrupted run must not leave a sentinel a later run would trust.
            _write_sentinel(index_dir, sentinel_inputs)

        retriever = await Retriever.open(
            store=store, embedder=embedder, vector_store=vectors, collection="beir"
        )
        per_query: dict[str, dict[str, float]] = {}
        latencies: list[float] = []
        for judgment in judgments:
            started = time.perf_counter()
            response = await retriever.search(judgment["query"], top_k=CANDIDATES, mode="dense")
            latencies.append((time.perf_counter() - started) * 1000)
            seen: set[str] = set()
            order: list[str] = []
            for result in response.results:
                doc = Path(result.source).name
                if doc not in seen:
                    seen.add(doc)
                    order.append(doc)
            per_query[judgment["query_id"]] = score_ranking(order, gold[judgment["query_id"]])
    finally:
        await store.close()

    latencies.sort()
    vector_bytes = sum(p.stat().st_size for p in lance_dir.rglob("*") if p.is_file())
    return {
        "model": model,
        "dimensions": dims,
        "chunks": chunks,
        "reused_index": not fresh,
        "embed_minutes": round(embed_seconds / 60, 2),
        "query_p50_ms": round(latencies[len(latencies) // 2], 1),
        "query_p95_ms": round(latencies[int(len(latencies) * 0.95)], 1),
        "vector_store_mb": round(vector_bytes / 1e6, 1),
        "per_query": per_query,
    }


def bootstrap(a: dict, b: dict, metric: str, rng: np.random.Generator) -> tuple:
    """Paired bootstrap over per-query deltas: (mean, lo, hi, p)."""
    deltas = np.array([a[q][metric] - b[q][metric] for q in a])
    resamples = 10000
    means = np.array(
        [rng.choice(deltas, size=len(deltas), replace=True).mean() for _ in range(resamples)]
    )
    lo, hi = np.percentile(means, [2.5, 97.5])
    non_positive = int((means <= 0).sum())
    non_negative = int((means >= 0).sum())
    # (r+1)/(B+1): a Monte Carlo estimate over B resamples cannot resolve a
    # probability below 1/B, so a zero tail count must not print as p = 0.0.
    # This is a floor on the true p-value, not proof that it is nonzero.
    p = min(1.0, 2.0 * (min(non_positive, non_negative) + 1) / (resamples + 1))
    return deltas.mean(), lo, hi, p


def _force_utf8_stdout() -> None:
    """Windows pipes default to cp1252, which cannot encode this script's output.

    Redirecting stdout to a file made the console encoding cp1252 and the first
    non-ASCII character killed the run before a single model had embedded. The
    output is fixed ASCII now, but reconfiguring is the belt to that braces: a
    model name or a future label containing a non-ASCII character must not be
    able to destroy a two-hour run.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


async def main() -> None:
    _force_utf8_stdout()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Adapted BEIR set.")
    parser.add_argument("--work", type=Path, required=True, help="Indexes + scores live here.")
    parser.add_argument("--baseline", default="nomic-embed-text", help="Model to compare against.")
    args = parser.parse_args()

    corpus = args.data / "corpus"
    judgments = [
        json.loads(line)
        for line in (args.data / "judgments.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    gold = {j["query_id"]: {g["doc"] for g in j["gold"]} for j in judgments}
    scores_dir = args.work / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    # The score cache is bound to the whole experiment, not just the model
    # name. Reusing a --work directory against a different --data set, edited
    # judgments, or changed scoring would otherwise return the previous
    # dataset's numbers purely because a file of that name exists -- and a
    # leaderboard assembled from those is wrong with nothing to show for it.
    # SCORING_VERSION is bumped by hand whenever `doc_level_scores` changes.
    experiment = {
        "scoring_version": SCORING_VERSION,
        "corpus_fingerprint": corpus_fingerprint(corpus.resolve()),
        "judgments_sha256": hashlib.sha256(
            (args.data / "judgments.jsonl").read_bytes()
        ).hexdigest(),
        "chunking_config": CFG.model_dump(),
        "candidates": CANDIDATES,
    }

    async def one_model(model: str, dims: int) -> dict | None:
        """Index and score a single model, caching the result. Never raises."""
        cached = scores_dir / f"{model.replace(':', '-')}.json"
        if cached.exists():
            try:
                payload = json.loads(cached.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict) and payload.get("experiment") == {
                **experiment,
                "dims": dims,
            }:
                print(f"{model:26s} cached", flush=True)
                return dict(payload["result"])
            print(f"{model:26s} cache stale (inputs changed), rescoring", flush=True)
        started = time.perf_counter()
        try:
            result = await index_and_score(model, dims, corpus, args.work, judgments, gold)
        except Exception as exc:  # one bad model must not lose the whole run
            print(f"{model:26s} FAILED: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
            return None
        # Written atomically and stamped with the inputs it was produced
        # from, so a later run can tell whether it still applies.
        tmp_path = cached.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(
                {"experiment": {**experiment, "dims": dims}, "result": result},
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(tmp_path, cached)
        print(f"{model:26s} done in {(time.perf_counter() - started) / 60:.1f} min", flush=True)
        return result

    results: list[dict] = []
    for wave_number, wave in enumerate(WAVES, start=1):
        names = ", ".join(model for model, _ in wave)
        print(f"\n── wave {wave_number}: {names} ──", flush=True)
        # ``gather`` is what makes a wave concurrent. Each model owns a separate
        # SQLite store and LanceDB directory, so the only shared resource is the
        # Ollama server itself — which is the thing being deliberately saturated.
        wave_results = await asyncio.gather(*(one_model(model, dims) for model, dims in wave))
        results.extend(r for r in wave_results if r is not None)

    print(
        f"\n{'model':26s} {'dims':>5s} | {'nDCG@10':>8s} {'MRR':>6s} {'R@10':>6s} {'R@1':>6s} "
        f"| {'embed':>7s} {'q p50':>7s} {'vectors':>8s}"
    )
    print("-" * 96)
    for r in sorted(
        results, key=lambda x: -np.mean([v["ndcg_at_10"] for v in x["per_query"].values()])
    ):
        pq = r["per_query"]
        cells = [np.mean([v[m] for v in pq.values()]) for m in METRICS]
        print(
            f"{r['model']:26s} {r['dimensions']:5d} | {cells[0]:8.4f} {cells[1]:6.3f} "
            f"{cells[3]:6.3f} {cells[2]:6.3f} | {r['embed_minutes']:5.1f}m "
            f"{r['query_p50_ms']:6.0f}ms {r['vector_store_mb']:7.0f}MB"
        )

    base = next((r for r in results if r["model"] == args.baseline), None)
    if base is None or len(results) < 2:
        return
    rng = np.random.default_rng(7)
    print(f"\npaired bootstrap vs {args.baseline}, n={len(judgments)}, 10,000 resamples, nDCG@10")
    for r in results:
        if r["model"] == args.baseline:
            continue
        mean, lo, hi, p = bootstrap(r["per_query"], base["per_query"], "ndcg_at_10", rng)
        verdict = "SIGNIFICANT" if (lo > 0 or hi < 0) else "not significant"
        print(f"  {r['model']:26s} {mean:+.4f}  [{lo:+.4f}, {hi:+.4f}]  p={p:.4f}  {verdict}")

    summary = args.work / "summary.json"
    summary.write_text(
        json.dumps([{k: v for k, v in r.items() if k != "per_query"} for r in results], indent=2),
        encoding="utf-8",
    )
    print(f"\nper-model scores: {scores_dir}\nsummary: {summary}")


if __name__ == "__main__":
    asyncio.run(main())

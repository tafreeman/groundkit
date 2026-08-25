"""Reproducible performance baseline for the pipeline-simplification work.

UNTRACKED, local-only. ``grk eval`` measures retrieval *quality*; this measures
the *cost* of producing it, which is the axis the simplification work moves.
Run it before and after each change and diff the JSON:

    uv run --no-sync python evals/perf/bench_pipeline.py --out evals/perf/<label>.json

Method note, and the reason this file is not a one-liner. A first cut reported
single-shot wall-clock, and three back-to-back runs of it disagreed by 48% on
``open_ms`` and 35% on ``search_p50_ms`` -- more than the ~20-25% ingest win the
contract-trimming work is expected to produce. A gate that cannot resolve the
change it exists to judge is decorative, so this harness does two things
instead:

* **Repeats and reports ``min`` alongside ``median``.** For CPU-bound work the
  minimum is the least-biased estimator -- noise only ever adds time -- while
  the median plus the observed spread says whether a run is trustworthy at all.
* **Records ``process_time`` next to wall-clock.** Removing a validator changes
  CPU cycles; it does not change SQLite fsync or the OS scheduler. CPU time
  isolates the axis the simplification work actually moves, and it is far
  quieter than wall-clock on Windows.

Read ``ingest_cpu_s`` for "did the contract change pay off", and the wall-clock
columns for "what does a user feel". A change is real when it moves ``min`` by
more than ``spread_pct`` of the baseline run it is compared against.

**Which numbers are gates, measured rather than assumed** (5 repeats, this
machine, 2026-08-21 -- re-derive on any other):

* **Gate on the 30k-chunk row only.** Across three full baseline runs its
  ``ingest_cpu_s`` spread was 8.6 / 11.3 / 7.4% and its ``search_p50_ms``
  spread 9.3 / 9.9 / 6.1% -- call the floor ~10% for both. The smaller
  corpora are not gates at any size: 7.5k ranged to 18.6% and 1.9k to 50%,
  because the work there is short enough for one scheduler excursion to
  dominate it.
* A change expected to buy 20-25% clears a ~10% floor, but only by about 2x.
  For anything landing inside that margin, do not argue from two separate runs
  of this harness -- measure both variants back-to-back inside a single
  process, where shared machine load cancels (the trick ``micro_ratios``
  relies on below).
* ``open_cpu_ms`` -- **not a gate.** Windows ``process_time`` has 15.625 ms
  granularity, which quantizes every value below ~1s, and the spread reached
  91% at 7.5k chunks. Judge the rebuild structurally instead: replacing it
  removes an O(corpus) step outright, a change no noise floor can hide.
* The absolute ``micro_us_per_op`` figures -- **not a gate**, 25-75% spread
  run-to-run. Their *ratios* are: ``metadata_guard/chunk_validated`` held
  0.54-0.66 across every run in this session, so "the metadata guard is
  roughly 55-65% of what constructing a Chunk costs" is a claim the machine
  supports and "it costs 4.19 us" is not.

Numbers are meaningful only on one machine; never compare across machines.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import random
import string
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from groundkit.contracts import Chunk, RetrievalResult
from groundkit.index.metadata import SQLiteMetadataStore
from groundkit.indexer import Indexer
from groundkit.ingestion.loaders import FileLoader
from groundkit.retrieval.search import Retriever

SEED = 7
VOCAB = 2000
CORPUS_SIZES = (30, 120, 480)  # documents; ~4000 words each
INGEST_REPEATS = 3  # a 480-doc ingest is ~7s, so repeats here are the expensive ones
QUERY_REPEATS = 5  # open + search batteries are cheap enough to repeat more
MICRO_N = 20_000
METADATA = {"source": "docs/a.md", "author": "x", "tags": ["a", "b", "c"], "n": 3}
TEXT = "x" * 512


def _words() -> list[str]:
    # S311: synthetic benchmark corpora, seeded for run-to-run comparability.
    # Nothing here is cryptographic.
    rng = random.Random(SEED)  # noqa: S311
    return ["".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 9))) for _ in range(VOCAB)]


def _micro(fn, n: int = MICRO_N) -> float:
    """Microseconds per operation, best of three.

    The micro numbers were already stable run-to-run (<2% spread), but ``min``
    costs nothing here and removes the occasional scheduler outlier.
    """
    return min(_one_micro(fn, n) for _ in range(3))


def _one_micro(fn, n: int) -> float:
    t0 = time.perf_counter()
    fn(n)
    return (time.perf_counter() - t0) / n * 1e6


def _stats(samples: list[float]) -> dict[str, float]:
    """Summarize repeated timings: the estimator, the typical value, the noise.

    ``spread_pct`` is the measurement floor. A later run has moved the number
    only if it moves ``min`` by more than this.
    """
    ordered = sorted(samples)
    low, high = ordered[0], ordered[-1]
    return {
        "min": round(low, 3),
        "median": round(ordered[len(ordered) // 2], 3),
        "max": round(high, 3),
        "spread_pct": round((high - low) / low * 100, 1) if low else 0.0,
    }


def micro_benchmarks() -> dict[str, float]:
    chunk = Chunk(document_id="d", chunk_index=0, content=TEXT, start_offset=0, end_offset=512)
    result = RetrievalResult(
        content=TEXT,
        score=0.5,
        document_id="d",
        chunk_id="c",
        source="s",
        start_offset=0,
        end_offset=512,
        metadata=METADATA,
    )

    def build_validated(n: int) -> None:
        for i in range(n):
            Chunk(
                document_id="d",
                chunk_index=i,
                content=TEXT,
                start_offset=0,
                end_offset=512,
                metadata=METADATA,
            )

    def build_construct(n: int) -> None:
        for i in range(n):
            Chunk.model_construct(
                chunk_id="c",
                document_id="d",
                chunk_index=i,
                content=TEXT,
                start_offset=0,
                end_offset=512,
                metadata=METADATA,
            )

    def metadata_guard(n: int) -> None:
        for _ in range(n):
            json.dumps(METADATA, allow_nan=False)
            copy.deepcopy(METADATA)

    def content_hash(n: int) -> None:
        # Bound the access so it is not a no-op expression ruff would flag --
        # and so the interpreter cannot elide the property call being timed.
        sink = ""
        for _ in range(n):
            sink = chunk.content_hash
        assert sink  # noqa: S101  # keeps `sink` live

    def citation(n: int) -> None:
        sink = None
        for _ in range(n):
            sink = result.citation
        assert sink is not None  # noqa: S101  # keeps `sink` live

    return {
        "chunk_validated_us": _micro(build_validated),
        "chunk_model_construct_us": _micro(build_construct),
        "metadata_guard_us": _micro(metadata_guard),
        "content_hash_us": _micro(content_hash),
        "citation_rebuild_us": _micro(citation),
    }


async def pipeline_benchmarks() -> list[dict[str, object]]:
    words = _words()
    rows: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for ndocs in CORPUS_SIZES:
            # One corpus on disk, reused by every repeat, so repeats measure the
            # pipeline rather than the filesystem warming up.
            rng = random.Random(SEED)  # noqa: S311  # synthetic corpus, see _words()
            src = tmp / f"src{ndocs}"
            src.mkdir()
            for i in range(ndocs):
                (src / f"d{i:04d}.md").write_text(
                    " ".join(rng.choices(words, k=4000)), encoding="utf-8"
                )
            corpus_bytes = sum(p.stat().st_size for p in src.glob("*.md"))
            queries = [" ".join(rng.choices(words, k=5)) for _ in range(40)]

            ingest_wall: list[float] = []
            ingest_cpu: list[float] = []
            chunks_written = 0
            db_bytes = 0

            for rep in range(INGEST_REPEATS):
                # A fresh collection per repeat: re-ingesting into a warm one hits
                # the unchanged-fingerprint skip path and would time nothing.
                name = f"c{ndocs}r{rep}"
                store = await SQLiteMetadataStore.open(index_dir=tmp, collection=name)
                try:
                    indexer = Indexer(store=store, loader=FileLoader(allowed_base_dir=tmp))
                    w0, c0 = time.perf_counter(), time.process_time()
                    report = await indexer.index_directory(str(src))
                    ingest_wall.append(time.perf_counter() - w0)
                    ingest_cpu.append(time.process_time() - c0)
                    chunks_written = report.chunks_written
                finally:
                    # On Windows an open SQLite handle blocks the enclosing
                    # TemporaryDirectory's cleanup, so a failure here would
                    # surface as a PermissionError from the cleanup and bury
                    # the benchmark's actual error. The other scripts in this
                    # directory already close in `finally`; this one did not.
                    await store.close()
                # Measured *after* the close, not before. The store runs in WAL
                # mode, so at the end of an ingest a large share of the
                # committed pages are still in `<name>.sqlite3-wal` and have not
                # been checkpointed into the main file. Stat-ing before the
                # close therefore reports whatever the auto-checkpoint happened
                # to have folded in -- measured here at 1,806,336 bytes against
                # a true 1,978,368 for 2,000 chunks, an 8.7% understatement,
                # with 4.2 MB still outstanding in the -wal at that moment.
                # Closing checkpoints and removes the WAL, so the main file is
                # then the whole database.
                db_bytes = (tmp / f"{name}.sqlite3").stat().st_size

            # Reopen the last-written collection for the query battery.
            open_wall: list[float] = []
            open_cpu: list[float] = []
            search_p50: list[float] = []
            search_p95: list[float] = []

            for _ in range(QUERY_REPEATS):
                store = await SQLiteMetadataStore.open(
                    index_dir=tmp, collection=f"c{ndocs}r{INGEST_REPEATS - 1}"
                )
                try:
                    w0, c0 = time.perf_counter(), time.process_time()
                    retriever = await Retriever.open(store=store)
                    open_wall.append((time.perf_counter() - w0) * 1000)
                    open_cpu.append((time.process_time() - c0) * 1000)

                    await retriever.search(queries[0], top_k=10)  # warm
                    lat = []
                    for query in queries:
                        t0 = time.perf_counter()
                        await retriever.search(query, top_k=10)
                        lat.append((time.perf_counter() - t0) * 1000)
                    lat.sort()
                    search_p50.append(lat[len(lat) // 2])
                    search_p95.append(lat[int(len(lat) * 0.95)])
                finally:
                    await store.close()

            rows.append(
                {
                    "documents": ndocs,
                    "chunks": chunks_written,
                    "corpus_mb": round(corpus_bytes / 1e6, 2),
                    "db_mb": round(db_bytes / 1e6, 2),
                    "ingest_wall_s": _stats(ingest_wall),
                    "ingest_cpu_s": _stats(ingest_cpu),
                    "open_wall_ms": _stats(open_wall),
                    "open_cpu_ms": _stats(open_cpu),
                    "search_p50_ms": _stats(search_p50),
                    "search_p95_ms": _stats(search_p95),
                }
            )
    return rows


#: What this machine can actually resolve, measured (see the module docstring).
#: A later run has moved a number only if it clears the floor for that metric.
MEASUREMENT_FLOOR = {
    "ingest_cpu_s@30000": "7-11% over three runs; ~10% floor -- gate",
    "search_p50_ms@30000": "6-10% over three runs; ~10% floor -- gate",
    "any_metric@<30000_chunks": "to 50% -- not a gate, too short to be stable",
    "open_cpu_ms": "20-91%, quantized by 15.625ms process_time -- not a gate",
    "micro_us_per_op": "25-75% absolute -- not a gate; use micro_ratios",
    "micro_ratios.metadata_guard_share": "~23% -- the usable micro claim",
}


def _micro_ratios(micro: dict[str, float]) -> dict[str, float]:
    """Ratios survive machine load in a way the absolute microsecond figures do not.

    Both terms of each ratio are measured microseconds apart under identical
    load, so a busy machine inflates numerator and denominator together and
    cancels out. The absolutes drifted 25-75% run-to-run in this session while
    ``metadata_guard_share`` held inside 23%.
    """
    validated = micro["chunk_validated_us"] or 1.0
    return {
        "validation_overhead_x": round(validated / (micro["chunk_model_construct_us"] or 1.0), 2),
        "metadata_guard_share": round(micro["metadata_guard_us"] / validated, 3),
    }


def _dirty_files() -> list[str]:
    """Paths git reports as not-clean, porcelain format.

    Read the output *unstripped*. Porcelain lines are ``XY<space>PATH``, so the
    path always starts at index 3 -- but an unstaged entry begins with a space
    (``" M f"``), and ``str.strip()`` on the whole captured stdout removes it
    from the first line only. That turned one capture's leading ``BACKLOG.md``
    into ``ACKLOG.md`` while every following line stayed correct, which is
    exactly the shape of bug that survives a glance at the output.
    """
    try:
        # S607: fixed literal argv, no untrusted input reaches the command line.
        raw = subprocess.run(
            ["git", "status", "--porcelain"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except Exception:
        return ["unknown"]
    return [line[3:] for line in raw.splitlines() if line.strip()]


def provenance() -> dict[str, object]:
    def git(*args: str) -> str:
        try:
            # S603/S607: fixed literal argv, arguments are this module's own
            # constants -- no untrusted input reaches the command line.
            return subprocess.run(  # noqa: S603
                ["git", *args],  # noqa: S607
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except Exception:
            return "unknown"

    return {
        "captured_at": datetime.now(UTC).isoformat(),
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_files": _dirty_files(),
        "python": sys.version.split()[0],
        "platform": sys.platform,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Write the JSON record here.")
    args = parser.parse_args()

    micro = micro_benchmarks()
    record = {
        "provenance": provenance(),
        "config": {
            "seed": SEED,
            "ingest_repeats": INGEST_REPEATS,
            "query_repeats": QUERY_REPEATS,
            "corpus_sizes": list(CORPUS_SIZES),
        },
        "micro_us_per_op": micro,
        "micro_ratios": _micro_ratios(micro),
        "measurement_floor": MEASUREMENT_FLOOR,
        "pipeline": await pipeline_benchmarks(),
    }

    print(f"{'chunks':>7s} | {'ingest CPU s':>22s} | {'open CPU ms':>22s} | {'search p50 ms':>22s}")
    print(
        f"{'':>7s} | {'min      med   +/-%':>22s} | {'min      med   +/-%':>22s} "
        f"| {'min      med   +/-%':>22s}"
    )
    print("-" * 82)
    for row in record["pipeline"]:
        cells = []
        for key in ("ingest_cpu_s", "open_cpu_ms", "search_p50_ms"):
            stat = row[key]
            cells.append(f"{stat['min']:8.2f} {stat['median']:8.2f} {stat['spread_pct']:5.1f}")
        print(f"{row['chunks']:7d} | {cells[0]:>22s} | {cells[1]:>22s} | {cells[2]:>22s}")

    print()
    print("wall-clock (what a user feels, noisier):")
    for row in record["pipeline"]:
        print(
            f"  {row['chunks']:7d} chunks  ingest {row['ingest_wall_s']['min']:6.2f}s  "
            f"open {row['open_wall_ms']['min']:7.0f}ms  "
            f"p95 {row['search_p95_ms']['min']:6.2f}ms  db {row['db_mb']:.1f}MB"
        )

    print()
    for key, value in record["micro_us_per_op"].items():
        print(f"  {key:34s} {value:8.2f} us/op   (absolute: not a gate)")
    ratios = record["micro_ratios"]
    print(f"  {'validation vs model_construct':34s} {ratios['validation_overhead_x']:8.2f} x")
    print(f"  {'metadata guard share of Chunk()':34s} {ratios['metadata_guard_share']:8.1%}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"\nrecord written to {args.out}")


if __name__ == "__main__":
    asyncio.run(main())

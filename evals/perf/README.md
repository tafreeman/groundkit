# Simplification baseline — 2026-08-21

Reference point for the pipeline-simplification work (contract trimming, two-tier
contracts, index-backend swap). Every later step is judged against these numbers.

**Tracked, and deliberately against convention.** `evals/results/` is gitignored so
that eval deltas are never cross-run — `stages[0]` is the intra-run baseline, by design
(SPEC.md §8). This directory breaks that rule on purpose, because a *performance*
refactor is the one case where cross-run comparison is the point. Nothing here may be
cited as a quality number in tracked docs.

(This directory was untracked when these numbers were produced, and said so. It is
committed now because it is the evidence behind a repositioning argument, and evidence
that only exists on one machine is not evidence. The rule above still holds: these are
*performance* baselines, and none of them is a retrieval-quality number.)

## Provenance

- commit `88fe5701e341` on `fix/gk-030-snapshot-read-nofollow` — **clean tree**
  (`dirty_files` = `['evals/perf/']`, this directory only)
- captured 2026-08-21T11:50:22.415080+00:00
- Python 3.11.9 · win32 · groundkit 0.1.0
- gates green at capture: 1669 passed / 27 skipped (exit 0, 96.94% coverage),
  `ruff check` + `ruff format --check` clean, `mypy --strict` clean (113 files)
- `corpus_hash` `47d95f2687f60673…` · `judgments_hash` `e2582a4a40cc57f4…`
  — a later run whose hashes differ is measuring a different corpus and is not comparable.

## Retrieval quality — `grk eval`

Golden corpus: 10 documents, 84 chunks, 44 judgments, 36 queries.

| stage | recall@1 | recall@5 | recall@10 | MRR | nDCG@10 | abstained | p50 |
|---|---|---|---|---|---|---|---|
| **bm25** *(baseline)* | 0.333 | 0.722 | 0.889 | 0.503 | 0.586 | 8/8 | 2.4ms |
| dense | 0.556 | 0.889 | 0.889 | 0.680 | 0.727 | 0/8 | 81.1ms |
| fusion | 0.556 | 0.889 | 0.944 | 0.679 | 0.738 | 0/8 | 81.2ms |

Dense and fusion ran through Ollama `nomic-embed-text` (768d) — a real measurement, not
the `inmemory` double, whose numbers SPEC.md §2 forbids treating as quality. The rerank
stage is absent: the `rerank` extra (sentence-transformers/torch) is not installed.

The metrics reproduced **identically** across two captures on different commits — only
latency moved. Retrieval is deterministic, so any metric change in a later run is a real
change, not run-to-run drift.

Two things to carry forward:

1. **Fusion is +0.151 nDCG@10 and +0.222 recall@1 over BM25** — the hybrid path earns
   its keep on this corpus.
2. **Fusion abstains on 0 of 8 no-answer queries; BM25 abstains on all 8.** The recall
   is bought with a willingness to answer when it should not. Any later change claiming
   a quality win must report both columns — this trade is a regression, not a wash.

Artifacts: `evals/results/baseline-bm25-2026-08-21.json` · `baseline-dense-2026-08-21.json`

## Pipeline cost — `evals/perf/bench_pipeline.py`

| chunks | ingest CPU s | open CPU ms | search p50 ms | search p95 ms | db MB |
|---|---|---|---|---|---|
| 1,886 | 0.16 | 94 | 3.65 | 5.52 | 1.7 |
| 7,543 | 1.09 | 453 | 6.40 | 8.60 | 6.5 |
| 30,180 ⬅ **gate** | 5.09 | 2031 | 22.45 | 27.88 | 27.4 |

All columns are `min` of repeated runs (3 ingests, 5 query batteries); the JSON carries median, max and spread.

- validation vs `model_construct`: **4.33×**
- metadata guard share of `Chunk(...)`: **56.0%** — held
  0.54–0.66 across every run in this session, so this is the robust micro claim. The
  absolute microsecond figures are **not**: they drifted 25–75% run-to-run.

`open` is O(corpus): BM25 postings are rebuilt in RAM on every `Retriever.open()`
(ADR-0002, deliberate). That is the scaling wall, and it is structural — no amount of
constant-factor work on the contracts touches it.

## What these numbers can and cannot judge

Measured on this machine over three full runs, not assumed. Mirrored in
`MEASUREMENT_FLOOR` in the harness:

- `ingest_cpu_s@30000` — 7-11% over three runs; ~10% floor -- gate
- `search_p50_ms@30000` — 6-10% over three runs; ~10% floor -- gate
- `any_metric@<30000_chunks` — to 50% -- not a gate, too short to be stable
- `open_cpu_ms` — 12-91%, quantized by 15.625ms process_time -- not a gate
- `micro_us_per_op` — 25-75% absolute -- not a gate; use micro_ratios
- `micro_ratios.metadata_guard_share` — ~23% -- the usable micro claim

The consequence worth acting on: a 20–25% ingest win clears the ~10% floor on the 30k
row, but only by about 2×. For anything landing inside that margin, do not diff two
separate runs of this harness — measure both variants back-to-back in a single process,
the way `micro_ratios` does, where shared machine load cancels in a ratio.

## Reproduce

```bash
uv run --no-sync grk eval --output evals/results/<label>-bm25.json
```

```bash
uv run --no-sync grk eval --dense --embed-provider ollama --embed-model nomic-embed-text --embed-dimensions 768 --output evals/results/<label>-dense.json
```

```bash
uv run --no-sync python evals/perf/bench_pipeline.py --out evals/perf/<label>.json
```

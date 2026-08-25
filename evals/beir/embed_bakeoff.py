"""Compare embedding models on an adapted BEIR set — one persisted index per model.

Tracked, alongside the rest of ``evals/beir/``.

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

**Known limitation: re-pulling an Ollama tag is invisible to both caches.** The
index sentinel and the score cache bind to the corpus content, the chunking
config, the model *tag* and its dimensions -- not to the weights behind the tag.
``ollama pull nomic-embed-text`` can replace those weights while the tag stays
identical, and both caches would then serve vectors and scores produced by the
previous model. Binding to the manifest digest would close this; it is not done
here because it needs a live daemon to resolve and could not be verified when
this was written. **Until it is: delete ``--work`` after re-pulling any model in
WAVES.** Everything else that can silently change under a cache is detected:
corpus edits and renames, a corrected dimension, the chunking *configuration*,
and -- since it is a distinct input, not the same one -- the chunker's
*behaviour* at an unchanged configuration. That last one is not theoretical:
this repository's own record shows 512/64 producing 25,028 spans before the
``_merge_parts`` fix and 20,219 after.
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
from contextlib import suppress
from functools import cache
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from groundkit.config import ChunkingConfig, EmbeddingConfig
from groundkit.contracts import Document
from groundkit.errors import EvalError
from groundkit.evals.corpus import Judgment, load_judgments
from groundkit.index.dense import LanceDBVectorStore
from groundkit.index.metadata import SQLiteMetadataStore
from groundkit.indexer import Indexer
from groundkit.ingestion.chunking import RecursiveChunker
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

#: Bumped by hand whenever anything changes what a cached result *means* -- how
#: a score is computed, or what the recorded fields report. It is part of the
#: score cache's identity, so bumping it invalidates every cached score rather
#: than letting two meanings share one leaderboard.
#:
#: 2: entries written before the build-timing fix carry ``embed_minutes: 0.0``
#:    whenever they were produced by a rescore against a reused index. Their
#:    identity is otherwise unchanged, so without this bump an existing
#:    ``--work`` directory would keep serving that zero and never receive the
#:    correction -- the fix would apply only to caches nobody had yet built.
#: 3: ``mrr`` was reciprocal rank over the *whole* distinct-document list while
#:    every other metric used the ``k`` cutoff. Cached entries therefore hold
#:    MRR values that are not MRR@k and are not comparable across chunking
#:    configurations, since the list's depth varies with chunks-per-document.
SCORING_VERSION = 3

#: Base seed every per-contrast generator is derived from (see ``contrast_rng``).
BOOTSTRAP_SEED = 7

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


class CachedMetrics(BaseModel):
    """One query's scores inside a cached result.

    Bounded to the unit interval, matching ``evals.schema.QueryMetrics`` --
    the authoritative statement of what a groundkit metric is -- and matching
    what ``score_ranking`` can actually return. ``allow_inf_nan=False`` alone
    admitted ``-1.0`` and ``2.0``, which are as unpublishable as a NaN and
    quieter: a NaN at least propagates visibly through the leaderboard, while
    an out-of-range float prints as a plausible number.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    ndcg_at_10: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    mrr: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    recall_at_1: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    recall_at_10: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class CachedResult(BaseModel):
    """Exactly what ``index_and_score`` returns and a later run reads back.

    Declared rather than hand-checked, and that is the point. Four rounds of
    review found four different shapes a hand-written check let through --
    a non-mapping, a missing ``result``, an empty ``{}``, and then values of
    the wrong type -- because each check tested the layer the previous crash
    happened at. A cache file is untrusted input read off disk, which is the
    boundary this project validates with a model everywhere else.

    ``extra="forbid"`` makes the drift bidirectional: a field added to the
    writer without being added here fails just as loudly as a field removed.
    ``allow_inf_nan=False`` matters because a NaN metric does not raise
    anywhere -- it silently poisons ``np.mean`` and every comparison drawn
    from it, which is the failure this whole effort exists to prevent.

    ``strict=True`` is not decoration either, and declaring the model without
    it was not enough: Pydantic's default lax mode coerces ``"1"`` to ``1.0``,
    so a JSON file whose metrics are strings still validated and still fed the
    leaderboard. Strict mode keeps the widening that is lossless (an ``int``
    where a ``float`` is declared, which is what JSON gives back for ``0``)
    and refuses the conversions that are guesses -- ``"768"`` as a dimension
    count, ``1`` as ``reused_index``.
    """

    model_config = ConfigDict(extra="forbid", strict=True, protected_namespaces=())

    model: str = Field(min_length=1)
    dimensions: int = Field(gt=0)
    # `gt`, not `ge`. A zero-chunk result is not a cheap edge case, it is the
    # signature of an empty corpus: every judgment scores as a miss, so the row
    # is all zeros, every zero is individually in range, and the whole thing
    # publishes as "this model scored 0.000" rather than "nothing was indexed".
    chunks: int = Field(gt=0)
    reused_index: bool
    # `ge=0.0`, not `gt`: SCORING_VERSION 2's note records that entries written
    # before the build-timing fix legitimately carry `embed_minutes: 0.0`.
    embed_minutes: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    query_p50_ms: float = Field(ge=0.0, allow_inf_nan=False)
    query_p95_ms: float = Field(ge=0.0, allow_inf_nan=False)
    vector_store_mb: float = Field(ge=0.0, allow_inf_nan=False)
    per_query: dict[str, CachedMetrics]

    @model_validator(mode="after")
    def _percentiles_are_ordered(self) -> CachedResult:
        """p95 cannot be below p50.

        Both are read out of one sorted latency array at monotone indices and
        rounded to the same precision, so this holds for every run the writer
        can produce -- which makes a violation proof the file was edited or
        truncated, not evidence of a slow run. Checked because a swapped pair
        is exactly the finite-but-impossible shape the bounds above close, and
        no per-field constraint can see it.
        """
        if self.query_p95_ms < self.query_p50_ms:
            raise ValueError(
                f"query_p95_ms ({self.query_p95_ms}) is below query_p50_ms ({self.query_p50_ms})"
            )
        return self


def validated_result(
    result: dict, expected_query_ids: set[str], *, model: str, dimensions: int
) -> dict:
    """The one place a result is judged fit to publish. Raises on anything unfit.

    A cache read only ever reached the leaderboard through ``usable_result`` --
    shape via ``CachedResult``, then the three identity checks below. A
    freshly-computed result from ``index_and_score`` reached the exact same
    leaderboard -- the same format strings, the same paired bootstrap -- through
    no check at all: it was a dict literal, returned, and trusted. Eight rounds
    of review on this file kept finding variations on that one asymmetry --
    a metric out of range, a wrong model name, a dimension mismatch, a
    per-query id set that didn't match the judgments -- each one invisible
    because the fresh path and the cache-read path were validated by two
    different amounts of code.

    This is now the only amount there is. ``usable_result`` (for a cache read)
    and ``assemble_result`` (for a fresh score) both call *this* function and
    nothing else, so the two paths cannot drift apart again -- there is no
    longer a second, weaker check for either of them to fall back onto.

    Raises:
        ValidationError: ``result`` does not have ``CachedResult``'s shape --
            a missing or extra field, a wrong type, or a metric outside its
            declared range.
        ValueError: ``result`` is shaped correctly but disagrees with what was
            asked for -- the wrong model, the wrong dimensions, or scored
            against a different set of query ids than ``expected_query_ids``.
    """
    parsed = CachedResult.model_validate(result)
    if parsed.model != model:
        raise ValueError(f"result names model {parsed.model!r}, expected {model!r}")
    if parsed.dimensions != dimensions:
        raise ValueError(f"result claims {parsed.dimensions} dimensions, expected {dimensions}")
    actual_ids = set(parsed.per_query)
    if actual_ids != expected_query_ids:
        missing = sorted(expected_query_ids - actual_ids)
        extra = sorted(actual_ids - expected_query_ids)
        raise ValueError(
            f"result's per_query ids disagree with the judgment set "
            f"(missing={missing!r}, extra={extra!r})"
        )
    return result


def usable_result(
    result: object, expected_query_ids: set[str], *, model: str, dimensions: int
) -> bool:
    """Whether a cached result is safe to publish without rescoring.

    A thin wrapper over :func:`validated_result` -- deliberately thin, because
    the two policies a result can be put to (a cache miss just means rescore;
    a validation failure on a *fresh* result means fail loudly, see
    ``assemble_result``) must read the same judgment, or they will eventually
    disagree about what "fit to publish" means. Sharing one function is what
    makes that structurally impossible rather than merely intended.

    ``expected_query_ids`` -- a result carrying the right shape for the wrong
    queries pairs cleanly in the bootstrap and reports a comparison nobody ran.

    ``model``/``dimensions`` -- the sentinel already binds both, but it binds
    them in ``payload["experiment"]``, which is written beside the result and
    not derived from it. So a body disagreeing with its own sentinel passes
    every check there was. It is then published under whatever name it
    carries, and since the paired bootstrap finds its baseline by
    ``r["model"] == args.baseline``, a wrong name there does not fail loudly
    -- it makes the baseline look absent and skips every comparison in the
    run.
    """
    if not isinstance(result, dict):
        return False
    try:
        validated_result(result, expected_query_ids, model=model, dimensions=dimensions)
    except (ValidationError, ValueError):
        return False
    return True


def write_cache_file(path: Path, payload: dict) -> bool:
    """Write *payload* atomically. ``True`` on success, ``False`` if it could not be.

    The counterpart to :func:`read_cache_file`, and total for the same reason
    it is. This ran *outside* ``one_model``'s ``except Exception`` and after
    scoring had already succeeded, so a stale ``.tmp`` directory, a locked
    target (routine on Windows, where a reader holds the file open) or a full
    disk did not merely lose this model's cache -- it propagated through
    ``asyncio.gather`` and cancelled the other models' index builds in the
    same wave, discarding tens of minutes of embedding work over a failed
    write of a file whose entire purpose is to save time on the *next* run.

    A failed write is therefore a rescore next run, never a lost wave. The
    temporary file is cleaned up on the way out so a failure does not leave
    the debris that makes the next attempt fail the same way.
    """
    tmp_path = path.with_suffix(".tmp")
    try:
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp_path, path)
    except (OSError, TypeError, ValueError):
        with suppress(OSError):
            tmp_path.unlink(missing_ok=True)
        return False
    return True


def score_ranking(order: list[str], gold: set[str], k: int = 10) -> dict[str, float]:
    """Binary-relevance IR metrics over a document ranking, all at depth ``k``.

    Binary is correct here rather than a simplification: SciFact's qrels score
    every judged pair ``1``, so graded gain would have nothing to grade.

    The ``mrr`` key is reciprocal rank at ``k``, not over the full ranking. The
    name is kept short because every metric here shares the one cutoff, but it
    is a cutoff metric like the rest.
    """
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(order[:k]) if d in gold)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(gold), k)))
    # Capped at `k` like every other metric here. `order` is distinct documents
    # collapsed from up to CANDIDATES chunks, so its depth depends on how many
    # chunks a document produced -- ~50 entries at whole-document chunking
    # against ~37 at 512/64. Searching it uncapped let one configuration earn
    # reciprocal-rank credit at depths another structurally could not reach,
    # which is not a difference in retrieval quality and is exactly the
    # comparison RESULTS-scifact draws.
    rank = next((i for i, d in enumerate(order[:k], 1) if d in gold), None)
    return {
        "ndcg_at_10": dcg / idcg if idcg else 0.0,
        "mrr": 1.0 / rank if rank else 0.0,
        "recall_at_1": len(set(order[:1]) & gold) / len(gold),
        "recall_at_10": len(set(order[:10]) & gold) / len(gold),
    }


def model_slug(model: str) -> str:
    """A filesystem-safe key for one model that cannot collide with another's.

    A plain ``replace(":", "-")`` flattens ``foo:bar`` and ``foo-bar`` onto the
    same name, and that name keys *both* the index directory and the score
    cache file. The consequence is not merely a shared cache: a stale sentinel
    now deletes the index directory, so two colliding models would take turns
    destroying each other's index and re-embedding it -- hours per cycle,
    every run, with nothing in the output saying why.

    A short digest of the exact name is appended only when flattening actually
    changed something, so every model in ``WAVES`` today (none contains ``:``
    or ``/``) keeps the readable directory it already has and no existing
    cache is invalidated by this fix.
    """
    flattened = model.replace(":", "-").replace("/", "-")
    if flattened == model:
        return flattened
    return f"{flattened}-{hashlib.sha256(model.encode('utf-8')).hexdigest()[:8]}"


@cache
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


@cache
def chunk_fingerprint(corpus: Path) -> str:
    """A hash of the chunk boundaries this chunker produces for this corpus *now*.

    ``chunking_config`` records the 512/64 values, not the algorithm that
    turns them into boundaries, and the two are not the same input. This
    repository's own benchmark record is the proof: at an unchanged 512/64,
    SciFact chunked to 25,028 spans before the ``_merge_parts`` fix and 20,219
    after (``RESULTS-scifact-2026-08-21.md``). A ``--work`` directory carried
    across that upgrade would have matched on config, reused the pre-fix index
    and the pre-fix scores, and published them as current -- the one failure
    this whole cache-identity effort exists to prevent.

    Fingerprinting the produced spans rather than a hand-maintained revision
    constant is deliberate: a constant only works if whoever changes the
    chunker remembers to bump it, and the change that motivated this was made
    in a different pull request by someone not looking at this file. Spans
    cannot forget.

    Costs one chunking pass over the corpus -- seconds against index builds
    measured in tens of minutes -- and is memoized, so a run pays it once
    rather than once per model.
    """
    chunker = RecursiveChunker()
    digest = hashlib.sha256()
    for path in sorted(p for p in corpus.rglob("*") if p.is_file()):
        name = path.relative_to(corpus).as_posix()
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            # Deliberately *not* the treatment `read_cache_file` gives a bad
            # cache. A damaged cache belongs to one model and means rebuild;
            # an unreadable corpus file is shared by every model in the run and
            # there is nothing to fall back to, so aborting is right. What is
            # not right is a bare UnicodeDecodeError -- it names the byte and
            # not the file, and it surfaces through `asyncio.gather` with no
            # indication of which document is at fault.
            raise RuntimeError(f"cannot read corpus file {name!r}: {exc}") from exc
        document = Document(source=name, content=content)
        for chunk in chunker.chunk(document, config=CFG):
            digest.update(f"{name}:{chunk.start_offset}:{chunk.end_offset}\0".encode())
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
        "chunk_fingerprint": chunk_fingerprint(resolved),
        "model": model,
        "dims": dims,
    }


def read_cache_file(path: Path) -> dict | None:
    """A cache file parsed as a JSON object, or ``None`` if it is unusable.

    **Every cache read in this module goes through here**, and the reason is
    that the two readers drifted apart twice. Both the index sentinel and the
    per-model score file are read *outside* ``one_model``'s ``except
    Exception`` -- the score-cache gate needs the sentinel before that handler
    is entered -- so anything either read raises propagates through
    ``asyncio.gather`` and aborts every other model in the wave, rather than
    rebuilding the one model whose cache is damaged.

    Total, therefore, and for all three ways a file resists being read:
    ``OSError`` for I/O, ``UnicodeDecodeError`` for bytes that are not UTF-8
    (a ``ValueError``, so an ``OSError`` clause does not cover it), and
    ``JSONDecodeError`` for text that is not JSON. A non-object payload is
    ``None`` too: every caller wants a mapping, and a bare list or string is
    no more usable than a truncated file.

    "Unusable cache" is a rebuild in every case this module has, never a
    crash. Sharing one reader is what stops that from being re-decided, and
    re-decided differently, at each call site.
    """
    if not path.exists():
        return None
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _read_sentinel(index_dir: Path) -> dict | None:
    """The parsed completion sentinel for one model's index, or ``None``."""
    return read_cache_file(index_dir / SENTINEL_NAME)


def _sentinel_build(index_dir: Path) -> dict:
    """What the run that built this index measured while building it.

    Kept beside the identity rather than mixed into it: these are *outputs* of
    the build, so comparing them would make every sentinel mismatch itself.
    Empty when the sentinel predates this field, which reads as "not measured"
    rather than as zero.
    """
    recorded = _read_sentinel(index_dir)
    build = recorded.get("build") if recorded else None
    return build if isinstance(build, dict) else {}


def _sentinel_valid(index_dir: Path, expected: dict) -> bool:
    """True only if a completion sentinel exists and matches the current inputs."""
    recorded = _read_sentinel(index_dir)
    if recorded is None:
        return False
    recorded = recorded.get("inputs")  # type: ignore[assignment]
    if not isinstance(recorded, dict):
        return False
    return recorded == expected


def _write_sentinel(index_dir: Path, inputs: dict, build: dict) -> None:
    """Write the completion sentinel atomically.

    A plain ``write_text`` can be interrupted mid-write (process kill, power
    loss), leaving a truncated or corrupt sentinel that ``_sentinel_valid``
    would then either wrongly trust (rare) or wrongly reject (harmless, just
    forces a re-index) -- but the truncated-then-trusted case is the one worth
    closing. Writing to a ``.tmp`` file and ``os.replace``-ing it over the real
    name means the sentinel is only ever fully-written or absent, never partial.

    ``inputs`` is the identity ``_sentinel_valid`` compares; ``build`` is what
    that build measured, carried so a later rescore against a reused index can
    still report the embedding cost rather than inventing a zero for it.
    """
    sentinel = index_dir / SENTINEL_NAME
    tmp_path = sentinel.with_suffix(".tmp")
    tmp_path.write_text(json.dumps({"inputs": inputs, "build": build}, indent=2), encoding="utf-8")
    os.replace(tmp_path, sentinel)


async def index_and_score(
    model: str, dims: int, corpus: Path, work: Path, judgments: list[Judgment], gold: dict
) -> dict:
    """Build (or reuse) this model's index, then score every query against it."""
    index_dir = work / "index" / model_slug(model)
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
        if index_dir.exists():
            # `ignore_errors` keeps a locked or permission-protected file from
            # raising, which would otherwise be fine -- except that the next
            # line reopens whatever survived and treats it as a fresh index.
            # Documents from the previous corpus are outside the new one's
            # pruning scope, so they would stay mixed into the rebuilt
            # collection and a completion sentinel would then certify it.
            # Refusing here costs this model a FAILED line; not refusing costs
            # a wrong number that looks like a right one.
            raise RuntimeError(
                f"could not fully remove the stale index at {index_dir}; refusing to "
                "reuse what is left of it. Close anything holding files open there "
                "(an editor, a previous run) and re-run."
            )
    index_dir.mkdir(parents=True, exist_ok=True)

    # Opened before the `try` on purpose, and everything after it is not: a
    # store left open on the way out of this function is not this function's
    # problem to solve twice, so exactly one `finally` covers every exit,
    # including the two lines right below that can themselves raise before
    # any of them ever ran. `build_embedder` can fail on a bad dimension and
    # `LanceDBVectorStore.open` can fail on a corrupt or inaccessible Lance
    # directory -- both were previously evaluated *before* this `try` began,
    # so either one raising leaked the SQLite handle for the rest of the
    # multi-model process. On Windows that handle blocks deleting the very
    # index directory an operator would reach for to recover from the failure
    # that leaked it -- `PermissionError: [WinError 32]` -- until the whole
    # process exits.
    store = await SQLiteMetadataStore.open(index_dir=index_dir, collection="beir")
    # `None`, not 0.0: an unmeasured cost and a zero cost are different claims,
    # and only one of them is ever true here.
    embed_seconds: float | None = None
    chunks = 0
    try:
        embedder = build_embedder(
            EmbeddingConfig(provider="ollama", model_name=model, dimensions=dims)
        )
        vectors = await LanceDBVectorStore.open(db_path=lance_dir)

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
            # interrupted run must not leave a sentinel a later run would
            # trust. The build cost travels with it so a later rescore can
            # report it rather than re-measure it.
            _write_sentinel(
                index_dir, sentinel_inputs, {"embed_seconds": embed_seconds, "chunks": chunks}
            )
        else:
            # The score cache and the index sentinel have different identities
            # on purpose -- changing the judgments, the candidate depth or
            # SCORING_VERSION invalidates the scores while the index stays
            # valid -- so this branch runs whenever a rescore reuses an index.
            # Both numbers below describe the *build*, and this run did not
            # build anything, so neither may be reported as zero: the chunk
            # count would understate the index and the embed time would erase
            # the ingest cost that is one of this bake-off's headline
            # comparisons. The count is authoritative from the store; the
            # timing can only come from the run that measured it.
            chunks = await store.count_chunks()
            embed_seconds = _sentinel_embed_seconds(index_dir)

        retriever = await Retriever.open(
            store=store, embedder=embedder, vector_store=vectors, collection="beir"
        )
        per_query: dict[str, dict[str, float]] = {}
        latencies: list[float] = []
        for judgment in judgments:
            started = time.perf_counter()
            response = await retriever.search(judgment.query, top_k=CANDIDATES, mode="dense")
            latencies.append((time.perf_counter() - started) * 1000)
            seen: set[str] = set()
            order: list[str] = []
            for result in response.results:
                doc = Path(result.source).name
                if doc not in seen:
                    seen.add(doc)
                    order.append(doc)
            per_query[judgment.query_id] = score_ranking(order, gold[judgment.query_id])
    finally:
        await store.close()

    vector_bytes = sum(p.stat().st_size for p in lance_dir.rglob("*") if p.is_file())
    return assemble_result(
        model=model,
        dimensions=dims,
        chunks=chunks,
        reused_index=not fresh,
        embed_seconds=embed_seconds,
        latencies=latencies,
        vector_bytes=vector_bytes,
        per_query=per_query,
        expected_query_ids=set(gold),
    )


def assemble_result(
    *,
    model: str,
    dimensions: int,
    chunks: int,
    reused_index: bool,
    embed_seconds: float | None,
    latencies: list[float],
    vector_bytes: int,
    per_query: dict[str, dict[str, float]],
    expected_query_ids: set[str],
) -> dict:
    """Build one model's published result and validate it before returning.

    This is the write side of the asymmetry :func:`validated_result` closes.
    Everything above it in ``index_and_score`` measures; this is where the
    measurements become the dict that gets cached, printed and fed to the
    bootstrap -- and it is handed to ``validated_result`` before it goes back
    to the caller, the same check a cache read has to pass. A fresh result
    used to skip that check entirely on the theory that it was just computed
    and therefore trustworthy, which is exactly backwards: it is the one
    result in the whole run that has never been checked by anything.

    ``latencies`` is sorted into a new list rather than in place and rather
    than trusted pre-sorted, so this function is idempotent -- calling it
    twice on the same measurements (a retry, a test) produces byte-identical
    output instead of depending on a mutation the caller already made.

    The rounding is unchanged from the dict literal this replaces: two
    decimal places for embed minutes, one for millisecond and megabyte
    figures, matching what a human comparing models in the printed table
    actually reads. Do not change it here without re-checking every cached
    score file's precision assumptions.

    Raises:
        ValidationError: The assembled dict does not have ``CachedResult``'s
            shape -- this would mean the arithmetic above produced a value
            outside a metric's declared range, since every field here is
            otherwise well-typed by construction.
        ValueError: The assembled dict disagrees with ``model``, ``dimensions``
            or ``expected_query_ids`` -- see :func:`validated_result`.
    """
    latencies = sorted(latencies)
    result = {
        "model": model,
        "dimensions": dimensions,
        "chunks": chunks,
        "reused_index": reused_index,
        # `null` rather than 0.0 when this run reused an index built before the
        # sentinel carried timings -- a reader can tell "not measured" from
        # "measured as free", which a zero cannot.
        "embed_minutes": None if embed_seconds is None else round(embed_seconds / 60, 2),
        "query_p50_ms": round(latencies[len(latencies) // 2], 1),
        "query_p95_ms": round(latencies[int(len(latencies) * 0.95)], 1),
        "vector_store_mb": round(vector_bytes / 1e6, 1),
        "per_query": per_query,
    }
    return validated_result(result, expected_query_ids, model=model, dimensions=dimensions)


class SentinelBuild(BaseModel):
    """The ``build`` half of an index sentinel, as read back.

    ``_sentinel_valid`` compares ``inputs`` and says nothing about ``build``,
    so these values reached the published result unchecked. A negative
    ``embed_seconds`` became a negative embedding time in the leaderboard --
    an impossible number, and one the cached-result schema would have caught
    had it come back through the cache instead of straight out of this run.
    A string was worse: it raised at ``embed_seconds / 60``, inside
    ``one_model``'s handler but *after* every query had been rescored, so the
    model was dropped having already paid the full cost.

    Lax about presence and strict about value, deliberately: a sentinel
    written before this file recorded timings legitimately has no
    ``embed_seconds``, which is why the result distinguishes ``null`` from
    ``0.0`` in the first place.
    """

    model_config = ConfigDict(extra="ignore", strict=True)

    embed_seconds: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)


def _sentinel_embed_seconds(index_dir: Path) -> float | None:
    """Build timing from *index_dir*'s sentinel, or ``None`` if unusable.

    Degrades rather than raises. The index itself is already known good --
    ``_sentinel_valid`` matched its ``inputs`` -- so a corrupt build stanza is
    a lost measurement, not a lost model, and "not measured" is a state this
    report can already represent.
    """
    try:
        return SentinelBuild.model_validate(_sentinel_build(index_dir)).embed_seconds
    except ValidationError:
        return None


def require_a_usable_corpus(corpus: Path) -> int:
    """Number of documents in *corpus*, refusing a directory that has none.

    Checked before the waves rather than discovered after them, because an
    empty corpus does not fail anywhere downstream -- it succeeds, wrongly, at
    every step. Both fingerprints hash an empty set (to `e3b0c442...`, the
    digest of nothing, which makes any two empty corpora indistinguishable to
    the cache identity), `index_directory` accepts an empty directory and
    builds a zero-chunk index, every judgment then scores as a miss, and the
    all-zero row that comes out is individually in range at every field. It
    would be cached and published as this model's measured performance.

    So the guard has to be here. The failure this prevents is not a crash, it
    is a number.

    Raises:
        EvalError: The directory is missing, is not a directory, or holds no
            document the adapter produces.
    """
    if not corpus.is_dir():
        raise EvalError(
            f"BEIR corpus directory {str(corpus)!r} does not exist. Adapt a dataset into "
            "it first with adapt_beir.py."
        )
    documents = [p for p in corpus.rglob("*") if p.is_file() and p.suffix.lower() == ".txt"]
    if not documents:
        raise EvalError(
            f"BEIR corpus directory {str(corpus)!r} contains no .txt documents. Indexing it "
            "would succeed with zero chunks and score every judgment as a miss, publishing "
            "an all-zero row as though it were measured performance. Re-adapt the dataset."
        )
    return len(documents)


def load_bakeoff_judgments(path: Path) -> tuple[list[Judgment], dict[str, set[str]]]:
    """Judgments and their gold map, refusing anything the scorer cannot pair.

    Loaded through the harness's own loader rather than ``json.loads`` per
    line. The raw read accepted a repeated ``query_id``, and the two consumers
    disagreed about what that meant: the gold map is a dict comprehension, so
    it silently kept only the last row, while the scoring loop iterated
    *every* row and wrote each result under the same ``per_query`` key. The
    earlier query was therefore scored against the later row's relevance set
    and then overwritten, leaving an aggregate over fewer queries than the
    ``n=len(judgments)`` printed beside it -- wrong in a way that reads as
    correct.

    ``load_judgments`` already rejects duplicates, and enforces the ascending
    order and id contract this file always assumed without checking. Same
    argument as :func:`read_cache_file`: one loader, so the rules are not
    re-decided -- and re-decided differently -- at each call site.

    Raises:
        EvalError: The file is missing, malformed, or repeats a ``query_id``.
    """
    judgments = load_judgments(path)
    return judgments, {j.query_id: {g.doc for g in j.gold} for j in judgments}


def contrast_rng(model: str) -> np.random.Generator:
    """A generator for one model's contrast, independent of every other's.

    One shared generator meant each contrast consumed whatever the previous
    ones left, so a model's confidence interval depended on how many models
    ran before it. If an earlier model failed, or ``WAVES`` was reordered, the
    survivors got different resamples from byte-identical paired inputs.
    Measured on a near-zero contrast: CI ``[-0.003427, +0.013830]`` when
    another model ran first versus ``[-0.003587, +0.014084]`` when it did not,
    p 0.2418 against 0.2580. No verdict flip was found searching 300 seeds, so
    the practical effect is small -- but a published interval should not depend
    on which unrelated models happened to complete, and
    ``groundkit.evals.significance`` already builds a fresh generator per call
    for exactly this reason.

    Seeded from the model name so the value is deterministic *and* positional
    independence is structural rather than a property of the loop's shape.
    """
    digest = hashlib.sha256(f"{BOOTSTRAP_SEED}:{model}".encode()).digest()[:8]
    return np.random.default_rng(int.from_bytes(digest, "big"))


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
    try:
        document_count = require_a_usable_corpus(corpus)
        judgments, gold = load_bakeoff_judgments(args.data / "judgments.jsonl")
    except EvalError as exc:
        parser.error(str(exc))
    print(f"corpus: {document_count} documents", flush=True)

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
        "chunk_fingerprint": chunk_fingerprint(corpus.resolve()),
        "candidates": CANDIDATES,
    }

    async def one_model(model: str, dims: int) -> dict | None:
        """Index and score a single model, caching the result. Never raises."""
        cached = scores_dir / f"{model_slug(model)}.json"
        # The exact model name, not just its dimensions and not just the slug
        # the filename is derived from. `model_slug` now makes a collision
        # unrepresentable, but recording the name is what lets an entry
        # written by an *older* build of this script -- when the slug was
        # lossy -- be recognised as belonging to a different embedder rather
        # than accepted on a dimension match.
        identity = {**experiment, "model": model, "dims": dims}
        # A cached score is only trustworthy while the index it was produced
        # from is still there and still valid. The module docstring tells
        # operators that deleting a model's index directory forces a re-embed --
        # and it did not: this shortcut returned the cached scores without ever
        # looking at the index, so an index deleted *because it was suspect*
        # left its scores publishable and the rebuild never happened. Requiring
        # the sentinel makes the documented remedy true.
        index_ready = _sentinel_valid(
            args.work / "index" / model_slug(model),
            _sentinel_inputs(corpus.resolve(), model, dims),
        )
        payload = read_cache_file(cached)
        # `read_cache_file` guarantees the *outer* object is a mapping and
        # nothing more. An entry whose identity matches but whose `result` is
        # missing or is not a mapping would raise KeyError or TypeError right
        # here -- outside `one_model`'s handler, so through `asyncio.gather`
        # and into every other model in the wave. Both halves are checked
        # together, so "usable cache" is one condition rather than an
        # assumption resting on a partial one.
        cached_result = payload.get("result") if payload is not None else None
        if (
            payload is not None
            and payload.get("experiment") == identity
            and usable_result(cached_result, set(gold), model=model, dimensions=dims)
        ):
            if index_ready:
                print(f"{model:26s} cached", flush=True)
                return dict(cached_result)
            print(f"{model:26s} scores cached but index missing or stale, rebuilding", flush=True)
        elif cached.exists():
            # A file that is present but unusable -- damaged, or written under
            # different inputs -- is worth announcing. An absent one is the
            # ordinary first run and says nothing.
            print(f"{model:26s} cache unusable or stale, rescoring", flush=True)
        started = time.perf_counter()
        try:
            result = await index_and_score(model, dims, corpus, args.work, judgments, gold)
        except Exception as exc:  # one bad model must not lose the whole run
            print(f"{model:26s} FAILED: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
            return None
        # Written atomically and stamped with the inputs it was produced
        # from, so a later run can tell whether it still applies. A failure
        # here costs a rescore next run; it must not cost this run's siblings.
        if not write_cache_file(cached, {"experiment": identity, "result": result}):
            print(f"{model:26s} scored, but its cache could not be written", flush=True)
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
        # `embed_minutes` is None when this run reused an index whose sentinel
        # predates the recorded build timing. Printed as `--`, which a reader
        # will not mistake for a measurement, and which also keeps the format
        # spec from raising on None.
        embed_cell = "    -- " if r["embed_minutes"] is None else f"{r['embed_minutes']:5.1f}m "
        print(
            f"{r['model']:26s} {r['dimensions']:5d} | {cells[0]:8.4f} {cells[1]:6.3f} "
            f"{cells[3]:6.3f} {cells[2]:6.3f} | {embed_cell}"
            f"{r['query_p50_ms']:6.0f}ms {r['vector_store_mb']:7.0f}MB"
        )

    # Written before the bootstrap gate below, not after it. A run that loses
    # the baseline or produces fewer than two results used to return early,
    # leaving a summary from some previous run in place -- which a reader, or
    # a later script, would take for this run's output while the console said
    # models had failed. It is rewritten unconditionally, so it always
    # describes the run that last finished, even when that run describes very
    # little.
    summary = args.work / "summary.json"
    summary.write_text(
        json.dumps([{k: v for k, v in r.items() if k != "per_query"} for r in results], indent=2),
        encoding="utf-8",
    )

    base = next((r for r in results if r["model"] == args.baseline), None)
    if base is None or len(results) < 2:
        print(
            f"\nskipping the paired bootstrap: baseline {args.baseline!r} "
            f"{'is missing' if base is None else 'is present'}, "
            f"{len(results)} model(s) scored, 2 needed."
        )
    else:
        print(
            f"\npaired bootstrap vs {args.baseline}, n={len(judgments)}, 10,000 resamples, nDCG@10"
        )
        for r in results:
            if r["model"] == args.baseline:
                continue
            mean, lo, hi, p = bootstrap(
                r["per_query"], base["per_query"], "ndcg_at_10", contrast_rng(r["model"])
            )
            verdict = "SIGNIFICANT" if (lo > 0 or hi < 0) else "not significant"
            print(f"  {r['model']:26s} {mean:+.4f}  [{lo:+.4f}, {hi:+.4f}]  p={p:.4f}  {verdict}")

    print(f"\nper-model scores: {scores_dir}\nsummary: {summary}")


if __name__ == "__main__":
    asyncio.run(main())

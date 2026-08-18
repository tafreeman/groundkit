# Concepts

Every other page on this site assumes you already know what a *chunk* is,
what *BM25* means, why an *ADR* gets written instead of just doing the
thing, and what makes a citation *verified* rather than merely *asserted*.
This page is where all of that gets explained, once, in plain language.
Read it start to finish in a few minutes and the rest of the documentation
should make sense on first pass — no background in search, retrieval, or
machine learning is assumed.

If you already know this territory, skim it anyway for what's specific to
*this* project. The exact grounding mechanism, the eval harness's baseline
discipline, and precisely where the LLM boundary is drawn are all narrower
and more particular than the general version of these ideas you probably
already carry around.

## Grounding, and why "verified" beats "asserted"

This is the idea groundkit is built around, so it goes first. The project's
own tagline — "grounded, citation-verifiable retrieval" — packs two words
into that claim, and this section is what they cash out to.

Ask a general-purpose language model a factual question and it answers
fluently, whether or not it actually knows. That confident-sounding wrong
answer is usually called **hallucination**: the model is very good at
producing text that *sounds* like a correct answer, and nothing forces the
words it generates to correspond to anything real. The failure mode that
matters isn't "wrong" so much as "wrong with no way to tell" — a
hallucinated answer and a correct one read identically until someone
checks.

A system that answers *from your documents* can fail the same way twice
over. It can invent a fact outright, or it can do something subtler and
just as damaging: **cite a source that doesn't actually say what was
claimed.** A citation nobody checks is not evidence. It's decoration.

groundkit's answer to both failure modes is the same mechanism. Every
passage it returns is tied to an exact **character range in a specific
source file** — not "somewhere in `policy.md`," but "`policy.md`, exactly
these characters." And that pointer isn't a shortcut for trusting the
index: groundkit can go back, re-open the source, re-read exactly that
character range, and confirm the text is still there before anything
treats it as a citation. If the document changed since it was indexed, the
check reports the citation as **drifted** rather than silently handing
back stale text; if the source can't be checked at all, it says so
(`unresolvable`) instead of guessing.

That's the distinction the whole project sits on:

- **Asserted** means "trust the index" — a passage is returned because it
  was retrieved, full stop.
- **Verified** means "we just checked" — the passage was re-read from its
  source and it still matches, at the moment it mattered.

!!! example "What this looks like in practice"
    Ask "how long do we keep deployment logs?" and a grounded answer
    doesn't just say "30 days" — it names *exactly* which file said so and
    where, in a form that can be re-checked rather than taken on faith.
    `fetch_chunk`, one of groundkit's four read-only tools, is the
    operation that performs this check: it re-reads the cited span and
    reports back one of three verdicts — `verified`, `drifted`, or
    `unresolvable` — never a silent pass.

This is why the rest of this page keeps coming back to **chunks** and
**offsets**: they're the substrate the verification depends on. See
[The LLM boundary](architecture/llm-boundary.md) for the full inventory of
what groundkit checks and doesn't, and the
[data model reference](reference/contracts.md) for the exact fields.

## Chunks and character offsets

A search index can't hand back a whole 40-page document as "the answer" —
the useful unit for search is a paragraph or two, not a file. So before
anything is indexed, groundkit splits each document into **chunks**:
overlapping windows of a few hundred to a couple of thousand characters,
cut on natural boundaries (paragraphs, headings) where it can rather than
mid-sentence.

The overlap matters as much as the splitting does. Cut a document into
strictly back-to-back pieces and an answer that happens to straddle the
cut — the end of one paragraph, the start of the next — gets sliced in
half, with neither half enough to answer the question alone. Overlapping
windows make that much less likely.

What makes a chunk more than "some text a script produced" is that it
carries **exactly where it came from**: a `start_offset` and `end_offset`
— character positions in the original document — such that a chunk's text
is *always* the literal substring of its parent document between those two
numbers. Not approximately: groundkit's data model enforces the
arithmetic, so a chunker with a bug can't even construct an invalid chunk
in the first place.

That's the payoff. Those offsets are precise enough to point back at,
which is what makes the grounding claim in the previous section possible
at all. A chunk with no reliable offset is just a quote; a chunk with one
is a **citation**.

## Two ways to find the right text

groundkit can search a collection two different ways, and they fail
differently — which is the whole reason to have both.

### BM25: matching words

**BM25** ("Best Match 25") is **lexical retrieval**: it matches on the
words actually in your query. Search "refund window" and BM25 looks for
chunks containing "refund" and "window," weighting a match more heavily
when the word is rare across the whole corpus (a common word like "the"
counts for almost nothing; an unusual word like a product code counts for
a lot) and slightly less when it's only a hit because the chunk is long.
This is decades-old, well-understood technology — the same family of
algorithm that powered web search before neural networks — and groundkit's
implementation is pure Python with no external service to run.

BM25 is excellent at exactly what it sounds like: names, product codes,
error messages, exact phrases. It is also **blind to synonyms**, by
construction — search "car" and a chunk that only ever says "automobile"
scores exactly zero, because BM25 has no idea the two words are related.
It never saw the word "car" in that chunk, and it isn't trying to know
what the words *mean*.

### Embeddings: matching meaning

An **embedding** model reads a chunk of text and produces a vector — a
long list of numbers representing where that text sits in a "meaning
space" the model learned during training. Two chunks about the same thing,
worded differently, land as nearby points; two unrelated chunks land far
apart. **Dense retrieval** (dense because that vector is packed with
numbers, unlike a chunk's sparse set of keyword matches) searches by
finding the vectors nearest to your query's own vector.

This is what lets "car" match a chunk that only says "automobile" — the
embedding model learned the two are related, so their vectors land near
each other regardless of the exact words used. The trade runs the opposite
way from BM25's: dense retrieval is good at paraphrase, and can miss an
exact product code or an unusual acronym the model was never trained to
treat specially.

|  | BM25 (lexical) | Embeddings (dense) |
|---|---|---|
| Matches on | shared words | shared meaning |
| Strong at | names, codes, exact phrases | paraphrase, synonyms |
| Blind spot | synonyms it never saw | exact identifiers it wasn't trained to weight |
| Needs to run | nothing extra | an embedding model (Ollama, locally, by default) |

## Combining them: hybrid retrieval and RRF

Because BM25 and embeddings fail in different, largely non-overlapping
ways, running both and combining the results beats running either alone —
this is **hybrid retrieval**.

```
query
  ├─ BM25 (lexical)     ─┐
  └─ embeddings (dense)  ─┼─► RRF fusion ─► optional rerank ─► citation-bearing results
                         ─┘
```

The hard part is combining the two results honestly. A BM25 score and a
cosine-similarity score from an embedding model aren't the same kind of
number — they sit on different, incomparable scales — so averaging them or
just picking whichever is bigger would be comparing apples to a
completely different fruit. groundkit uses **Reciprocal Rank Fusion
(RRF)**, which sidesteps the problem by throwing the scores away entirely
and looking only at *rank*: where did this chunk place in each list?

In one sentence: every result earns `1 / (k + its rank)` points from each
list it appears in, for a small constant `k`, and the points are added up
across lists — so a chunk that ranks well in *both* lists comes out on
top, and a chunk that ranks well in only one still earns some credit
rather than none. You don't need the formula to use groundkit; the useful
intuition is that RRF cares about *position*, never about *how confident*
either retriever felt.

Even though hybrid measures better than BM25 alone on this project's own
quality metrics, `grk search` still defaults to plain BM25 — a deliberate
choice, not an oversight, because hybrid needs a running embedding model
and BM25 needs nothing at all, and the default has to keep working on a
machine with zero setup. See [Retrieval modes](guides/retrieval-modes.md)
for the practical guidance and [ADR-0007](adr/ADR-0007-default-retrieval-mode.md)
for the full reasoning — a good first example if you want to see what an
ADR actually looks like (more on those later on this page).

## Reranking: a slower, more careful second pass

Both BM25 and dense retrieval score a query against *many* chunks
independently — each chunk is scored on its own, without ever being
compared side by side with the query in one pass. That independence is
what makes them fast enough to run over a whole collection.

A **cross-encoder** reranker does the opposite trade: it reads the query
*and* one candidate chunk together, in a single pass, and produces one
relevance score for that specific pairing. Reading the two together,
rather than separately, catches relevance signals that independent scoring
misses — at a cost. A cross-encoder is far too slow to run over an entire
collection; it only makes sense as a **second pass** over a small
shortlist something faster has already narrowed down.

The usual pattern, and groundkit's: retrieve a wider set of candidates
cheaply (BM25, dense, or hybrid), then let a cross-encoder re-score just
that shortlist and reorder it. You trade a little latency on a handful of
candidates for meaningfully better ordering at the top, where it matters
most.

One disambiguation worth making explicit: **a cross-encoder is not a
language model.** It's a small local scoring model — it reads a (query,
passage) pair and outputs a number, and never generates text. It runs
entirely offline once its weights are cached, no different in kind from
BM25 or an embedding model, and it belongs to the deterministic side of
this project, not the LLM side (more on that distinction later on this
page too). Today, groundkit's reranker is wired into the eval harness — so
its effect can be measured — but not yet into `grk search` itself; see
[Retrieval modes](guides/retrieval-modes.md) and
[ADR-0012](adr/ADR-0012-rerank-eval-stage-reorders-upstream-stage.md).

## Measuring quality: the eval harness

### Why measure before you build

It's tempting to add a feature that sounds like it should help — dense
retrieval, hybrid fusion, reranking — and just assume it does. groundkit's
answer is to refuse the assumption: before any of those features existed,
the project first built the tool that measures whether a retrieval change
actually helps, over a fixed, known-answer test set. Every retrieval
feature since has reported a measured result against that same reference
point in the same generated report — including on occasions where a
feature didn't clearly beat what came before it. (At least one of
groundkit's own optional features has, on a specific measured
configuration, landed in exactly that "didn't clearly help" bucket — and
the harness said so plainly in the generated report rather than quietly
omitting it.) The point isn't that every feature has to win. It's that the
project can't fool itself about which ones did.

### The golden corpus

The **golden corpus** is a small, hand-curated set of documents plus a
list of questions someone has already answered by hand — for each
question, which document and which exact quote answers it. Because it's
fixed and versioned, the same questions can be run against the same
documents after every change, and the score is comparable run to run. The
corpus deliberately includes not just straightforward questions but
ambiguous ones, and — importantly — questions the corpus *cannot* answer
at all, to measure whether a retrieval mode is honest enough to come back
empty rather than confidently returning nonsense.

### recall@k and nDCG: what "good" means, in plain terms

**recall@k** answers one yes/no question per query: did a relevant chunk
show up *anywhere* in the top `k` results? groundkit reports this at a few
values of `k` — a small `k` like 1 is strict, since the right answer
basically has to be the very first result; a larger `k` is more forgiving.
It's deliberately a hit-rate rather than "what fraction of everything
relevant did we find": otherwise a single correct answer that happens to
straddle a chunk boundary would silently make the score depend on chunk
size rather than on retrieval quality.

**nDCG@10** ("normalized discounted cumulative gain") is the metric that
*does* care about position: a ranking that puts the right answer first
scores higher than one that buries it near the bottom of the top 10, even
though recall@10 would call both of those a hit. It gives credit for
where in the list the right answer landed, not just whether it landed.

Between the two: recall@k tells you whether the retriever found the
answer at all; nDCG tells you whether it had the good sense to put it near
the top.

### The baseline, and what "delta" means

groundkit measures every retrieval mode against one fixed reference: plain
BM25, with nothing else enabled. Every report line for a later stage —
dense, hybrid, reranked — includes its **delta**: that stage's score minus
the BM25 baseline's score, on the identical questions, in the identical
report. "Beat the baseline by this much" and "didn't beat the baseline"
are both acceptable things for the report to say; a feature that shipped
unmeasured is not.

See [Eval harness](guides/evals.md) for how to run it yourself.

## MCP: how an assistant like Claude searches your documents

**MCP (Model Context Protocol)** is an open standard that lets an AI
assistant call out to external tools mid-conversation, instead of only
generating text from what it already knows. Rather than every assistant
needing its own bespoke integration for every tool, both sides just need
to speak MCP.

groundkit runs an MCP server exposing four tools, all read-only:

| Tool | What it does |
|---|---|
| `search` | Search a collection and return citation-bearing results |
| `fetch_chunk` | Fetch one chunk and re-verify its citation against its source, right now |
| `list_collections` | List the collections this server can search |
| `index_status` | Report a collection's document/chunk counts and embedding identity |

Point an MCP-capable assistant — Claude Desktop or Claude Code, for
instance — at a running groundkit server, and it can search your indexed
documents as part of a conversation and get back grounded,
citation-bearing passages, instead of you copy-pasting text into the chat
and instead of the assistant guessing. See
[MCP clients](guides/mcp-clients.md) to connect one.

## Local-first: no cloud credentials required

groundkit's default retrieval path — plain BM25 — makes no network calls
at all. The moment you opt into dense or hybrid retrieval, the only call
it makes is to a locally-running Ollama process on the same machine
(`127.0.0.1`) — loopback traffic that never reaches an actual network
interface. Every cloud provider, for embeddings or for the optional chat
features, is opt-in configuration, never a requirement.

Why this is worth caring about, beyond principle:

- **Privacy.** Nothing you index or search for is sent anywhere unless you
  deliberately point groundkit at a cloud provider.
- **Cost.** No per-token bill for indexing or searching your own
  documents.
- **No lock-in.** Swap embedding or chat providers, or run with none
  configured at all — the retrieval path itself never requires one.

This is a default, not an unconditional guarantee. The exact inventory of
every path text can leave the process — including exactly what changes the
moment you configure a cloud provider, and what is *not* closed even then
— is [The LLM boundary](architecture/llm-boundary.md). Read it before
pointing anything at a cloud endpoint; it says plainly what is and isn't
covered.

## Deterministic core, LLM at the boundary

This is the organizing principle behind almost every other decision on
this page, so it's worth naming directly.

Most of what groundkit does — reading your files, splitting them into
chunks, scoring BM25 matches, computing embedding distances, fusing
rankings, re-verifying a citation — is ordinary **deterministic** code:
the same input produces the same output, every time, with nothing
"creative" or probabilistic about it. That includes the cross-encoder
reranker from earlier on this page — it's a model in the machine-learning
sense, but it scores deterministically and never generates text, so it
sits on this side of the line too.

A language model is different in kind. It's genuinely useful for
open-ended language tasks, and genuinely unsuited to being a step that has
to be *right* every single time — it can phrase the same request two
different ways on two different calls, and nothing forces it to be
consistent. That makes it a poor fit for anything downstream that depends
on being exact, like deciding which sixteen characters of a document
actually matched a query.

So groundkit draws a hard line: **no model runs anywhere in the retrieval
or citation path.** Finding chunks, scoring them, fusing rankings, and
verifying a citation are all deterministic, every time, no exceptions. A
language model only ever appears at the *boundary* — optionally rewriting
a query before it's searched, optionally writing a synthesized answer from
chunks retrieval already found and verified, or optionally acting as an
advisory judge that checks whether a synthesized answer stayed faithful to
its citations. Every one of those uses is skippable; turn them off and
nothing about search, ranking, or citation changes.

The reason this matters practically: deterministic code can be tested
exhaustively — the same regression test passes or fails the same way
forever, so a reviewer can trust that a green test suite means what it
says. An LLM call can't be pinned down that way. Keeping the LLM confined
to a narrow, optional, clearly-labeled boundary is what lets the rest of
the system make the "verified, not asserted" claim from the top of this
page with a straight face.

## ADRs: recording a decision so it stays one

An **ADR (Architecture Decision Record)** is a short document that records
one decision: what was decided, what alternatives were considered and
rejected, and why. It isn't a design spec for a feature — it's a
permanent note explaining *why the code looks the way it does*, written
for whoever reads the code next (very often the same person, eighteen
months later, having forgotten their own reasoning).

A concrete example: hybrid retrieval measures better than BM25 alone on
this project's own quality metrics, so why does `grk search` still default
to plain BM25? [ADR-0007](adr/ADR-0007-default-retrieval-mode.md) is the
answer — hybrid needs a running embedding model and BM25 needs nothing at
all, and the default has to keep working on a machine with zero setup.
Without the ADR, a future reader sees a default that looks like it's
ignoring the project's own measurements, with no way to tell "considered
and rejected" apart from "nobody got around to it." With it, the reasoning
survives the people who had it in their heads at the time.

Every irreversible decision in groundkit gets one; the full list is the
[ADR index](adr/index.md).

## Where to go next

- [Architecture](architecture/index.md) — how these pieces fit together in
  the actual codebase
- [The LLM boundary](architecture/llm-boundary.md) — the exhaustive
  inventory of where text can leave the process
- [Retrieval modes](guides/retrieval-modes.md),
  [Eval harness](guides/evals.md), and
  [MCP clients](guides/mcp-clients.md) — using each of the above
- [ADR index](adr/index.md) — every recorded decision, not just the two
  used as examples here
- [Security](security.md) and [Known limitations](limitations.md) — the
  honest, current statement of what's guarded and what isn't
- [SPEC.md](https://github.com/tafreeman/groundkit/blob/main/SPEC.md) — the
  formal, load-bearing spec this project holds itself to, for readers who
  want the primary source rather than a plain-language tour of it

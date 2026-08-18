# The LLM boundary

**In plain terms, this page answers one question: when does anything you've
indexed, or anything you type into a query, leave this machine — and what,
if anything, gets stripped out first?** If you're about to index sensitive
material, or turn on a cloud embedding or chat provider, this is the page
to read first, not after the fact. New to terms like *embedding*,
*reranker*, or what "the LLM boundary" even means? [Concepts](../concepts.md)
explains each of those in plain language; this page assumes you know them
and gives the complete, current — and occasionally uncomfortable — inventory
built on top of them. One term this page introduces itself: **redaction**
means a pattern-based pass that swaps things like email addresses and phone
numbers for placeholder tokens before text reaches a cloud provider — not a
promise that everything sensitive is caught, as the sections below make
precise.

SPEC.md §2 makes two promises that only mean something if the boundary they
describe is written down: *"Deterministic core, LLM at the boundary"* and
*"Anonymization at the LLM boundary."* The second one names this document as
its obligation — the boundary must be documented explicitly: **where text can
leave the process, what is redacted, and what is not.**

This page is that inventory. It describes what the code does today, not what
the design intends. Where the two differ, the difference is stated rather than
smoothed over — a boundary document that describes an aspiration is worse than
none, because it is the document a reader trusts when deciding what to index.

!!! note "Current state: the redaction pass exists, on exactly one boundary"

    Phase 5 built `groundkit.providers.redaction` and wired it into **cloud
    chat egress**: `build_chat` wraps every `openai_compatible` chat
    provider in `RedactingChat`, with no operator opt-out (ADR-0017). The
    **embedding** boundary is deliberately *not* redacted — a recorded
    SPEC §2 deviation with its reasoning in ADR-0017 decision 5, restated
    in row 2's section below. Every "Redacted?" cell in the inventory is a
    statement about code that runs today.

## The one-sentence version

With default configuration groundkit makes no network calls to anything except
a loopback Ollama endpoint, and nothing leaves the machine. Change
`EmbeddingConfig.provider` to `openai_compatible` and **your full document
text and every query go to that endpoint verbatim, unredacted**; change
`ChatConfig.provider` to `openai_compatible` and the prompt text `grk answer`
sends goes through the pattern-based redaction pass first — which is a
narrower promise than it sounds, and its limits are stated below.

## Egress inventory

Every point at which bytes derived from your content can leave the process,
plus the one path where the process reaches out based on content you gave it
(URL ingestion, row 6). This list is exhaustive as of the v0.1.0 release
(SPEC.md §9); nothing is queued to extend it.

| # | Path | What leaves | Default | Redacted? |
|---|------|-------------|---------|-----------|
| 1 | `OllamaEmbedder` | Chunk text at ingest; query text at search | **On** — loopback only | No — and it does not leave the machine |
| 2 | `OpenAICompatibleEmbedder` | Chunk text at ingest; query text at search | Off (opt-in) | **No — deliberate deviation, ADR-0017 decision 5** |
| 3 | `CrossEncoderReranker` model load | Model *name* only, to a model host | Off (`rerank` extra) | N/A — no content in the request |
| 4 | `OllamaChat` (`grk answer`, `grk eval --synthesis`) | Query, retrieved chunk text, candidate answers — whatever the rewrite/synthesis/judge prompt carries | Off (opt-in commands) — loopback only | No — and it does not leave the machine |
| 5 | `OpenAICompatChat`, always inside `RedactingChat` | The same prompt text, after the pattern pass | Off (opt-in) | **Pattern-based** — structural values by default, person names only via configured patterns |
| 6 | `UrlLoader` (`grk ingest <url>`, ADR-0016 Wave 4) | The URL itself — host, path, query string — to whatever endpoint it resolves to. Inbound fetch, not content egress: nothing from your existing corpus is sent | On whenever a URL is ingested (opt-in per invocation) | N/A — no content to redact; guarded by the SSRF checks in section 6 below instead |

### 1. Ollama embeddings — the default, and the reason the default is safe

`OllamaEmbedder` POSTs to `{base_url}/api/embed` with the raw text of every
chunk at ingest time and the raw query at search time. `base_url` defaults to
`http://127.0.0.1:11434` (`config.DEFAULT_OLLAMA_BASE_URL`).

That default is the whole local-first guarantee: the text is sent, but it is
sent to loopback, so it never reaches a network interface. SPEC.md §7 records
this endpoint as *the* named, deliberate exception to the SSRF guard — the
guard scopes around it rather than over it, precisely because "this one
private address is allowed" is a decision that should be visible rather than
implicit.

!!! note "`base_url` is operator-controlled and not validated as loopback"

    Nothing in `EmbeddingConfig` constrains `base_url` to a local address. An
    operator who points the *Ollama* provider at a remote host gets exactly
    the egress described in row 2 — full document text over the network —
    while still reading `provider = "ollama"` in their config and reasonably
    believing they are in local mode. The provider name is not the boundary;
    the URL is.

### 2. OpenAI-compatible embeddings — the real boundary crossing

`OpenAICompatibleEmbedder` POSTs to `{base_url}/v1/embeddings`. This is the
path SPEC.md §2's anonymization requirement is written for, and it is the path
where that requirement is **deliberately not met** — a recorded deviation, not
a gap waiting for wiring (ADR-0017 decision 5). Three independent reasons:
redaction tokens are allocated in encounter order, so the same value redacts
to different tokens in the ingest process and the search process, splitting
the semantic space; a vector computed over redacted text stops describing the
chunk the store actually holds; and `CollectionManifest` records nothing about
redaction, so a collection mixing redacted and unredacted vectors would pass
ADR-0004's identity verification — the exact silent-corruption class that
check exists to catch, arriving through the mitigation. Reopening this needs
value-derived stable tokens *and* a redaction marker in the manifest,
together.

**What is sent:** the full text of every chunk of every document you ingest,
and the full text of every query you search with. Batched
(`EmbeddingConfig.batch_size`, default 32), otherwise unmodified. No
truncation, no tokenization, no transformation — embedding requires the text.

**What is redacted: nothing, on purpose.** The redaction pass exists
(`providers/redaction.py`) and runs on the chat path below; it is kept off
this path for the reasons above.

**What is protected:** credentials, not content. The API key is read from
`os.environ[config.api_key_env]` at call time and is never stored in the
config object, never logged, and never written to an index. Credential
scrubbing covers exception messages *and* the `__cause__`/`__context__` chains
that carry them (ADR-0001 hazard 6) — an HTTP failure cannot leak the key
through a re-raised traceback. `base_url` is scrubbed from error messages on
the same path, because it is free-form operator-controlled configuration and
can itself contain a credential in the userinfo component.

That distinction is the honest summary of the boundary today: **groundkit
protects your key from the logs; it does not protect your documents from the
provider.**

### 3. Cross-encoder rerank — model download, not content upload

`CrossEncoderReranker` is a *local* model (see
[Concepts](../concepts.md#reranking-a-slower-more-careful-second-pass) for
what a cross-encoder is and why it counts as deterministic rather than as an
LLM). It scores query/passage pairs in-process, and SPEC.md §2's "no LLM in
the retrieval path" holds: a cross-encoder is a scoring model, not a
generative one, and no chunk text is transmitted anywhere to rerank it.

The one network event is the first load. `sentence_transformers.CrossEncoder`
resolves the configured model name against a model host and downloads weights
to a local cache. Only the model name crosses the wire; none of your content
does. On an air-gapped machine, pre-populate the cache — the load is the only
step that needs the network, and it happens once.

### 4–5. The chat boundary — rewrite, synthesis, and the judge share one seam

`grk answer` (and `grk eval --synthesis`) drive every LLM feature through one
`ChatProtocol` provider: the optional query rewrite sends the query, synthesis
sends the query plus every retrieved chunk's sanitized text, and the advisory
judge sends the query, the candidate answer, and the source texts. Which
feature is calling does not change the boundary — the provider is the egress
point, so the inventory lists providers, not features.

**Local (`OllamaChat`, the default):** the prompt goes to loopback verbatim,
unredacted, exactly like row 1 — it does not leave the machine, and wrapping
it in a redaction pass would add cost to every call for no benefit
(ADR-0017).

**Cloud (`OpenAICompatChat`):** `build_chat` always wraps it in
`RedactingChat` — there is no bare-cloud-chat construction path and no
opt-out flag, and a cloud chat config with an explicitly empty pattern set is
refused at construction. Each call constructs a fresh `Redactor`, redacts the
prompt (and system text), and restores tokens in the completion. Fresh-per-
call is load-bearing: a long-lived redactor would let `restore()` expand a
token minted in one request into a value captured in another — a
cross-request disclosure manufactured by the mitigation itself
(regression-tested).

**What the pattern pass actually covers:** the default floor is structural —
emails, E.164/US phone shapes, IPv4 addresses, long secret-shaped tokens.
**Person names are redacted only via operator-configured patterns.** No
default name regex ships, because free-text name detection by regex is
unreliable enough that shipping one under the label "redacts names" would be
a false promise. SPEC.md §2's "names → tokens" is met through its own
"configurable patterns" clause, and an operator who configures none should
assume names cross the wire.

### 6. URL ingestion — an inbound fetch, guarded like the outbound ones

`UrlLoader` (`grk ingest <url>`, ADR-0016 Wave 4) is not an egress path for
your existing content — it fetches a *new* document in, and stores a local
snapshot of what it read. It is listed here anyway because "the process
opens a socket to an address derived from user input" is exactly the shape
SPEC.md §7's SSRF guard exists for, and this loader shares the guard's
implementation (`utils/url_safety.py`) with the cloud-provider embedding
path in row 2 rather than reinventing it.

Every fetch is checked immediately before the request, not once at
construction — DNS can change in between — via
`ensure_safe_endpoint(..., allow_private_endpoint=False)`, unconditionally:
unlike `OllamaEmbedder`, `UrlLoader` has no named private-endpoint exception.
Redirects are refused rather than followed (`follow_redirects=False` per
request; a 3xx status raises instead of being chased). A URL carrying
userinfo (`user:pass@host`) is refused before the fetch. A URL whose query
string contains a credential-shaped parameter name — `token`, `key`,
`secret`, `password`, and the rest of `url_safety.CREDENTIAL_QUERY_PARAMS` —
is refused for the same reason, rather than redacted: `documents.source` is
`TEXT UNIQUE NOT NULL`, and redacting the query would collapse two distinct
URLs onto one stored identity. That denylist is deliberately non-exhaustive
(`code` and `state` are excluded as ambiguous — OAuth credentials in one
context, an ordinary country or US-state code in another), so a credential
under an unlisted parameter name still gets through; this narrows a
documented leak, it does not close every spelling of one. The response body
is bounded by a byte cap and refused, never truncated, past it.

What this does *not* close: DNS rebinding between `url_safety`'s resolution
and the connection `httpx` actually makes, and the general proxy-bypass
residual risk `trust_env` carries. Both are shared with the embedding-egress
path and are stated plainly in [SECURITY.md](../security.md) rather than
implied closed.

## What is *not* on the list

Stating the negative space explicitly, because "it isn't in the table" is
easy to misread as "nobody checked":

- **File ingestion never fetches.** `FileLoader.load` reads from disk, inside
  an `allowed_base_dir` containment check, and opens no socket. URL ingestion
  does fetch, and is not a gap in this inventory — it is row 6 above, guarded
  on every request.
- **The index is local files.** SQLite metadata and the LanceDB table are
  written to `index_dir`. Nothing in the persistence layer opens a socket.
- **The retrieval path is pure.** BM25 scoring, RRF fusion, and citation
  resolution are deterministic in-process code with no I/O beyond the store.
- **Logging never carries content.** SPEC.md §3: document content and queries
  are never logged at info level.

## Paths that used to extend this boundary

Nothing is currently queued to extend this boundary further — the table this
section used to carry is empty because everything in it has landed. The
Phase 5 rows — the redaction pass, query rewrite, synthesis, the judge — have
moved up into the inventory as rows 4–5. URL ingestion (ADR-0016 Wave 4) has
moved up as row 6. The REST + MCP surfaces landed read-only with **no
mutating operation on either transport** (ADR-0014), so SPEC.md §7's
shared-secret obligation is dormant until a mutating operation exists.

Phase 5 changed row 5, not row 2. SPEC.md §9 expected the embedding row to
be the cell that stopped saying **No**; ADR-0017 decision 5 records why it
deliberately still says it, and the chat row is where the redaction pass
actually runs. Treat the OpenAI-compatible *embedding* provider as sending
your corpus to a third party in the clear, because that is what it does.

## Practical guidance

- **Indexing anything sensitive?** Stay on the default. Local Ollama plus a
  file-based index means no text leaves the machine, and that is not a
  configuration you have to maintain — it is what happens if you change
  nothing.
- **Using a cloud embedding provider?** Assume the provider sees everything
  you index and everything you search for, and decide on that basis. Check
  `base_url` as carefully as you check `provider`.
- **Air-gapped?** Everything works except the first cross-encoder model load.
  Pre-seed the model cache and the `rerank` extra runs offline too.
- **Using a cloud chat provider with `grk answer`?** The pattern pass covers
  structural identifiers, not meaning: the provider still sees your retrieved
  content's substance, your questions, and — unless you configure name
  patterns — every person name in them. Redaction narrows what crosses the
  boundary; it does not make cloud synthesis private.
- **Ingesting a URL?** The fetch is guarded (SSRF checks, refused redirects,
  refused userinfo and credential-shaped query parameters, a byte cap), but
  the credential-parameter check is a denylist, not a closed set, and DNS
  rebinding between the safety check and the actual connection is not
  closed. Don't rely on it against a URL you don't trust the operator of.

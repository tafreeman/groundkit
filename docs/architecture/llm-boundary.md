# The LLM boundary

SPEC.md §2 makes two promises that only mean something if the boundary they
describe is written down: *"Deterministic core, LLM at the boundary"* and
*"Anonymization at the LLM boundary."* The second one names this document as
its obligation — the boundary must be documented explicitly: **where text can
leave the process, what is redacted, and what is not.**

This page is that inventory. It describes what the code does today, not what
the design intends. Where the two differ, the difference is stated rather than
smoothed over — a boundary document that describes an aspiration is worse than
none, because it is the document a reader trusts when deciding what to index.

!!! warning "Current state: the redaction pass is not implemented"

    `groundkit.providers.redaction` is a docstring and nothing else. It is
    Phase 5 work. **No redaction runs on any path today.** Every claim below
    about what is redacted is therefore a claim about *nothing being
    redacted*, and the local-first defaults are doing all of the work.

## The one-sentence version

With default configuration groundkit makes no network calls to anything except
a loopback Ollama endpoint, and nothing leaves the machine. Change
`EmbeddingConfig.provider` to `openai_compatible` and **your full document
text and every query go to that endpoint verbatim, unredacted**.

## Egress inventory

Every point at which bytes derived from your content can leave the process.
This list is exhaustive as of Phase 3; the "not built yet" section below names
the paths that will extend it.

| # | Path | What leaves | Default | Redacted? |
|---|------|-------------|---------|-----------|
| 1 | `OllamaEmbedder` | Chunk text at ingest; query text at search | **On** — loopback only | No — and it does not leave the machine |
| 2 | `OpenAICompatibleEmbedder` | Chunk text at ingest; query text at search | Off (opt-in) | **No** |
| 3 | `CrossEncoderReranker` model load | Model *name* only, to a model host | Off (`rerank` extra) | N/A — no content in the request |

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
where that requirement is currently unmet.

**What is sent:** the full text of every chunk of every document you ingest,
and the full text of every query you search with. Batched
(`EmbeddingConfig.batch_size`, default 32), otherwise unmodified. No
truncation, no tokenization, no transformation — embedding requires the text.

**What is redacted: nothing.** There is no redaction pass on this path.

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

`CrossEncoderReranker` is a *local* model. It scores query/passage pairs
in-process, and SPEC.md §2's "no LLM in the retrieval path" holds: a
cross-encoder is a scoring model, not a generative one, and no chunk text is
transmitted anywhere to rerank it.

The one network event is the first load. `sentence_transformers.CrossEncoder`
resolves the configured model name against a model host and downloads weights
to a local cache. Only the model name crosses the wire; none of your content
does. On an air-gapped machine, pre-populate the cache — the load is the only
step that needs the network, and it happens once.

## What is *not* on the list

Stating the negative space explicitly, because "it isn't in the table" is
easy to misread as "nobody checked":

- **Ingestion never fetches.** `FileSystemLoader.load` reads from disk, inside
  an `allowed_base_dir` containment check. SPEC.md §4 puts URL ingestion in
  v1 scope, but it is not built — there is no HTTP client on the ingestion
  path at all today, and so no inbound-fetch surface to guard yet.
- **The index is local files.** SQLite metadata and the LanceDB table are
  written to `index_dir`. Nothing in the persistence layer opens a socket.
- **The retrieval path is pure.** BM25 scoring, RRF fusion, and citation
  resolution are deterministic in-process code with no I/O beyond the store.
- **Logging never carries content.** SPEC.md §3: document content and queries
  are never logged at info level.

## Paths that will extend this boundary

These do not exist yet. Listed so this page can be checked against them when
they land, rather than quietly falling out of date:

| Path | Phase | Boundary obligation |
|---|---|---|
| Redaction pass (`providers/redaction.py`) | 5 | Names → tokens, configurable patterns; must run **before** any text reaches a cloud provider |
| Optional query rewrite | 5 | Sends the query to an LLM; must be skippable |
| Optional cited synthesis | 5 | Sends retrieved spans to an LLM; may cite only retrieved spans |
| Advisory faithfulness judge | 5 | LLM-as-judge over candidate answers; advisory only, gates nothing |
| URL ingestion | 4 | Inbound fetch — SSRF guard with `redirect: error` (SPEC.md §7) |
| REST + MCP surfaces | 4 | Mutating operations behind a shared-secret header, binding 127.0.0.1 by default |

When Phase 5 lands the redaction pass, row 2 of the egress inventory is the
row that changes, and the "Redacted?" column is the cell that has to stop
saying **No**. Until then, treat the OpenAI-compatible provider as sending
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

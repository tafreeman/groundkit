# Index

Two search backends, plus the store both are rebuilt from. Neither backend
persists its own on-disk form: `BM25Index.from_store()` and a freshly
populated in-memory vector index are both derived, at open time, from
[`SQLiteMetadataStore`](#groundkit.index.metadata.SQLiteMetadataStore) — the
durable truth for documents and chunks (ADR-0002). That is a deliberate
anti-pattern guard, not an optimization deferred: no in-process index state
may shadow persisted state (SPEC.md §2), so postings and vectors can never
drift out of sync with what SQLite actually holds.

## Lexical (BM25)

Ported near-verbatim from ARP (ADR-0001) — pure Python, zero dependencies.
`BM25Index.search` returns `(Chunk, score)` pairs rather than constructing a
`RetrievalResult` itself; joining a chunk back to its document's source path
is the retrieval layer's job; see [Retrieval](retrieval.md).

::: groundkit.index.bm25

## Dense (vector)

`InMemoryVectorStore` is the offline/dev double, importable with no optional
extra installed. `LanceDBVectorStore` is the persisted backend behind the
`dense` extra, with a lazily-imported `lancedb`/`pyarrow` so a BM25-only
install never pays for either. Both implementations close two ported ARP
defects (ADR-0001): hazard 3, a metadata filter accepted through `**kwargs`
and silently never applied, and hazard 5, a document ID reaching an
unescaped LanceDB delete predicate. Filtering happens *before* the top-k
truncation on both stores, not after — filter-then-truncate, so a filtered
query can't lose real matches to unfiltered ones that happened to score
higher.

::: groundkit.index.dense

## Metadata store

The durable truth for documents and chunks (ADR-0002): one SQLite connection
per collection, `foreign_keys=ON`, `journal_mode=WAL`, every call serialized
through a single `asyncio.Lock` held across the awaited call — the lock, not
the flag, is what makes concurrent awaits safe against sqlite3's own
thread-safety limits. Also owns the ADR-0004 collection manifest, which binds
a collection to the embedding `(provider, model_name, dimensions)` triple it
was built with; see [Configuration](config.md) for the identity model itself.

::: groundkit.index.metadata

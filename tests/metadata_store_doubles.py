"""One shared base for every hand-built metadata-store double in the suite.

Why this module exists, stated plainly because it is the point rather than a
tidying: for three phases the recorded reason for *not* widening
``MetadataStoreProtocol`` was test maintenance — "several hand-built doubles
implement only the pre-existing member set, so a new required member breaks
every one of them". That is a true cost and it was correctly weighed, but it
turned into a design constraint on production code: ``Retriever`` grew an
``isinstance`` capability fork with a silent-downgrade branch in the retrieval
hot path, and ``COUNT(*)`` aggregates were reachable only by holding a concrete
:class:`~groundkit.index.metadata.SQLiteMetadataStore` (GK-019).

The constraint was never really about protocols. It was about there being no
single place a protocol member had to be added on the test side. There is one
now:

- :class:`RefusingMetadataStore` — every
  :class:`~groundkit.index.protocols.MetadataStoreProtocol` member, each
  refusing loudly. A double subclasses it and overrides only the members its
  test actually drives, so the double states its own scope and any call
  outside that scope is a named ``NotImplementedError`` rather than a
  fabricated empty answer. Deliberately does **not** implement
  :class:`~groundkit.index.protocols.DocumentRecordStoreProtocol`, so a double
  built on it exercises the *fallback* branch of every optional-capability
  fork.
- :class:`RefusingDocumentRecordStore` — the same, plus the optional
  capability, for a double that must exercise the *keyed* branch instead.
- :class:`DelegatingMetadataStore` — forwards every member of both protocols to
  a real store, for a double whose job is to intercept exactly one call
  (count it, stall it, make it raise) and behave normally otherwise.

``DelegatingMetadataStore`` forwards explicitly rather than through
``__getattr__``. A ``__getattr__`` wrapper is shorter and defeats the purpose:
it returns ``Any``, so mypy stops checking the delegation entirely and a
protocol member the wrapper was supposed to cover can silently vanish from the
real store's surface without anything failing. Explicit forwarders make the
next protocol widening a compile-time-visible edit in exactly one file, which
is the property this module is here to provide.

Not a test module — no ``test_`` prefix, so pytest does not collect it. It is
imported by module name (``from metadata_store_doubles import ...``), the same
flat-import convention ``test_protocol_conformance``'s shared
``assert_signature_parity`` already uses, which ``pyproject.toml``'s
``mypy_path`` includes ``tests`` for.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from groundkit.contracts import Chunk, CollectionManifest, EmbeddingIdentity, SourceClass
    from groundkit.index.metadata import SQLiteMetadataStore
    from groundkit.index.protocols import DocumentRecord


def _refuse(store: object, member: str) -> NotImplementedError:
    """Build the refusal a double raises for a member its test never wired up.

    Names the class and the member, because the useful information when this
    fires is *which double was asked for what* — a bare ``NotImplementedError``
    from inside a protocol seam is one of the least informative failures a
    suite can produce.
    """
    return NotImplementedError(
        f"{type(store).__name__} does not implement {member}(). This double only "
        "supports the members its test overrides; if the code under test now calls "
        f"{member}(), override it rather than widening the shared base."
    )


class RefusingMetadataStore:
    """Structurally a ``MetadataStoreProtocol``; every member refuses until overridden.

    Refusing rather than returning an empty default is deliberate. An
    unoverridden ``get_document_sources`` returning ``{}`` would let a double
    silently answer a question its test never meant it to be asked, and the
    answer — "this collection has no documents" — is precisely the shape of
    every silent-absence defect this repo fails closed against. A double should
    be unable to accidentally assert something about the system it was not
    built to model.
    """

    async def upsert_document(
        self,
        source: str,
        document_id: str,
        content_hash: str,
        *,
        source_class: SourceClass = "text",
        extractor: str | None = None,
    ) -> None:
        raise _refuse(self, "upsert_document")

    async def get_document_hash(self, source: str) -> str | None:
        raise _refuse(self, "get_document_hash")

    async def get_document_id(self, source: str) -> str | None:
        raise _refuse(self, "get_document_id")

    async def get_document_sources(self) -> dict[str, str]:
        raise _refuse(self, "get_document_sources")

    async def add_chunks(self, chunks: list[Chunk], source: str) -> None:
        raise _refuse(self, "add_chunks")

    async def replace_document(
        self,
        source: str,
        document_id: str,
        content_hash: str,
        chunks: list[Chunk],
        *,
        source_class: SourceClass = "text",
        extractor: str | None = None,
    ) -> None:
        raise _refuse(self, "replace_document")

    async def get_chunks(self) -> list[Chunk]:
        raise _refuse(self, "get_chunks")

    async def get_chunk(self, chunk_id: str) -> Chunk | None:
        raise _refuse(self, "get_chunk")

    async def delete_document(self, document_id: str) -> int:
        raise _refuse(self, "delete_document")

    async def write_manifest(self, identity: EmbeddingIdentity) -> None:
        raise _refuse(self, "write_manifest")

    async def verify_manifest(self, identity: EmbeddingIdentity) -> CollectionManifest | None:
        raise _refuse(self, "verify_manifest")

    async def get_manifest(self) -> CollectionManifest | None:
        raise _refuse(self, "get_manifest")

    async def get_generation(self) -> int | None:
        """Return ``None`` — freshness genuinely unanswerable, not merely unimplemented.

        The one member with a real default instead of a refusal, because
        ``None`` is the honest answer for a double with no durable state
        (ADR-0013), and every caller already treats it as "assume it changed".
        Returning it is a statement about the double, not a stand-in for one.
        """
        return None

    async def close(self) -> None:
        """No-op: a double owns no connection to close."""
        return None


class RefusingDocumentRecordStore(RefusingMetadataStore):
    """:class:`RefusingMetadataStore` plus the optional record/count capability.

    Use this when the test needs the code under test to take the **keyed**
    branch of an ``isinstance(store, DocumentRecordStoreProtocol)`` fork.
    Building on :class:`RefusingMetadataStore` instead would silently route the
    test down the whole-table fallback and prove nothing about the keyed path.
    """

    async def get_document_record(self, document_id: str) -> DocumentRecord | None:
        raise _refuse(self, "get_document_record")

    async def get_document_records(self) -> dict[str, DocumentRecord]:
        raise _refuse(self, "get_document_records")

    async def count_documents(self) -> int:
        raise _refuse(self, "count_documents")

    async def count_chunks(self) -> int:
        raise _refuse(self, "count_chunks")


class DelegatingMetadataStore:
    """Forwards every store member to a real :class:`SQLiteMetadataStore`.

    For a double whose job is to intercept exactly one call — count it, stall
    it, make it raise — while everything else behaves like the real store.
    Subclass and override that one member.

    ``db_path`` is forwarded too: :class:`~groundkit.runtime.CollectionRuntime`
    reads it for diagnostics and for the ADR-0013 legacy-store warning, so a
    wrapper missing it fails only on the error path, which is the path least
    likely to be covered.
    """

    def __init__(self, inner: SQLiteMetadataStore) -> None:
        self._inner = inner

    @property
    def db_path(self) -> Path:
        return self._inner.db_path

    async def upsert_document(
        self,
        source: str,
        document_id: str,
        content_hash: str,
        *,
        source_class: SourceClass = "text",
        extractor: str | None = None,
    ) -> None:
        await self._inner.upsert_document(
            source, document_id, content_hash, source_class=source_class, extractor=extractor
        )

    async def get_document_hash(self, source: str) -> str | None:
        return await self._inner.get_document_hash(source)

    async def get_document_id(self, source: str) -> str | None:
        return await self._inner.get_document_id(source)

    async def get_document_sources(self) -> dict[str, str]:
        return await self._inner.get_document_sources()

    async def add_chunks(self, chunks: list[Chunk], source: str) -> None:
        await self._inner.add_chunks(chunks, source)

    async def replace_document(
        self,
        source: str,
        document_id: str,
        content_hash: str,
        chunks: list[Chunk],
        *,
        source_class: SourceClass = "text",
        extractor: str | None = None,
    ) -> None:
        await self._inner.replace_document(
            source,
            document_id,
            content_hash,
            chunks,
            source_class=source_class,
            extractor=extractor,
        )

    async def get_chunks(self) -> list[Chunk]:
        return await self._inner.get_chunks()

    async def get_chunk(self, chunk_id: str) -> Chunk | None:
        return await self._inner.get_chunk(chunk_id)

    async def delete_document(self, document_id: str) -> int:
        return await self._inner.delete_document(document_id)

    async def write_manifest(self, identity: EmbeddingIdentity) -> None:
        await self._inner.write_manifest(identity)

    async def verify_manifest(self, identity: EmbeddingIdentity) -> CollectionManifest | None:
        return await self._inner.verify_manifest(identity)

    async def get_manifest(self) -> CollectionManifest | None:
        return await self._inner.get_manifest()

    async def get_generation(self) -> int | None:
        return await self._inner.get_generation()

    async def get_document_record(self, document_id: str) -> DocumentRecord | None:
        return await self._inner.get_document_record(document_id)

    async def get_document_records(self) -> dict[str, DocumentRecord]:
        return await self._inner.get_document_records()

    async def count_documents(self) -> int:
        return await self._inner.count_documents()

    async def count_chunks(self) -> int:
        return await self._inner.count_chunks()

    async def close(self) -> None:
        await self._inner.close()

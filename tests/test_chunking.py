"""RecursiveChunker tests — offset-preserving chunking and the hard-split
regression test for ADR-0001 hazard 1 (the reproduced infinite loop)."""

from __future__ import annotations

import asyncio
import string
from itertools import pairwise

import pytest

from groundkit.config import ChunkingConfig
from groundkit.contracts import Chunk, Document
from groundkit.errors import ChunkingError
from groundkit.ingestion.chunking import RecursiveChunker
from groundkit.ingestion.protocols import ChunkerProtocol

#: Timeout for the hard-split regression test — a regression to the ADR-0001
#: infinite-loop bug should fail this test in seconds, not hang the suite.
_HARD_SPLIT_TIMEOUT_SECONDS = 5.0


def _assert_offset_invariant(document: Document, chunks: list[Chunk]) -> None:
    for chunk in chunks:
        assert chunk.content == document.content[chunk.start_offset : chunk.end_offset]
        assert chunk.document_id == document.document_id


def _assert_sequential_index(chunks: list[Chunk]) -> None:
    for idx, chunk in enumerate(chunks):
        assert chunk.chunk_index == idx


class TestBasicChunking:
    def test_short_document_single_chunk(self) -> None:
        doc = Document(source="a.md", content="Hello world")
        chunks = RecursiveChunker().chunk(doc)

        assert len(chunks) == 1
        assert chunks[0].content == "Hello world"
        assert chunks[0].start_offset == 0
        assert chunks[0].end_offset == len("Hello world")
        _assert_offset_invariant(doc, chunks)

    def test_long_document_multiple_chunks(self) -> None:
        content = ("word " * 200).strip()
        doc = Document(source="a.md", content=content)
        config = ChunkingConfig(chunk_size=100, chunk_overlap=20)

        chunks = RecursiveChunker().chunk(doc, config=config)

        assert len(chunks) > 1
        _assert_offset_invariant(doc, chunks)
        _assert_sequential_index(chunks)

    def test_split_on_paragraph_boundary(self) -> None:
        content = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        doc = Document(source="a.md", content=content)
        config = ChunkingConfig(chunk_size=30, chunk_overlap=0)

        chunks = RecursiveChunker().chunk(doc, config=config)

        assert len(chunks) >= 2
        _assert_offset_invariant(doc, chunks)

    def test_whitespace_only_split_segment_is_dropped(self) -> None:
        # "\n\n" splits this into ["AAAA", "   ", "BBBB"]; the middle,
        # whitespace-only part must not surface as its own chunk.
        content = "AAAA\n\n   \n\nBBBB"
        doc = Document(source="a.md", content=content)
        config = ChunkingConfig(chunk_size=5, chunk_overlap=0)

        chunks = RecursiveChunker().chunk(doc, config=config)

        assert [c.content for c in chunks] == ["AAAA", "BBBB"]
        _assert_offset_invariant(doc, chunks)

    def test_all_whitespace_document_yields_no_chunks(self) -> None:
        doc = Document(source="a.md", content="   \n\t\n   ")
        chunks = RecursiveChunker().chunk(doc)
        assert chunks == []

    def test_overlap_produces_overlapping_offsets(self) -> None:
        content = ("word " * 200).strip()
        doc = Document(source="a.md", content=content)
        config = ChunkingConfig(chunk_size=100, chunk_overlap=30)

        chunks = RecursiveChunker().chunk(doc, config=config)

        assert len(chunks) > 1
        overlaps = [
            chunks[i].end_offset > chunks[i + 1].start_offset for i in range(len(chunks) - 1)
        ]
        assert any(overlaps)


class TestMetadataAndIdentity:
    def test_metadata_inherited_from_document(self) -> None:
        doc = Document(source="a.md", content="Hello world", metadata={"author": "test"})
        chunks = RecursiveChunker().chunk(doc)

        assert chunks[0].metadata["source"] == "a.md"
        assert chunks[0].metadata["author"] == "test"

    def test_colliding_source_key_does_not_overwrite_authoritative_source(self) -> None:
        # GK-004 regression: a Document whose own metadata carries a "source"
        # key must not let that value win over document.source in the
        # resulting chunk metadata. document.source is the authoritative
        # value joined against downstream (dense metadata_filter, SQLite);
        # a caller-supplied "source" silently overwriting it produced wrong
        # or zero filtered results with nothing raised anywhere.
        doc = Document(
            source="real/path.md",
            content="Hello world",
            metadata={"source": "ATTACKER-LABEL"},
        )
        chunks = RecursiveChunker().chunk(doc)

        assert chunks[0].metadata["source"] == "real/path.md"

    def test_chunk_ids_unique(self) -> None:
        content = ("word " * 200).strip()
        doc = Document(source="a.md", content=content)
        config = ChunkingConfig(chunk_size=100, chunk_overlap=10)

        chunks = RecursiveChunker().chunk(doc, config=config)

        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_conforms_to_chunker_protocol(self) -> None:
        assert isinstance(RecursiveChunker(), ChunkerProtocol)


class TestInvalidConfig:
    def test_non_chunking_config_kwarg_raises_chunking_error(self) -> None:
        doc = Document(source="a.md", content="Hello world")
        with pytest.raises(ChunkingError, match="ChunkingConfig"):
            RecursiveChunker().chunk(doc, config="not-a-config")


class TestHardSplitRegression:
    """ADR-0001 hazard 1: separator-free text with overlap > 0 must terminate,
    advance start monotonically, and fully cover the text — never loop forever
    re-emitting the same [start:end) window."""

    def test_hard_split_terminates_and_covers_separator_free_text(self) -> None:
        chunk_size = 100
        overlap = 30
        alphabet = string.ascii_letters + string.digits  # no separators at all
        content = "".join(alphabet[i % len(alphabet)] for i in range(10 * chunk_size))
        doc = Document(source="blob.txt", content=content)
        config = ChunkingConfig(
            chunk_size=chunk_size, chunk_overlap=overlap, separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunker = RecursiveChunker()

        async def _run() -> list[Chunk]:
            return await asyncio.wait_for(
                asyncio.to_thread(chunker.chunk, doc, config=config),
                timeout=_HARD_SPLIT_TIMEOUT_SECONDS,
            )

        chunks = asyncio.run(_run())

        assert len(chunks) > 1
        _assert_offset_invariant(doc, chunks)
        _assert_sequential_index(chunks)

        starts = [c.start_offset for c in chunks]
        assert starts == sorted(set(starts)), "starts must be strictly monotonic increasing"

        assert chunks[0].start_offset == 0
        assert chunks[-1].end_offset == len(content)
        for prev, nxt in pairwise(chunks):
            assert nxt.start_offset <= prev.end_offset, "no gap may exist between windows"

    def test_hard_split_terminates_when_overlap_would_equal_step(self) -> None:
        # chunk_size - overlap == 1: the smallest legal strictly-positive step.
        chunk_size = 50
        overlap = 49
        content = "x" * (6 * chunk_size)
        doc = Document(source="blob.txt", content=content)
        config = ChunkingConfig(chunk_size=chunk_size, chunk_overlap=overlap, separators=[""])
        chunker = RecursiveChunker()

        async def _run() -> list[Chunk]:
            return await asyncio.wait_for(
                asyncio.to_thread(chunker.chunk, doc, config=config),
                timeout=_HARD_SPLIT_TIMEOUT_SECONDS,
            )

        chunks = asyncio.run(_run())

        assert len(chunks) > 1
        _assert_offset_invariant(doc, chunks)
        starts = [c.start_offset for c in chunks]
        assert starts == sorted(set(starts))


class TestOffsetInvariantAcrossDocuments:
    """Property-style check: for every (document, config) pair, every emitted
    chunk's content is exactly document.content[start:end]."""

    @pytest.mark.parametrize(
        ("content", "config"),
        [
            pytest.param("Hello world", ChunkingConfig(), id="short-default"),
            pytest.param(
                "# Title\n\nFirst paragraph with some words.\n\n"
                "Second paragraph, longer, with more words in it to split.\n\n"
                "## Subheading\n\nThird paragraph.",
                ChunkingConfig(chunk_size=40, chunk_overlap=10),
                id="markdown-structure",
            ),
            pytest.param(
                "héllo wörld 😀 — café résumé naïve 北京 東京 " * 20,
                ChunkingConfig(chunk_size=60, chunk_overlap=15),
                id="unicode-heavy",
            ),
            pytest.param(
                "one two three four five six seven eight nine ten " * 50,
                ChunkingConfig(chunk_size=80, chunk_overlap=0),
                id="word-repeat-no-overlap",
            ),
            pytest.param(
                "line1\nline2\nline3\n" * 30,
                ChunkingConfig(chunk_size=25, chunk_overlap=5, separators=["\n", ""]),
                id="line-separators-only",
            ),
        ],
    )
    def test_offset_invariant_holds(self, content: str, config: ChunkingConfig) -> None:
        doc = Document(source="prop.md", content=content)
        chunks = RecursiveChunker().chunk(doc, config=config)

        assert len(chunks) >= 1
        _assert_offset_invariant(doc, chunks)
        _assert_sequential_index(chunks)
        for chunk in chunks:
            assert chunk.content.strip() != ""

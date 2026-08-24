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


class TestOversizedNeighborDoesNotOrphanShortLeadingPart:
    """A short part must not be stranded as its own chunk by an oversized neighbor.

    ``_merge_parts`` flushes the accumulated run whenever the incoming part
    would overflow ``chunk_size``. When that part is oversized *on its own* it
    gets recursively re-split regardless, so flushing first only guarantees the
    accumulated run is emitted alone. For ``heading\n\nlong body`` -- the shape
    of essentially every markdown document -- that meant the bare heading became
    a chunk. Measured on BEIR SciFact it put 21.8% of chunks under 128 characters
    and cost 0.027 nDCG@10 over 300 queries (p = 0.0004).
    """

    HEADING = "Sodium intake and its association with blood pressure"

    def _document(self) -> Document:
        body = "Dietary sodium reduction lowers systolic blood pressure in adults. " * 20
        return Document(source="paper.md", content=f"{self.HEADING}\n\n{body}")

    def test_heading_is_merged_into_the_first_chunk_not_emitted_alone(self) -> None:
        doc = self._document()
        chunks = RecursiveChunker().chunk(
            doc, config=ChunkingConfig(chunk_size=512, chunk_overlap=64)
        )

        assert chunks[0].content != self.HEADING, (
            "the heading was emitted as a standalone chunk -- the oversized-neighbor "
            "branch in _merge_parts is missing or inverted"
        )
        assert self.HEADING in chunks[0].content
        assert len(chunks[0].content) > len(self.HEADING)
        _assert_offset_invariant(doc, chunks)
        _assert_sequential_index(chunks)

    def test_merging_the_heading_does_not_break_the_size_ceiling(self) -> None:
        """Folding the run in must not emit a chunk larger than ``chunk_size``."""
        doc = self._document()
        config = ChunkingConfig(chunk_size=512, chunk_overlap=64)
        chunks = RecursiveChunker().chunk(doc, config=config)

        assert all(len(chunk.content) <= config.chunk_size for chunk in chunks)


class TestNestedRecursionDoesNotRepeatAPrefix:
    """A flush must not emit a chunk that the next one wholly contains.

    The oversized-neighbor branch handles the case where the *incoming* part
    exceeds ``chunk_size`` on its own. It does not handle the second way the
    same duplication arises, which shows up one level down in the recursion:
    the incoming part fits, so the run is flushed -- but ``_carry_overlap``
    then retains *all* of what was just flushed, so the next span begins where
    that chunk began and repeats it verbatim.

    The reported reproduction is exact rather than illustrative: before the
    fix, ``"Title here\n1234567. tailxx"`` at 13/6 produced spans
    ``(0, 5), (0, 13), (7, 18), (20, 26)`` -- ``"Title"`` orphaned at ``(0, 5)``
    and then repeated inside ``(0, 13)``, which is the very shape
    ``TestOversizedNeighborDoesNotOrphanShortLeadingPart`` exists to prevent.
    """

    TEXT = "Title here\n1234567. tailxx"
    CONFIG = ChunkingConfig(chunk_size=13, chunk_overlap=6)

    def _chunks(self) -> tuple[Document, list[Chunk]]:
        document = Document(source="t.md", content=self.TEXT)
        return document, RecursiveChunker().chunk(document, config=self.CONFIG)

    def test_the_short_prefix_is_not_emitted_alone(self) -> None:
        document, chunks = self._chunks()

        assert [c.content for c in chunks] != ["Title"], "sanity: expected several chunks"
        assert "Title" not in [c.content for c in chunks], (
            "'Title' was emitted as its own chunk and is repeated inside the next one -- "
            "the carry-retains-everything branch in _merge_parts is missing"
        )
        _assert_offset_invariant(document, chunks)
        _assert_sequential_index(chunks)

    def test_no_chunk_is_wholly_contained_in_another(self) -> None:
        """The general property the reported case violates.

        Overlapping neighbours are expected and fine; one chunk *containing*
        another is never useful -- it is a duplicate embedding and a duplicate
        retrieval candidate.
        """
        _, chunks = self._chunks()
        spans = [(c.start_offset, c.end_offset) for c in chunks]

        contained = [
            (inner, outer)
            for index, inner in enumerate(spans)
            for outer in spans[index + 1 :]
            if outer[0] <= inner[0] and inner[1] <= outer[1]
        ]
        assert contained == [], f"chunks contained in a later chunk: {contained}"

    def test_the_property_holds_at_the_configuration_evals_actually_use(self) -> None:
        """512/64 is ``EVAL_CHUNKING_CONFIG``'s shape, and the one every
        published number was produced at."""
        text = "Title here\n\n" + "sentence body. " * 80
        document = Document(source="t.md", content=text)
        chunks = RecursiveChunker().chunk(
            document, config=ChunkingConfig(chunk_size=512, chunk_overlap=64)
        )
        spans = [(c.start_offset, c.end_offset) for c in chunks]

        contained = [
            (inner, outer)
            for index, inner in enumerate(spans)
            for outer in spans[index + 1 :]
            if outer[0] <= inner[0] and inner[1] <= outer[1]
        ]
        assert contained == []
        _assert_offset_invariant(document, chunks)


class TestFoldedFlushDoesNotRepeatItsOwnTail:
    """After folding a part in and recursing, nothing may be carried forward.

    ``_flush`` has already emitted the whole combined span, and its recursion
    applied overlap *within* what it split. Carrying a tail on top of that
    re-emits text the recursion covered: the next chunk starts inside the span
    just written, so it runs backwards and contains its predecessor.

    The oversized branch always cleared the carry in practice -- a part larger
    than ``chunk_size`` also exceeds ``overlap``, so ``_carry_overlap`` came
    back empty -- but by accident rather than intent. The carry-retains-
    everything branch admits parts *smaller* than ``overlap``, where that
    accident does not hold, which is what this pins.

    Reported reproduction, exact: ``"aa\n a"`` at 4/3 with separators
    ``["\n", " ", ""]`` produced ``(0,3), (4,5), (3,5)`` -- the final chunk
    moving backward and wholly repeating the one before it.
    """

    TEXT = "aa\n a"
    CONFIG = ChunkingConfig(chunk_size=4, chunk_overlap=3, separators=["\n", " ", ""])

    def _chunks(self) -> tuple[Document, list[Chunk]]:
        document = Document(source="t.md", content=self.TEXT)
        return document, RecursiveChunker().chunk(document, config=self.CONFIG)

    def test_chunk_starts_never_move_backwards(self) -> None:
        _, chunks = self._chunks()
        starts = [c.start_offset for c in chunks]

        assert starts == sorted(starts), (
            f"chunk starts run backwards ({starts}) -- a folded flush carried a tail "
            "its own recursion had already emitted"
        )

    def test_no_chunk_repeats_one_already_emitted(self) -> None:
        document, chunks = self._chunks()
        spans = [(c.start_offset, c.end_offset) for c in chunks]

        contained = [
            (a, b)
            for index, a in enumerate(spans)
            for b in spans[index + 1 :]
            if (b[0] <= a[0] and a[1] <= b[1]) or (a[0] <= b[0] and b[1] <= a[1])
        ]
        assert contained == [], f"chunks repeat one another: {contained}"
        _assert_offset_invariant(document, chunks)
        _assert_sequential_index(chunks)

    def test_every_non_separator_character_is_still_covered(self) -> None:
        """Clearing the carry must drop duplicates, never content.

        Separator characters are consumed by the split and are legitimately
        absent from every chunk, so they are excluded from the check.
        """
        document, chunks = self._chunks()
        covered: set[int] = set()
        for chunk in chunks:
            covered.update(range(chunk.start_offset, chunk.end_offset))
        separators = set("".join(s for s in self.CONFIG.separators if s))

        missing = [
            index
            for index, char in enumerate(document.content)
            if char.strip() and char not in separators and index not in covered
        ]
        assert missing == [], f"content characters dropped at offsets {missing}"


class TestOverlapIsAppliedOnceNotTwice:
    """A carry may never survive a flush that recursed.

    Whoever emits the chunks owns the overlap. When ``_flush`` emits ``current``
    directly, this loop applies it. When ``current`` was oversized, ``_flush``
    recursed and the recursion applied overlap *within* the span it split -- so
    a tail carried on top re-emits covered text, and the next flush starts
    inside the span just written.

    Three reports were the same rule seen from three branches: the oversized
    branch cleared the carry by accident, the carry-retains-everything branch
    had to be taught to, and the ordinary branch was still carrying across a
    recursion. Deciding from what ``_flush`` did, rather than which branch
    called it, is what makes the property hold generally.

    Reported reproduction, exact: ``"  \n . ... bbb\n..bb b\n"`` at 13/11
    produced ``(0,13), (3,16), (5,18), (19,20), (14,21)`` -- the last chunk
    running backwards and containing ``(19,20)``.
    """

    TEXT = "  \n . ... bbb\n..bb b\n"
    CONFIG = ChunkingConfig(chunk_size=13, chunk_overlap=11)

    def _chunks(self) -> tuple[Document, list[Chunk]]:
        document = Document(source="t.md", content=self.TEXT)
        return document, RecursiveChunker().chunk(document, config=self.CONFIG)

    def test_starts_never_move_backwards(self) -> None:
        _, chunks = self._chunks()
        starts = [c.start_offset for c in chunks]

        assert starts == sorted(starts), (
            f"chunk starts run backwards ({starts}) -- an outer carry survived a "
            "flush whose recursion had already emitted it"
        )

    def test_no_chunk_contains_another(self) -> None:
        document, chunks = self._chunks()
        spans = [(c.start_offset, c.end_offset) for c in chunks]

        contained = [
            (a, b)
            for index, a in enumerate(spans)
            for b in spans[index + 1 :]
            if (b[0] <= a[0] and a[1] <= b[1]) or (a[0] <= b[0] and b[1] <= a[1])
        ]
        assert contained == [], f"chunks repeat one another: {contained}"
        _assert_offset_invariant(document, chunks)
        _assert_sequential_index(chunks)

    @pytest.mark.parametrize(
        ("text", "size", "overlap", "separators"),
        [
            ("  \n . ... bbb\n..bb b\n", 13, 11, ["\n\n", "\n", ". ", " ", ""]),
            ("aa\n a", 4, 3, ["\n", " ", ""]),
            ("Title here\n1234567. tailxx", 13, 6, ["\n\n", "\n", ". ", " ", ""]),
            ("a\nword.bbb\n\nword\n\n . \n\n", 8, 6, ["\n", " ", ""]),
        ],
    )
    def test_every_reported_shape_holds_both_properties(
        self, text: str, size: int, overlap: int, separators: list[str]
    ) -> None:
        """Each of these came from a separate review round, and each was a
        different branch reaching the same defect. Pinned together so a future
        change to one branch cannot quietly reopen another."""
        document = Document(source="t.md", content=text)
        chunks = RecursiveChunker().chunk(
            document,
            config=ChunkingConfig(chunk_size=size, chunk_overlap=overlap, separators=separators),
        )
        spans = [(c.start_offset, c.end_offset) for c in chunks]
        starts = [s for s, _ in spans]

        assert starts == sorted(starts), f"backward starts for {text!r}: {spans}"
        contained = [
            (a, b)
            for index, a in enumerate(spans)
            for b in spans[index + 1 :]
            if (b[0] <= a[0] and a[1] <= b[1]) or (a[0] <= b[0] and b[1] <= a[1])
        ]
        assert contained == [], f"contained spans for {text!r}: {contained}"
        _assert_offset_invariant(document, chunks)


class TestBlankPrefixIsDroppedNotPromoted:
    """A whitespace-only run must never be folded into an oversized part.

    ``_part_offsets`` keeps blank parts like any other, so ``current`` can hold
    nothing but separators and indentation. ``_flush`` drops such a span by its
    own blank check -- but the fold bypasses ``_flush``'s judgement and
    *prepends* the blank run to the part's first sub-chunk, promoting a
    droppable gap into content and shifting every boundary after it.

    Reported reproduction, exact: 99 spaces, then ``". "``, then 101 ``x``, at
    100/0 produced ``(0,100), (100,200), (200,202)`` -- a leading chunk whose
    entire content is one full stop. ``main`` produced the two body chunks and
    nothing else, so this was introduced by this branch's first commit.
    """

    TEXT = " " * 99 + ". " + "x" * 101
    CONFIG = ChunkingConfig(chunk_size=100, chunk_overlap=0)

    def _chunks(self) -> tuple[Document, list[Chunk]]:
        document = Document(source="t.md", content=self.TEXT)
        return document, RecursiveChunker().chunk(document, config=self.CONFIG)

    def test_no_chunk_is_only_punctuation_and_whitespace(self) -> None:
        _, chunks = self._chunks()

        empty = [c.content for c in chunks if not c.content.strip(" \t\n.")]
        assert empty == [], (
            f"chunks carrying no content: {empty!r} -- a blank prefix was folded "
            "into the oversized part instead of dropped"
        )

    def test_the_leading_blank_run_is_not_prepended_to_the_body(self) -> None:
        document, chunks = self._chunks()

        assert chunks[0].start_offset == 101, (
            f"first chunk starts at {chunks[0].start_offset}, not at the body -- "
            "the blank run was promoted into content"
        )
        _assert_offset_invariant(document, chunks)
        _assert_sequential_index(chunks)

    def test_a_non_blank_prefix_is_still_folded(self) -> None:
        """The guard must not disable the fold this branch exists for: a real
        heading beside an oversized body still gets merged in."""
        heading = "Sodium intake and blood pressure"
        document = Document(source="t.md", content=f"{heading}\n\n{'x' * 1000}")
        chunks = RecursiveChunker().chunk(
            document, config=ChunkingConfig(chunk_size=512, chunk_overlap=64)
        )

        assert chunks[0].content != heading
        assert heading in chunks[0].content

"""IngestionPipeline tests: load -> chunk orchestration, ARP's exception-
wrapping shape, and the directory-scale entry point ARP never had."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from groundkit.contracts import Chunk, Document
from groundkit.errors import IngestionError
from groundkit.ingestion.chunking import RecursiveChunker
from groundkit.ingestion.loaders import FileLoader
from groundkit.ingestion.pipeline import IngestionPipeline
from groundkit.ingestion.protocols import ChunkerProtocol, LoaderProtocol


class _RaisingLoader:
    """Fake loader whose load() raises a configurable exception."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    @property
    def supported_extensions(self) -> list[str]:
        return [".txt"]

    async def load(self, _source: str) -> list[Document]:
        raise self._exc


class _SingleDocLoader:
    """Fake loader that always returns one fixed Document, ignoring the path."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".txt"]

    async def load(self, source: str) -> list[Document]:
        return [Document(source=source, content="fixed content")]


class _RaisingChunker:
    """Fake chunker whose chunk() always raises."""

    def chunk(self, document: Document, **_kwargs: Any) -> list[Chunk]:
        raise RuntimeError(f"bad chunk for {document.document_id}")


class _ConcurrencyTrackingLoader:
    """Fake loader that records the peak number of concurrently in-flight loads."""

    def __init__(self, delay_seconds: float) -> None:
        self._delay = delay_seconds
        self._active = 0
        self.max_seen = 0

    @property
    def supported_extensions(self) -> list[str]:
        return [".txt"]

    async def load(self, source: str) -> list[Document]:
        self._active += 1
        self.max_seen = max(self.max_seen, self._active)
        await asyncio.sleep(self._delay)
        self._active -= 1
        return [Document(source=source, content="x")]


def _write_files(root: Path, *relative_paths: str) -> None:
    for rel in relative_paths:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"content of {rel}", encoding="utf-8")


class TestIngestSingleSource:
    def test_ingest_markdown_file(self, tmp_path: Path) -> None:
        md_file = tmp_path / "test.md"
        md_file.write_text("# Hello\n\nParagraph one.\n\nParagraph two.", encoding="utf-8")

        pipeline = IngestionPipeline(
            loader=FileLoader(allowed_base_dir=tmp_path),
            chunker=RecursiveChunker(),
        )
        chunks = asyncio.run(pipeline.ingest(str(md_file)))

        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_ingest_uses_default_chunker_when_none_given(self, tmp_path: Path) -> None:
        md_file = tmp_path / "test.md"
        md_file.write_text("Some content.", encoding="utf-8")

        pipeline = IngestionPipeline(loader=FileLoader(allowed_base_dir=tmp_path))
        chunks = asyncio.run(pipeline.ingest(str(md_file)))

        assert len(chunks) == 1
        assert chunks[0].content == "Some content."

    def test_empty_file_yields_no_chunks(self, tmp_path: Path) -> None:
        md_file = tmp_path / "empty.md"
        md_file.write_text("", encoding="utf-8")

        pipeline = IngestionPipeline(loader=FileLoader(allowed_base_dir=tmp_path))
        assert asyncio.run(pipeline.ingest(str(md_file))) == []


class TestExceptionWrapping:
    def test_ingestion_error_from_loader_reraised_unchanged(self) -> None:
        original = IngestionError("original failure")
        pipeline = IngestionPipeline(loader=_RaisingLoader(original))

        with pytest.raises(IngestionError) as excinfo:
            asyncio.run(pipeline.ingest("whatever.txt"))

        assert excinfo.value is original

    def test_other_loader_exception_wrapped_with_source_context(self) -> None:
        pipeline = IngestionPipeline(loader=_RaisingLoader(ValueError("oops")))

        with pytest.raises(IngestionError, match="Loader failed") as excinfo:
            asyncio.run(pipeline.ingest("whatever.txt"))

        assert isinstance(excinfo.value.__cause__, ValueError)

    def test_chunking_failure_wrapped_with_document_id(self) -> None:
        pipeline = IngestionPipeline(loader=_SingleDocLoader(), chunker=_RaisingChunker())

        with pytest.raises(IngestionError, match="Chunking failed for document"):
            asyncio.run(pipeline.ingest("source.txt"))


class TestIngestDirectory:
    def test_nested_dirs_and_mixed_extensions(self, tmp_path: Path) -> None:
        _write_files(
            tmp_path,
            "a.md",
            "sub/b.txt",
            "sub/nested/c.markdown",
            "sub/ignored.bin",
        )
        pipeline = IngestionPipeline(loader=FileLoader(allowed_base_dir=tmp_path))

        chunks = asyncio.run(pipeline.ingest_directory(str(tmp_path)))

        sources = {c.metadata["source"] for c in chunks}
        assert any(s.endswith("a.md") for s in sources)
        assert any(s.endswith("b.txt") for s in sources)
        assert any(s.endswith("c.markdown") for s in sources)
        assert not any(s.endswith("ignored.bin") for s in sources)

    def test_skips_hidden_directories(self, tmp_path: Path) -> None:
        _write_files(tmp_path, "visible.md", ".hidden/secret.md")
        pipeline = IngestionPipeline(loader=FileLoader(allowed_base_dir=tmp_path))

        chunks = asyncio.run(pipeline.ingest_directory(str(tmp_path)))

        sources = {c.metadata["source"] for c in chunks}
        assert any(s.endswith("visible.md") for s in sources)
        assert not any("secret.md" in s for s in sources)

    def test_empty_directory_returns_empty_list(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        pipeline = IngestionPipeline(loader=FileLoader(allowed_base_dir=tmp_path))

        assert asyncio.run(pipeline.ingest_directory(str(empty_dir))) == []

    def test_directory_with_no_matching_extensions_returns_empty_list(self, tmp_path: Path) -> None:
        (tmp_path / "data.bin").write_bytes(b"\x00")
        pipeline = IngestionPipeline(loader=FileLoader(allowed_base_dir=tmp_path))

        assert asyncio.run(pipeline.ingest_directory(str(tmp_path))) == []

    def test_missing_directory_raises_ingestion_error(self, tmp_path: Path) -> None:
        pipeline = IngestionPipeline(loader=FileLoader(allowed_base_dir=tmp_path))

        with pytest.raises(IngestionError, match="Directory not found"):
            asyncio.run(pipeline.ingest_directory(str(tmp_path / "nope")))

    def test_file_path_instead_of_directory_raises(self, tmp_path: Path) -> None:
        a_file = tmp_path / "a.md"
        a_file.write_text("x", encoding="utf-8")
        pipeline = IngestionPipeline(loader=FileLoader(allowed_base_dir=tmp_path))

        with pytest.raises(IngestionError, match="Directory not found"):
            asyncio.run(pipeline.ingest_directory(str(a_file)))

    def test_deterministic_ordering(self, tmp_path: Path) -> None:
        _write_files(tmp_path, "b.txt", "a.txt", "c.txt")
        pipeline = IngestionPipeline(loader=FileLoader(allowed_base_dir=tmp_path))

        chunks = asyncio.run(pipeline.ingest_directory(str(tmp_path)))
        sources_in_order = [c.metadata["source"] for c in chunks]

        assert sources_in_order == sorted(sources_in_order)

    def test_invalid_max_concurrent_raises_value_error(self, tmp_path: Path) -> None:
        pipeline = IngestionPipeline(loader=FileLoader(allowed_base_dir=tmp_path))

        with pytest.raises(ValueError, match="max_concurrent"):
            asyncio.run(pipeline.ingest_directory(str(tmp_path), max_concurrent=0))

    def test_concurrency_is_bounded(self, tmp_path: Path) -> None:
        _write_files(tmp_path, *(f"f{i}.txt" for i in range(6)))
        loader = _ConcurrencyTrackingLoader(delay_seconds=0.02)
        pipeline = IngestionPipeline(loader=loader)

        asyncio.run(pipeline.ingest_directory(str(tmp_path), max_concurrent=2))

        assert loader.max_seen == 2

    def test_directory_error_propagates_from_underlying_ingest(self, tmp_path: Path) -> None:
        _write_files(tmp_path, "a.txt")
        pipeline = IngestionPipeline(
            loader=_SingleDocLoader(),
            chunker=_RaisingChunker(),
        )

        with pytest.raises(IngestionError, match="Chunking failed"):
            asyncio.run(pipeline.ingest_directory(str(tmp_path)))


class TestProtocolConformance:
    def test_file_loader_conforms_to_loader_protocol(self) -> None:
        assert isinstance(FileLoader(), LoaderProtocol)

    def test_recursive_chunker_conforms_to_chunker_protocol(self) -> None:
        assert isinstance(RecursiveChunker(), ChunkerProtocol)

    def test_pipeline_accepts_protocol_typed_components(self, tmp_path: Path) -> None:
        loader: LoaderProtocol = FileLoader(allowed_base_dir=tmp_path)
        chunker: ChunkerProtocol = RecursiveChunker()
        pipeline = IngestionPipeline(loader=loader, chunker=chunker)

        md_file = tmp_path / "a.md"
        md_file.write_text("hello", encoding="utf-8")
        chunks = asyncio.run(pipeline.ingest(str(md_file)))
        assert len(chunks) == 1

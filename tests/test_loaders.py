"""FileLoader tests — the single parametrized loader replacing ARP's
duplicated MarkdownLoader/TextLoader (ADR-0001, loaders.py row), including
the path-traversal test ARP never wrote."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest

from groundkit.contracts import Document
from groundkit.errors import IngestionError
from groundkit.ingestion.loaders import DEFAULT_EXTENSIONS, FileLoader
from groundkit.ingestion.protocols import LoaderProtocol


def _load(loader: FileLoader, source: str) -> list[Document]:
    return asyncio.run(loader.load(source))


class TestFileLoaderBasics:
    def test_load_markdown_file(self, tmp_path: Path) -> None:
        md_file = tmp_path / "test.md"
        md_file.write_text("# Title\n\nSome content here.", encoding="utf-8")

        loader = FileLoader(allowed_base_dir=tmp_path)
        docs = _load(loader, str(md_file))

        assert len(docs) == 1
        assert docs[0].content == "# Title\n\nSome content here."
        assert docs[0].source == str(md_file.resolve())
        assert docs[0].metadata["file_name"] == "test.md"
        assert docs[0].metadata["file_extension"] == ".md"

    def test_load_text_file(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "notes.txt"
        txt_file.write_text("Plain text body.", encoding="utf-8")

        loader = FileLoader(allowed_base_dir=tmp_path)
        docs = _load(loader, str(txt_file))

        assert len(docs) == 1
        assert docs[0].content == "Plain text body."
        assert docs[0].metadata["file_extension"] == ".txt"

    def test_load_markdown_extension_variant(self, tmp_path: Path) -> None:
        md_file = tmp_path / "long.markdown"
        md_file.write_text("Body.", encoding="utf-8")

        loader = FileLoader(allowed_base_dir=tmp_path)
        docs = _load(loader, str(md_file))

        assert docs[0].content == "Body."

    def test_supported_extensions_defaults(self) -> None:
        loader = FileLoader()
        assert loader.supported_extensions == list(DEFAULT_EXTENSIONS)

    def test_custom_extensions(self, tmp_path: Path) -> None:
        rst_file = tmp_path / "doc.rst"
        rst_file.write_text("reStructuredText body", encoding="utf-8")

        loader = FileLoader(allowed_base_dir=tmp_path, extensions=(".rst",))
        docs = _load(loader, str(rst_file))

        assert loader.supported_extensions == [".rst"]
        assert docs[0].content == "reStructuredText body"

    def test_unsupported_extension_rejected(self, tmp_path: Path) -> None:
        bin_file = tmp_path / "data.bin"
        bin_file.write_bytes(b"\x00\x01")

        loader = FileLoader(allowed_base_dir=tmp_path)
        with pytest.raises(IngestionError, match="Unsupported extension"):
            _load(loader, str(bin_file))

    def test_allowed_base_dir_property_resolved(self, tmp_path: Path) -> None:
        loader = FileLoader(allowed_base_dir=tmp_path)
        assert loader.allowed_base_dir == tmp_path.resolve()
        assert loader.allowed_base_dir.is_absolute()

    def test_conforms_to_loader_protocol(self) -> None:
        assert isinstance(FileLoader(), LoaderProtocol)


class TestFileLoaderEmptyFileHandling:
    """Empty/whitespace-only files behave identically across every supported extension."""

    @pytest.mark.parametrize("suffix", [".md", ".markdown", ".txt"])
    def test_empty_file_returns_empty_list(
        self, tmp_path: Path, suffix: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        empty_file = tmp_path / f"empty{suffix}"
        empty_file.write_text("", encoding="utf-8")

        loader = FileLoader(allowed_base_dir=tmp_path)
        with caplog.at_level(logging.WARNING):
            docs = _load(loader, str(empty_file))

        assert docs == []
        assert any("Empty or whitespace-only" in rec.message for rec in caplog.records)

    @pytest.mark.parametrize("suffix", [".md", ".markdown", ".txt"])
    def test_whitespace_only_file_returns_empty_list(self, tmp_path: Path, suffix: str) -> None:
        blank_file = tmp_path / f"blank{suffix}"
        blank_file.write_text("   \n\t\n  ", encoding="utf-8")

        loader = FileLoader(allowed_base_dir=tmp_path)
        assert _load(loader, str(blank_file)) == []


class TestFileLoaderSizeCap:
    def test_oversized_file_rejected(self, tmp_path: Path) -> None:
        big_file = tmp_path / "big.txt"
        big_file.write_text("a" * 100, encoding="utf-8")

        loader = FileLoader(allowed_base_dir=tmp_path, max_bytes=50)
        with pytest.raises(IngestionError, match="exceeds max size"):
            _load(loader, str(big_file))

    def test_file_at_exact_cap_is_accepted(self, tmp_path: Path) -> None:
        exact_file = tmp_path / "exact.txt"
        exact_file.write_text("a" * 50, encoding="utf-8")

        loader = FileLoader(allowed_base_dir=tmp_path, max_bytes=50)
        docs = _load(loader, str(exact_file))
        assert docs[0].content == "a" * 50


class TestFileLoaderErrors:
    def test_nonexistent_file_raises(self, tmp_path: Path) -> None:
        loader = FileLoader(allowed_base_dir=tmp_path)
        with pytest.raises(IngestionError, match="not found"):
            _load(loader, str(tmp_path / "missing.md"))

    def test_directory_path_raises(self, tmp_path: Path) -> None:
        subdir = tmp_path / "sub.md"
        subdir.mkdir()

        loader = FileLoader(allowed_base_dir=tmp_path)
        with pytest.raises(IngestionError, match="Not a file"):
            _load(loader, str(subdir))

    def test_path_traversal_rejected(self, tmp_path: Path) -> None:
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        outside = tmp_path / "secret.md"
        outside.write_text("top secret", encoding="utf-8")

        loader = FileLoader(allowed_base_dir=sandbox)
        escape_path = sandbox / ".." / "secret.md"
        with pytest.raises(IngestionError, match="escapes base"):
            _load(loader, str(escape_path))

    def test_absolute_path_outside_base_rejected(self, tmp_path: Path) -> None:
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        outside = tmp_path / "outside.md"
        outside.write_text("nope", encoding="utf-8")

        loader = FileLoader(allowed_base_dir=sandbox)
        with pytest.raises(IngestionError, match="escapes base"):
            _load(loader, str(outside))


class TestFileLoaderInvalidUtf8:
    """A file that is not valid UTF-8 must raise the typed ``IngestionError``,
    not a raw ``UnicodeDecodeError`` — ``UnicodeDecodeError`` is a
    ``ValueError`` subclass, not an ``OSError``, so a naive ``except OSError``
    around the read lets it escape uncaught."""

    @pytest.mark.parametrize("suffix", [".md", ".txt"])
    def test_invalid_utf8_raises_ingestion_error(self, tmp_path: Path, suffix: str) -> None:
        bad_file = tmp_path / f"bad{suffix}"
        bad_file.write_bytes(b"\xff\xfe\x00invalid")

        loader = FileLoader(allowed_base_dir=tmp_path)
        with pytest.raises(IngestionError, match="not valid UTF-8") as exc_info:
            _load(loader, str(bad_file))
        assert isinstance(exc_info.value.__cause__, UnicodeDecodeError)

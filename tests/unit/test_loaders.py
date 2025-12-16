#
# Copyright RODMENA LIMITED 2025
# SPDX-License-Identifier: Apache-2.0
#
"""
Unit tests for ragit.loaders module.
"""

import pytest

from ragit import Document
from ragit.loaders import (
    chunk_by_separator,
    chunk_document,
    chunk_rst_sections,
    chunk_text,
    load_directory,
    load_text,
)


class TestLoadText:
    """Tests for load_text function."""

    def test_load_text_file(self, tmp_path):
        """Test loading a text file."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("Hello, World!")

        doc = load_text(file_path)

        assert doc.id == "test"
        assert doc.content == "Hello, World!"
        assert doc.metadata["filename"] == "test.txt"
        assert "source" in doc.metadata

    def test_load_text_with_path_string(self, tmp_path):
        """Test loading with string path."""
        file_path = tmp_path / "doc.md"
        file_path.write_text("# Title\n\nContent")

        doc = load_text(str(file_path))

        assert doc.id == "doc"
        assert "# Title" in doc.content

    def test_load_text_preserves_content(self, tmp_path):
        """Test that content is preserved exactly."""
        content = "Line 1\nLine 2\n\nParagraph 2"
        file_path = tmp_path / "multi.txt"
        file_path.write_text(content)

        doc = load_text(file_path)

        assert doc.content == content


class TestLoadDirectory:
    """Tests for load_directory function."""

    def test_load_directory_txt(self, tmp_path):
        """Test loading .txt files from directory."""
        (tmp_path / "a.txt").write_text("Content A")
        (tmp_path / "b.txt").write_text("Content B")
        (tmp_path / "c.md").write_text("Content C")  # Should be ignored

        docs = load_directory(tmp_path, "*.txt")

        assert len(docs) == 2
        ids = [d.id for d in docs]
        assert "a" in ids
        assert "b" in ids

    def test_load_directory_multiple_patterns(self, tmp_path):
        """Test loading with different patterns."""
        (tmp_path / "doc.rst").write_text("RST content")
        (tmp_path / "readme.md").write_text("MD content")

        rst_docs = load_directory(tmp_path, "*.rst")
        md_docs = load_directory(tmp_path, "*.md")

        assert len(rst_docs) == 1
        assert len(md_docs) == 1

    def test_load_directory_recursive(self, tmp_path):
        """Test recursive directory loading."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "root.txt").write_text("Root")
        (subdir / "nested.txt").write_text("Nested")

        docs_flat = load_directory(tmp_path, "*.txt", recursive=False)
        docs_recursive = load_directory(tmp_path, "**/*.txt", recursive=True)

        assert len(docs_flat) == 1
        assert len(docs_recursive) == 2

    def test_load_directory_empty(self, tmp_path):
        """Test loading from empty directory."""
        docs = load_directory(tmp_path, "*.txt")

        assert docs == []


class TestChunkText:
    """Tests for chunk_text function."""

    def test_chunk_text_basic(self):
        """Test basic text chunking."""
        text = "A" * 100
        chunks = chunk_text(text, chunk_size=30, chunk_overlap=10)

        assert len(chunks) > 1
        assert all(len(c.content) <= 30 for c in chunks)

    def test_chunk_text_overlap(self):
        """Test that chunks overlap correctly."""
        text = "0123456789" * 10  # 100 chars
        chunks = chunk_text(text, chunk_size=20, chunk_overlap=5)

        # Check overlap
        if len(chunks) > 1:
            end_of_first = chunks[0].content[-5:]
            start_of_second = chunks[1].content[:5]
            # They should share some content
            assert end_of_first in chunks[1].content or start_of_second in chunks[0].content

    def test_chunk_text_doc_id(self):
        """Test that doc_id is set correctly."""
        chunks = chunk_text("Hello world", chunk_size=100, chunk_overlap=10, doc_id="mydoc")

        assert all(c.doc_id == "mydoc" for c in chunks)

    def test_chunk_text_indices(self):
        """Test that chunk indices are sequential."""
        text = "A" * 200
        chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)

        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_text_invalid_overlap(self):
        """Test that invalid overlap raises error."""
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            chunk_text("text", chunk_size=10, chunk_overlap=10)

    def test_chunk_text_empty(self):
        """Test chunking empty text."""
        chunks = chunk_text("", chunk_size=100, chunk_overlap=10)

        assert chunks == []

    def test_chunk_text_whitespace_only(self):
        """Test chunking whitespace-only text."""
        chunks = chunk_text("   \n\t  ", chunk_size=100, chunk_overlap=10)

        assert chunks == []


class TestChunkDocument:
    """Tests for chunk_document function."""

    def test_chunk_document_basic(self):
        """Test chunking a document."""
        doc = Document(id="doc1", content="A" * 200)
        chunks = chunk_document(doc, chunk_size=50, chunk_overlap=10)

        assert len(chunks) > 1
        assert all(c.doc_id == "doc1" for c in chunks)

    def test_chunk_document_preserves_metadata_id(self):
        """Test that document ID is used for chunks."""
        doc = Document(id="special-doc", content="Content here")
        chunks = chunk_document(doc)

        assert chunks[0].doc_id == "special-doc"


class TestChunkBySeparator:
    """Tests for chunk_by_separator function."""

    def test_chunk_by_paragraph(self):
        """Test chunking by paragraph."""
        text = "Para 1\n\nPara 2\n\nPara 3"
        chunks = chunk_by_separator(text, separator="\n\n")

        assert len(chunks) == 3
        assert chunks[0].content == "Para 1"
        assert chunks[1].content == "Para 2"
        assert chunks[2].content == "Para 3"

    def test_chunk_by_custom_separator(self):
        """Test chunking by custom separator."""
        text = "Section 1\n---\nSection 2\n---\nSection 3"
        chunks = chunk_by_separator(text, separator="\n---\n")

        assert len(chunks) == 3

    def test_chunk_by_separator_empty_parts(self):
        """Test that empty parts are skipped."""
        text = "A\n\n\n\nB"  # Multiple separators
        chunks = chunk_by_separator(text, separator="\n\n")

        contents = [c.content for c in chunks]
        assert "" not in contents

    def test_chunk_by_separator_doc_id(self):
        """Test doc_id is set correctly."""
        chunks = chunk_by_separator("A\n\nB", separator="\n\n", doc_id="test")

        assert all(c.doc_id == "test" for c in chunks)


class TestChunkRstSections:
    """Tests for chunk_rst_sections function."""

    def test_chunk_rst_basic(self):
        """Test basic RST section chunking."""
        text = """Introduction
============

This is the intro.

Getting Started
---------------

This is getting started.
"""
        chunks = chunk_rst_sections(text)

        assert len(chunks) >= 2

    def test_chunk_rst_no_sections(self):
        """Test RST with no sections returns whole text."""
        text = "Just plain text without any section headers."
        chunks = chunk_rst_sections(text)

        assert len(chunks) == 1
        assert chunks[0].content == text

    def test_chunk_rst_empty(self):
        """Test empty RST."""
        chunks = chunk_rst_sections("")

        assert chunks == []

    def test_chunk_rst_doc_id(self):
        """Test doc_id is set correctly."""
        text = """Title
=====

Content.
"""
        chunks = chunk_rst_sections(text, doc_id="rst_doc")

        assert all(c.doc_id == "rst_doc" for c in chunks)

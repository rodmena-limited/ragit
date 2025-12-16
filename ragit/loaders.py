#
# Copyright RODMENA LIMITED 2025
# SPDX-License-Identifier: Apache-2.0
#
"""
Document loading and chunking utilities.

Provides simple functions to load documents from files and chunk text.
"""

import re
from pathlib import Path

from ragit.core.experiment.experiment import Chunk, Document


def load_text(path: str | Path) -> Document:
    """
    Load a single text file as a Document.

    Parameters
    ----------
    path : str or Path
        Path to the text file (.txt, .md, .rst, etc.)

    Returns
    -------
    Document
        Document with file content and metadata.

    Examples
    --------
    >>> doc = load_text("docs/tutorial.rst")
    >>> print(doc.id, len(doc.content))
    """
    path = Path(path)
    content = path.read_text(encoding="utf-8")
    return Document(id=path.stem, content=content, metadata={"source": str(path), "filename": path.name})


def load_directory(path: str | Path, pattern: str = "*.txt", recursive: bool = False) -> list[Document]:
    """
    Load all matching files from a directory as Documents.

    Parameters
    ----------
    path : str or Path
        Directory path.
    pattern : str
        Glob pattern for files (default: "*.txt").
    recursive : bool
        If True, search recursively (default: False).

    Returns
    -------
    list[Document]
        List of loaded documents.

    Examples
    --------
    >>> docs = load_directory("docs/", "*.rst")
    >>> docs = load_directory("docs/", "**/*.md", recursive=True)
    """
    path = Path(path)
    glob_method = path.rglob if recursive else path.glob
    documents = []

    for file_path in sorted(glob_method(pattern)):
        if file_path.is_file():
            documents.append(load_text(file_path))

    return documents


def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 50, doc_id: str = "doc") -> list[Chunk]:
    """
    Split text into overlapping chunks.

    Parameters
    ----------
    text : str
        Text to chunk.
    chunk_size : int
        Maximum characters per chunk (default: 512).
    chunk_overlap : int
        Overlap between chunks (default: 50).
    doc_id : str
        Document ID for the chunks (default: "doc").

    Returns
    -------
    list[Chunk]
        List of text chunks.

    Examples
    --------
    >>> chunks = chunk_text("Long document...", chunk_size=256, chunk_overlap=50)
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")

    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append(Chunk(content=chunk_text, doc_id=doc_id, chunk_index=chunk_idx))
            chunk_idx += 1

        start = end - chunk_overlap
        if start >= len(text) - chunk_overlap:
            break

    return chunks


def chunk_document(doc: Document, chunk_size: int = 512, chunk_overlap: int = 50) -> list[Chunk]:
    """
    Split a Document into overlapping chunks.

    Parameters
    ----------
    doc : Document
        Document to chunk.
    chunk_size : int
        Maximum characters per chunk.
    chunk_overlap : int
        Overlap between chunks.

    Returns
    -------
    list[Chunk]
        List of chunks from the document.
    """
    return chunk_text(doc.content, chunk_size, chunk_overlap, doc.id)


def chunk_by_separator(text: str, separator: str = "\n\n", doc_id: str = "doc") -> list[Chunk]:
    """
    Split text by a separator (e.g., paragraphs, sections).

    Parameters
    ----------
    text : str
        Text to split.
    separator : str
        Separator string (default: double newline for paragraphs).
    doc_id : str
        Document ID for the chunks.

    Returns
    -------
    list[Chunk]
        List of chunks.

    Examples
    --------
    >>> chunks = chunk_by_separator(text, separator="\\n---\\n")
    """
    parts = text.split(separator)
    chunks = []

    for idx, part in enumerate(parts):
        content = part.strip()
        if content:
            chunks.append(Chunk(content=content, doc_id=doc_id, chunk_index=idx))

    return chunks


def chunk_rst_sections(text: str, doc_id: str = "doc") -> list[Chunk]:
    """
    Split RST document by section headers.

    Parameters
    ----------
    text : str
        RST document text.
    doc_id : str
        Document ID for the chunks.

    Returns
    -------
    list[Chunk]
        List of section chunks.
    """
    # Match RST section headers (title followed by underline of =, -, ~, etc.)
    pattern = r"\n([^\n]+)\n([=\-~`\'\"^_*+#]+)\n"

    # Find all section positions
    matches = list(re.finditer(pattern, text))

    if not matches:
        # No sections found, return whole text as one chunk
        return [Chunk(content=text.strip(), doc_id=doc_id, chunk_index=0)] if text.strip() else []

    chunks = []

    # Handle content before first section
    first_pos = matches[0].start()
    if first_pos > 0:
        pre_content = text[:first_pos].strip()
        if pre_content:
            chunks.append(Chunk(content=pre_content, doc_id=doc_id, chunk_index=0))

    # Extract each section
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        section_content = text[start:end].strip()
        if section_content:
            chunks.append(Chunk(content=section_content, doc_id=doc_id, chunk_index=len(chunks)))

    return chunks

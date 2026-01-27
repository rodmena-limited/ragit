#
# Copyright RODMENA LIMITED 2025
# SPDX-License-Identifier: Apache-2.0
#
"""
Document loading and chunking utilities.

Provides simple functions to load documents from files and chunk text.

Includes ai4rag-inspired patterns:
- Auto-generated document IDs via SHA256 hash
- Sequence numbering for chunk ordering
- Deduplication via content hashing
"""

import hashlib
import re
from pathlib import Path

from ragit.core.experiment.experiment import Chunk, Document


def generate_document_id(content: str) -> str:
    """
    Generate a unique document ID from content using SHA256 hash.

    Pattern from ai4rag langchain_chunker.py.

    Parameters
    ----------
    content : str
        Document content to hash.

    Returns
    -------
    str
        16-character hex string (first 64 bits of SHA256).

    Examples
    --------
    >>> doc_id = generate_document_id("Hello, world!")
    >>> len(doc_id)
    16
    """
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def deduplicate_documents(documents: list[Document]) -> list[Document]:
    """
    Remove duplicate documents based on content hash.

    Pattern from ai4rag chroma.py.

    Parameters
    ----------
    documents : list[Document]
        Documents to deduplicate.

    Returns
    -------
    list[Document]
        Unique documents (first occurrence kept).

    Examples
    --------
    >>> unique_docs = deduplicate_documents(docs)
    >>> print(f"Removed {len(docs) - len(unique_docs)} duplicates")
    """
    seen_hashes: set[str] = set()
    unique_docs: list[Document] = []

    for doc in documents:
        content_hash = generate_document_id(doc.content)
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_docs.append(doc)

    return unique_docs


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


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    doc_id: str | None = None,
    include_metadata: bool = True,
) -> list[Chunk]:
    """
    Split text into overlapping chunks with rich metadata.

    Includes ai4rag-inspired metadata:
    - document_id: SHA256 hash for deduplication and window search
    - sequence_number: Order within the document
    - chunk_start/chunk_end: Character positions in original text

    Parameters
    ----------
    text : str
        Text to chunk.
    chunk_size : int
        Maximum characters per chunk (default: 512).
    chunk_overlap : int
        Overlap between chunks (default: 50).
    doc_id : str, optional
        Document ID for the chunks. If None, generates from content hash.
    include_metadata : bool
        Include rich metadata in chunks (default: True).

    Returns
    -------
    list[Chunk]
        List of text chunks with metadata.

    Examples
    --------
    >>> chunks = chunk_text("Long document...", chunk_size=256)
    >>> print(chunks[0].metadata)
    {'document_id': 'a1b2c3...', 'sequence_number': 0, 'chunk_start': 0, 'chunk_end': 256}
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")

    # Generate document ID if not provided
    effective_doc_id = doc_id or generate_document_id(text)

    chunks = []
    start = 0
    sequence_number = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_content = text[start:end].strip()

        if chunk_content:
            metadata = {}
            if include_metadata:
                metadata = {
                    "document_id": effective_doc_id,
                    "sequence_number": sequence_number,
                    "chunk_start": start,
                    "chunk_end": end,
                }

            chunks.append(
                Chunk(
                    content=chunk_content,
                    doc_id=effective_doc_id,
                    chunk_index=sequence_number,
                    metadata=metadata,
                )
            )
            sequence_number += 1

        start = end - chunk_overlap
        if start >= len(text) - chunk_overlap:
            break

    return chunks


def chunk_document(
    doc: Document,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    include_metadata: bool = True,
) -> list[Chunk]:
    """
    Split a Document into overlapping chunks with rich metadata.

    Parameters
    ----------
    doc : Document
        Document to chunk.
    chunk_size : int
        Maximum characters per chunk.
    chunk_overlap : int
        Overlap between chunks.
    include_metadata : bool
        Include rich metadata in chunks (default: True).

    Returns
    -------
    list[Chunk]
        List of chunks from the document with metadata.
    """
    chunks = chunk_text(doc.content, chunk_size, chunk_overlap, doc.id, include_metadata)

    # Merge document metadata into chunk metadata
    if doc.metadata and include_metadata:
        for chunk in chunks:
            chunk.metadata = {**doc.metadata, **chunk.metadata}

    return chunks


def chunk_by_separator(
    text: str,
    separator: str = "\n\n",
    doc_id: str | None = None,
    include_metadata: bool = True,
) -> list[Chunk]:
    """
    Split text by a separator (e.g., paragraphs, sections).

    Parameters
    ----------
    text : str
        Text to split.
    separator : str
        Separator string (default: double newline for paragraphs).
    doc_id : str, optional
        Document ID for the chunks. If None, generates from content hash.
    include_metadata : bool
        Include rich metadata in chunks (default: True).

    Returns
    -------
    list[Chunk]
        List of chunks with metadata.

    Examples
    --------
    >>> chunks = chunk_by_separator(text, separator="\\n---\\n")
    """
    effective_doc_id = doc_id or generate_document_id(text)
    parts = text.split(separator)
    chunks = []
    current_pos = 0

    for _idx, part in enumerate(parts):
        content = part.strip()
        if content:
            metadata = {}
            if include_metadata:
                # Find actual position in original text
                part_start = text.find(part, current_pos)
                part_end = part_start + len(part) if part_start >= 0 else current_pos + len(part)
                metadata = {
                    "document_id": effective_doc_id,
                    "sequence_number": len(chunks),
                    "chunk_start": part_start if part_start >= 0 else current_pos,
                    "chunk_end": part_end,
                }
                current_pos = part_end

            chunks.append(
                Chunk(
                    content=content,
                    doc_id=effective_doc_id,
                    chunk_index=len(chunks),
                    metadata=metadata,
                )
            )

    return chunks


def chunk_rst_sections(
    text: str,
    doc_id: str | None = None,
    include_metadata: bool = True,
) -> list[Chunk]:
    """
    Split RST document by section headers with rich metadata.

    Parameters
    ----------
    text : str
        RST document text.
    doc_id : str, optional
        Document ID for the chunks. If None, generates from content hash.
    include_metadata : bool
        Include rich metadata in chunks (default: True).

    Returns
    -------
    list[Chunk]
        List of section chunks with metadata.
    """
    effective_doc_id = doc_id or generate_document_id(text)

    # Match RST section headers (title followed by underline of =, -, ~, etc.)
    pattern = r"\n([^\n]+)\n([=\-~`\'\"^_*+#]+)\n"

    # Find all section positions
    matches = list(re.finditer(pattern, text))

    if not matches:
        # No sections found, return whole text as one chunk
        if text.strip():
            metadata = {}
            if include_metadata:
                metadata = {
                    "document_id": effective_doc_id,
                    "sequence_number": 0,
                    "chunk_start": 0,
                    "chunk_end": len(text),
                }
            return [Chunk(content=text.strip(), doc_id=effective_doc_id, chunk_index=0, metadata=metadata)]
        return []

    chunks = []

    # Handle content before first section
    first_pos = matches[0].start()
    if first_pos > 0:
        pre_content = text[:first_pos].strip()
        if pre_content:
            metadata = {}
            if include_metadata:
                metadata = {
                    "document_id": effective_doc_id,
                    "sequence_number": 0,
                    "chunk_start": 0,
                    "chunk_end": first_pos,
                }
            chunks.append(Chunk(content=pre_content, doc_id=effective_doc_id, chunk_index=0, metadata=metadata))

    # Extract each section
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        section_content = text[start:end].strip()
        if section_content:
            metadata = {}
            if include_metadata:
                metadata = {
                    "document_id": effective_doc_id,
                    "sequence_number": len(chunks),
                    "chunk_start": start,
                    "chunk_end": end,
                }
            chunks.append(
                Chunk(
                    content=section_content,
                    doc_id=effective_doc_id,
                    chunk_index=len(chunks),
                    metadata=metadata,
                )
            )

    return chunks

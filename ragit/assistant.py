#
# Copyright RODMENA LIMITED 2025
# SPDX-License-Identifier: Apache-2.0
#
"""
High-level RAG Assistant for document Q&A and code generation.

Provides a simple interface for RAG-based tasks.

Thread Safety:
    This class uses lock-free atomic operations for thread safety.
    The IndexState is immutable, and all mutations create a new state
    that is atomically swapped. Python's GIL ensures reference assignment
    is atomic, making concurrent reads and writes safe.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ragit.core.experiment.experiment import Chunk, Document
from ragit.exceptions import IndexingError
from ragit.loaders import chunk_document, chunk_rst_sections, load_directory, load_text
from ragit.logging import log_operation, logger
from ragit.providers.base import BaseEmbeddingProvider, BaseLLMProvider
from ragit.providers.function_adapter import FunctionProvider

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class IndexState:
    """Immutable snapshot of index state for lock-free thread safety.

    This class holds all mutable index data in a single immutable structure.
    Updates create a new IndexState instance, and the reference swap is
    atomic under Python's GIL, ensuring thread-safe reads and writes.

    Attributes:
        chunks: Tuple of indexed chunks (immutable).
        embedding_matrix: Pre-normalized numpy array of embeddings, or None if empty.
    """

    chunks: tuple[Chunk, ...]
    embedding_matrix: NDArray[np.float64] | None


class RAGAssistant:
    """
    High-level RAG assistant for document Q&A and generation.

    Handles document indexing, retrieval, and LLM generation in one simple API.

    Parameters
    ----------
    documents : list[Document] or str or Path
        Documents to index. Can be:
        - List of Document objects
        - Path to a single file
        - Path to a directory (will load all .txt, .md, .rst files)
    embed_fn : Callable[[str], list[float]], optional
        Function that takes text and returns an embedding vector.
        If provided, creates a FunctionProvider internally.
    generate_fn : Callable, optional
        Function for text generation. Supports (prompt) or (prompt, system_prompt).
        If provided without embed_fn, must also provide embed_fn.
    provider : BaseEmbeddingProvider, optional
        Provider for embeddings (and optionally LLM). If embed_fn is provided,
        this is ignored for embeddings.
    embedding_model : str, optional
        Embedding model name (used with provider).
    llm_model : str, optional
        LLM model name (used with provider).
    chunk_size : int, optional
        Chunk size for splitting documents (default: 512).
    chunk_overlap : int, optional
        Overlap between chunks (default: 50).

    Raises
    ------
    ValueError
        If neither embed_fn nor provider is provided.

    Thread Safety
    -------------
    This class uses lock-free atomic operations for thread safety.
    Multiple threads can safely call retrieve() while another thread
    calls add_documents(). The IndexState is immutable, and reference
    swaps are atomic under Python's GIL.

    Examples
    --------
    >>> # With custom embedding function (retrieval-only)
    >>> assistant = RAGAssistant(docs, embed_fn=my_embed)
    >>> results = assistant.retrieve("query")
    >>>
    >>> # With custom embedding and LLM functions (full RAG)
    >>> assistant = RAGAssistant(docs, embed_fn=my_embed, generate_fn=my_llm)
    >>> answer = assistant.ask("What is X?")
    >>>
    >>> # With Ollama provider (supports nomic-embed-text)
    >>> from ragit.providers import OllamaProvider
    >>> assistant = RAGAssistant(docs, provider=OllamaProvider())
    >>>
    >>> # Save and load index for persistence
    >>> assistant.save_index("/path/to/index")
    >>> loaded = RAGAssistant.load_index("/path/to/index", provider=OllamaProvider())
    """

    def __init__(
        self,
        documents: list[Document] | str | Path,
        embed_fn: Callable[[str], list[float]] | None = None,
        generate_fn: Callable[..., str] | None = None,
        provider: BaseEmbeddingProvider | BaseLLMProvider | None = None,
        embedding_model: str | None = None,
        llm_model: str | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        # Resolve provider from embed_fn/generate_fn or explicit provider
        self._embedding_provider: BaseEmbeddingProvider
        self._llm_provider: BaseLLMProvider | None = None

        if embed_fn is not None:
            # Create FunctionProvider from provided functions
            function_provider = FunctionProvider(
                embed_fn=embed_fn,
                generate_fn=generate_fn,
            )
            self._embedding_provider = function_provider
            if generate_fn is not None:
                self._llm_provider = function_provider
            elif provider is not None and isinstance(provider, BaseLLMProvider):
                # Use explicit provider for LLM if function_provider doesn't have LLM
                self._llm_provider = provider
        elif provider is not None:
            # Use explicit provider
            if not isinstance(provider, BaseEmbeddingProvider):
                raise ValueError(
                    "Provider must implement BaseEmbeddingProvider for embeddings. Alternatively, provide embed_fn."
                )
            self._embedding_provider = provider
            if isinstance(provider, BaseLLMProvider):
                self._llm_provider = provider
        else:
            raise ValueError(
                "Must provide embed_fn or provider for embeddings. "
                "Examples:\n"
                "  RAGAssistant(docs, embed_fn=my_embed_function)\n"
                "  RAGAssistant(docs, provider=OllamaProvider())"
            )

        self.embedding_model = embedding_model or "default"
        self.llm_model = llm_model or "default"
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Load documents if path provided
        self.documents = self._load_documents(documents)

        # Thread-safe index state (immutable, atomic reference swap)
        self._state: IndexState = IndexState(chunks=(), embedding_matrix=None)
        self._build_index()

    def _load_documents(self, documents: list[Document] | str | Path) -> list[Document]:
        """Load documents from various sources."""
        if isinstance(documents, list):
            return documents

        path = Path(documents)

        if path.is_file():
            return [load_text(path)]

        if path.is_dir():
            docs: list[Document] = []
            for pattern in (
                "*.txt",
                "*.md",
                "*.rst",
                "*.py",
                "*.js",
                "*.ts",
                "*.go",
                "*.java",
                "*.c",
                "*.cpp",
                "*.h",
                "*.hpp",
            ):
                docs.extend(load_directory(path, pattern))
            return docs

        raise ValueError(f"Invalid documents source: {documents}")

    def _build_index(self) -> None:
        """Build vector index from documents using batch embedding.

        Raises:
            IndexingError: If embedding count doesn't match chunk count.
        """
        all_chunks: list[Chunk] = []

        for doc in self.documents:
            # Use RST section chunking for .rst files, otherwise regular chunking
            if doc.metadata.get("filename", "").endswith(".rst"):
                chunks = chunk_rst_sections(doc.content, doc.id)
            else:
                chunks = chunk_document(doc, self.chunk_size, self.chunk_overlap)
            all_chunks.extend(chunks)

        if not all_chunks:
            logger.warning("No chunks produced from documents - index will be empty")
            self._state = IndexState(chunks=(), embedding_matrix=None)
            return

        # Batch embed all chunks at once (single API call)
        texts = [chunk.content for chunk in all_chunks]
        responses = self._embedding_provider.embed_batch(texts, self.embedding_model)

        # CRITICAL: Validate embedding count matches chunk count
        if len(responses) != len(all_chunks):
            raise IndexingError(
                f"Embedding count mismatch: expected {len(all_chunks)} embeddings, "
                f"got {len(responses)}. Index may be corrupted."
            )

        # Build embedding matrix directly (skip storing in chunks to avoid duplication)
        embedding_matrix = np.array([response.embedding for response in responses], dtype=np.float64)

        # Additional validation: matrix shape
        if embedding_matrix.shape[0] != len(all_chunks):
            raise IndexingError(
                f"Matrix row count {embedding_matrix.shape[0]} doesn't match chunk count {len(all_chunks)}"
            )

        # Pre-normalize for fast cosine similarity (normalize once, use many times)
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero

        # Atomic state update (thread-safe under GIL)
        self._state = IndexState(
            chunks=tuple(all_chunks),
            embedding_matrix=embedding_matrix / norms,
        )

    def add_documents(self, documents: list[Document] | str | Path) -> int:
        """Add documents to the existing index incrementally.

        This method is thread-safe. It creates a new IndexState and atomically
        swaps the reference, ensuring readers always see a consistent state.

        Args:
            documents: Documents to add.

        Returns:
            Number of chunks added.

        Raises:
            IndexingError: If embedding count doesn't match chunk count.
        """
        new_docs = self._load_documents(documents)
        if not new_docs:
            return 0

        self.documents.extend(new_docs)

        # Chunk new docs
        new_chunks: list[Chunk] = []
        for doc in new_docs:
            if doc.metadata.get("filename", "").endswith(".rst"):
                chunks = chunk_rst_sections(doc.content, doc.id)
            else:
                chunks = chunk_document(doc, self.chunk_size, self.chunk_overlap)
            new_chunks.extend(chunks)

        if not new_chunks:
            return 0

        # Embed new chunks
        texts = [chunk.content for chunk in new_chunks]
        responses = self._embedding_provider.embed_batch(texts, self.embedding_model)

        # Validate embedding count
        if len(responses) != len(new_chunks):
            raise IndexingError(
                f"Embedding count mismatch: expected {len(new_chunks)} embeddings, "
                f"got {len(responses)}. Index update aborted."
            )

        new_matrix = np.array([response.embedding for response in responses], dtype=np.float64)

        # Normalize
        norms = np.linalg.norm(new_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        new_matrix_norm = new_matrix / norms

        # Read current state (atomic read)
        current_state = self._state

        # Build new state
        combined_chunks = current_state.chunks + tuple(new_chunks)
        if current_state.embedding_matrix is None:
            combined_matrix = new_matrix_norm
        else:
            combined_matrix = np.vstack((current_state.embedding_matrix, new_matrix_norm))

        # Atomic state swap (thread-safe under GIL)
        self._state = IndexState(chunks=combined_chunks, embedding_matrix=combined_matrix)

        return len(new_chunks)

    def remove_documents(self, source_path_pattern: str) -> int:
        """Remove documents matching a source path pattern.

        This method is thread-safe. It creates a new IndexState and atomically
        swaps the reference.

        Args:
            source_path_pattern: Glob pattern to match 'source' metadata.

        Returns:
            Number of chunks removed.
        """
        import fnmatch

        # Read current state (atomic read)
        current_state = self._state

        if not current_state.chunks:
            return 0

        indices_to_keep = []
        kept_chunks = []
        removed_count = 0

        for i, chunk in enumerate(current_state.chunks):
            source = chunk.metadata.get("source", "")
            if not source or not fnmatch.fnmatch(source, source_path_pattern):
                indices_to_keep.append(i)
                kept_chunks.append(chunk)
            else:
                removed_count += 1

        if removed_count == 0:
            return 0

        # Build new embedding matrix
        if current_state.embedding_matrix is not None:
            new_matrix = None if not kept_chunks else current_state.embedding_matrix[indices_to_keep]
        else:
            new_matrix = None

        # Atomic state swap (thread-safe under GIL)
        self._state = IndexState(chunks=tuple(kept_chunks), embedding_matrix=new_matrix)

        # Also remove from self.documents
        self.documents = [
            doc for doc in self.documents if not fnmatch.fnmatch(doc.metadata.get("source", ""), source_path_pattern)
        ]

        return removed_count

    def update_documents(self, documents: list[Document] | str | Path) -> int:
        """Update existing documents (remove old, add new).

        Uses document source path to identify what to remove.

        Args:
            documents: New versions of documents.

        Returns:
            Number of chunks added.
        """
        new_docs = self._load_documents(documents)
        if not new_docs:
            return 0

        # Identify sources to remove
        sources_to_remove = set()
        for doc in new_docs:
            source = doc.metadata.get("source")
            if source:
                sources_to_remove.add(source)

        # Remove old versions
        for source in sources_to_remove:
            self.remove_documents(source)

        # Add new versions
        return self.add_documents(new_docs)

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        """
        Retrieve relevant chunks for a query.

        Uses vectorized cosine similarity for fast search over all chunks.
        This method is thread-safe - it reads a consistent snapshot of the index.

        Parameters
        ----------
        query : str
            Search query.
        top_k : int
            Number of chunks to return (default: 3).

        Returns
        -------
        list[tuple[Chunk, float]]
            List of (chunk, similarity_score) tuples, sorted by relevance.

        Examples
        --------
        >>> results = assistant.retrieve("how to create a route")
        >>> for chunk, score in results:
        ...     print(f"{score:.2f}: {chunk.content[:100]}...")
        """
        # Atomic state read - get consistent snapshot
        state = self._state

        if not state.chunks or state.embedding_matrix is None:
            return []

        # Get query embedding and normalize
        query_response = self._embedding_provider.embed(query, self.embedding_model)
        query_vec = np.array(query_response.embedding, dtype=np.float64)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []
        query_normalized = query_vec / query_norm

        # Fast cosine similarity: matrix is pre-normalized, just dot product
        similarities = state.embedding_matrix @ query_normalized

        # Get top_k indices using argpartition (faster than full sort for large arrays)
        if len(similarities) <= top_k:
            top_indices = np.argsort(similarities)[::-1]
        else:
            # Partial sort - only find top_k elements
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            # Sort the top_k by score
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        return [(state.chunks[i], float(similarities[i])) for i in top_indices]

    def retrieve_with_context(
        self,
        query: str,
        top_k: int = 3,
        window_size: int = 1,
        min_score: float = 0.0,
    ) -> list[tuple[Chunk, float]]:
        """
        Retrieve chunks with adjacent context expansion (window search).

        For each retrieved chunk, also includes adjacent chunks from the
        same document to provide more context. This is useful when relevant
        information spans multiple chunks.

        Pattern inspired by ai4rag window_search.

        Parameters
        ----------
        query : str
            Search query.
        top_k : int
            Number of initial chunks to retrieve (default: 3).
        window_size : int
            Number of adjacent chunks to include on each side (default: 1).
            Set to 0 to disable window expansion.
        min_score : float
            Minimum similarity score threshold (default: 0.0).

        Returns
        -------
        list[tuple[Chunk, float]]
            List of (chunk, similarity_score) tuples, sorted by relevance.
            Adjacent chunks have slightly lower scores.

        Examples
        --------
        >>> # Get chunks with 1 adjacent chunk on each side
        >>> results = assistant.retrieve_with_context("query", window_size=1)
        >>> for chunk, score in results:
        ...     print(f"{score:.2f}: {chunk.content[:50]}...")
        """
        # Get consistent state snapshot
        state = self._state

        with log_operation("retrieve_with_context", query_len=len(query), top_k=top_k, window_size=window_size) as ctx:
            # Get initial results (more than top_k to account for filtering)
            results = self.retrieve(query, top_k * 2)

            # Apply minimum score threshold
            if min_score > 0:
                results = [(chunk, score) for chunk, score in results if score >= min_score]

            if window_size == 0 or not results:
                ctx["expanded_chunks"] = len(results)
                return results[:top_k]

            # Build chunk index for fast lookup
            chunk_to_idx = {id(chunk): i for i, chunk in enumerate(state.chunks)}

            expanded_results: list[tuple[Chunk, float]] = []
            seen_indices: set[int] = set()

            for chunk, score in results[:top_k]:
                chunk_idx = chunk_to_idx.get(id(chunk))
                if chunk_idx is None:
                    expanded_results.append((chunk, score))
                    continue

                # Get window of adjacent chunks from same document
                start_idx = max(0, chunk_idx - window_size)
                end_idx = min(len(state.chunks), chunk_idx + window_size + 1)

                for idx in range(start_idx, end_idx):
                    if idx in seen_indices:
                        continue

                    adjacent_chunk = state.chunks[idx]
                    # Only include adjacent chunks from same document
                    if adjacent_chunk.doc_id == chunk.doc_id:
                        seen_indices.add(idx)
                        # Original chunk keeps full score, adjacent get 80%
                        adj_score = score if idx == chunk_idx else score * 0.8
                        expanded_results.append((adjacent_chunk, adj_score))

            # Sort by score (highest first)
            expanded_results.sort(key=lambda x: (-x[1], state.chunks.index(x[0]) if x[0] in state.chunks else 0))
            ctx["expanded_chunks"] = len(expanded_results)

            return expanded_results

    def get_context_with_window(
        self,
        query: str,
        top_k: int = 3,
        window_size: int = 1,
        min_score: float = 0.0,
    ) -> str:
        """
        Get formatted context with adjacent chunk expansion.

        Merges overlapping text from adjacent chunks intelligently.

        Parameters
        ----------
        query : str
            Search query.
        top_k : int
            Number of initial chunks to retrieve.
        window_size : int
            Number of adjacent chunks on each side.
        min_score : float
            Minimum similarity score threshold.

        Returns
        -------
        str
            Formatted context string with merged chunks.
        """
        # Get consistent state snapshot
        state = self._state

        results = self.retrieve_with_context(query, top_k, window_size, min_score)

        if not results:
            return ""

        # Group chunks by document to merge properly
        doc_chunks: dict[str, list[tuple[Chunk, float]]] = {}
        for chunk, score in results:
            doc_id = chunk.doc_id or "unknown"
            if doc_id not in doc_chunks:
                doc_chunks[doc_id] = []
            doc_chunks[doc_id].append((chunk, score))

        merged_sections: list[str] = []

        for _doc_id, chunks in doc_chunks.items():
            # Sort chunks by their position in the original list
            chunks.sort(key=lambda x: state.chunks.index(x[0]) if x[0] in state.chunks else 0)

            # Merge overlapping text
            merged_content: list[str] = []
            for chunk, _ in chunks:
                if merged_content:
                    # Check for overlap with previous chunk
                    prev_content = merged_content[-1]
                    non_overlapping = self._get_non_overlapping_text(prev_content, chunk.content)
                    if non_overlapping != chunk.content:
                        # Found overlap, extend previous chunk
                        merged_content[-1] = prev_content + non_overlapping
                    else:
                        # No overlap, add as new section
                        merged_content.append(chunk.content)
                else:
                    merged_content.append(chunk.content)

            merged_sections.append("\n".join(merged_content))

        return "\n\n---\n\n".join(merged_sections)

    def _get_non_overlapping_text(self, str1: str, str2: str) -> str:
        """
        Find non-overlapping portion of str2 when appending after str1.

        Detects overlap where the end of str1 matches the beginning of str2,
        and returns only the non-overlapping portion of str2.

        Pattern from ai4rag vector_store/utils.py.

        Parameters
        ----------
        str1 : str
            First string (previous content).
        str2 : str
            Second string (content to potentially append).

        Returns
        -------
        str
            Non-overlapping portion of str2, or full str2 if no overlap.
        """
        # Limit overlap search to avoid O(n^2) for large strings
        max_overlap = min(len(str1), len(str2), 200)

        for i in range(max_overlap, 0, -1):
            if str1[-i:] == str2[:i]:
                return str2[i:]

        return str2

    def get_context(self, query: str, top_k: int = 3) -> str:
        """
        Get formatted context string from retrieved chunks.

        Parameters
        ----------
        query : str
            Search query.
        top_k : int
            Number of chunks to include.

        Returns
        -------
        str
            Formatted context string.
        """
        results = self.retrieve(query, top_k)
        return "\n\n---\n\n".join(chunk.content for chunk, _ in results)

    def _ensure_llm(self) -> BaseLLMProvider:
        """Ensure LLM provider is available."""
        if self._llm_provider is None:
            raise NotImplementedError(
                "No LLM configured. Provide generate_fn or a provider with LLM support "
                "to use ask(), generate(), or generate_code() methods."
            )
        return self._llm_provider

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate text using the LLM (without retrieval).

        Parameters
        ----------
        prompt : str
            User prompt.
        system_prompt : str, optional
            System prompt for context.
        temperature : float
            Sampling temperature (default: 0.7).

        Returns
        -------
        str
            Generated text.

        Raises
        ------
        NotImplementedError
            If no LLM is configured.
        """
        llm = self._ensure_llm()
        response = llm.generate(
            prompt=prompt,
            model=self.llm_model,
            system_prompt=system_prompt,
            temperature=temperature,
        )
        return response.text

    def ask(
        self,
        question: str,
        system_prompt: str | None = None,
        top_k: int = 3,
        temperature: float = 0.7,
    ) -> str:
        """
        Ask a question using RAG (retrieve + generate).

        Parameters
        ----------
        question : str
            Question to answer.
        system_prompt : str, optional
            System prompt. Defaults to a helpful assistant prompt.
        top_k : int
            Number of context chunks to retrieve (default: 3).
        temperature : float
            Sampling temperature (default: 0.7).

        Returns
        -------
        str
            Generated answer.

        Raises
        ------
        NotImplementedError
            If no LLM is configured.

        Examples
        --------
        >>> answer = assistant.ask("How do I create a REST API?")
        >>> print(answer)
        """
        # Retrieve context
        context = self.get_context(question, top_k)

        # Default system prompt
        if system_prompt is None:
            system_prompt = """You are a helpful assistant. Answer questions based on the provided context.
If the context doesn't contain enough information, say so. Be concise and accurate."""

        # Build prompt with context
        prompt = f"""Context:
{context}

Question: {question}

Answer:"""

        return self.generate(prompt, system_prompt, temperature)

    def generate_code(
        self,
        request: str,
        language: str = "python",
        top_k: int = 3,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate code based on documentation context.

        Parameters
        ----------
        request : str
            Description of what code to generate.
        language : str
            Programming language (default: "python").
        top_k : int
            Number of context chunks to retrieve.
        temperature : float
            Sampling temperature.

        Returns
        -------
        str
            Generated code (cleaned, without markdown).

        Raises
        ------
        NotImplementedError
            If no LLM is configured.

        Examples
        --------
        >>> code = assistant.generate_code("create a REST API with user endpoints")
        >>> print(code)
        """
        context = self.get_context(request, top_k)

        system_prompt = f"""You are an expert {language} developer. Generate ONLY valid {language} code.

RULES:
1. Output PURE CODE ONLY - no explanations, no markdown code blocks
2. Include necessary imports
3. Write clean, production-ready code
4. Add brief comments for clarity"""

        prompt = f"""Documentation:
{context}

Request: {request}

Generate the {language} code:"""

        response = self.generate(prompt, system_prompt, temperature)

        # Clean up response - remove markdown if present
        code = response
        if f"```{language}" in code:
            code = code.split(f"```{language}")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]

        return code.strip()

    @property
    def num_chunks(self) -> int:
        """Return number of indexed chunks."""
        return len(self._state.chunks)

    @property
    def chunk_count(self) -> int:
        """Number of chunks in index (alias for num_chunks)."""
        return len(self._state.chunks)

    @property
    def is_indexed(self) -> bool:
        """Check if index has any documents."""
        return len(self._state.chunks) > 0

    @property
    def num_documents(self) -> int:
        """Return number of loaded documents."""
        return len(self.documents)

    @property
    def has_llm(self) -> bool:
        """Check if LLM is configured."""
        return self._llm_provider is not None

    def save_index(self, path: str | Path) -> None:
        """Save index to disk for later restoration.

        Saves the index in an efficient format:
        - chunks.json: Chunk metadata and content
        - embeddings.npy: Numpy array of embeddings (binary format)
        - metadata.json: Index configuration

        Args:
            path: Directory path to save index files.

        Example:
            >>> assistant.save_index("/path/to/index")
            >>> # Later...
            >>> loaded = RAGAssistant.load_index("/path/to/index", provider=provider)
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        state = self._state

        # Save chunks as JSON
        chunks_data = [
            {
                "content": chunk.content,
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata,
            }
            for chunk in state.chunks
        ]
        (path / "chunks.json").write_text(json.dumps(chunks_data, indent=2))

        # Save embeddings as numpy binary (efficient for large arrays)
        if state.embedding_matrix is not None:
            np.save(path / "embeddings.npy", state.embedding_matrix)

        # Save metadata for validation and configuration restoration
        metadata = {
            "chunk_count": len(state.chunks),
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "version": "1.0",
        }
        (path / "metadata.json").write_text(json.dumps(metadata, indent=2))

        logger.info(f"Index saved to {path} ({len(state.chunks)} chunks)")

    @classmethod
    def load_index(
        cls,
        path: str | Path,
        provider: BaseEmbeddingProvider | BaseLLMProvider | None = None,
    ) -> "RAGAssistant":
        """Load a previously saved index.

        Args:
            path: Directory path containing saved index files.
            provider: Provider for embeddings/LLM (required for new queries).

        Returns:
            RAGAssistant instance with loaded index.

        Raises:
            IndexingError: If loaded index is corrupted (count mismatch).
            FileNotFoundError: If index files don't exist.

        Example:
            >>> loaded = RAGAssistant.load_index("/path/to/index", provider=OllamaProvider())
            >>> results = loaded.retrieve("query")
        """
        path = Path(path)

        # Load metadata
        metadata = json.loads((path / "metadata.json").read_text())

        # Load chunks
        chunks_data = json.loads((path / "chunks.json").read_text())
        chunks = tuple(
            Chunk(
                content=c["content"],
                doc_id=c.get("doc_id", ""),
                chunk_index=c.get("chunk_index", 0),
                metadata=c.get("metadata", {}),
            )
            for c in chunks_data
        )

        # Load embeddings
        embeddings_path = path / "embeddings.npy"
        embedding_matrix: NDArray[np.float64] | None = None
        if embeddings_path.exists():
            embedding_matrix = np.load(embeddings_path)

        # Validate consistency
        if embedding_matrix is not None and embedding_matrix.shape[0] != len(chunks):
            raise IndexingError(
                f"Loaded index corrupted: {embedding_matrix.shape[0]} embeddings but {len(chunks)} chunks"
            )

        # Create instance without calling __init__ (skip indexing)
        instance = object.__new__(cls)

        # Initialize required attributes
        instance._state = IndexState(chunks=chunks, embedding_matrix=embedding_matrix)
        instance.embedding_model = metadata.get("embedding_model", "default")
        instance.llm_model = metadata.get("llm_model", "default")
        instance.chunk_size = metadata.get("chunk_size", 512)
        instance.chunk_overlap = metadata.get("chunk_overlap", 50)
        instance.documents = []  # Original docs not saved

        # Set up providers
        instance._embedding_provider = provider if isinstance(provider, BaseEmbeddingProvider) else None  # type: ignore
        instance._llm_provider = provider if isinstance(provider, BaseLLMProvider) else None

        logger.info(f"Index loaded from {path} ({len(chunks)} chunks)")
        return instance

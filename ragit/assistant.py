#
# Copyright RODMENA LIMITED 2025
# SPDX-License-Identifier: Apache-2.0
#
"""
High-level RAG Assistant for document Q&A and code generation.

Provides a simple interface for RAG-based tasks.

Note: This class is NOT thread-safe. Do not share instances across threads.
"""

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ragit.core.experiment.experiment import Chunk, Document
from ragit.loaders import chunk_document, chunk_rst_sections, load_directory, load_text
from ragit.providers.base import BaseEmbeddingProvider, BaseLLMProvider
from ragit.providers.function_adapter import FunctionProvider

if TYPE_CHECKING:
    from numpy.typing import NDArray


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

    Note
    ----
    This class is NOT thread-safe. Each thread should have its own instance.

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
    >>> # With explicit provider
    >>> from ragit.providers import OllamaProvider
    >>> assistant = RAGAssistant(docs, provider=OllamaProvider())
    >>>
    >>> # With SentenceTransformers (offline)
    >>> from ragit.providers import SentenceTransformersProvider
    >>> assistant = RAGAssistant(docs, provider=SentenceTransformersProvider())
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
                    "Provider must implement BaseEmbeddingProvider for embeddings. "
                    "Alternatively, provide embed_fn."
                )
            self._embedding_provider = provider
            if isinstance(provider, BaseLLMProvider):
                self._llm_provider = provider
        else:
            raise ValueError(
                "Must provide embed_fn or provider for embeddings. "
                "Examples:\n"
                "  RAGAssistant(docs, embed_fn=my_embed_function)\n"
                "  RAGAssistant(docs, provider=OllamaProvider())\n"
                "  RAGAssistant(docs, provider=SentenceTransformersProvider())"
            )

        self.embedding_model = embedding_model or "default"
        self.llm_model = llm_model or "default"
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Load documents if path provided
        self.documents = self._load_documents(documents)

        # Index chunks - embeddings stored as pre-normalized numpy matrix for fast search
        self._chunks: tuple[Chunk, ...] = ()
        self._embedding_matrix: NDArray[np.float64] | None = None  # Pre-normalized
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
            for pattern in ("*.txt", "*.md", "*.rst", "*.py", "*.js", "*.ts", "*.go", "*.java", "*.c", "*.cpp", "*.h", "*.hpp"):
                docs.extend(load_directory(path, pattern))
            return docs

        raise ValueError(f"Invalid documents source: {documents}")

    def _build_index(self) -> None:
        """Build vector index from documents using batch embedding."""
        all_chunks: list[Chunk] = []

        for doc in self.documents:
            # Use RST section chunking for .rst files, otherwise regular chunking
            if doc.metadata.get("filename", "").endswith(".rst"):
                chunks = chunk_rst_sections(doc.content, doc.id)
            else:
                chunks = chunk_document(doc, self.chunk_size, self.chunk_overlap)
            all_chunks.extend(chunks)

        if not all_chunks:
            self._chunks = ()
            self._embedding_matrix = None
            return

        # Batch embed all chunks at once (single API call)
        texts = [chunk.content for chunk in all_chunks]
        responses = self._embedding_provider.embed_batch(texts, self.embedding_model)

        # Build embedding matrix directly (skip storing in chunks to avoid duplication)
        embedding_matrix = np.array([response.embedding for response in responses], dtype=np.float64)

        # Pre-normalize for fast cosine similarity (normalize once, use many times)
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero

        # Store as immutable tuple and pre-normalized numpy matrix
        self._chunks = tuple(all_chunks)
        self._embedding_matrix = embedding_matrix / norms

    def add_documents(self, documents: list[Document] | str | Path) -> int:
        """Add documents to the existing index incrementally.

        Args:
            documents: Documents to add.

        Returns:
            Number of chunks added.
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
        
        new_matrix = np.array([response.embedding for response in responses], dtype=np.float64)
        
        # Normalize
        norms = np.linalg.norm(new_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        new_matrix_norm = new_matrix / norms
        
        # Update state
        current_chunks = list(self._chunks)
        current_chunks.extend(new_chunks)
        self._chunks = tuple(current_chunks)
        
        if self._embedding_matrix is None:
            self._embedding_matrix = new_matrix_norm
        else:
            self._embedding_matrix = np.vstack((self._embedding_matrix, new_matrix_norm))
            
        return len(new_chunks)

    def remove_documents(self, source_path_pattern: str) -> int:
        """Remove documents matching a source path pattern.

        Args:
            source_path_pattern: Glob pattern to match 'source' metadata.

        Returns:
            Number of chunks removed.
        """
        import fnmatch
        
        if not self._chunks:
            return 0
            
        indices_to_keep = []
        kept_chunks = []
        removed_count = 0
        
        for i, chunk in enumerate(self._chunks):
            source = chunk.metadata.get("source", "")
            if not source or not fnmatch.fnmatch(source, source_path_pattern):
                indices_to_keep.append(i)
                kept_chunks.append(chunk)
            else:
                removed_count += 1
                
        if removed_count == 0:
            return 0
            
        self._chunks = tuple(kept_chunks)
        
        if self._embedding_matrix is not None:
            if not kept_chunks:
                self._embedding_matrix = None
            else:
                self._embedding_matrix = self._embedding_matrix[indices_to_keep]
            
        # Also remove from self.documents
        self.documents = [
            doc for doc in self.documents 
            if not fnmatch.fnmatch(doc.metadata.get("source", ""), source_path_pattern)
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
        if not self._chunks or self._embedding_matrix is None:
            return []

        # Get query embedding and normalize
        query_response = self._embedding_provider.embed(query, self.embedding_model)
        query_vec = np.array(query_response.embedding, dtype=np.float64)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []
        query_normalized = query_vec / query_norm

        # Fast cosine similarity: matrix is pre-normalized, just dot product
        similarities = self._embedding_matrix @ query_normalized

        # Get top_k indices using argpartition (faster than full sort for large arrays)
        if len(similarities) <= top_k:
            top_indices = np.argsort(similarities)[::-1]
        else:
            # Partial sort - only find top_k elements
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            # Sort the top_k by score
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        return [(self._chunks[i], float(similarities[i])) for i in top_indices]

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
        return len(self._chunks)

    @property
    def num_documents(self) -> int:
        """Return number of loaded documents."""
        return len(self.documents)

    @property
    def has_llm(self) -> bool:
        """Check if LLM is configured."""
        return self._llm_provider is not None

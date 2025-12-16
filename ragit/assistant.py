#
# Copyright RODMENA LIMITED 2025
# SPDX-License-Identifier: Apache-2.0
#
"""
High-level RAG Assistant for document Q&A and code generation.

Provides a simple interface for RAG-based tasks.
"""

from pathlib import Path

import numpy as np

from ragit.config import config
from ragit.core.experiment.experiment import Chunk, Document
from ragit.loaders import chunk_document, chunk_rst_sections, load_directory, load_text
from ragit.providers import OllamaProvider


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
    provider : OllamaProvider, optional
        LLM/embedding provider. Defaults to OllamaProvider().
    embedding_model : str, optional
        Embedding model name. Defaults to config.DEFAULT_EMBEDDING_MODEL.
    llm_model : str, optional
        LLM model name. Defaults to config.DEFAULT_LLM_MODEL.
    chunk_size : int, optional
        Chunk size for splitting documents (default: 512).
    chunk_overlap : int, optional
        Overlap between chunks (default: 50).

    Examples
    --------
    >>> # From documents
    >>> assistant = RAGAssistant([Document(id="doc1", content="...")])
    >>> answer = assistant.ask("What is X?")

    >>> # From file
    >>> assistant = RAGAssistant("docs/tutorial.rst")
    >>> answer = assistant.ask("How do I do Y?")

    >>> # From directory
    >>> assistant = RAGAssistant("docs/")
    >>> answer = assistant.ask("Explain Z")
    """

    def __init__(
        self,
        documents: list[Document] | str | Path,
        provider: OllamaProvider | None = None,
        embedding_model: str | None = None,
        llm_model: str | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        self.provider = provider or OllamaProvider()
        self.embedding_model = embedding_model or config.DEFAULT_EMBEDDING_MODEL
        self.llm_model = llm_model or config.DEFAULT_LLM_MODEL
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Load documents if path provided
        self.documents = self._load_documents(documents)

        # Index chunks
        self._chunks: list[Chunk] = []
        self._embeddings: list[list[float]] = []
        self._build_index()

    def _load_documents(self, documents: list[Document] | str | Path) -> list[Document]:
        """Load documents from various sources."""
        if isinstance(documents, list):
            return documents

        path = Path(documents)

        if path.is_file():
            return [load_text(path)]

        if path.is_dir():
            docs = []
            for pattern in ["*.txt", "*.md", "*.rst"]:
                docs.extend(load_directory(path, pattern))
            return docs

        raise ValueError(f"Invalid documents source: {documents}")

    def _build_index(self) -> None:
        """Build vector index from documents."""
        self._chunks = []
        self._embeddings = []

        for doc in self.documents:
            # Use RST section chunking for .rst files, otherwise regular chunking
            if doc.metadata.get("filename", "").endswith(".rst"):
                chunks = chunk_rst_sections(doc.content, doc.id)
            else:
                chunks = chunk_document(doc, self.chunk_size, self.chunk_overlap)

            for chunk in chunks:
                response = self.provider.embed(chunk.content, self.embedding_model)
                chunk.embedding = response.embedding
                self._chunks.append(chunk)
                self._embeddings.append(response.embedding)

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between vectors."""
        a_arr, b_arr = np.array(a), np.array(b)
        return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        """
        Retrieve relevant chunks for a query.

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
        if not self._chunks:
            return []

        query_response = self.provider.embed(query, self.embedding_model)
        query_emb = query_response.embedding

        scores = [
            (chunk, self._cosine_similarity(query_emb, emb))
            for chunk, emb in zip(self._chunks, self._embeddings, strict=True)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]

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
        return "\n\n---\n\n".join([chunk.content for chunk, _ in results])

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
        """
        response = self.provider.generate(
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

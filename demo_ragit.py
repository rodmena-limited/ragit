#!/usr/bin/env python3
"""
Ragit Demo: Automatic RAG Optimization Engine
================================================

This demo showcases the core concept of Ragit - an automatic hyperparameter
optimization system for RAG (Retrieval-Augmented Generation) pipelines.

Ragit solves the problem of finding optimal RAG configurations by automatically
testing different combinations of:
- Chunking strategies (chunk size, overlap)
- Embedding models
- Retrieval methods (simple vs window, number of chunks)
- Generation parameters

This demo uses Ollama for LLM inference and embeddings.
"""

import json
import time
import requests
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from itertools import product


# =============================================================================
# Configuration
# =============================================================================

OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "qwen3-vl:235b-instruct-cloud"
EMBEDDING_MODEL = "nomic-embed-text:latest"


# =============================================================================
# Sample Documents (Knowledge Base)
# =============================================================================

SAMPLE_DOCUMENTS = [
    {
        "id": "doc1",
        "title": "Introduction to Machine Learning",
        "content": """Machine learning is a subset of artificial intelligence (AI) that provides systems
the ability to automatically learn and improve from experience without being explicitly programmed.
Machine learning focuses on the development of computer programs that can access data and use it
to learn for themselves. The process of learning begins with observations or data, such as examples,
direct experience, or instruction, in order to look for patterns in data and make better decisions
in the future based on the examples that we provide. The primary aim is to allow the computers to
learn automatically without human intervention or assistance and adjust actions accordingly.

There are three main types of machine learning:
1. Supervised Learning: The algorithm learns from labeled training data, and makes predictions.
2. Unsupervised Learning: The algorithm finds patterns in unlabeled data.
3. Reinforcement Learning: The algorithm learns by interacting with an environment and receiving rewards."""
    },
    {
        "id": "doc2",
        "title": "Deep Learning Fundamentals",
        "content": """Deep learning is part of a broader family of machine learning methods based on
artificial neural networks with representation learning. Learning can be supervised, semi-supervised
or unsupervised. Deep learning architectures such as deep neural networks, recurrent neural networks,
convolutional neural networks and transformers have been applied to fields including computer vision,
speech recognition, natural language processing, machine translation, bioinformatics, drug design,
medical image analysis, climate science and board game programs.

Neural networks are composed of layers of interconnected nodes or neurons. Each connection has a weight
that is adjusted during training. The "deep" in deep learning refers to the number of layers through
which the data is transformed. Deep learning has revolutionized AI by enabling end-to-end learning
directly from raw data without manual feature engineering."""
    },
    {
        "id": "doc3",
        "title": "Natural Language Processing",
        "content": """Natural Language Processing (NLP) is a subfield of linguistics, computer science,
and artificial intelligence concerned with the interactions between computers and human language,
in particular how to program computers to process and analyze large amounts of natural language data.

Key NLP tasks include:
- Text Classification: Categorizing text into predefined classes
- Named Entity Recognition: Identifying entities like names, dates, locations
- Sentiment Analysis: Determining the emotional tone of text
- Machine Translation: Translating text between languages
- Question Answering: Answering questions based on context
- Text Summarization: Creating concise summaries of longer texts

Modern NLP heavily relies on transformer architectures like BERT, GPT, and T5, which use
self-attention mechanisms to understand context and relationships between words."""
    },
    {
        "id": "doc4",
        "title": "RAG Systems Overview",
        "content": """Retrieval-Augmented Generation (RAG) is a technique that combines the power of
large language models with external knowledge retrieval to produce more accurate and up-to-date responses.

The RAG pipeline typically consists of:
1. Indexing Phase: Documents are chunked, embedded, and stored in a vector database
2. Retrieval Phase: Given a query, relevant document chunks are retrieved using similarity search
3. Generation Phase: The LLM generates a response using the retrieved context

RAG addresses key limitations of LLMs:
- Knowledge cutoff: RAG can access current information
- Hallucination: Grounding in retrieved documents reduces false information
- Domain specificity: Can be customized with domain-specific knowledge bases

Hyperparameters that affect RAG performance include chunk size, chunk overlap, number of
retrieved documents, embedding model choice, and retrieval strategy."""
    },
    {
        "id": "doc5",
        "title": "Vector Databases",
        "content": """Vector databases are specialized databases designed to store, manage, and query
high-dimensional vector embeddings efficiently. They are essential components of modern AI applications,
particularly for similarity search and RAG systems.

Popular vector databases include:
- Milvus: Open-source, highly scalable vector database
- Pinecone: Managed vector database service
- Chroma: Lightweight, embedded vector database
- Elasticsearch: Full-text search with vector capabilities
- FAISS: Facebook's library for efficient similarity search

Key features of vector databases:
- Approximate Nearest Neighbor (ANN) search for fast retrieval
- Support for various distance metrics (cosine, euclidean, dot product)
- Filtering and metadata support
- Scalability for billions of vectors

The choice of vector database and its configuration significantly impacts RAG system performance."""
    }
]

# Benchmark Q&A pairs for evaluation
BENCHMARK_QA = [
    {
        "question": "What are the three main types of machine learning?",
        "ground_truth": "The three main types of machine learning are: 1) Supervised Learning - the algorithm learns from labeled training data, 2) Unsupervised Learning - the algorithm finds patterns in unlabeled data, and 3) Reinforcement Learning - the algorithm learns by interacting with an environment and receiving rewards.",
        "relevant_docs": ["doc1"]
    },
    {
        "question": "What is deep learning and how does it differ from traditional machine learning?",
        "ground_truth": "Deep learning is part of machine learning based on artificial neural networks with representation learning. The 'deep' refers to multiple layers through which data is transformed. Unlike traditional ML, deep learning enables end-to-end learning directly from raw data without manual feature engineering.",
        "relevant_docs": ["doc2"]
    },
    {
        "question": "What are the main components of a RAG pipeline?",
        "ground_truth": "A RAG pipeline consists of three main phases: 1) Indexing Phase - documents are chunked, embedded, and stored in a vector database, 2) Retrieval Phase - relevant document chunks are retrieved using similarity search, and 3) Generation Phase - the LLM generates a response using the retrieved context.",
        "relevant_docs": ["doc4"]
    },
    {
        "question": "Name some popular vector databases used in AI applications.",
        "ground_truth": "Popular vector databases include Milvus (open-source, highly scalable), Pinecone (managed service), Chroma (lightweight, embedded), Elasticsearch (full-text search with vector capabilities), and FAISS (Facebook's library for efficient similarity search).",
        "relevant_docs": ["doc5"]
    },
    {
        "question": "What NLP tasks can be performed using modern transformer models?",
        "ground_truth": "Key NLP tasks include text classification, named entity recognition, sentiment analysis, machine translation, question answering, and text summarization. Modern NLP relies on transformer architectures like BERT, GPT, and T5.",
        "relevant_docs": ["doc3"]
    }
]


# =============================================================================
# Ollama Client
# =============================================================================

class OllamaClient:
    """Simple client for Ollama API"""

    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url

    def generate(self, model: str, prompt: str, system: str = None) -> str:
        """Generate text using Ollama"""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        if system:
            payload["system"] = system

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        return response.json()["response"]

    def embed(self, model: str, text: str) -> list[float]:
        """Get embeddings using Ollama"""
        response = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": model, "input": text},
            timeout=60
        )
        response.raise_for_status()
        return response.json()["embeddings"][0]


# =============================================================================
# RAG Components
# =============================================================================

@dataclass
class Chunk:
    """A document chunk"""
    content: str
    doc_id: str
    chunk_id: int
    embedding: Optional[list[float]] = None


@dataclass
class RetrievalResult:
    """Result of retrieval"""
    chunks: list[Chunk]
    scores: list[float]


def chunk_document(text: str, doc_id: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Split document into overlapping chunks"""
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        if chunk_text.strip():
            chunks.append(Chunk(
                content=chunk_text.strip(),
                doc_id=doc_id,
                chunk_id=chunk_id
            ))
            chunk_id += 1

        start = end - chunk_overlap
        if start >= len(text) - chunk_overlap:
            break

    return chunks


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class SimpleVectorStore:
    """In-memory vector store for demo purposes"""

    def __init__(self):
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk]):
        """Add chunks to the store"""
        self.chunks.extend(chunks)

    def search(self, query_embedding: list[float], top_k: int = 5) -> RetrievalResult:
        """Search for similar chunks"""
        scores = []
        for chunk in self.chunks:
            score = cosine_similarity(query_embedding, chunk.embedding)
            scores.append((chunk, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        top_chunks = [s[0] for s in scores[:top_k]]
        top_scores = [s[1] for s in scores[:top_k]]

        return RetrievalResult(chunks=top_chunks, scores=top_scores)


# =============================================================================
# RAG Pattern Configuration
# =============================================================================

@dataclass
class RAGConfig:
    """Configuration for a RAG pattern"""
    name: str
    chunk_size: int
    chunk_overlap: int
    retrieval_method: str  # "simple" or "window"
    num_chunks: int
    window_size: int = 0  # For window retrieval


@dataclass
class EvaluationScore:
    """Evaluation scores for a RAG pattern"""
    answer_correctness: float
    context_relevance: float
    faithfulness: float
    combined_score: float


@dataclass
class PatternResult:
    """Results from evaluating a RAG pattern"""
    config: RAGConfig
    scores: EvaluationScore
    sample_responses: list[dict] = field(default_factory=list)
    execution_time: float = 0.0


# =============================================================================
# RAG Evaluation
# =============================================================================

def evaluate_answer(
    client: OllamaClient,
    question: str,
    generated_answer: str,
    ground_truth: str,
    context: str
) -> EvaluationScore:
    """
    Evaluate the quality of a RAG response.
    Uses LLM-as-judge approach for evaluation.
    """

    # Evaluate answer correctness (compared to ground truth)
    correctness_prompt = f"""Evaluate how correct this generated answer is compared to the ground truth.

Question: {question}

Ground Truth Answer: {ground_truth}

Generated Answer: {generated_answer}

Rate the correctness from 0 to 100, where:
- 100 = Perfectly correct, covers all key points
- 75 = Mostly correct with minor omissions
- 50 = Partially correct
- 25 = Mostly incorrect
- 0 = Completely wrong

Respond with ONLY a number between 0 and 100."""

    try:
        correctness_response = client.generate(LLM_MODEL, correctness_prompt)
        # Extract number from response
        correctness = float(''.join(c for c in correctness_response if c.isdigit() or c == '.') or '50')
        correctness = min(100, max(0, correctness)) / 100
    except:
        correctness = 0.5

    # Evaluate context relevance
    relevance_prompt = f"""Evaluate how relevant this context is to answering the question.

Question: {question}

Context: {context[:1500]}

Rate the relevance from 0 to 100, where:
- 100 = Highly relevant, contains all needed information
- 50 = Somewhat relevant
- 0 = Not relevant at all

Respond with ONLY a number between 0 and 100."""

    try:
        relevance_response = client.generate(LLM_MODEL, relevance_prompt)
        relevance = float(''.join(c for c in relevance_response if c.isdigit() or c == '.') or '50')
        relevance = min(100, max(0, relevance)) / 100
    except:
        relevance = 0.5

    # Evaluate faithfulness (is answer grounded in context?)
    faithfulness_prompt = f"""Evaluate if this answer is faithful to (grounded in) the provided context.

Context: {context[:1500]}

Generated Answer: {generated_answer}

Rate faithfulness from 0 to 100, where:
- 100 = Completely grounded in context, no hallucinations
- 50 = Partially grounded
- 0 = Contains information not in context (hallucinated)

Respond with ONLY a number between 0 and 100."""

    try:
        faithfulness_response = client.generate(LLM_MODEL, faithfulness_prompt)
        faithfulness = float(''.join(c for c in faithfulness_response if c.isdigit() or c == '.') or '50')
        faithfulness = min(100, max(0, faithfulness)) / 100
    except:
        faithfulness = 0.5

    # Combined score (weighted average)
    combined = 0.4 * correctness + 0.3 * relevance + 0.3 * faithfulness

    return EvaluationScore(
        answer_correctness=correctness,
        context_relevance=relevance,
        faithfulness=faithfulness,
        combined_score=combined
    )


# =============================================================================
# Ragit Core: Hyperparameter Optimization
# =============================================================================

class RagitOptimizer:
    """
    Ragit: Automatic RAG Optimization Engine

    This class demonstrates the core functionality of Ragit - automatically
    finding the best hyperparameters for a RAG system through systematic
    evaluation of different configurations.
    """

    def __init__(
        self,
        documents: list[dict],
        benchmark_qa: list[dict],
        ollama_client: OllamaClient
    ):
        self.documents = documents
        self.benchmark_qa = benchmark_qa
        self.client = ollama_client
        self.results: list[PatternResult] = []

    def define_search_space(self) -> list[RAGConfig]:
        """
        Define the hyperparameter search space.
        This creates all combinations of parameters to evaluate.
        """
        # Define parameter ranges (simplified for demo)
        chunk_sizes = [256, 512]
        chunk_overlaps = [50, 100]
        retrieval_methods = ["simple"]
        num_chunks_options = [2, 3]

        configs = []
        pattern_num = 1

        for chunk_size, overlap, method, num_chunks in product(
            chunk_sizes, chunk_overlaps, retrieval_methods, num_chunks_options
        ):
            # Ensure overlap is less than chunk size
            if overlap >= chunk_size:
                continue

            configs.append(RAGConfig(
                name=f"Pattern_{pattern_num}",
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                retrieval_method=method,
                num_chunks=num_chunks
            ))
            pattern_num += 1

        return configs

    def build_index(self, config: RAGConfig) -> SimpleVectorStore:
        """Build vector index with given configuration"""
        store = SimpleVectorStore()

        all_chunks = []
        for doc in self.documents:
            chunks = chunk_document(
                doc["content"],
                doc["id"],
                config.chunk_size,
                config.chunk_overlap
            )
            all_chunks.extend(chunks)

        # Embed all chunks
        print(f"    Embedding {len(all_chunks)} chunks...")
        for chunk in all_chunks:
            chunk.embedding = self.client.embed(EMBEDDING_MODEL, chunk.content)

        store.add(all_chunks)
        return store

    def generate_answer(self, question: str, context: str) -> str:
        """Generate answer using RAG"""
        system_prompt = """You are a helpful assistant. Answer questions based ONLY on the provided context.
If the context doesn't contain enough information, say so. Be concise and accurate."""

        user_prompt = f"""Context:
{context}

Question: {question}

Answer based on the context above:"""

        return self.client.generate(LLM_MODEL, user_prompt, system_prompt)

    def evaluate_pattern(self, config: RAGConfig) -> PatternResult:
        """Evaluate a single RAG pattern configuration"""
        print(f"\n  Evaluating {config.name}:")
        print(f"    chunk_size={config.chunk_size}, overlap={config.chunk_overlap}, "
              f"retrieval={config.retrieval_method}, num_chunks={config.num_chunks}")

        start_time = time.time()

        # Build index
        store = self.build_index(config)

        # Evaluate on benchmark
        scores_list = []
        sample_responses = []

        for i, qa in enumerate(self.benchmark_qa):
            print(f"    Processing question {i+1}/{len(self.benchmark_qa)}...")

            # Get query embedding
            query_embedding = self.client.embed(EMBEDDING_MODEL, qa["question"])

            # Retrieve relevant chunks
            results = store.search(query_embedding, top_k=config.num_chunks)

            # Build context from retrieved chunks
            context = "\n\n".join([
                f"[From {chunk.doc_id}]: {chunk.content}"
                for chunk in results.chunks
            ])

            # Generate answer
            answer = self.generate_answer(qa["question"], context)

            # Evaluate
            score = evaluate_answer(
                self.client,
                qa["question"],
                answer,
                qa["ground_truth"],
                context
            )
            scores_list.append(score)

            # Store sample response
            sample_responses.append({
                "question": qa["question"],
                "generated_answer": answer,
                "ground_truth": qa["ground_truth"],
                "retrieved_docs": [c.doc_id for c in results.chunks],
                "scores": {
                    "correctness": score.answer_correctness,
                    "relevance": score.context_relevance,
                    "faithfulness": score.faithfulness
                }
            })

        # Aggregate scores
        avg_scores = EvaluationScore(
            answer_correctness=np.mean([s.answer_correctness for s in scores_list]),
            context_relevance=np.mean([s.context_relevance for s in scores_list]),
            faithfulness=np.mean([s.faithfulness for s in scores_list]),
            combined_score=np.mean([s.combined_score for s in scores_list])
        )

        execution_time = time.time() - start_time

        print(f"    Scores: correctness={avg_scores.answer_correctness:.2f}, "
              f"relevance={avg_scores.context_relevance:.2f}, "
              f"faithfulness={avg_scores.faithfulness:.2f}")
        print(f"    Combined Score: {avg_scores.combined_score:.3f}")

        return PatternResult(
            config=config,
            scores=avg_scores,
            sample_responses=sample_responses,
            execution_time=execution_time
        )

    def run_optimization(self, max_patterns: int = None) -> PatternResult:
        """
        Run the Ragit optimization process.
        Evaluates different RAG configurations and returns the best one.
        """
        print("=" * 70)
        print("Ragit: Automatic RAG Optimization Engine")
        print("=" * 70)

        # Define search space
        configs = self.define_search_space()
        if max_patterns:
            configs = configs[:max_patterns]

        print(f"\nSearch Space: {len(configs)} configurations to evaluate")
        print(f"Documents: {len(self.documents)}")
        print(f"Benchmark Questions: {len(self.benchmark_qa)}")
        print(f"LLM Model: {LLM_MODEL}")
        print(f"Embedding Model: {EMBEDDING_MODEL}")

        # Evaluate each pattern
        print("\n" + "-" * 70)
        print("Starting Hyperparameter Optimization...")
        print("-" * 70)

        for config in configs:
            result = self.evaluate_pattern(config)
            self.results.append(result)

        # Sort by combined score
        self.results.sort(key=lambda x: x.scores.combined_score, reverse=True)

        # Report results
        print("\n" + "=" * 70)
        print("OPTIMIZATION RESULTS")
        print("=" * 70)

        print("\nAll Patterns Ranked by Combined Score:")
        print("-" * 70)
        for i, result in enumerate(self.results, 1):
            print(f"{i}. {result.config.name}: {result.scores.combined_score:.3f}")
            print(f"   Config: chunk_size={result.config.chunk_size}, "
                  f"overlap={result.config.chunk_overlap}, "
                  f"num_chunks={result.config.num_chunks}")
            print(f"   Scores: correctness={result.scores.answer_correctness:.2f}, "
                  f"relevance={result.scores.context_relevance:.2f}, "
                  f"faithfulness={result.scores.faithfulness:.2f}")
            print(f"   Time: {result.execution_time:.1f}s")

        best = self.results[0]
        print("\n" + "=" * 70)
        print("BEST RAG PATTERN FOUND")
        print("=" * 70)
        print(f"\n{best.config.name}")
        print(f"  - Chunk Size: {best.config.chunk_size}")
        print(f"  - Chunk Overlap: {best.config.chunk_overlap}")
        print(f"  - Retrieval Method: {best.config.retrieval_method}")
        print(f"  - Number of Chunks: {best.config.num_chunks}")
        print(f"\nScores:")
        print(f"  - Answer Correctness: {best.scores.answer_correctness:.2%}")
        print(f"  - Context Relevance: {best.scores.context_relevance:.2%}")
        print(f"  - Faithfulness: {best.scores.faithfulness:.2%}")
        print(f"  - Combined Score: {best.scores.combined_score:.2%}")

        return best


# =============================================================================
# Interactive Demo
# =============================================================================

def run_interactive_demo(best_pattern: PatternResult, client: OllamaClient):
    """Run an interactive demo with the best pattern"""
    print("\n" + "=" * 70)
    print("INTERACTIVE RAG DEMO")
    print("=" * 70)
    print("\nUsing the optimized RAG configuration:")
    print(f"  chunk_size={best_pattern.config.chunk_size}, "
          f"overlap={best_pattern.config.chunk_overlap}, "
          f"num_chunks={best_pattern.config.num_chunks}")

    # Build index with best config
    store = SimpleVectorStore()
    for doc in SAMPLE_DOCUMENTS:
        chunks = chunk_document(
            doc["content"],
            doc["id"],
            best_pattern.config.chunk_size,
            best_pattern.config.chunk_overlap
        )
        for chunk in chunks:
            chunk.embedding = client.embed(EMBEDDING_MODEL, chunk.content)
        store.add(chunks)

    print("\nSample Q&A with optimized RAG:\n")

    test_questions = [
        "What is the difference between supervised and unsupervised learning?",
        "How does RAG help reduce hallucinations in LLMs?"
    ]

    for question in test_questions:
        print(f"Q: {question}")

        # Retrieve
        query_emb = client.embed(EMBEDDING_MODEL, question)
        results = store.search(query_emb, top_k=best_pattern.config.num_chunks)

        context = "\n\n".join([
            f"[{chunk.doc_id}]: {chunk.content}"
            for chunk in results.chunks
        ])

        # Generate
        system = "You are a helpful assistant. Answer based only on the provided context."
        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        answer = client.generate(LLM_MODEL, prompt, system)
        print(f"A: {answer}\n")
        print(f"Sources: {[c.doc_id for c in results.chunks]}")
        print("-" * 50 + "\n")


# =============================================================================
# Main
# =============================================================================

def main():
    print("""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                                                                       ║
    ║     █████╗ ██╗██╗  ██╗██████╗  █████╗  ██████╗                        ║
    ║    ██╔══██╗██║██║  ██║██╔══██╗██╔══██╗██╔════╝                        ║
    ║    ███████║██║███████║██████╔╝███████║██║  ███╗                       ║
    ║    ██╔══██║██║╚════██║██╔══██╗██╔══██║██║   ██║                       ║
    ║    ██║  ██║██║     ██║██║  ██║██║  ██║╚██████╔╝                       ║
    ║    ╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝                        ║
    ║                                                                       ║
    ║         Automatic RAG Optimization Engine by RODMENA LIMITED                      ║
    ║                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """)

    # Initialize client
    client = OllamaClient()

    # Verify Ollama is running
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
        print("✓ Connected to Ollama server")
    except Exception as e:
        print(f"✗ Error connecting to Ollama: {e}")
        print("  Make sure Ollama is running: ollama serve")
        return

    # Check models
    print(f"✓ LLM Model: {LLM_MODEL}")
    print(f"✓ Embedding Model: {EMBEDDING_MODEL}")

    # Create optimizer
    optimizer = RagitOptimizer(
        documents=SAMPLE_DOCUMENTS,
        benchmark_qa=BENCHMARK_QA,
        ollama_client=client
    )

    # Run optimization (limit patterns for demo)
    best_pattern = optimizer.run_optimization(max_patterns=4)

    # Run interactive demo
    run_interactive_demo(best_pattern, client)

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("""
This demo showcased Ragit's core capability: automatically finding
the optimal hyperparameters for a RAG system by evaluating different
configurations and selecting the best one based on metrics like:
- Answer Correctness
- Context Relevance
- Faithfulness (grounding)

In production, Ragit supports:
- More chunking strategies (recursive, semantic)
- Multiple embedding models
- Various vector databases (Milvus, Chroma, Elasticsearch)
- More retrieval methods (window-based, hybrid)
- Larger search spaces with sophisticated optimizers
- Generated deployment code for the best pattern
""")


if __name__ == "__main__":
    main()

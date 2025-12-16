#!/usr/bin/env python3
"""
Ragit Simple Demo - Shows the core concept quickly
"""

import requests
import numpy as np
import time

OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "qwen3-vl:235b-instruct-cloud"
EMBEDDING_MODEL = "nomic-embed-text:latest"

# Sample knowledge base
DOCUMENTS = [
    {"id": "doc1", "content": "Machine learning is a subset of AI that allows systems to learn from data. The three main types are: supervised learning (learns from labeled data), unsupervised learning (finds patterns in unlabeled data), and reinforcement learning (learns through rewards)."},
    {"id": "doc2", "content": "RAG (Retrieval-Augmented Generation) combines LLMs with external knowledge retrieval. The pipeline has 3 phases: 1) Indexing - chunk and embed documents, 2) Retrieval - find relevant chunks via similarity search, 3) Generation - LLM answers using retrieved context."},
    {"id": "doc3", "content": "Vector databases like Milvus, Pinecone, and Chroma store embeddings for similarity search. They use Approximate Nearest Neighbor (ANN) algorithms for fast retrieval and support various distance metrics like cosine similarity."},
]

# Test questions
QUESTIONS = [
    {"q": "What are the types of machine learning?", "expected_doc": "doc1"},
    {"q": "How does RAG work?", "expected_doc": "doc2"},
]


def embed(text):
    """Get embedding from Ollama"""
    r = requests.post(f"{OLLAMA_BASE_URL}/api/embed",
                      json={"model": EMBEDDING_MODEL, "input": text}, timeout=60)
    return r.json()["embeddings"][0]


def generate(prompt, system="You are a helpful assistant. Be concise."):
    """Generate text from Ollama"""
    r = requests.post(f"{OLLAMA_BASE_URL}/api/generate",
                      json={"model": LLM_MODEL, "prompt": prompt, "system": system, "stream": False},
                      timeout=120)
    return r.json()["response"]


def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def chunk_text(text, chunk_size, overlap):
    """Simple chunking"""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
        if start >= len(text) - overlap:
            break
    return chunks


def run_rag_pattern(chunk_size, overlap, top_k):
    """Run RAG with specific hyperparameters and measure quality"""
    print(f"\n  Testing: chunk_size={chunk_size}, overlap={overlap}, top_k={top_k}")

    # Index phase - chunk and embed documents
    indexed = []
    for doc in DOCUMENTS:
        chunks = chunk_text(doc["content"], chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            indexed.append({
                "doc_id": doc["id"],
                "chunk_id": i,
                "text": chunk,
                "embedding": embed(chunk)
            })
    print(f"    Indexed {len(indexed)} chunks")

    # Evaluate on questions
    correct = 0
    for qa in QUESTIONS:
        # Retrieve phase
        q_emb = embed(qa["q"])
        scores = [(c, cosine_sim(q_emb, c["embedding"])) for c in indexed]
        scores.sort(key=lambda x: x[1], reverse=True)
        top_chunks = scores[:top_k]

        # Check if correct doc was retrieved
        retrieved_docs = [c[0]["doc_id"] for c in top_chunks]
        if qa["expected_doc"] in retrieved_docs:
            correct += 1

        # Generate phase
        context = "\n".join([c[0]["text"] for c in top_chunks])
        prompt = f"Context:\n{context}\n\nQuestion: {qa['q']}\n\nAnswer briefly:"
        answer = generate(prompt)
        print(f"    Q: {qa['q']}")
        print(f"    A: {answer[:150]}...")

    retrieval_accuracy = correct / len(QUESTIONS)
    print(f"    Retrieval Accuracy: {retrieval_accuracy:.0%}")
    return retrieval_accuracy


def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║  Ragit Demo: Automatic RAG Hyperparameter Optimization      ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Verify connection
    try:
        requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        print(f"✓ Ollama connected | LLM: {LLM_MODEL} | Embeddings: {EMBEDDING_MODEL}")
    except:
        print("✗ Cannot connect to Ollama. Run: ollama serve")
        return

    print("\n" + "="*60)
    print("HYPERPARAMETER SEARCH")
    print("="*60)
    print("\nRagit evaluates different RAG configurations to find the best one.")
    print("Testing combinations of: chunk_size, overlap, num_retrieved_chunks\n")

    # Search space (simplified)
    configs = [
        (200, 50, 2),   # small chunks, small overlap, 2 results
        (400, 100, 2),  # larger chunks, more overlap, 2 results
        (200, 50, 3),   # small chunks, 3 results
    ]

    results = []
    for chunk_size, overlap, top_k in configs:
        score = run_rag_pattern(chunk_size, overlap, top_k)
        results.append((chunk_size, overlap, top_k, score))

    # Find best
    results.sort(key=lambda x: x[3], reverse=True)
    best = results[0]

    print("\n" + "="*60)
    print("OPTIMIZATION RESULT")
    print("="*60)
    print(f"\n🏆 Best Configuration Found:")
    print(f"   • Chunk Size: {best[0]}")
    print(f"   • Chunk Overlap: {best[1]}")
    print(f"   • Retrieved Chunks: {best[2]}")
    print(f"   • Score: {best[3]:.0%}")

    print("\n" + "="*60)
    print("WHAT Ragit DOES")
    print("="*60)
    print("""
Ragit automates RAG pipeline optimization by:

1. DEFINING SEARCH SPACE
   - Chunking: method, size, overlap
   - Embedding: model selection
   - Retrieval: method, num_chunks, window_size
   - Generation: model, prompts

2. SYSTEMATIC EVALUATION
   - Tests each configuration on benchmark Q&A pairs
   - Measures: Answer Correctness, Context Relevance, Faithfulness

3. BEST PATTERN SELECTION
   - Ranks all configurations by combined score
   - Returns optimal hyperparameters

4. CODE GENERATION
   - Generates deployable Python code for the best RAG pattern
   - Ready for production use
""")


if __name__ == "__main__":
    main()

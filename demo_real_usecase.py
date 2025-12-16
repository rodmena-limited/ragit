#!/usr/bin/env python3
"""
REAL-WORLD RAG EXAMPLE: Customer Support Chatbot
=================================================

Scenario: You're building a chatbot for "TechGadget Inc." that can answer
customer questions about products, return policies, troubleshooting, etc.

The problem: ChatGPT/Claude don't know about YOUR company's specific policies.
The solution: RAG - retrieve relevant info from YOUR documents, then generate answers.
"""

import requests
import numpy as np

OLLAMA_URL = "http://localhost:11434"
LLM = "qwen3-vl:235b-instruct-cloud"
EMBEDDER = "nomic-embed-text:latest"

# ============================================================================
# YOUR COMPANY'S KNOWLEDGE BASE (normally loaded from PDFs, databases, etc.)
# ============================================================================

COMPANY_DOCUMENTS = {
    "return_policy": """
    TechGadget Inc. Return Policy (Updated 2024)

    - All products can be returned within 30 days of purchase
    - Items must be in original packaging with all accessories
    - Refunds are processed within 5-7 business days
    - Digital products and opened software cannot be returned
    - Shipping costs for returns are covered by customer unless item is defective
    - For defective items, we provide free return shipping and full refund
    - Holiday purchases (Nov 15 - Dec 31) can be returned until January 31
    """,

    "warranty_info": """
    TechGadget Inc. Warranty Information

    - Standard warranty: 1 year from purchase date
    - Premium warranty (purchasable): 3 years coverage
    - Warranty covers manufacturing defects only
    - Warranty does NOT cover: water damage, drops, misuse
    - To claim warranty: visit support.techgadget.com/warranty
    - Keep your receipt as proof of purchase
    - Replacement or repair at TechGadget's discretion
    """,

    "smartwatch_x1_specs": """
    SmartWatch X1 - Product Specifications

    - Display: 1.5" AMOLED, 400x400 resolution
    - Battery: Up to 7 days normal use, 2 days with GPS
    - Water resistance: 5ATM (swimming OK, no diving)
    - Sensors: Heart rate, SpO2, accelerometer, GPS
    - Compatibility: iOS 14+ and Android 10+
    - Price: $299 (Premium bundle: $349 with extra bands)
    - Colors: Black, Silver, Rose Gold
    - Storage: 32GB for music and apps
    """,

    "smartwatch_troubleshooting": """
    SmartWatch X1 - Troubleshooting Guide

    Problem: Watch won't turn on
    Solution: Hold power button for 15 seconds. If still dead, charge for 2 hours then try again.

    Problem: Battery draining fast
    Solution: Disable always-on display. Turn off continuous heart rate. Update firmware.

    Problem: Not syncing with phone
    Solution: 1) Ensure Bluetooth is on. 2) Restart both devices. 3) Unpair and re-pair in app.

    Problem: Heart rate sensor not working
    Solution: Clean sensor with soft cloth. Wear watch tighter. If persists, contact support.

    Problem: GPS inaccurate
    Solution: Wait 30 seconds outdoors before starting activity. Ensure clear sky view.
    """,

    "contact_support": """
    TechGadget Inc. - Contact Information

    - Customer Support Phone: 1-800-TECHGAD (1-800-832-4423)
    - Hours: Monday-Friday 8AM-8PM EST, Saturday 9AM-5PM EST
    - Email: support@techgadget.com (response within 24 hours)
    - Live Chat: Available on website during business hours
    - Support Portal: support.techgadget.com
    - For urgent issues with recent orders: priority@techgadget.com
    """
}

# ============================================================================
# RAG SYSTEM IMPLEMENTATION
# ============================================================================

def embed(text):
    """Convert text to vector embedding"""
    r = requests.post(f"{OLLAMA_URL}/api/embed",
                      json={"model": EMBEDDER, "input": text}, timeout=60)
    return np.array(r.json()["embeddings"][0])

def generate(prompt, system):
    """Generate response using LLM"""
    r = requests.post(f"{OLLAMA_URL}/api/generate",
                      json={"model": LLM, "prompt": prompt, "system": system, "stream": False},
                      timeout=120)
    return r.json()["response"]

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class CustomerSupportRAG:
    """RAG-powered customer support system"""

    def __init__(self, documents: dict):
        print("📚 Indexing company documents...")
        self.index = []
        for doc_name, content in documents.items():
            embedding = embed(content)
            self.index.append({
                "name": doc_name,
                "content": content,
                "embedding": embedding
            })
        print(f"✅ Indexed {len(self.index)} documents\n")

    def retrieve(self, query: str, top_k: int = 2):
        """Find most relevant documents for the query"""
        query_embedding = embed(query)

        # Calculate similarity with each document
        scores = []
        for doc in self.index:
            sim = cosine_similarity(query_embedding, doc["embedding"])
            scores.append((doc, sim))

        # Sort by similarity (highest first)
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def answer(self, question: str):
        """Answer a customer question using RAG"""
        # Step 1: RETRIEVE relevant documents
        retrieved = self.retrieve(question, top_k=2)

        # Step 2: Build context from retrieved docs
        context = "\n\n---\n\n".join([
            f"[{doc['name']}]\n{doc['content']}"
            for doc, score in retrieved
        ])

        # Step 3: GENERATE answer using LLM + context
        system = """You are a helpful customer support agent for TechGadget Inc.
Answer questions based ONLY on the provided company documents.
Be friendly, concise, and accurate. If the information isn't in the documents, say so."""

        prompt = f"""Company Documents:
{context}

Customer Question: {question}

Answer:"""

        answer = generate(prompt, system)

        return {
            "question": question,
            "answer": answer,
            "sources": [doc["name"] for doc, _ in retrieved]
        }


# ============================================================================
# DEMO: Show RAG vs No-RAG
# ============================================================================

def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║   REAL-WORLD RAG EXAMPLE: Customer Support Chatbot                    ║
║                                                                       ║
║   Scenario: TechGadget Inc. needs a chatbot that can answer           ║
║   questions about THEIR specific products, policies, and support.     ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
""")

    # Initialize RAG system
    rag = CustomerSupportRAG(COMPANY_DOCUMENTS)

    # Customer questions to test
    customer_questions = [
        "How long do I have to return a product?",
        "My SmartWatch X1 won't turn on, what should I do?",
        "What's the phone number for customer support?",
        "Is the SmartWatch X1 waterproof? Can I swim with it?",
    ]

    print("=" * 70)
    print("CUSTOMER SUPPORT CHATBOT (Powered by RAG)")
    print("=" * 70)

    for q in customer_questions:
        print(f"\n👤 CUSTOMER: {q}")
        result = rag.answer(q)
        print(f"\n🤖 SUPPORT: {result['answer']}")
        print(f"\n   📄 Sources: {', '.join(result['sources'])}")
        print("-" * 70)

    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║   WHY THIS MATTERS (RAG Benefits):                                    ║
║                                                                       ║
║   ✓ Answers are based on YOUR actual company data                     ║
║   ✓ No hallucinations - grounded in real documents                    ║
║   ✓ Easy to update - just change the documents                        ║
║   ✓ Traceable - you can see which sources were used                   ║
║   ✓ Domain-specific - knows YOUR products, policies, prices           ║
║                                                                       ║
║   Without RAG, the LLM would make up answers or say "I don't know"    ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝

WHERE Ragit FITS IN:
---------------------
The RAG system above has HYPERPARAMETERS that affect quality:

  - How to chunk documents? (by paragraph? by 500 chars? by topic?)
  - How many documents to retrieve? (1? 3? 5?)
  - Which embedding model? (there are dozens)
  - What similarity threshold?

Ragit AUTOMATICALLY finds the BEST settings by testing many combinations
and measuring which one gives the most accurate answers.

Think of it like this:
  - RAG = The car
  - Ragit = The mechanic that tunes the car for best performance
""")


if __name__ == "__main__":
    main()

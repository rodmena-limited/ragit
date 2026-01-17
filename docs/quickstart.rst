Quickstart Guide
================

Get started with ragit in minutes.

Installation
------------

.. code-block:: bash

   pip install ragit

   # For offline embedding (no API required)
   pip install ragit[transformers]

Your First RAG Assistant
------------------------

You must provide an embedding source. Here are three options:

**Option 1: Custom embedding function**

.. code-block:: python

   from ragit import RAGAssistant

   def my_embed(text: str) -> list[float]:
       # Use any embedding API: OpenAI, Cohere, HuggingFace, etc.
       return your_embedding_api(text)

   assistant = RAGAssistant("docs/", embed_fn=my_embed)
   results = assistant.retrieve("search query")

**Option 2: Offline with SentenceTransformers**

Models download automatically on first use (~90MB for default model).

.. code-block:: python

   from ragit import RAGAssistant
   from ragit.providers import SentenceTransformersProvider

   assistant = RAGAssistant(
       "docs/",
       provider=SentenceTransformersProvider()  # Uses all-MiniLM-L6-v2
   )
   results = assistant.retrieve("search query")

Available models: ``all-MiniLM-L6-v2`` (384d), ``all-mpnet-base-v2`` (768d)

**Option 3: With LLM for Q&A**

.. code-block:: python

   from ragit import RAGAssistant

   def my_embed(text: str) -> list[float]:
       return your_embedding_api(text)

   def my_generate(prompt: str, system_prompt: str = "") -> str:
       return your_llm_api(prompt, system_prompt)

   assistant = RAGAssistant(
       "docs/",
       embed_fn=my_embed,
       generate_fn=my_generate
   )
   answer = assistant.ask("How does authentication work?")

Retrieval
---------

.. code-block:: python

   # Get relevant chunks
   results = assistant.retrieve("database configuration", top_k=5)

   for chunk, score in results:
       print(f"Score: {score:.3f}")
       print(f"Content: {chunk.content[:200]}...")
       print(f"Source: {chunk.doc_id}")

   # Get formatted context string
   context = assistant.get_context("database configuration", top_k=3)

Generation (requires LLM)
-------------------------

.. code-block:: python

   # Q&A with retrieval
   answer = assistant.ask("How do I configure the database?")

   # Direct generation
   response = assistant.generate("Explain RAG in simple terms")

   # Code generation
   code = assistant.generate_code("Write a fibonacci function")

Document Loading
----------------

.. code-block:: python

   from ragit import load_text, load_directory, chunk_text

   # Load single file
   doc = load_text("README.md")

   # Load directory
   docs = load_directory("docs/", pattern="*.md")

   # Custom chunking
   chunks = chunk_text(text, chunk_size=512, chunk_overlap=50, doc_id="doc1")

Chunk Settings
--------------

.. code-block:: python

   assistant = RAGAssistant(
       "docs/",
       embed_fn=my_embed,
       chunk_size=1024,      # Characters per chunk
       chunk_overlap=100     # Overlap between chunks
   )

Hyperparameter Optimization
---------------------------

.. code-block:: python

   from ragit import RagitExperiment, Document, BenchmarkQuestion

   def my_embed(text: str) -> list[float]:
       return your_embedding_api(text)

   def my_generate(prompt: str, system_prompt: str = "") -> str:
       return your_llm_api(prompt, system_prompt)

   docs = [Document(id="doc1", content="...")]
   benchmark = [BenchmarkQuestion(question="...", ground_truth="...")]

   experiment = RagitExperiment(
       docs, benchmark,
       embed_fn=my_embed,
       generate_fn=my_generate
   )
   results = experiment.run(max_configs=20)
   print(results[0])  # Best config

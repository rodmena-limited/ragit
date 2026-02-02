RAGAssistant API
================

The ``RAGAssistant`` class provides a high-level interface for RAG operations.

.. note::

   ``RAGAssistant`` is **thread-safe** as of v0.11.0. It uses lock-free atomic operations
   with immutable state, allowing concurrent reads while another thread writes.
   See :doc:`../integration` for usage patterns.

Class Reference
---------------

.. autoclass:: ragit.RAGAssistant
   :members:
   :undoc-members:
   :show-inheritance:

Quick Reference
---------------

Constructor
^^^^^^^^^^^

.. code-block:: python

   from ragit import RAGAssistant

   assistant = RAGAssistant(
       source,                          # str, Path, or list of sources
       chunk_size=512,                  # Characters per chunk
       chunk_overlap=50,                # Overlap between chunks
       llm_model="llama3",              # LLM model name
       embedding_model="mxbai-embed-large",  # Embedding model name
       pattern="*.txt",                 # File pattern for directories
       recursive=False                  # Recursive directory search
   )

Methods
^^^^^^^

ask()
"""""

Ask a question and get an answer using RAG.

.. code-block:: python

   answer = assistant.ask("What is the installation process?")
   print(answer)

retrieve()
""""""""""

Retrieve relevant chunks without generating an answer.

.. code-block:: python

   results = assistant.retrieve("database configuration", top_k=5)

   for chunk, score in results:
       print(f"Score: {score:.3f}")
       print(f"Source: {chunk.doc_id}")
       print(f"Content: {chunk.content[:100]}...")

generate()
""""""""""

Direct LLM generation without retrieval.

.. code-block:: python

   response = assistant.generate("Explain what RAG means")
   print(response)

generate_code()
"""""""""""""""

Generate code with context from documents.

.. code-block:: python

   code = assistant.generate_code("Write a function to parse JSON")
   print(code)

Examples
--------

Basic Usage
^^^^^^^^^^^

.. code-block:: python

   from ragit import RAGAssistant

   # Create assistant
   assistant = RAGAssistant("docs/")

   # Ask questions
   answer = assistant.ask("How do I get started?")
   print(answer)

Custom Configuration
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from ragit import RAGAssistant

   assistant = RAGAssistant(
       "docs/",
       chunk_size=1024,                    # Larger chunks
       chunk_overlap=100,                  # More overlap
       llm_model="llama3:70b",             # Larger model
       embedding_model="nomic-embed-text"  # Different embeddings
   )

Multiple Document Sources
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from ragit import RAGAssistant

   # Load from multiple sources
   assistant = RAGAssistant([
       "README.md",           # Single file
       "docs/",               # Directory
       "examples/tutorial/"   # Another directory
   ])

Recursive Loading
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from ragit import RAGAssistant

   # Load all markdown files recursively
   assistant = RAGAssistant(
       "project/",
       pattern="**/*.md",
       recursive=True
   )

Getting Sources with Answers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from ragit import RAGAssistant

   assistant = RAGAssistant("docs/")

   question = "How do I configure logging?"

   # Get sources
   sources = assistant.retrieve(question, top_k=3)

   # Get answer
   answer = assistant.ask(question)

   print(f"Answer: {answer}\n")
   print("Based on:")
   for chunk, score in sources:
       print(f"  - {chunk.doc_id} (relevance: {score:.2f})")

Index Persistence
-----------------

Save and load indexes to avoid re-computing embeddings:

.. code-block:: python

   from ragit import RAGAssistant
   from ragit.providers import OllamaProvider

   # Build and save index
   assistant = RAGAssistant("docs/", provider=OllamaProvider())
   assistant.save_index("./my_index")

   # Load index later (much faster)
   loaded = RAGAssistant.load_index("./my_index", provider=OllamaProvider())
   results = loaded.retrieve("query")

Thread Safety
-------------

RAGAssistant is thread-safe as of v0.11.0:

.. code-block:: python

   import threading
   from ragit import RAGAssistant

   assistant = RAGAssistant("docs/", provider=provider)

   # Safe: concurrent reads
   for _ in range(10):
       threading.Thread(target=lambda: assistant.retrieve("query")).start()

   # Safe: read while writing
   threading.Thread(target=lambda: assistant.add_documents([doc])).start()
   threading.Thread(target=lambda: assistant.retrieve("query")).start()

Internal Attributes
-------------------

These attributes are available but considered internal:

.. code-block:: python

   # Access index state (immutable IndexState dataclass)
   state = assistant._state
   print(f"Total chunks: {len(state.chunks)}")
   print(f"Matrix shape: {state.embedding_matrix.shape}")

   # Use public properties instead
   print(f"Chunk count: {assistant.chunk_count}")
   print(f"Is indexed: {assistant.is_indexed}")

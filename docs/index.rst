ragit Documentation
===================

**ragit** is a RAG (Retrieval-Augmented Generation) toolkit for Python. Build and optimize RAG pipelines with any embedding or LLM provider.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   concepts
   configuration
   optimization
   integration

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/assistant
   api/experiment
   api/providers
   api/loaders

.. toctree::
   :maxdepth: 1
   :caption: Community

   contributing
   changelog


Key Features
------------

- **Provider Agnostic**: Use any embedding API (OpenAI, Cohere, HuggingFace) or Ollama with nomic-embed-text
- **RAG Hyperparameter Optimization**: Find optimal chunk size, overlap, and retrieval parameters
- **High-Level API**: Simple ``RAGAssistant`` for document Q&A
- **Document Loading**: Built-in utilities for loading and chunking documents

Quick Example
-------------

.. code-block:: python

   from ragit import RAGAssistant

   def my_embed(text: str) -> list[float]:
       # Use any embedding API
       return your_embedding_api(text)

   assistant = RAGAssistant("docs/", embed_fn=my_embed)
   results = assistant.retrieve("How do I create a new user?")

Or with Ollama (nomic-embed-text):

.. code-block:: python

   from ragit import RAGAssistant
   from ragit.providers import OllamaProvider

   assistant = RAGAssistant("docs/", provider=OllamaProvider())
   results = assistant.retrieve("How do I create a new user?")


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

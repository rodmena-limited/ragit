Installation
============

Prerequisites
-------------

- Python 3.12 or higher (up to 3.14)
- pip package manager

Installing from PyPI
--------------------

.. code-block:: bash

   pip install ragit

For offline embedding (no external API required):

.. code-block:: bash

   pip install ragit[transformers]

This installs sentence-transformers for local embedding. Models download automatically on first use (~90MB for default model).

Installing from Source
----------------------

.. code-block:: bash

   git clone https://github.com/rodmena-limited/ragit.git
   cd ragit
   pip install .

Development Installation
------------------------

.. code-block:: bash

   git clone https://github.com/rodmena-limited/ragit.git
   cd ragit
   pip install -e ".[dev]"

Verification
------------

.. code-block:: python

   import ragit
   print(ragit.__version__)

   # Test with offline embedding
   from ragit import RAGAssistant
   from ragit.providers import SentenceTransformersProvider

   assistant = RAGAssistant(
       ".",  # Current directory
       provider=SentenceTransformersProvider()
   )
   print(f"Loaded {assistant.num_chunks} chunks")

Next Steps
----------

- Follow the :doc:`quickstart` guide to build your first RAG application
- Read about :doc:`configuration` options for customizing ragit

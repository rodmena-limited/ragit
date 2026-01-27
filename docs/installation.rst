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

   # Test with Ollama provider
   from ragit import RAGAssistant
   from ragit.providers import OllamaProvider

   assistant = RAGAssistant(
       ".",  # Current directory
       provider=OllamaProvider()
   )
   print(f"Loaded {assistant.num_chunks} chunks")

Next Steps
----------

- Follow the :doc:`quickstart` guide to build your first RAG application
- Read about :doc:`configuration` options for customizing ragit

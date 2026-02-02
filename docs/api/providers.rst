Providers API
=============

Providers are responsible for LLM and embedding operations. ragit includes an Ollama provider and abstract base classes for creating custom providers.

OllamaProvider
--------------

The primary provider for both LLM and embedding operations.

.. autoclass:: ragit.providers.OllamaProvider
   :members:
   :undoc-members:
   :show-inheritance:

Quick Reference
^^^^^^^^^^^^^^^

.. code-block:: python

   from ragit.providers import OllamaProvider

   # Create with defaults (uses environment variables)
   provider = OllamaProvider()

   # Create with custom settings
   provider = OllamaProvider(
       base_url="http://localhost:11434",
       embedding_url="http://localhost:11434",  # Can be different
       api_key=None,                            # For cloud providers
       timeout=120                              # Request timeout
   )

Checking Availability
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from ragit.providers import OllamaProvider

   provider = OllamaProvider()

   if provider.is_available():
       print("Ollama is running")
       print(f"URL: {provider.base_url}")
   else:
       print("Ollama is not available")
       print("Start with: ollama serve")

Text Generation
^^^^^^^^^^^^^^^

.. code-block:: python

   from ragit.providers import OllamaProvider

   provider = OllamaProvider()

   # Basic generation
   response = provider.generate(
       prompt="Explain quantum computing",
       model="llama3"
   )
   print(response.text)

   # With parameters
   response = provider.generate(
       prompt="Write a haiku about programming",
       model="llama3",
       temperature=0.9,      # Higher = more creative
       max_tokens=100,       # Limit response length
       system_prompt="You are a poet."
   )
   print(response.text)
   print(f"Model: {response.model}")
   print(f"Provider: {response.provider}")

Creating Embeddings
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from ragit.providers import OllamaProvider

   provider = OllamaProvider()

   # Single embedding
   response = provider.embed(
       text="This is a sample sentence",
       model="mxbai-embed-large"
   )

   print(f"Dimensions: {response.dimensions}")
   print(f"Embedding type: {type(response.embedding)}")  # tuple
   print(f"First 5 values: {response.embedding[:5]}")

   # Batch embeddings (more efficient)
   texts = [
       "First document",
       "Second document",
       "Third document"
   ]

   responses = provider.embed_batch(texts, model="mxbai-embed-large")
   print(f"Created {len(responses)} embeddings")

   for i, resp in enumerate(responses):
       print(f"  {i}: {resp.dimensions} dimensions")

Resource Management
^^^^^^^^^^^^^^^^^^^

``OllamaProvider`` supports the context manager protocol for automatic cleanup:

.. code-block:: python

   from ragit.providers import OllamaProvider

   # Recommended: Use context manager for automatic cleanup
   with OllamaProvider() as provider:
       response = provider.generate("Hello", model="llama3")
       embeddings = provider.embed_batch(texts, model="mxbai-embed-large")
   # Session automatically closed on exit

   # Alternative: Manual cleanup
   provider = OllamaProvider()
   try:
       response = provider.generate("Hello", model="llama3")
   finally:
       provider.close()

Performance Features
^^^^^^^^^^^^^^^^^^^^

**Connection Pooling**

``OllamaProvider`` uses ``requests.Session()`` for HTTP connection pooling:

.. code-block:: python

   from ragit.providers import OllamaProvider

   with OllamaProvider() as provider:
       # All requests reuse the same TCP connection
       for text in texts:
           provider.embed(text, model="mxbai-embed-large")

**Async Parallel Embedding**

For large batches, use ``embed_batch_async()`` with asyncio:

.. code-block:: python

   import asyncio
   from ragit.providers import OllamaProvider

   provider = OllamaProvider()

   async def embed_many():
       texts = ["doc1...", "doc2...", "doc3..."]
       return await provider.embed_batch_async(
           texts,
           model="mxbai-embed-large"
       )

   results = asyncio.run(embed_many())

**Embedding Cache**

Embeddings are cached automatically using an LRU cache (2048 entries):

.. code-block:: python

   from ragit.providers import OllamaProvider

   provider = OllamaProvider(use_cache=True)  # Default

   # First call hits API
   provider.embed("Hello", model="mxbai-embed-large")

   # Second call returns cached result
   provider.embed("Hello", model="mxbai-embed-large")

   # View cache statistics
   info = OllamaProvider.embedding_cache_info()
   print(info)  # {'hits': 1, 'misses': 1, 'maxsize': 2048, 'currsize': 1}

   # Clear cache
   OllamaProvider.clear_embedding_cache()

Base Classes
------------

Abstract base classes for creating custom providers.

BaseLLMProvider
^^^^^^^^^^^^^^^

.. autoclass:: ragit.providers.base.BaseLLMProvider
   :members:
   :undoc-members:
   :show-inheritance:

BaseEmbeddingProvider
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: ragit.providers.base.BaseEmbeddingProvider
   :members:
   :undoc-members:
   :show-inheritance:

Response Classes
----------------

LLMResponse
^^^^^^^^^^^

.. autoclass:: ragit.providers.base.LLMResponse
   :members:
   :undoc-members:

.. code-block:: python

   from ragit.providers import OllamaProvider

   provider = OllamaProvider()
   response = provider.generate("Hello", model="llama3")

   # Access response attributes
   print(response.text)       # Generated text
   print(response.model)      # "llama3"
   print(response.provider)   # "ollama"

EmbeddingResponse
^^^^^^^^^^^^^^^^^

.. autoclass:: ragit.providers.base.EmbeddingResponse
   :members:
   :undoc-members:

.. code-block:: python

   from ragit.providers import OllamaProvider

   provider = OllamaProvider()
   response = provider.embed("Sample text", model="mxbai-embed-large")

   # Access response attributes
   print(response.embedding)   # tuple[float, ...]
   print(response.dimensions)  # 1024 (for mxbai-embed-large)
   print(response.model)       # "mxbai-embed-large"
   print(response.provider)    # "ollama"

.. note::

   ``EmbeddingResponse`` is immutable (frozen dataclass) with tuple embeddings
   to prevent accidental modification and ensure thread-safety of the data.

Creating Custom Providers
-------------------------

To add support for a new LLM service, inherit from the base classes:

.. code-block:: python

   from ragit.providers.base import (
       BaseLLMProvider,
       BaseEmbeddingProvider,
       LLMResponse,
       EmbeddingResponse
   )

   class OpenAIProvider(BaseLLMProvider, BaseEmbeddingProvider):
       """Provider for OpenAI API."""

       def __init__(self, api_key: str):
           self.api_key = api_key
           self._dimensions = 1536  # text-embedding-ada-002

       @property
       def provider_name(self) -> str:
           return "openai"

       @property
       def dimensions(self) -> int:
           return self._dimensions

       def generate(
           self,
           prompt: str,
           model: str = "gpt-4",
           temperature: float = 0.7,
           max_tokens: int | None = None,
           system_prompt: str | None = None,
       ) -> LLMResponse:
           # Implementation using OpenAI API
           import openai

           messages = []
           if system_prompt:
               messages.append({"role": "system", "content": system_prompt})
           messages.append({"role": "user", "content": prompt})

           response = openai.ChatCompletion.create(
               model=model,
               messages=messages,
               temperature=temperature,
               max_tokens=max_tokens
           )

           return LLMResponse(
               text=response.choices[0].message.content,
               model=model,
               provider=self.provider_name
           )

       def embed(self, text: str, model: str = "text-embedding-ada-002") -> EmbeddingResponse:
           import openai

           response = openai.Embedding.create(
               input=text,
               model=model
           )

           embedding = response.data[0].embedding

           return EmbeddingResponse(
               embedding=tuple(embedding),  # Must be tuple
               model=model,
               provider=self.provider_name,
               dimensions=len(embedding)
           )

       def embed_batch(
           self,
           texts: list[str],
           model: str = "text-embedding-ada-002"
       ) -> list[EmbeddingResponse]:
           import openai

           response = openai.Embedding.create(
               input=texts,
               model=model
           )

           return [
               EmbeddingResponse(
                   embedding=tuple(item.embedding),
                   model=model,
                   provider=self.provider_name,
                   dimensions=len(item.embedding)
               )
               for item in response.data
           ]

       def is_available(self) -> bool:
           try:
               import openai
               openai.Model.list()
               return True
           except Exception:
               return False

Using Custom Providers
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from ragit import RagitExperiment

   # Create custom provider
   provider = OpenAIProvider(api_key="sk-...")

   # Use with experiment
   experiment = RagitExperiment(
       documents,
       benchmark,
       provider=provider
   )
   results = experiment.run()

Embedding Model Dimensions
--------------------------

ragit includes a mapping of known embedding model dimensions:

.. code-block:: python

   from ragit.providers.ollama import EMBEDDING_DIMENSIONS

   print(EMBEDDING_DIMENSIONS)
   # {
   #     "mxbai-embed-large": 1024,
   #     "nomic-embed-text": 768,
   #     "all-minilm": 384,
   #     ...
   # }

When using a model not in this mapping, ragit queries the model to determine dimensions automatically.

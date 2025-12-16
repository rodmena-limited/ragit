# ragit

Automatic RAG (Retrieval-Augmented Generation) hyperparameter optimization engine.

## What it does

ragit finds the best configuration for your RAG pipeline by testing different combinations of:
- Chunk sizes and overlaps
- Number of retrieved chunks
- Embedding models
- LLM models

You provide documents and benchmark questions, ragit evaluates different configurations and returns the best one.

## Install

```bash
pip install ragit
```

## Usage

```python
from ragit import RagitExperiment, Document, BenchmarkQuestion

documents = [
    Document(id="doc1", content="Your document text here..."),
]

benchmark = [
    BenchmarkQuestion(
        question="A question about your documents?",
        ground_truth="The expected answer."
    ),
]

experiment = RagitExperiment(documents, benchmark)
results = experiment.run()

print(results[0])  # Best configuration
```

## License

Apache-2.0 - RODMENA LIMITED

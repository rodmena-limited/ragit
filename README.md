> [!IMPORTANT]
> Work in progress ...

<div align="center">

# `ragit`
### RAG Templates Optimization Engine

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)
[![GitHub](https://img.shields.io/github/license/rodmena-limited/ragit?style=for-the-badge)](https://github.com/rodmena-limited/ragit)

[![RAG Builder](https://img.shields.io/badge/RAG%20Builder-10B981?style=flat-square)](#)
[![HPO](https://img.shields.io/badge/Hyperparameter%20Optimization-F59E0B?style=flat-square)](#)
[![AutoML](https://img.shields.io/badge/AutoML%20for%20RAG-8B5CF6?style=flat-square)](#)

**Initialises RAG Template with optimal parameters**

[Quick Start](#quick-start) • [Documentation](#how-to-use) • [Examples](#example) • [Contributing](#contribution)

</div>

---

## What is ragit?

`ragit` is an **optimization engine for RAG Templates** that is LLM and Vector Database provider agnostic.
It accepts variety of RAG Templates and search space definition. Returns initialised RAG Template with optimal parameters' values (called RAG Pattern).

## Quick Start

```bash
pip install ragit
```

## How to use

```python
from ragit import RagitExperiment, Document, BenchmarkQuestion, OllamaProvider

# Prepare your documents
documents = [
    Document(id="doc1", content="Python is a programming language created by Guido van Rossum."),
    Document(id="doc2", content="Machine learning enables systems to learn from data."),
]

# Define benchmark questions with expected answers
benchmark = [
    BenchmarkQuestion(
        question="Who created Python?",
        ground_truth="Guido van Rossum created Python."
    ),
]

# Run the optimization experiment
experiment = RagitExperiment(documents, benchmark)
results = experiment.run()

# Get the best configuration
best = results[0]
print(f"Best config: {best.pattern_name}, Score: {best.final_score:.3f}")
```

## Example

See the `examples/` directory for more detailed examples.


## Contribution
Pull requests are very welcome! Make sure your patches are well tested. Ideally create a topic branch for every separate change you make. For example:

1. Fork the repo
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Added some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create new Pull Request

See more details in [contributing section](contributing.md).

---

Copyright (c) 2025 RODMENA LIMITED

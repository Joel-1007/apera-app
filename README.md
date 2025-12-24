# APERA PoC - Agentic RAG for Ethical Academic Research

## Overview
Student-led proof-of-concept for building an ethical RAG system for academic research.

## Timeline
- **Weeks 1-2**: Planning & Setup ✅ (YOU ARE HERE!)
- **Weeks 3-5**: Core Development
- **Weeks 6-8**: Integration & Ethics
- **Weeks 9-11**: Testing
- **Weeks 12-14**: Deployment

## Quick Start
```bash
# Install dependencies
poetry install

# Fetch papers
python src/data_fetch.py

# Run tests
pytest
```

## Stack
- **Orchestration**: LlamaIndex
- **Embeddings**: SentenceTransformers
- **Indexing**: FAISS
- **Ethics**: AIF360, toxic-bert
- **UI**: Gradio

## License
MIT

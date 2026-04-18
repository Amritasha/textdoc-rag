# textdoc-rag

A minimal Retrieval-Augmented Generation (RAG) pipeline in a single Python script. No frameworks, no cloud — just FAISS, SentenceTransformers, and a local LLM via Ollama.

## How it works

```
Documents (PDF/TXT)
     │
     ▼ chunk (400 words, 50-word overlap)
     │
     ▼ embed (all-MiniLM-L6-v2, 384-dim)
     │
     ▼ store (FAISS IndexFlatIP — exact cosine search)
     │
  [saved to disk: rag_index.faiss + rag_index.pkl]
     │
  query → embed → top-k retrieval → prompt → Ollama LLM → answer
```

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com) running locally with a model pulled

## Setup

```bash
# 1. Clone
git clone git@github.com:Amritasha/textdoc-rag.git
cd textdoc-rag

# 2. Install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install faiss-cpu sentence-transformers pymupdf requests numpy

# 3. Start Ollama and pull a model
ollama pull mistral
# ollama serve   ← only if not already running
```

## Usage

```bash
# Index one or more documents
python rag.py index report.pdf notes.txt

# Ask a question
python rag.py query "What are the key findings?"
```

Multiple `index` calls append to the same index — you don't need to re-index existing files.

## Quick test with the included sample

```bash
python rag.py index sample.txt
python rag.py query "What is RAG and how does it work?"
python rag.py query "What is FAISS used for?"
python rag.py query "What are the two stages of a RAG pipeline?"
```

## Configuration

Edit the constants at the top of `rag.py`:

| Variable | Default | Description |
|---|---|---|
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model (~80 MB, CPU-friendly) |
| `CHUNK_SIZE` | `400` | Words per chunk |
| `CHUNK_OVERLAP` | `50` | Word overlap between chunks |
| `TOP_K` | `5` | Chunks retrieved per query |
| `OLLAMA_MODEL` | `mistral` | LLM model name in Ollama |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama endpoint |

## Stack

| Component | Library |
|---|---|
| Embeddings | `sentence-transformers` |
| Vector search | `faiss-cpu` |
| PDF parsing | `pymupdf` |
| LLM | Ollama (local) |

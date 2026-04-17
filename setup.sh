#!/usr/bin/env bash
# Zero-friction setup for the local RAG pipeline.
# Run once: bash setup.sh

set -e

echo ""
echo "=== RAG Setup ==="
echo ""

# ── 1. Python version check ────────────────────────────────────────────────────
PYTHON=$(command -v python3.11 || command -v python3.10 || command -v python3.9 || echo "")
if [ -z "$PYTHON" ]; then
  echo "[error] Python 3.9+ not found."
  echo "  Install via: brew install python@3.11"
  exit 1
fi
echo "[ok] Using Python: $($PYTHON --version) at $PYTHON"

# ── 2. Create virtual environment ─────────────────────────────────────────────
if [ ! -d ".venv" ]; then
  echo "[setup] Creating virtual environment..."
  $PYTHON -m venv .venv
else
  echo "[ok] Virtual environment already exists."
fi

source .venv/bin/activate

# ── 3. Install dependencies ────────────────────────────────────────────────────
echo "[setup] Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet faiss-cpu sentence-transformers pymupdf requests numpy

echo "[ok] All dependencies installed."

# ── 4. Ollama check ───────────────────────────────────────────────────────────
echo ""
if command -v ollama &>/dev/null; then
  echo "[ok] Ollama is installed: $(ollama --version 2>&1 | head -1)"
  echo "     Make sure it's running:  ollama serve  (in a separate terminal)"
  echo "     And the model is pulled: ollama pull mistral"
else
  echo "[warn] Ollama not found. Install it from https://ollama.com"
  echo "       Then run:  ollama pull mistral"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  source .venv/bin/activate"
echo "  python rag.py index sample.txt"
echo "  python rag.py query \"What is RAG and how does it work?\""
echo ""

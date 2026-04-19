"""
Minimal local RAG pipeline — no frameworks, no cloud.

Dependencies:
    pip install faiss-cpu sentence-transformers pymupdf requests

LLM backend (pick one):
    Ollama:  https://ollama.com  →  ollama pull mistral
    OR set USE_OLLAMA=False and point OPENAI_BASE_URL at any OpenAI-compat server.

Usage:
    # Index documents
    python rag.py index doc1.txt doc2.pdf

    # Query
    python rag.py query "What is the main topic?"
"""
from __future__ import annotations  # enables list[str] on Python 3.9+

import os
import sys
import time
import pickle
import argparse
import textwrap
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

@contextmanager
def timer(label: str):
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    print(f"[time]  {label}: {elapsed * 1000:.1f} ms")

import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer

# ─── Configuration ────────────────────────────────────────────────────────────

INDEX_PATH   = "rag_index"        # base name; saves .faiss + .pkl
EMBED_MODEL  = "all-MiniLM-L6-v2" # ~80 MB, CPU-friendly, dim=384
CHUNK_SIZE   = 400                 # words per chunk
CHUNK_OVERLAP = 50                 # word overlap between consecutive chunks
TOP_K        = 5                   # retrieved chunks per query

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"

# ─── Document Loading ─────────────────────────────────────────────────────────

def _load_pdf(p: Path) -> str:
    try:
        import fitz
    except ImportError:
        sys.exit("Install pymupdf:  pip install pymupdf")

    doc = fitz.open(str(p))
    pages_text = []
    for page in doc:
        text = page.get_text().strip()
        if text:
            pages_text.append(text)
        else:
            # Page has no selectable text — likely scanned; fall back to OCR
            pages_text.append(_ocr_page(page))
    return "\n".join(pages_text)


def _ocr_page(page) -> str:
    """Render a pymupdf page to an image and run Tesseract OCR on it."""
    try:
        import pytesseract
        from PIL import Image
        import io
    except ImportError:
        sys.exit(
            "Scanned PDF detected. Install OCR deps:\n"
            "  pip install pytesseract pillow\n"
            "  brew install tesseract   # macOS"
        )
    pix = page.get_pixmap(dpi=300)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img)


def _load_docx(p: Path) -> str:
    try:
        from docx import Document
    except ImportError:
        sys.exit("Install python-docx:  pip install python-docx")
    doc = Document(str(p))
    return "\n".join(para.text for para in doc.paragraphs if para.text.strip())


def _load_pptx(p: Path) -> str:
    try:
        from pptx import Presentation
    except ImportError:
        sys.exit("Install python-pptx:  pip install python-pptx")
    prs = Presentation(str(p))
    slides = []
    for slide in prs.slides:
        slide_text = " ".join(
            shape.text for shape in slide.shapes if hasattr(shape, "text") and shape.text.strip()
        )
        if slide_text.strip():
            slides.append(slide_text)
    return "\n".join(slides)


def load_file(path: str) -> str:
    """Return plain text from a PDF, DOCX, PPTX, or plain-text file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = p.suffix.lower()
    if ext == ".pdf":
        return _load_pdf(p)
    if ext == ".docx":
        return _load_docx(p)
    if ext == ".pptx":
        return _load_pptx(p)

    # Fallback: plain text (.txt, .md, .csv, etc.)
    return p.read_text(encoding="utf-8", errors="replace")

# ─── Chunking ────────────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping word-count chunks.
    Overlap avoids cutting a sentence's context in half at boundaries.
    """
    words = text.split()
    chunks: list[str] = []
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

# ─── Embedding ───────────────────────────────────────────────────────────────

_embedder: Optional[SentenceTransformer] = None

def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print(f"[embed] Loading model '{EMBED_MODEL}' (first run downloads ~80 MB)...")
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder

def embed(texts: list[str]) -> np.ndarray:
    """Return L2-normalised float32 embeddings (n × dim)."""
    vecs = get_embedder().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(vecs, dtype="float32")

# ─── FAISS Index ─────────────────────────────────────────────────────────────

def _dim() -> int:
    return get_embedder().get_sentence_embedding_dimension()

def load_index() -> tuple[faiss.Index, list[str]]:
    """Load a previously saved index. Exits cleanly if not found."""
    faiss_file = f"{INDEX_PATH}.faiss"
    meta_file  = f"{INDEX_PATH}.pkl"
    if not Path(faiss_file).exists():
        sys.exit(f"No index found at '{faiss_file}'. Run:  python rag.py index <files>")
    index  = faiss.read_index(faiss_file)
    chunks = pickle.loads(Path(meta_file).read_bytes())
    return index, chunks

def save_index(index: faiss.Index, chunks: list[str]) -> None:
    faiss.write_index(index, f"{INDEX_PATH}.faiss")
    Path(f"{INDEX_PATH}.pkl").write_bytes(pickle.dumps(chunks))

def build_or_extend_index(new_chunks: list[str], new_vecs: np.ndarray) -> None:
    """Add vectors to an existing index, or create a new one."""
    faiss_file = f"{INDEX_PATH}.faiss"
    if Path(faiss_file).exists():
        index, existing_chunks = load_index()
        existing_chunks.extend(new_chunks)
    else:
        # IndexFlatIP = exact cosine similarity (on unit vectors)
        index = faiss.IndexFlatIP(_dim())
        existing_chunks = new_chunks[:]

    index.add(new_vecs)
    save_index(index, existing_chunks)
    return index.ntotal, existing_chunks

# ─── Retrieval ────────────────────────────────────────────────────────────────

def retrieve(query: str, k: int = TOP_K) -> list[tuple[float, str]]:
    """Return (score, chunk) pairs, best match first."""
    index, chunks = load_index()
    q_vec = embed([query])
    scores, indices = index.search(q_vec, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results.append((float(score), chunks[idx]))
    return results

# ─── LLM Call (Ollama) ───────────────────────────────────────────────────────

def call_llm(prompt: str) -> str:
    """Send prompt to a locally running Ollama instance and return the response."""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        sys.exit(
            f"\n[error] Cannot reach Ollama at {OLLAMA_URL}.\n"
            "  Start it with:  ollama serve\n"
            f"  Then pull a model:  ollama pull {OLLAMA_MODEL}\n"
        )

def build_prompt(query: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    return textwrap.dedent(f"""\
        You are a helpful assistant. Use only the context below to answer the question.
        If the answer is not in the context, say "I don't know based on the provided documents."

        CONTEXT:
        {context}

        QUESTION:
        {query}

        ANSWER:
    """)

# ─── CLI Commands ─────────────────────────────────────────────────────────────

def cmd_index(files: list[str]) -> None:
    all_chunks: list[str] = []

    for path in files:
        print(f"[index] Loading  {path}")
        with timer("load + chunk"):
            text   = load_file(path)
            chunks = chunk_text(text)
        print(f"[index]   → {len(chunks)} chunks")
        all_chunks.extend(chunks)

    if not all_chunks:
        sys.exit("[index] No chunks produced. Check your files.")

    print(f"[embed] Embedding {len(all_chunks)} chunks...")
    with timer(f"embed {len(all_chunks)} chunks"):
        vecs = embed(all_chunks)

    with timer("faiss add + save"):
        total, _ = build_or_extend_index(all_chunks, vecs)
    print(f"[index] Done. Index now holds {total} vectors → saved to '{INDEX_PATH}.*'")

def cmd_query(question: str) -> None:
    print(f"[query] Retrieving top-{TOP_K} chunks...")
    with timer("embed query + faiss search"):
        hits = retrieve(question, k=TOP_K)

    if not hits:
        sys.exit("[query] No results returned from index.")

    print(f"[query] Top match score: {hits[0][0]:.4f}\n")

    context_chunks = [chunk for _, chunk in hits]
    prompt = build_prompt(question, context_chunks)

    print("[llm]   Generating answer...\n")
    with timer("llm generation"):
        answer = call_llm(prompt)

    print("=" * 60)
    print(answer)
    print("=" * 60)

# ─── Entry Point ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal local RAG — FAISS + SentenceTransformers + Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python rag.py index report.pdf notes.txt
              python rag.py query "What are the key findings?"
        """),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Embed and index documents")
    p_index.add_argument("files", nargs="+", help="PDF or TXT files to index")

    p_query = sub.add_parser("query", help="Ask a question against indexed docs")
    p_query.add_argument("question", help="Natural language question")

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args.files)
    elif args.command == "query":
        cmd_query(args.question)

if __name__ == "__main__":
    main()

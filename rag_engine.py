"""
RAG query engine — retrieve context and generate answers locally.

Pipeline: search ChromaDB -> build prompt -> call Ollama (free local LLM).
Requires Ollama installed: https://ollama.com
  Example:  ollama pull llama3.2
"""

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

from document_loader import load_documents
from text_chunker import chunk_documents
from vector_store import (
    get_or_create_store,
    search_documents,
    store_documents,
)

load_dotenv()

# --- Config (override in .env) ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
DEFAULT_TOP_K = int(os.getenv("TOP_K", "3"))
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))


def build_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into one context block for the LLM.

    Args:
        chunks: Results from search_documents().

    Returns:
        A single string with numbered source excerpts.
    """
    if not chunks:
        return ""

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = os.path.basename(chunk.get("source", "unknown"))
        page = chunk.get("page", "?")
        content = chunk.get("content", "").strip()
        score = chunk.get("score", 0)

        parts.append(
            f"[{i}] (source: {source}, page: {page}, relevance: {score})\n{content}"
        )

    return "\n\n".join(parts)


def build_prompt(question: str, context: str) -> str:
    """
    Build the full prompt sent to the local LLM.

    Instructs the model to answer only from the provided context.
    """
    return f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the context does not contain enough information, say "I don't have enough information in the provided documents."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


def check_ollama_available() -> bool:
    """Return True if Ollama is running and reachable."""
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def generate_answer(prompt: str, model: str | None = None) -> str:
    """
    Send a prompt to Ollama and return the generated text.

    Args:
        prompt: Full prompt string.
        model: Ollama model name (default from .env).

    Returns:
        The model's answer text.
    """
    model_name = model or OLLAMA_MODEL
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"

    payload = json.dumps(
        {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
        }
    ).encode("utf-8")

    print(f"Calling Ollama model '{model_name}'...")

    try:
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as response:
            data = json.loads(response.read().decode("utf-8"))

        answer = data.get("response", "").strip()
        if not answer:
            return "No response received from Ollama."
        print("  Answer received.\n")
        return answer

    except urllib.error.URLError as e:
        msg = (
            f"Could not reach Ollama at {OLLAMA_BASE_URL}. "
            "Install from https://ollama.com and run: ollama serve"
        )
        print(f"  Error: {msg} ({e})")
        raise ConnectionError(msg) from e
    except Exception as e:
        print(f"  Error generating answer: {e}")
        raise


def index_documents(
    path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    db_path: str | None = None,
    collection_name: str | None = None,
) -> int:
    """
    Full indexing pipeline: load -> chunk -> embed -> store in ChromaDB.

    Args:
        path: File or folder path (same as document_loader).
        chunk_size: Characters per chunk.
        chunk_overlap: Overlap between chunks.
        db_path: ChromaDB folder (default from .env).
        collection_name: Chroma collection name (default from .env).

    Returns:
        Number of chunks stored.
    """
    print("=" * 60)
    print("Indexing documents")
    print("=" * 60 + "\n")

    docs = load_documents(path)
    if not docs:
        print("Nothing to index.")
        return 0

    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        print("No chunks created.")
        return 0

    _, collection = get_or_create_store(db_path=db_path, collection_name=collection_name)
    stored = store_documents(chunks, collection=collection)

    print(f"Indexing complete. {stored} chunk(s) in the vector store.\n")
    return stored


def ask(
    question: str,
    collection=None,
    top_k: int | None = None,
    use_llm: bool = True,
    db_path: str | None = None,
    collection_name: str | None = None,
) -> dict:
    """
    Ask a question against your indexed documents (full RAG flow).

    1. Retrieve similar chunks from ChromaDB
    2. Build a context prompt
    3. Optionally generate an answer with Ollama

    Args:
        question: User question.
        collection: Chroma collection (opens default if None).
        top_k: How many chunks to retrieve.
        use_llm: If False, return retrieval only (no Ollama call).
        db_path: ChromaDB folder if collection is None.
        collection_name: Collection name if collection is None.

    Returns:
        Dict with keys: question, answer, context, sources, chunks.
    """
    question = question.strip()
    if not question:
        print("Empty question.")
        return {"question": "", "answer": "", "context": "", "sources": [], "chunks": []}

    print("=" * 60)
    print(f"Question: {question}")
    print("=" * 60 + "\n")

    if collection is None:
        _, collection = get_or_create_store(db_path=db_path, collection_name=collection_name)

    # Step 1: retrieve
    chunks = search_documents(question, collection=collection, top_k=top_k)
    context = build_context(chunks)

    # Unique source file names for citations
    sources = []
    seen = set()
    for c in chunks:
        src = c.get("source", "")
        label = f"{os.path.basename(src)} (page {c.get('page', '?')})"
        if label not in seen:
            seen.add(label)
            sources.append(label)

    result = {
        "question": question,
        "answer": "",
        "context": context,
        "sources": sources,
        "chunks": chunks,
    }

    if not chunks:
        result["answer"] = "No relevant documents found. Index your files first with index_documents()."
        print(result["answer"] + "\n")
        return result

    if not use_llm:
        result["answer"] = "(Retrieval only — LLM skipped)"
        print("Retrieved context (LLM disabled):\n")
        print(context[:500] + ("..." if len(context) > 500 else ""))
        print()
        return result

    # Step 2 & 3: prompt + generate
    if not check_ollama_available():
        result["answer"] = (
            "Ollama is not running. Start it with 'ollama serve' and pull a model, e.g. "
            f"'ollama pull {OLLAMA_MODEL}'. Retrieved context is still available in 'context'."
        )
        print(result["answer"] + "\n")
        return result

    prompt = build_prompt(question, context)
    try:
        result["answer"] = generate_answer(prompt)
    except ConnectionError as e:
        result["answer"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Quick test:  python rag_engine.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import shutil
    import tempfile

    print("=" * 60)
    print("RAG Engine — self-test")
    print("=" * 60 + "\n")

    test_db = "./chroma_db_rag_test"
    if os.path.isdir(test_db):
        shutil.rmtree(test_db)

    with tempfile.TemporaryDirectory() as tmp_dir:
        sample = Path(tmp_dir) / "faq.txt"
        sample.write_text(
            "Our company was founded in 2020. "
            "We offer a free local RAG system using Python, ChromaDB, and Ollama. "
            "Support hours are Monday to Friday, 9am to 5pm.",
            encoding="utf-8",
        )

        print("1) Index sample document\n")
        index_documents(
            tmp_dir,
            chunk_size=60,
            chunk_overlap=10,
            db_path=test_db,
            collection_name="rag_test",
        )

        print("2) Ask a question\n")
        result = ask(
            "When was the company founded?",
            use_llm=check_ollama_available(),
            db_path=test_db,
            collection_name="rag_test",
        )

        print("Answer:")
        print("-" * 40)
        print(result["answer"])
        print()
        print("Sources:", ", ".join(result["sources"]) or "(none)")
        print()

        if not check_ollama_available():
            print(
                "Tip: Install Ollama and run 'ollama pull llama3.2' "
                "to enable full answer generation in this test."
            )

    if os.path.isdir(test_db):
        shutil.rmtree(test_db)

    print("=" * 60)
    print("Self-test finished.")

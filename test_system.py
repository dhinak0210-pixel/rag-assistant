#!/usr/bin/env python3
"""End-to-end test for the complete RAG system."""

import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def ok(msg):
    print(f"  ✅ {msg}")


def fail(msg):
    print(f"  ❌ {msg}")
    sys.exit(1)


def main():
    print("=" * 50)
    print("RAG System — full test")
    print("=" * 50)

    # 1. Imports
    print("\n1. Module imports")
    try:
        from document_loader import load_documents
        from text_chunker import chunk_documents
        from vector_store import get_or_create_store, search_documents
        from groq_chain import ask, GROQ_API_KEY
        from rag_pipeline import index_path, get_stats
        ok("All modules imported")
    except Exception as e:
        fail(f"Import error: {e}")

    # 2. Index sample
    print("\n2. Index sample document")
    test_db = "./chroma_db_test_run"
    os.environ["CHROMA_DB_PATH"] = test_db

    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "test.txt").write_text(
            "The magic number is 42. RAG means Retrieval Augmented Generation.",
            encoding="utf-8",
        )
        count = index_path(tmp, quiet=True)
    if count > 0:
        ok(f"Indexed {count} chunks")
    else:
        fail("Indexing returned 0")

    # 3. Search
    print("\n3. Vector search")
    _, col = get_or_create_store()
    hits = search_documents("magic number", collection=col, verbose=False)
    if hits:
        ok(f"Found {len(hits)} hit(s), top score={hits[0]['score']}")
    else:
        fail("Search returned no results")

    # 4. Groq answer
    print("\n4. Groq RAG answer")
    if not GROQ_API_KEY:
        print("  ⚠️  GROQ_API_KEY not set — skipping LLM test")
    else:
        r = ask("What is the magic number?", collection=col, verbose=False)
        if r["answer"] and "42" in r["answer"]:
            ok(f"Answer: {r['answer'][:80]}...")
        else:
            fail(f"Unexpected answer: {r['answer']}")

    # Cleanup test db
    import shutil
    if os.path.isdir(test_db):
        shutil.rmtree(test_db)

    print("\n" + "=" * 50)
    print("All tests passed! Run:  streamlit run app.py")
    print("=" * 50)


if __name__ == "__main__":
    main()

"""
Unified RAG pipeline — load, chunk, index, search, answer.

Use this module for a single entry point to the whole system.
"""

import contextlib
import io
import os

from dotenv import load_dotenv

load_dotenv()


@contextlib.contextmanager
def _quiet(quiet: bool):
    """Suppress stdout when quiet=True (for Streamlit UI)."""
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            yield
    else:
        yield


def index_path(
    path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    quiet: bool = False,
) -> int:
    """
    Full indexing: load -> chunk -> embed -> store.

    Args:
        path: File or folder with PDF/TXT/MD.
        chunk_size: Max characters per chunk.
        chunk_overlap: Overlap between chunks.
        quiet: Suppress terminal output.

    Returns:
        Number of chunks stored.
    """
    from document_loader import load_documents
    from text_chunker import chunk_documents
    from vector_store import get_or_create_store, store_documents

    with _quiet(quiet):
        docs = load_documents(path)
    if not docs:
        return 0

    with _quiet(quiet):
        chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        return 0

    with _quiet(quiet):
        _, collection = get_or_create_store()
        stored = store_documents(chunks, collection=collection)
    return stored


def search(question: str, top_k: int | None = None, quiet: bool = True) -> list[dict]:
    """Search ChromaDB for relevant chunks."""
    from vector_store import get_or_create_store, search_documents

    _, collection = get_or_create_store()
    with _quiet(quiet):
        return search_documents(question, collection=collection, top_k=top_k)


def answer(question: str, top_k: int | None = None, stream: bool = False, quiet: bool = True):
    """
    Ask a question (Groq RAG).

    Returns:
        If stream=False: dict with answer, sources, chunks.
        If stream=True: generator of tokens (use ask_stream from groq_chain).
    """
    from groq_chain import ask, ask_stream, retrieve_context
    from vector_store import get_or_create_store

    _, collection = get_or_create_store()
    with _quiet(quiet):
        ctx = retrieve_context(question, collection=collection, top_k=top_k)

    if stream:
        return ask_stream(
            question,
            collection=collection,
            top_k=top_k,
            verbose=False,
            chunks=ctx["chunks"],
        )

    with _quiet(quiet):
        result = ask(
            question,
            collection=collection,
            top_k=top_k,
            verbose=False,
            chunks=ctx["chunks"],
        )
    result["source_chunks"] = ctx["chunks"]
    return result


def get_stats() -> dict:
    """Return vector store statistics."""
    from vector_store import CHROMA_DB_PATH, COLLECTION_NAME, get_or_create_store

    _, collection = get_or_create_store()
    return {
        "chunks": collection.count(),
        "db_path": CHROMA_DB_PATH,
        "collection": COLLECTION_NAME,
    }

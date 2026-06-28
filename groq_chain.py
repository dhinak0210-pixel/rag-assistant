"""
RAG chain powered by Groq API (free tier).

Retrieves context from ChromaDB, then answers using Groq's hosted LLM.
Get a free API key at: https://console.groq.com
"""

import os
from typing import Generator

from dotenv import load_dotenv
from groq import Groq

from vector_store import get_or_create_store, search_documents

load_dotenv()

# --- Config ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
DEFAULT_TOP_K = int(os.getenv("TOP_K", "3"))
TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.1"))
MAX_HISTORY_TURNS = 3  # last 3 user/assistant pairs

# In-memory chat history for multi-turn conversations
_chat_history: list[dict] = []

# Cached Groq client
_groq_client: Groq | None = None

SYSTEM_PROMPT = """You are a precise document assistant for a RAG system.

Rules:
1. Answer ONLY using the CONTEXT provided in the user's message.
2. If the context does not contain the answer, say: "I don't have enough information in the provided documents."
3. Do not invent facts or use outside knowledge.
4. Be concise and accurate.
5. When relevant, mention which source numbers ([1], [2], etc.) support your answer."""


def _get_client() -> Groq:
    """Create or return the Groq API client."""
    global _groq_client
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to your .env file. "
            "Get a free key at https://console.groq.com"
        )
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


def build_context(chunks: list[dict]) -> str:
    """Format search hits into a numbered context block."""
    if not chunks:
        return "(No relevant documents found.)"

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = os.path.basename(chunk.get("source", "unknown"))
        page = chunk.get("page", "?")
        content = chunk.get("content", "").strip()
        score = chunk.get("score", 0)
        parts.append(
            f"[{i}] Source: {source} | Page: {page} | Relevance: {score}\n{content}"
        )
    return "\n\n".join(parts)


def extract_sources(chunks: list[dict]) -> list[str]:
    """Build a readable list of sources used in retrieval."""
    sources = []
    seen = set()
    for chunk in chunks:
        src = chunk.get("source", "")
        label = f"{os.path.basename(src)} (page {chunk.get('page', '?')}, score {chunk.get('score', 0)})"
        if label not in seen:
            seen.add(label)
            sources.append(label)
    return sources


def _get_history_messages() -> list[dict]:
    """Return the last 3 conversation turns as Groq message dicts."""
    # Each turn = 1 user + 1 assistant message (2 entries)
    max_messages = MAX_HISTORY_TURNS * 2
    return list(_chat_history[-max_messages:])


def _build_user_message(question: str, context: str) -> str:
    """User message with embedded retrieval context."""
    return f"""CONTEXT:
{context}

QUESTION:
{question}

Answer using only the context above. Cite source numbers like [1] when helpful."""


def _retrieve(question: str, collection=None, top_k: int | None = None) -> list[dict]:
    """Search the vector store for relevant chunks."""
    if collection is None:
        _, collection = get_or_create_store()
    return search_documents(question, collection=collection, top_k=top_k, verbose=False)


def _save_turn(question: str, answer: str) -> None:
    """Append this Q&A to history and trim to last 3 turns."""
    global _chat_history
    _chat_history.append({"role": "user", "content": question})
    _chat_history.append({"role": "assistant", "content": answer})
    max_messages = MAX_HISTORY_TURNS * 2
    _chat_history = _chat_history[-max_messages:]


def clear_chat_history() -> None:
    """Reset conversation memory."""
    global _chat_history
    _chat_history = []
    print("Chat history cleared.")


def retrieve_context(
    question: str,
    collection=None,
    top_k: int | None = None,
) -> dict:
    """
    Retrieve relevant chunks without calling the LLM.

    Returns:
        Dict with chunks, sources, and context keys.
    """
    chunks = _retrieve(question, collection=collection, top_k=top_k)
    return {
        "chunks": chunks,
        "sources": extract_sources(chunks),
        "context": build_context(chunks),
    }


def ask(
    question: str,
    collection=None,
    top_k: int | None = None,
    verbose: bool = True,
    chunks: list[dict] | None = None,
    history: list[dict] | None = None,
    system_prompt: str | None = None,
) -> dict:
    """
    Ask a question: retrieve context, call Groq, return answer + sources.

    Args:
        question: User question.
        collection: Optional ChromaDB collection.
        top_k: Number of chunks to retrieve.
        verbose: Print progress to the terminal.
        chunks: Optional pre-retrieved chunks.
        history: Optional session-isolated chat history.

    Returns:
        Dict with keys: question, answer, sources, chunks, context.
    """
    question = question.strip()
    if not question:
        return {
            "question": "",
            "answer": "Please provide a question.",
            "sources": [],
            "chunks": [],
            "context": "",
        }

    if verbose:
        print("=" * 60)
        print(f"Question: {question}")
        print("=" * 60 + "\n")

    # Step 1: retrieve (skip if chunks already provided)
    if chunks is None:
        chunks = _retrieve(question, collection=collection, top_k=top_k)
    context = build_context(chunks)
    sources = extract_sources(chunks)

    result = {
        "question": question,
        "answer": "",
        "sources": sources,
        "chunks": chunks,
        "context": context,
    }

    if not chunks:
        result["answer"] = (
            "No relevant documents found. Index your files first "
            "(see rag_engine.index_documents or vector_store.store_documents)."
        )
        if verbose:
            print(result["answer"] + "\n")
        return result

    if verbose:
        print("Sources used:")
        for s in sources:
            print(f"  - {s}")
        print()

    # Step 2: build messages (system + history + current question with context)
    chat_history = history if history is not None else _chat_history
    max_messages = MAX_HISTORY_TURNS * 2
    history_turns = chat_history[-max_messages:]
    
    sys_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": sys_prompt},
    ]
    for msg in history_turns:
        messages.append({"role": msg["role"], "content": msg["content"]})
        
    messages.append({"role": "user", "content": _build_user_message(question, context)})

    # Step 3: call Groq
    try:
        client = _get_client()
        if verbose:
            print(f"Calling Groq ({GROQ_MODEL}, temperature={TEMPERATURE})...\n")

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=TEMPERATURE,
        )

        answer = response.choices[0].message.content or ""
        result["answer"] = answer.strip()
        
        if history is None:
            _save_turn(question, result["answer"])

        if verbose:
            print("Answer:")
            print("-" * 40)
            print(result["answer"])
            print()

    except ValueError as e:
        result["answer"] = str(e)
        if verbose:
            print(f"Error: {e}\n")
    except Exception as e:
        error_name = type(e).__name__
        result["answer"] = f"Groq API error ({error_name}): {e}"
        if verbose:
            print(f"  API Error: {e}\n")

    return result


def ask_stream(
    question: str,
    collection=None,
    top_k: int | None = None,
    verbose: bool = True,
    chunks: list[dict] | None = None,
    history: list[dict] | None = None,
    system_prompt: str | None = None,
) -> Generator[str, None, dict]:
    """
    Stream the answer token by token from Groq.

    Yields:
        Text tokens as they arrive.

    Returns (via generator return value):
        Final result dict with question, answer, sources, chunks, context.
        Note: use manual accumulation when iterating; see self-test below.

    Example:
        tokens = []
        gen = ask_stream("What is RAG?")
        for token in gen:
            print(token, end="", flush=True)
            tokens.append(token)
    """
    question = question.strip()
    empty_result = {
        "question": question,
        "answer": "",
        "sources": [],
        "chunks": [],
        "context": "",
    }

    if not question:
        yield "Please provide a question."
        return empty_result

    if verbose:
        print("=" * 60)
        print(f"Question (streaming): {question}")
        print("=" * 60 + "\n")

    if chunks is None:
        chunks = _retrieve(question, collection=collection, top_k=top_k)
    context = build_context(chunks)
    sources = extract_sources(chunks)

    result = {
        "question": question,
        "answer": "",
        "sources": sources,
        "chunks": chunks,
        "context": context,
    }

    if not chunks:
        msg = "No relevant documents found. Index your files first."
        result["answer"] = msg
        yield msg
        return result

    if verbose:
        print("Sources used:")
        for s in sources:
            print(f"  - {s}")
        print()

    chat_history = history if history is not None else _chat_history
    max_messages = MAX_HISTORY_TURNS * 2
    history_turns = chat_history[-max_messages:]

    sys_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": sys_prompt},
    ]
    for msg in history_turns:
        messages.append({"role": msg["role"], "content": msg["content"]})
        
    messages.append({"role": "user", "content": _build_user_message(question, context)})

    full_answer = []

    try:
        client = _get_client()
        if verbose:
            print(f"Streaming from Groq ({GROQ_MODEL})...\n")

        stream = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            stream=True,
        )

        for event in stream:
            delta = event.choices[0].delta
            token = getattr(delta, "content", None) or ""
            if token:
                full_answer.append(token)
                yield token

        result["answer"] = "".join(full_answer).strip()
        
        if history is None:
            _save_turn(question, result["answer"])
            
        if verbose:
            print("\n")

    except ValueError as e:
        err = str(e)
        result["answer"] = err
        yield err
    except Exception as e:
        err = f"Groq API error ({type(e).__name__}): {e}"
        result["answer"] = err
        yield err

    return result


# ---------------------------------------------------------------------------
# Quick test:  python groq_chain.py
# Requires GROQ_API_KEY in .env and indexed documents in chroma_db
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import shutil
    import tempfile
    from pathlib import Path

    from rag_engine import index_documents

    print("=" * 60)
    print("Groq RAG Chain — self-test")
    print("=" * 60 + "\n")

    if not GROQ_API_KEY:
        print("Set GROQ_API_KEY in .env to run the full test.")
        print("Get a free key: https://console.groq.com\n")
    else:
        print(f"API key found. Model: {GROQ_MODEL}\n")

    test_db = "./chroma_db_groq_test"
    if os.path.isdir(test_db):
        shutil.rmtree(test_db)

    with tempfile.TemporaryDirectory() as tmp_dir:
        sample = Path(tmp_dir) / "company.txt"
        sample.write_text(
            "Acme Corp was founded in 2019. "
            "Our flagship product is a free RAG toolkit built with Python and Groq. "
            "Headquarters are in Austin, Texas.",
            encoding="utf-8",
        )

        print("1) Index sample document\n")
        index_documents(
            tmp_dir,
            chunk_size=60,
            chunk_overlap=10,
            db_path=test_db,
            collection_name="groq_test",
        )

        _, collection = get_or_create_store(db_path=test_db, collection_name="groq_test")

        if GROQ_API_KEY:
            clear_chat_history()

            print("2) ask() — non-streaming\n")
            out = ask("When was Acme Corp founded?", collection=collection)
            print("Returned sources:", out["sources"])
            print()

            print("3) ask_stream() — streaming\n")
            print("Stream: ", end="")
            collected = []
            for piece in ask_stream("Where is the headquarters?", collection=collection):
                print(piece, end="", flush=True)
                collected.append(piece)
            print("\n")
        else:
            print("2) Skipping Groq calls (no API key)\n")
            chunks = _retrieve("When was Acme founded?", collection=collection)
            print("Retrieval-only test:")
            print(build_context(chunks))

    if os.path.isdir(test_db):
        shutil.rmtree(test_db)

    print("=" * 60)
    print("Self-test finished.")

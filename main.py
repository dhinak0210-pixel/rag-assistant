#!/usr/bin/env python3
"""
RAG System — command-line interface.

Usage:
  python main.py index ./my_documents
  python main.py ask "What is this document about?"
  python main.py chat
  python main.py status
"""

import argparse
import sys

from dotenv import load_dotenv

load_dotenv()


def cmd_index(args) -> int:
    """Load, chunk, and store documents in ChromaDB."""
    from rag_engine import index_documents

    count = index_documents(
        args.path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    if count == 0:
        print("Indexing failed or no documents found.")
        return 1
    print(f"Ready. {count} chunk(s) indexed. You can now ask questions.")
    return 0


def cmd_ask(args) -> int:
    """Ask a single question (Groq)."""
    from groq_chain import ask
    from vector_store import get_or_create_store

    _, collection = get_or_create_store()
    if collection.count() == 0:
        print("No documents indexed. Run: python main.py index <folder>")
        return 1

    result = ask(args.question, collection=collection, top_k=args.top_k)
    if args.quiet:
        print(result["answer"])
        if result["sources"]:
            print("\nSources:", file=sys.stderr)
            for s in result["sources"]:
                print(f"  - {s}", file=sys.stderr)
    return 0 if result["answer"] and "error" not in result["answer"].lower()[:20] else 1


def cmd_chat(args) -> int:
    """Interactive chat loop with optional streaming."""
    from groq_chain import ask, ask_stream, clear_chat_history
    from vector_store import get_or_create_store

    _, collection = get_or_create_store()
    if collection.count() == 0:
        print("No documents indexed. Run: python main.py index <folder>")
        return 1

    clear_chat_history()
    print("RAG Chat (Groq) — type 'quit' or 'exit' to stop")
    print("Commands: /clear (reset history), /stream on|off")
    print("-" * 50)

    use_stream = args.stream

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue

        lower = question.lower()
        if lower in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        if lower == "/clear":
            clear_chat_history()
            print("Chat history cleared.")
            continue

        if lower.startswith("/stream"):
            parts = lower.split()
            if len(parts) == 2 and parts[1] in ("on", "off"):
                use_stream = parts[1] == "on"
                print(f"Streaming {'enabled' if use_stream else 'disabled'}.")
            else:
                print("Usage: /stream on  or  /stream off")
            continue

        if use_stream:
            print("\nAssistant: ", end="", flush=True)
            for token in ask_stream(question, collection=collection, top_k=args.top_k):
                print(token, end="", flush=True)
            print()
        else:
            ask(question, collection=collection, top_k=args.top_k)

    return 0


def cmd_status(args) -> int:
    """Show vector store status."""
    from vector_store import get_or_create_store, CHROMA_DB_PATH, COLLECTION_NAME

    _, collection = get_or_create_store()
    count = collection.count()
    print(f"Database:  {CHROMA_DB_PATH}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Chunks:    {count}")
    if count == 0:
        print("\nIndex documents with: python main.py index <folder>")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Local RAG system — index documents and ask questions (Groq free tier).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py index ./documents
  python main.py ask "Summarize the main topics"
  python main.py chat
  python main.py status
        """,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # index
    p_index = sub.add_parser("index", help="Index a file or folder into ChromaDB")
    p_index.add_argument("path", help="Path to PDF/TXT/MD file or folder")
    p_index.add_argument("--chunk-size", type=int, default=500)
    p_index.add_argument("--chunk-overlap", type=int, default=50)
    p_index.set_defaults(func=cmd_index)

    # ask
    p_ask = sub.add_parser("ask", help="Ask one question")
    p_ask.add_argument("question", help="Your question")
    p_ask.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve")
    p_ask.add_argument("-q", "--quiet", action="store_true", help="Print answer only")
    p_ask.set_defaults(func=cmd_ask)

    # chat
    p_chat = sub.add_parser("chat", help="Interactive chat session")
    p_chat.add_argument("--stream", action="store_true", help="Stream responses by default")
    p_chat.add_argument("--top-k", type=int, default=None)
    p_chat.set_defaults(func=cmd_chat)

    # status
    p_status = sub.add_parser("status", help="Show index status")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

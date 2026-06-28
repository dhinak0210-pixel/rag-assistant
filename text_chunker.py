"""
Text chunker for RAG applications.

Takes documents from document_loader (content, source, page) and splits them
into smaller overlapping chunks suitable for embedding and retrieval.
"""

import os

# Default sizes — tune these for your model and use case
DEFAULT_CHUNK_SIZE = 500      # max characters per chunk
DEFAULT_CHUNK_OVERLAP = 50    # characters shared between neighboring chunks


def split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """
    Split a string into overlapping chunks by character count.

    Tries to break at spaces so words are not cut in half.

    Args:
        text: The full text to split.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: How many characters repeat between consecutive chunks.

    Returns:
        List of text chunk strings.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    # Short text fits in one chunk — no splitting needed
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Where this chunk would end if we took a full window
        end = min(start + chunk_size, text_length)

        # If we are not at the end yet, snap back to the last space (word boundary)
        if end < text_length:
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Reached the end of the document
        if end >= text_length:
            break

        # Move forward, but step back by overlap so context carries over
        next_start = end - chunk_overlap
        if next_start <= start:
            # Safety: always advance at least one character to avoid an infinite loop
            next_start = end if end > start else start + 1

        start = next_start

    return chunks


def chunk_document(
    document: dict,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split one loaded document dict into smaller chunk dicts.

    Args:
        document: Dict with 'content', 'source', and 'page' (from document_loader).
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of chunk dicts with content, source, page, and chunk_index.
    """
    chunks_out = []

    try:
        content = document.get("content", "")
        source = document.get("source", "unknown")
        page = document.get("page", 1)

        if not isinstance(content, str):
            print(f"  Warning: Skipping document with non-string content from {source}")
            return chunks_out

        text_pieces = split_text(content, chunk_size, chunk_overlap)

        for index, piece in enumerate(text_pieces):
            chunks_out.append(
                {
                    "content": piece,
                    "source": source,
                    "page": page,
                    "chunk_index": index,
                }
            )

    except Exception as e:
        source = document.get("source", "unknown")
        print(f"  Error: Failed to chunk document from '{source}': {e}")

    return chunks_out


def chunk_documents(
    documents: list[dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """
    Chunk a list of documents (output from document_loader.load_documents).

    Args:
        documents: List of dicts with 'content', 'source', and 'page'.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        Flat list of all chunk dicts ready for embedding.
    """
    all_chunks = []

    if not documents:
        print("No documents to chunk.")
        return all_chunks

    print(f"\nChunking {len(documents)} document(s)")
    print(f"  chunk_size={chunk_size}, chunk_overlap={chunk_overlap}\n")

    for i, doc in enumerate(documents, start=1):
        source_name = os.path.basename(doc.get("source", "unknown"))
        page = doc.get("page", "?")

        try:
            doc_chunks = chunk_document(doc, chunk_size, chunk_overlap)

            if not doc_chunks:
                print(f"[{i}/{len(documents)}] Skipped (empty): {source_name} page {page}")
                continue

            all_chunks.extend(doc_chunks)
            print(
                f"[{i}/{len(documents)}] {source_name} page {page} "
                f"-> {len(doc_chunks)} chunk(s)"
            )

        except Exception as e:
            print(f"[{i}/{len(documents)}] Error: {source_name} page {page}: {e}")

    print(f"\nDone. Created {len(all_chunks)} chunk(s) total.\n")
    return all_chunks


# ---------------------------------------------------------------------------
# Quick test:  python text_chunker.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    from document_loader import load_documents

    print("=" * 60)
    print("Text Chunker — self-test")
    print("=" * 60)

    # Long sample text so we get multiple chunks (chunk_size=100 for demo)
    long_paragraph = (
        "Retrieval-Augmented Generation combines search with language models. "
        "First you load documents, then you split them into chunks. "
        "Each chunk gets an embedding vector stored in a database. "
        "When a user asks a question, you find similar chunks and pass them "
        "to the model as context. This helps answers stay grounded in your data."
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        sample_file = Path(tmp_dir) / "rag_intro.txt"
        sample_file.write_text(long_paragraph, encoding="utf-8")

        print(f"\nTest file: {sample_file}\n")

        # Step 1: load (previous pipeline step)
        documents = load_documents(tmp_dir)

        # Step 2: chunk with a small size so the test shows multiple chunks
        chunks = chunk_documents(documents, chunk_size=100, chunk_overlap=20)

        print("Sample chunks:")
        print("-" * 40)
        for c in chunks:
            preview = c["content"][:70].replace("\n", " ")
            if len(c["content"]) > 70:
                preview += "..."
            print(f"  chunk_index: {c['chunk_index']}")
            print(f"  page:        {c['page']}")
            print(f"  content:     {preview}")
            print()

        print(f"Total chunks in test: {len(chunks)}")
        print("=" * 60)
        print("Self-test finished. Use chunk_documents() after load_documents().")

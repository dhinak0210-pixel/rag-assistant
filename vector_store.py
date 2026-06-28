"""
Embedding and vector store for RAG (free, local).

Uses sentence-transformers (all-MiniLM-L6-v2) for embeddings and ChromaDB
for persistent local storage. No OpenAI or paid APIs required.
"""

import hashlib
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load settings from .env file in the project folder (if it exists)
load_dotenv()

# --- Config (override in .env) ---
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_documents")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_TOP_K = int(os.getenv("TOP_K", "3"))

# Cached model so we only load it once per program run
_embedding_model = None


def _get_embedding_model() -> SentenceTransformer:
    """Load the sentence-transformers model once and reuse it."""
    global _embedding_model
    if _embedding_model is None:
        print(f"Loading embedding model '{EMBEDDING_MODEL}' (first run may download weights)...")
        try:
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            print("  Model loaded successfully.\n")
        except Exception as e:
            print(f"  Error: Could not load model '{EMBEDDING_MODEL}': {e}")
            raise
    return _embedding_model


def create_embeddings(texts: list[str], show_progress: bool = True) -> list[list[float]]:
    """
    Turn a list of text strings into embedding vectors.

    Args:
        texts: List of chunk texts to embed.
        show_progress: Print progress messages.

    Returns:
        List of embedding vectors (each vector is a list of floats).
    """
    if not texts:
        print("No texts to embed.")
        return []

    # Filter out empty strings but keep track for alignment if needed
    valid_texts = [t.strip() for t in texts if t and t.strip()]
    if not valid_texts:
        print("All texts were empty — nothing to embed.")
        return []

    if show_progress:
        print(f"Creating embeddings for {len(valid_texts)} text(s)...")

    try:
        model = _get_embedding_model()
        # encode returns a numpy array; convert to plain Python lists for ChromaDB
        vectors = model.encode(valid_texts, show_progress_bar=show_progress)
        embeddings = [vec.tolist() for vec in vectors]

        if show_progress:
            dim = len(embeddings[0]) if embeddings else 0
            print(f"  Done. {len(embeddings)} embedding(s), dimension={dim}\n")

        return embeddings

    except Exception as e:
        print(f"  Error during embedding: {e}")
        return []


def get_or_create_store(
    db_path: str | None = None,
    collection_name: str | None = None,
) -> tuple[chromadb.PersistentClient, chromadb.Collection]:
    """
    Open (or create) a persistent ChromaDB database and collection.

    Data is saved under ./chroma_db by default.

    Args:
        db_path: Folder for ChromaDB files (default from .env or ./chroma_db).
        collection_name: Name of the collection (default from .env).

    Returns:
        Tuple of (client, collection).
    """
    path = db_path or CHROMA_DB_PATH
    name = collection_name or COLLECTION_NAME

    # Ensure the database folder exists
    Path(path).mkdir(parents=True, exist_ok=True)

    print(f"Connecting to ChromaDB at: {os.path.abspath(path)}")
    print(f"  Collection: {name}")

    try:
        client = chromadb.PersistentClient(path=path)
        # cosine works well with sentence-transformers normalized vectors
        collection = client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        try:
            count = collection.count()
        except Exception as e:
            print(f"  Warning: Collection '{name}' index is corrupted or missing: {e}")
            print(f"  Recreating collection '{name}' to restore consistency...")
            try:
                client.delete_collection(name=name)
            except Exception as delete_err:
                print(f"  Could not delete collection: {delete_err}")
            collection = client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
            count = collection.count()
            
        print(f"  Collection ready ({count} document(s) stored).\n")
        return client, collection

    except Exception as e:
        print(f"  Error: Could not open ChromaDB: {e}")
        raise


def _make_chunk_id(chunk: dict) -> str:
    """Build a stable unique ID from source + page + chunk_index."""
    key = f"{chunk.get('source', '')}|{chunk.get('page', 0)}|{chunk.get('chunk_index', 0)}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def store_documents(
    chunks: list[dict],
    collection: chromadb.Collection | None = None,
) -> int:
    """
    Embed chunk dicts and save them in ChromaDB.

    Each chunk should have: content, source, page, chunk_index
    (as produced by text_chunker.chunk_documents).

    Args:
        chunks: List of chunk dictionaries.
        collection: Chroma collection (creates one if None).

    Returns:
        Number of chunks successfully stored.
    """
    if not chunks:
        print("No chunks to store.")
        return 0

    if collection is None:
        _, collection = get_or_create_store()

    # Prepare parallel lists for ChromaDB batch add
    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        content = chunk.get("content", "").strip()
        if not content:
            continue

        ids.append(_make_chunk_id(chunk))
        documents.append(content)
        metadatas.append(
            {
                "source": str(chunk.get("source", "")),
                "page": int(chunk.get("page", 1)),
                "chunk_index": int(chunk.get("chunk_index", 0)),
            }
        )

    if not documents:
        print("No non-empty chunks to store.")
        return 0

    print(f"Storing {len(documents)} chunk(s) in ChromaDB...")

    try:
        embeddings = create_embeddings(documents, show_progress=True)
        if len(embeddings) != len(documents):
            print("  Error: Embedding count mismatch — storage aborted.")
            return 0

        # upsert avoids duplicate ID errors if you re-index the same files
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        print(f"  Stored {len(documents)} chunk(s). Total in collection: {collection.count()}\n")
        return len(documents)

    except Exception as e:
        print(f"  Error storing documents: {e}")
        return 0


def search_documents(
    query: str,
    collection: chromadb.Collection | None = None,
    top_k: int | None = None,
    verbose: bool = True,
) -> list[dict]:
    """
    Find the most similar chunks to a natural-language query.

    Args:
        query: User question or search text.
        collection: Chroma collection (opens default store if None).
        top_k: How many results to return (default from .env TOP_K or 3).
        verbose: Print progress messages.

    Returns:
        List of dicts with content, source, page, chunk_index, score, distance, id.
        Higher score means more similar (cosine-based).
    """
    k = top_k if top_k is not None else DEFAULT_TOP_K
    query = query.strip()

    if not query:
        if verbose:
            print("Empty query — no search performed.")
        return []

    if collection is None:
        _, collection = get_or_create_store()

    try:
        count = collection.count()
    except Exception as e:
        if verbose:
            print(f"  Warning: search_documents failed because collection count raised an error: {e}")
            print("  Re-initializing store...")
        _, collection = get_or_create_store()
        count = collection.count()

    if count == 0:
        if verbose:
            print("Collection is empty. Run store_documents() first.")
        return []

    if verbose:
        print(f"Searching for: \"{query}\"")
        print(f"  Returning top {k} result(s)...")

    try:
        query_embedding = create_embeddings([query], show_progress=False)
        if not query_embedding:
            if verbose:
                print("  Error: Could not embed query.")
            return []

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        # Chroma returns nested lists (one list per query)
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        hits = []
        for i, doc_id in enumerate(ids):
            distance = distances[i] if i < len(distances) else 0.0
            # Cosine distance: 0 = identical. Convert to a friendly similarity score.
            score = max(0.0, 1.0 - distance)

            meta = metadatas[i] if i < len(metadatas) else {}
            hits.append(
                {
                    "id": doc_id,
                    "content": documents[i] if i < len(documents) else "",
                    "source": meta.get("source", ""),
                    "page": meta.get("page", 1),
                    "chunk_index": meta.get("chunk_index", 0),
                    "distance": round(float(distance), 4),
                    "score": round(float(score), 4),
                }
            )

        if verbose:
            print(f"  Found {len(hits)} match(es).\n")
        return hits

    except Exception as e:
        if verbose:
            print(f"  Error during search: {e}")
        return []


# ---------------------------------------------------------------------------
# Quick test:  python vector_store.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import shutil
    import tempfile

    from document_loader import load_documents
    from text_chunker import chunk_documents

    print("=" * 60)
    print("Vector Store — self-test")
    print("=" * 60)

    # Use a separate test DB so we don't overwrite your real data
    test_db = "./chroma_db_test"
    if os.path.isdir(test_db):
        shutil.rmtree(test_db)

    with tempfile.TemporaryDirectory() as tmp_dir:
        sample = Path(tmp_dir) / "knowledge.txt"
        sample.write_text(
            "Python is a popular programming language. "
            "RAG systems retrieve relevant documents before generating answers. "
            "ChromaDB stores vectors locally without cloud APIs. "
            "Sentence transformers create embeddings for free on your machine.",
            encoding="utf-8",
        )

        print(f"\n1) Load and chunk sample file\n")
        docs = load_documents(tmp_dir)
        chunks = chunk_documents(docs, chunk_size=80, chunk_overlap=10)

        print("2) Create store and save chunks\n")
        _, collection = get_or_create_store(db_path=test_db, collection_name="test_rag")
        stored = store_documents(chunks, collection=collection)
        print(f"   Stored count: {stored}")

        print("3) Search\n")
        results = search_documents(
            "What database stores vectors locally?",
            collection=collection,
            top_k=2,
        )

        print("Top results:")
        print("-" * 40)
        for rank, hit in enumerate(results, start=1):
            preview = hit["content"][:60].replace("\n", " ")
            if len(hit["content"]) > 60:
                preview += "..."
            print(f"  #{rank} score={hit['score']} distance={hit['distance']}")
            print(f"      {preview}")
            print()

    # Clean up test database
    if os.path.isdir(test_db):
        shutil.rmtree(test_db)

    print("=" * 60)
    print("Self-test finished. Production DB path:", os.path.abspath(CHROMA_DB_PATH))

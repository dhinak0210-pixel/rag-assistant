import os
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from src.embeddings import embed_documents, embed_query

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_documents")

_client = None
_collection = None

def get_chroma_client():
    global _client, _collection
    if _client is None:
        try:
            os.makedirs(CHROMA_PATH, exist_ok=True)
            _client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))
            _collection = _client.get_or_create_collection(name=COLLECTION_NAME)
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
    return _collection

def store_documents(chunks):
    """
    Stores chunks in ChromaDB. Automatically creates embeddings.
    """
    if not chunks:
        print("No chunks to store.")
        return
        
    try:
        collection = get_chroma_client()
        if collection is None:
            return
            
        texts = [chunk['content'] for chunk in chunks]
        ids = [chunk['chunk_id'] for chunk in chunks]
        metadatas = [{
            "source": chunk.get('source', ''),
            "filename": chunk.get('filename', ''),
            "page": chunk.get('page', 1),
            "chunk_num": chunk.get('chunk_num', 1)
        } for chunk in chunks]
        
        # We manually embed to have control and show progress
        embeddings = embed_documents(texts).tolist()
        
        # Batch add to chroma to avoid size limits
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end_idx = min(i + batch_size, len(ids))
            collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                metadatas=metadatas[i:end_idx],
                documents=texts[i:end_idx]
            )
            print(f"Stored chunks {i} to {end_idx} out of {len(ids)}")
            
        print(f"Successfully stored {len(chunks)} chunks in ChromaDB.")
    except Exception as e:
        print(f"Error storing documents in ChromaDB: {e}")

def search(query, top_k=5):
    """
    Searches ChromaDB for similar chunks.
    """
    results_out = []
    try:
        collection = get_chroma_client()
        if collection is None:
            return results_out
            
        query_embedding = embed_query(query).tolist()
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        if results and results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                results_out.append({
                    "content": results['documents'][0][i],
                    "score": 1.0 - results['distances'][0][i] if 'distances' in results and results['distances'] else 0.0,
                    "source": results['metadatas'][0][i].get('source', ''),
                    "filename": results['metadatas'][0][i].get('filename', ''),
                    "page": results['metadatas'][0][i].get('page', 1)
                })
    except Exception as e:
        print(f"Error searching ChromaDB: {e}")
        
    return results_out

def get_stats():
    """Returns database stats."""
    try:
        collection = get_chroma_client()
        if collection:
            count = collection.count()
            return {"status": "ok", "total_chunks": count}
    except Exception as e:
        print(f"Error getting ChromaDB stats: {e}")
    return {"status": "error", "total_chunks": 0}

def clear():
    """Deletes all documents from the collection."""
    try:
        global _client, _collection
        if _client is None:
            _client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))
        try:
            _client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            pass # Ignore if doesn't exist
        _client = None
        _collection = None
        print("ChromaDB collection cleared successfully.")
    except Exception as e:
        print(f"Error clearing ChromaDB: {e}")

if __name__ == "__main__":
    print("Testing vectorstore.py")
    stats = get_stats()
    print(f"Current stats: {stats}")

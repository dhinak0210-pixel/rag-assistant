import chromadb
import os
from .embeddings import EmbeddingModel

class VectorStore:
    def __init__(self, path="./chroma_db"):
        os.makedirs(path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=path)
        self._init_collection()
        self.embedding_model = EmbeddingModel()
        print(f"Storage location: {path}")

    def _init_collection(self):
        """Initializes the collection and ensures it is healthy and not missing index files."""
        try:
            self.collection = self.client.get_or_create_collection(name="rag_docs")
            # Verify collection is healthy on disk by calling count()
            self.collection.count()
        except Exception as e:
            print(f"Warning: Collection index is corrupted or missing: {e}")
            print("Recreating collection 'rag_docs' to restore consistency...")
            try:
                self.client.delete_collection("rag_docs")
            except Exception as delete_err:
                print(f"Could not delete collection: {delete_err}")
            self.collection = self.client.get_or_create_collection(name="rag_docs")

    def store(self, chunks: list):
        self.clear()
        
        ids = []
        documents = []
        embeddings = []
        metadatas = []
        
        print("Generating embeddings for chunks...")
        texts = [c["content"] for c in chunks]
        vectors = self.embedding_model.embed_texts(texts)
        
        for i, chunk in enumerate(chunks):
            ids.append(chunk["chunk_id"])
            documents.append(chunk["content"])
            embeddings.append(vectors[i].tolist())
            metadatas.append({
                "source": chunk["source"],
                "filename": chunk["filename"],
                "page": str(chunk.get("page", "1"))
            })
            
        batch_size = 50
        total_stored = 0
        for i in range(0, len(ids), batch_size):
            self.collection.add(
                ids=ids[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size],
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )
            total_stored += len(ids[i:i+batch_size])
            print(f"Stored {total_stored}/{len(ids)} chunks...")
            
        return total_stored

    def search(self, query: str, top_k=5):
        # Self-heal if count fails due to a missing/corrupted collection index
        try:
            count = self.collection.count()
        except Exception as e:
            print(f"Search failed because collection count raised an error: {e}")
            print("Self-healing: Reinitializing the collection...")
            self._init_collection()
            count = self.collection.count()

        if count == 0:
            return []
            
        q_emb = self.embedding_model.embed_query(query).tolist()
        results = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k
        )
        
        formatted_results = []
        if results['documents'] and len(results['documents']) > 0:
            for i in range(len(results['documents'][0])):
                meta = results['metadatas'][0][i] if results['metadatas'] else {}
                dist = results['distances'][0][i] if results['distances'] else 0.0
                formatted_results.append({
                    "content": results['documents'][0][i],
                    "source": meta.get("source", ""),
                    "filename": meta.get("filename", ""),
                    "page": meta.get("page", "1"),
                    "score": 1.0 / (1.0 + dist)
                })
        return formatted_results

    def get_stats(self):
        try:
            count = self.collection.count()
        except Exception as e:
            print(f"get_stats failed: {e}. Reinitializing...")
            self._init_collection()
            count = self.collection.count()
        return {"total": count, "ready": count > 0}

    def clear(self):
        try:
            self.client.delete_collection("rag_docs")
            self.collection = self.client.get_or_create_collection(name="rag_docs")
        except Exception as e:
            print(f"Error clearing collection: {e}")
            self.collection = self.client.get_or_create_collection(name="rag_docs")

_vector_store = None

def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store

def clear():
    get_vector_store().clear()

def store_documents(chunks):
    return get_vector_store().store(chunks)

def search(query, top_k=5):
    return get_vector_store().search(query, top_k)

def get_stats():
    return get_vector_store().get_stats()

def get_chroma_client():
    vs_instance = get_vector_store()
    try:
        vs_instance.collection.count()
    except Exception as e:
        print(f"get_chroma_client caught error: {e}. Reinitializing...")
        vs_instance._init_collection()
    return vs_instance.collection

if __name__ == "__main__":
    vs = VectorStore()
    sample_chunks = [{
        "chunk_id": "test_1",
        "content": "This is test content for the vector store.",
        "source": "test.txt",
        "filename": "test.txt",
        "page": 1
    }]
    vs.store(sample_chunks)
    res = vs.search("test")
    print(res)

import chromadb
import os
from .embeddings import EmbeddingModel

class VectorStore:
    def __init__(self, path="./chroma_db"):
        os.makedirs(path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(name="rag_docs")
        self.embedding_model = EmbeddingModel()
        print(f"Storage location: {path}")

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
        if self.collection.count() == 0:
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
        count = self.collection.count()
        return {"total": count, "ready": count > 0}

    def clear(self):
        try:
            self.client.delete_collection("rag_docs")
            self.collection = self.client.create_collection("rag_docs")
        except:
            pass

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

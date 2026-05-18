from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingModel:
    def __init__(self):
        print("Loading FREE local embedding model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.dimension = 384
        print(f"Dimension size: {self.dimension}")

    def embed_texts(self, texts: list):
        embeddings = self.model.encode(
            texts, 
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=True
        )
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, query: str):
        embedding = self.model.encode(
            query, 
            normalize_embeddings=True
        )
        return np.array(embedding, dtype=np.float32)

if __name__ == "__main__":
    em = EmbeddingModel()
    texts = ["I love AI", "Machine learning is great", "Apples are tasty"]
    query = "Tell me about artificial intelligence"
    
    doc_embs = em.embed_texts(texts)
    q_emb = em.embed_query(query)
    
    similarities = np.dot(doc_embs, q_emb)
    best_idx = np.argmax(similarities)
    print(f"Query: {query}")
    print(f"Most similar text: {texts[best_idx]} (Score: {similarities[best_idx]:.4f})")

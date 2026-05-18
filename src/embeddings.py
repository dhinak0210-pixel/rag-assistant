import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Uses FREE local model: all-MiniLM-L6-v2
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

_model = None

def get_model():
    """Lazy load the sentence transformer model."""
    global _model
    if _model is None:
        try:
            print(f"Loading embedding model: {MODEL_NAME}...")
            # Downloads automatically on first run
            _model = SentenceTransformer(MODEL_NAME)
            print(f"Model loaded successfully. Dimension: {_model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            raise
    return _model

def embed_documents(texts, batch_size=32):
    """
    Embeds a list of texts into a numpy array.
    Uses batch processing and normalization.
    """
    try:
        model = get_model()
        print(f"Embedding {len(texts)} documents in batches of {batch_size}...")
        embeddings = model.encode(
            texts, 
            batch_size=batch_size, 
            show_progress_bar=True,
            normalize_embeddings=True
        )
        return embeddings
    except Exception as e:
        print(f"Error embedding documents: {e}")
        return []

def embed_query(query):
    """
    Embeds a single query string.
    """
    try:
        model = get_model()
        embedding = model.encode(
            query,
            normalize_embeddings=True
        )
        return embedding
    except Exception as e:
        print(f"Error embedding query: {e}")
        return None

if __name__ == "__main__":
    print("Testing embeddings.py")
    test_docs = ["This is a test document.", "Another test string."]
    embeddings = embed_documents(test_docs)
    print(f"Generated embeddings shape: {embeddings.shape if hasattr(embeddings, 'shape') else 'N/A'}")
    
    q_emb = embed_query("Test query")
    print(f"Query embedding length: {len(q_emb) if q_emb is not None else 'N/A'}")

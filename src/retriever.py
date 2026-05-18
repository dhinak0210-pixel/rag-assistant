from rank_bm25 import BM25Okapi
import src.vectorstore as vs

class HybridRetriever:
    def __init__(self):
        self.bm25 = None
        self.corpus_docs = []
        self._initialize_bm25()

    def _initialize_bm25(self):
        """Initializes BM25 with current documents in Chroma."""
        try:
            collection = vs.get_chroma_client()
            if collection and collection.count() > 0:
                results = collection.get()
                self.corpus_docs = []
                tokenized_corpus = []
                
                if results and 'documents' in results and results['documents']:
                    for i, doc in enumerate(results['documents']):
                        self.corpus_docs.append({
                            "content": doc,
                            "source": results['metadatas'][i].get('source', ''),
                            "filename": results['metadatas'][i].get('filename', ''),
                            "page": results['metadatas'][i].get('page', 1)
                        })
                        tokenized_corpus.append(doc.lower().split())
                
                if tokenized_corpus:
                    self.bm25 = BM25Okapi(tokenized_corpus)
                    print(f"BM25 initialized with {len(tokenized_corpus)} documents.")
        except Exception as e:
            print(f"Error initializing BM25: {e}")

    def _bm25_search(self, query, top_k=5):
        if not self.bm25 or not self.corpus_docs:
            return []
            
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0: # Only include if there is some match
                doc = self.corpus_docs[idx].copy()
                doc['score'] = float(scores[idx]) # Normalizing BM25 score is hard, just return raw for now
                results.append(doc)
        return results

    def retrieve(self, query, top_k=5):
        """
        Retrieves documents using Reciprocal Rank Fusion (70% Vector, 30% BM25).
        """
        # Re-initialize BM25 in case new documents were added
        if not self.bm25:
            self._initialize_bm25()
            
        print(f"Running Hybrid Retrieval for query: '{query}'")
        vector_results = vs.search(query, top_k=top_k*2)
        bm25_results = self._bm25_search(query, top_k=top_k*2)
        
        # Reciprocal Rank Fusion (RRF)
        # score = w1 * (1 / (k + rank_vector)) + w2 * (1 / (k + rank_bm25))
        # where k is a constant, typically 60
        rrf_k = 60
        fused_scores = {}
        doc_map = {}
        
        # Process Vector Results (Weight: 0.7)
        for rank, doc in enumerate(vector_results):
            doc_id = doc['content'][:100] # Use content prefix as ID for simplicity
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
                doc_map[doc_id] = doc
            fused_scores[doc_id] += 0.7 * (1.0 / (rrf_k + rank + 1))
            
        # Process BM25 Results (Weight: 0.3)
        for rank, doc in enumerate(bm25_results):
            doc_id = doc['content'][:100]
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
                doc_map[doc_id] = doc
            fused_scores[doc_id] += 0.3 * (1.0 / (rrf_k + rank + 1))
            
        # Sort by fused score
        sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_results = []
        for doc_id, score in sorted_results[:top_k]:
            doc = doc_map[doc_id]
            doc['fusion_score'] = score
            final_results.append(doc)
            
        print(f"Retrieved {len(final_results)} fused results.")
        return final_results

if __name__ == "__main__":
    print("Testing retriever.py")
    retriever = HybridRetriever()
    res = retriever.retrieve("test", top_k=2)
    print(f"Got {len(res)} results.")

import os
from src.ingest import load_folder
from src.chunker import chunk_documents
from src.embeddings import get_model
import src.vectorstore as vs
from src.evaluator import RAGEvaluator

def run_pipeline(folder_path="data/documents"):
    """Rebuilds the entire knowledge base from a folder."""
    print("="*50)
    print("🚀 STARTING RAG PIPELINE")
    print("="*50)
    
    # 1. Clear existing database
    print("\n[1/4] Clearing existing Vector DB...")
    vs.clear()
    
    # 2. Load Documents
    print(f"\n[2/4] Loading documents from {folder_path}...")
    docs = load_folder(folder_path)
    if not docs:
        print("No documents found. Pipeline stopped.")
        return
        
    # 3. Chunking
    print("\n[3/4] Chunking documents...")
    chunks = chunk_documents(docs)
    
    # 4. Store in Vector DB (Embeddings happen here)
    print("\n[4/4] Creating embeddings and storing in ChromaDB...")
    vs.store_documents(chunks)
    
    stats = vs.get_stats()
    print("\n✅ PIPELINE COMPLETE!")
    print(f"Total Chunks in DB: {stats.get('total_chunks', 0)}")

def run_evaluation():
    """Runs evaluation with sample questions."""
    print("\n" + "="*50)
    print("📊 RUNNING EVALUATION")
    print("="*50)
    
    from src.rag_chain import RAGChain
    rag = RAGChain()
    evaluator = RAGEvaluator()
    
    test_cases = [
        "What is RAG?",
        "How does RAG improve language models?",
        "What is a vector database?",
        "Explain the chunking process.",
        "What are embeddings?"
    ]
    
    results_list = []
    
    test_cases_eval = []
    
    for q in test_cases:
        print(f"\nQ: {q}")
        res = rag.ask(q)
        print(f"A: {res['answer'][:100]}...")
        
        # Format context for evaluator
        context = res['sources']
        scores = evaluator.evaluate(q, res['answer'], context)
        scores['question'] = q
        results_list.append(scores)
        test_cases_eval.append({"question": q, "answer": res['answer'], "context": context})
        
        print(f"Scores -> Relevancy: {scores['answer_relevancy']}, Faithfulness: {scores['faithfulness']}, Precision: {scores['context_precision']}")
        
    evaluator.last_results = results_list
    summary = evaluator.evaluate_batch(test_cases_eval) # Pass the correct format
    
    print("\n✅ EVALUATION COMPLETE!")
    print("Average Scores:")
    print(f"  - Relevancy:   {summary['avg_relevancy']:.4f}")
    print(f"  - Faithfulness:{summary['avg_faithfulness']:.4f}")
    print(f"  - Precision:   {summary['avg_precision']:.4f}")
    
    # Create plot
    evaluator.plot_results()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation step")
    args = parser.parse_args()
    
    run_pipeline()
    if not args.skip_eval:
        run_evaluation()

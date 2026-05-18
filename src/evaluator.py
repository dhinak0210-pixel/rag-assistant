import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from src.embeddings import get_model

class RAGEvaluator:
    def __init__(self):
        self.model = get_model()

    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def evaluate(self, question, answer, context):
        """Evaluates a single Q&A pair."""
        if not answer or not context:
            return {"answer_relevancy": 0.0, "faithfulness": 0.0, "context_precision": 0.0}
            
        context_text = " ".join([c.get('content', '') for c in context])
        
        # 1. Answer Relevancy: Similarity between question and answer
        q_emb = self.model.encode(question)
        a_emb = self.model.encode(answer)
        relevancy = float(self._cosine_similarity(q_emb, a_emb))
        # Map to 0-1 range roughly, it's already -1 to 1 but usually >0 for related text
        relevancy = max(0.0, relevancy)
        
        # 2. Faithfulness: Similarity between answer and context
        c_emb = self.model.encode(context_text)
        faithfulness = float(self._cosine_similarity(a_emb, c_emb))
        faithfulness = max(0.0, faithfulness)
        
        # 3. Context Precision: Similarity between question and context
        precision = float(self._cosine_similarity(q_emb, c_emb))
        precision = max(0.0, precision)
        
        return {
            "answer_relevancy": round(relevancy, 4),
            "faithfulness": round(faithfulness, 4),
            "context_precision": round(precision, 4)
        }

    def evaluate_batch(self, test_cases):
        """
        Evaluates a list of test cases.
        test_cases format: [{"question": "...", "answer": "...", "context": [...]}, ...]
        """
        results = []
        for case in test_cases:
            scores = self.evaluate(case["question"], case["answer"], case["context"])
            scores["question"] = case["question"]
            results.append(scores)
            
        self.last_results = results
        
        summary = {
            "avg_relevancy": np.mean([r["answer_relevancy"] for r in results]),
            "avg_faithfulness": np.mean([r["faithfulness"] for r in results]),
            "avg_precision": np.mean([r["context_precision"] for r in results])
        }
        return summary

    def generate_report(self):
        """Returns results as a pandas DataFrame."""
        if not hasattr(self, 'last_results') or not self.last_results:
            return pd.DataFrame()
        return pd.DataFrame(self.last_results)

    def plot_results(self, save_path="evaluation_chart.png"):
        """Creates a bar chart of the evaluation metrics."""
        if not hasattr(self, 'last_results') or not self.last_results:
            print("No evaluation results to plot.")
            return
            
        df = self.generate_report()
        metrics = ["answer_relevancy", "faithfulness", "context_precision"]
        means = df[metrics].mean()
        
        plt.figure(figsize=(8, 5))
        bars = plt.bar(metrics, means, color=['blue', 'green', 'orange'])
        plt.ylim(0, 1)
        plt.title('RAG Evaluation Metrics')
        plt.ylabel('Score (0-1)')
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 2), ha='center')
            
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Chart saved to {save_path}")

if __name__ == "__main__":
    print("Testing evaluator.py")
    ev = RAGEvaluator()
    res = ev.evaluate("What is RAG?", "RAG is a technique.", [{"content": "RAG is a technique to improve LLMs."}])
    print(res)

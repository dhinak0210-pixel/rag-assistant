import os
import time
from groq import Groq
from .vectorstore import VectorStore
from .retriever import HybridRetriever

class RAGChain:
    def __init__(self):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("Warning: GROQ_API_KEY not found in environment!")
        self.client = Groq(api_key=api_key)
        self.model = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
        self.vector_store = VectorStore()
        self.retriever = HybridRetriever()
        self.history = []
        self.temperature = 0.1
        self.max_tokens = 1024
        print(f"RAG Chain Ready! (Using model: {self.model})")

    def _build_context(self, results):
        parts = []
        for i, res in enumerate(results):
            parts.append(f"[Source {i+1}: {res['filename']}, Page {res['page']}]\n{res['content']}")
        return "\n\n".join(parts)

    def _system_prompt(self):
        return """You are a helpful assistant.
        Answer ONLY from the provided context.
        If answer not in context say exactly:
        'I dont have this information in the documents.'
        Always cite which source you used.
        Be clear and concise."""

    def _build_prompt(self, question, context):
        prompt = "Context information is below.\n---------------------\n"
        prompt += context
        prompt += "\n---------------------\n"
        prompt += f"Question: {question}\nAnswer:"
        return prompt

    def ask(self, question: str, top_k=5):
        start_time = time.time()
        if len(question) < 3:
            return {"answer": "Question too short.", "sources": []}
            
        results = self.retriever.retrieve(question, top_k=top_k)
        if not results:
            return {
                "answer": "I dont have this information in the documents.",
                "sources": [],
                "latency": 0.0,
                "model": self.model,
                "chunks_used": 0
            }
            
        context = self._build_context(results)
        prompt = self._build_prompt(question, context)
        
        messages = [{"role": "system", "content": self._system_prompt()}]
        
        for msg in self.history[-6:]:
            messages.append(msg)
            
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            answer = response.choices[0].message.content
            
            self.history.append({"role": "user", "content": question})
            self.history.append({"role": "assistant", "content": answer})
            if len(self.history) > 6:
                self.history = self.history[-6:]
                
            latency = time.time() - start_time
            return {
                "answer": answer,
                "sources": results,
                "latency": latency,
                "model": self.model,
                "chunks_used": len(results)
            }
        except Exception as e:
            if "RateLimit" in str(e) or "429" in str(e):
                err = "Please wait 60 seconds (Rate Limit)"
            elif "Authentication" in str(e) or "401" in str(e):
                err = "Check your GROQ_API_KEY"
            else:
                err = f"Error: {e}"
            return {"answer": err, "sources": []}

    def ask_stream(self, question: str, top_k=5):
        if len(question) < 3:
            yield "Question too short."
            return
            
        results = self.retriever.retrieve(question, top_k=top_k)
        if not results:
            yield "I dont have this information in the documents."
            return
            
        context = self._build_context(results)
        prompt = self._build_prompt(question, context)
        
        messages = [{"role": "system", "content": self._system_prompt()}]
        for msg in self.history[-6:]:
            messages.append(msg)
        messages.append({"role": "user", "content": prompt})
        
        full_answer = ""
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    token = chunk.choices[0].delta.content
                    full_answer += token
                    yield token
                    
            self.history.append({"role": "user", "content": question})
            self.history.append({"role": "assistant", "content": full_answer})
            if len(self.history) > 6:
                self.history = self.history[-6:]
                
        except Exception as e:
            if "RateLimit" in str(e) or "429" in str(e):
                yield "Please wait 60 seconds (Rate Limit)"
            elif "Authentication" in str(e) or "401" in str(e):
                yield "Check your GROQ_API_KEY"
            else:
                yield f"Error: {e}"

    def clear_history(self):
        self.history = []

if __name__ == "__main__":
    rag = RAGChain()
    print(rag.ask("Hello?"))

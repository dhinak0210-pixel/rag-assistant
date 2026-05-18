import os
import time
from groq import Groq
from dotenv import load_dotenv
from src.retriever import HybridRetriever

load_dotenv()

class RAGChain:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = os.getenv("LLM_MODEL", "llama-3.1-70b-versatile")
        self.client = None
        self.retriever = HybridRetriever()
        self.chat_history = []
        
        if not self.api_key or self.api_key == "your_groq_key_here":
            print("WARNING: Invalid or missing GROQ_API_KEY")
        else:
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception as e:
                print(f"Error initializing Groq client: {e}")

    def clear_history(self):
        """Reset conversation history."""
        self.chat_history = []
        print("Chat history cleared.")

    def _build_prompt(self, question, contexts):
        """Constructs the prompt with context and instructions."""
        context_text = "\n\n---\n\n".join([
            f"[Source: {c.get('filename', 'Unknown')} | Page: {c.get('page', 1)}]\n{c.get('content', '')}" 
            for c in contexts
        ])
        
        sys_prompt = f"""You are a helpful and precise AI assistant.
Answer ONLY from the provided context.
Show source citations in your answer (e.g. [filename, page]).
If the information is not in the context, explicitly say: "I don't have this info in my context."
Do not hallucinate or use outside knowledge.

CONTEXT:
{context_text}"""
        
        messages = [{"role": "system", "content": sys_prompt}]
        
        # Add history (last 3 turns = 6 messages)
        for msg in self.chat_history[-6:]:
            messages.append(msg)
            
        messages.append({"role": "user", "content": question})
        return messages

    def ask(self, question, top_k=5):
        """Standard synchronous request."""
        start_time = time.time()
        
        if not self.client:
            return {
                "answer": "Error: Groq API key is invalid or missing. Please check your .env file.",
                "sources": [],
                "latency": 0,
                "model": self.model
            }
            
        try:
            contexts = self.retriever.retrieve(question, top_k=top_k)
            messages = self._build_prompt(question, contexts)
            
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=1024,
                top_p=0.9
            )
            
            answer = completion.choices[0].message.content
            
            # Update history
            self.chat_history.append({"role": "user", "content": question})
            self.chat_history.append({"role": "assistant", "content": answer})
            
            latency = round(time.time() - start_time, 2)
            
            return {
                "answer": answer,
                "sources": contexts,
                "latency": latency,
                "model": self.model
            }
        except Exception as e:
            error_msg = str(e)
            if "rate limit" in error_msg.lower():
                return {"answer": "Rate limit exceeded. Please wait a moment and try again.", "sources": [], "latency": 0, "model": self.model}
            return {"answer": f"An error occurred: {error_msg}", "sources": [], "latency": 0, "model": self.model}

    def ask_stream(self, question, top_k=5):
        """Streaming request generator."""
        if not self.client:
            yield "Error: Groq API key is invalid or missing. Please check your .env file."
            return
            
        try:
            contexts = self.retriever.retrieve(question, top_k=top_k)
            messages = self._build_prompt(question, contexts)
            
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=1024,
                top_p=0.9,
                stream=True
            )
            
            full_answer = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_answer += content
                    yield content
                    
            # Update history
            self.chat_history.append({"role": "user", "content": question})
            self.chat_history.append({"role": "assistant", "content": full_answer})
            
        except Exception as e:
            error_msg = str(e)
            if "rate limit" in error_msg.lower():
                yield "Rate limit exceeded. Please wait a moment and try again."
            else:
                yield f"An error occurred: {error_msg}"

if __name__ == "__main__":
    print("Testing rag_chain.py")
    chain = RAGChain()
    print("Initialized RAG Chain.")

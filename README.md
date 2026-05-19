---
title: Free RAG Assistant
emoji: 🤖
colorFrom: purple
colorTo: blue
sdk: docker
pinned: true
license: mit
---

# 🤖 Free RAG Assistant


> Chat with your documents using AI - 100% FREE!

## 🌐 Live Demo
[Click here to try it!](https://huggingface.co/spaces/USERNAME/rag-assistant)

## ✨ Features
- Upload PDF/TXT documents
- Ask questions in natural language
- Get answers with source citations
- Streaming responses
- Chat history
- 100% free to use

## 🛠️ Tech Stack (All FREE)
| Component | Tool | Cost |
|-----------|------|------|
| LLM | Groq + LLaMA 3.1 | FREE |
| Embeddings | sentence-transformers | FREE |
| Vector DB | ChromaDB | FREE |
| UI | Streamlit | FREE |
| Hosting | HuggingFace Spaces | FREE |
| Total | | $0.00 |

## 🚀 Run Locally
```bash
git clone your-repo
cd rag-app
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key" > .env
streamlit run app.py
```

## 🔑 Get FREE API Key
1. Go to console.groq.com
2. Sign up free
3. Create API key
4. Add to .env file

## 📊 Architecture
User Question
     ↓
Input Guardrail Check
     ↓  
ChromaDB Vector Search
     ↓
BM25 Keyword Search
     ↓
Combine Results (RRF)
     ↓
Build Context + Prompt
     ↓
Groq LLaMA 3.1 (FREE)
     ↓
Stream Answer to User

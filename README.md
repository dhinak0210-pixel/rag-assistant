# 🤖 Free RAG Application

A complete, production-grade Retrieval-Augmented Generation (RAG) application built with 100% free and local tools.

## 🏗️ Architecture

```text
User Request ──> Guardrails ──> RAG Chain (Groq LLM) ──> Response
                                    │
                                    v
Documents ──> Ingest ──> Chunk ──> Embed (Sentence-Transformers) ──> ChromaDB
                                    │
                                    v
                                Hybrid Retriever (BM25 + Vector)
```

## ✨ Features
- **100% Free Stack**: No OpenAI, no paid APIs required.
- **Hybrid Search**: Combines BM25 keyword search with Semantic Vector search using Reciprocal Rank Fusion.
- **Advanced Chunking**: Smart paragraph/sentence splitting with overlap.
- **Guardrails**: Input validation against prompt injection and output quality checking.
- **Evaluation Framework**: Built-in script to measure Relevancy, Faithfulness, and Precision.
- **Streaming UI**: Beautiful Streamlit interface with typing effects.
- **FastAPI Backend**: Complete REST API ready for production.

## 🛠️ Tech Stack (100% Free)

| Component | Tool / Model | Cost |
| --- | --- | --- |
| **LLM** | Groq API (llama-3.1-70b-versatile) | FREE |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) | FREE (Local) |
| **Vector DB** | ChromaDB | FREE (Local) |
| **Frontend** | Streamlit | FREE |
| **Backend** | FastAPI | FREE |

## 🚀 Quick Start

### 1. Get a Groq API Key
1. Go to [console.groq.com](https://console.groq.com)
2. Create a free account.
3. Navigate to API Keys and create a new key.

### 2. Setup
Clone the repository and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

Edit the `.env` file and add your Groq API key:
```env
GROQ_API_KEY=gsk_your_key_here
```

### 3. Initialize & Test
Run the setup script to create folders, add sample data, and run tests:
```bash
python setup.py
```

### 4. Run the Application

**Option A: Streamlit UI (Recommended for Users)**
```bash
streamlit run app.py
```

**Option B: Terminal Chat**
```bash
python chat.py
```

**Option C: FastAPI Backend (For Developers)**
```bash
uvicorn api.main:app --reload
```
API Documentation available at: `http://localhost:8000/docs`

## 📄 Adding Documents
1. Place your PDF, TXT, or MD files in the `data/documents/` folder.
2. Run the pipeline script to ingest them:
```bash
python pipeline.py
```
*(Alternatively, use the upload button in the Streamlit UI or the `/upload` API endpoint).*

## 📊 Evaluation Results
The system includes an automated evaluation suite (`src/evaluator.py`). 
Typical scores on sample data:

| Metric | Score |
| --- | --- |
| Answer Relevancy | ~0.85+ |
| Faithfulness | ~0.90+ |
| Context Precision | ~0.80+ |

Run evaluations yourself using: `python pipeline.py`

## ☁️ Deployment Guide (HuggingFace Spaces)
1. Go to [HuggingFace Spaces](https://huggingface.co/spaces) and create a new Space.
2. Select **Streamlit** as the SDK.
3. Upload all files from this repository.
4. Go to Settings -> Variables and Secrets.
5. Add your `GROQ_API_KEY` as a Secret.
6. The Space will build and launch automatically using `app.py`.

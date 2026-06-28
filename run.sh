#!/usr/bin/env bash
# Start the RAG Streamlit app
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
fi

.venv/bin/pip install -q -r requirements.txt
echo "Starting RAG Chatbot at http://localhost:8501"
.venv/bin/streamlit run app.py

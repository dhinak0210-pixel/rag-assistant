import streamlit as st
import os
import time
import pandas as pd
from src.rag_chain import RAGChain
import src.vectorstore as vs
from src.ingest import load_pdf, load_txt
from src.chunker import chunk_documents

st.set_page_config(page_title="Free RAG Assistant", page_icon="🤖", layout="wide")

# Custom CSS for aesthetics
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: #fafafa;
}
.free-badge {
    background-color: #1b5e20;
    color: white;
    padding: 4px 8px;
    border-radius: 4px;
    font-weight: bold;
    font-size: 0.8em;
}
.user-msg {
    background-color: #1e3a8a;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
}
.assistant-msg {
    background-color: #3b2c6a;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_rag_chain():
    return RAGChain()

rag = get_rag_chain()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 🤖 Free RAG Assistant <span class='free-badge'>FREE</span>", unsafe_allow_html=True)
    
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader("Upload PDF/TXT/MD", accept_multiple_files=True, type=['pdf', 'txt', 'md'])
    
    if st.button("Process & Index"):
        if uploaded_files:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            os.makedirs("data/documents", exist_ok=True)
            
            all_chunks = []
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name}...")
                file_path = f"data/documents/{file.name}"
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
                    
                ext = file.name.lower().split('.')[-1]
                docs = load_pdf(file_path) if ext == 'pdf' else load_txt(file_path)
                chunks = chunk_documents(docs)
                all_chunks.extend(chunks)
                progress_bar.progress((i + 1) / len(uploaded_files))
                
            status_text.text(f"Storing {len(all_chunks)} chunks in Vector DB...")
            vs.store_documents(all_chunks)
            rag.retriever._initialize_bm25() # update bm25
            status_text.text("Done!")
            st.success("Indexing complete!")
        else:
            st.warning("Please upload files first.")
            
    st.header("2. Knowledge Base Stats")
    stats = vs.get_stats()
    st.metric("Total Chunks", stats.get("total_chunks", 0))
    
    st.header("3. Settings")
    model_choice = st.selectbox("Model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"])
    rag.model = model_choice
    
    top_k = st.slider("Top-K Sources", 1, 10, 5)
    
    if st.button("Clear History"):
        rag.clear_history()
        st.session_state.messages = []
        st.success("History cleared.")
        
    st.markdown("### Tech Stack")
    st.table(pd.DataFrame({
        "Component": ["LLM", "Embeddings", "Vector DB", "Frontend"],
        "Tool (Free)": ["Groq API", "Sentence-Transformers", "ChromaDB", "Streamlit"]
    }))

# --- MAIN AREA ---
st.title("Chat with your Data")

if "messages" not in st.session_state:
    st.session_state.messages = []

if stats.get("total_chunks", 0) == 0:
    st.info("Welcome! Please upload and index some documents in the sidebar to get started.")

# Display chat messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("📚 Sources"):
                    for i, s in enumerate(msg["sources"]):
                        st.markdown(f"**[{i+1}] {s.get('filename', 'Unknown')} (Page {s.get('page', 1)})**")
                        st.text(s.get('content', '')[:200] + "...")
                        st.caption(f"Score: {s.get('fusion_score', s.get('score', 0)):.4f}")
            if "latency" in msg:
                st.caption(f"⏱️ Latency: {msg['latency']}s")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user msg
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)
        
    # Generate response
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            start_time = time.time()
            # Stream response
            for chunk in rag.ask_stream(prompt, top_k=top_k):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            latency = round(time.time() - start_time, 2)
            
            # Fetch sources by doing a dummy ask to get the context used
            # (In a real app, streaming method should return sources too, handling via state here)
            sources = rag.retriever.retrieve(prompt, top_k=top_k)
            
            if sources:
                with st.expander("📚 Sources"):
                    for i, s in enumerate(sources):
                        st.markdown(f"**[{i+1}] {s.get('filename', 'Unknown')} (Page {s.get('page', 1)})**")
                        st.text(s.get('content', '')[:200] + "...")
                        st.caption(f"Score: {s.get('fusion_score', s.get('score', 0)):.4f}")
                        
            st.caption(f"⏱️ Latency: {latency}s")
            
            # Save to state
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "sources": sources,
                "latency": latency
            })
        except Exception as e:
            st.error(f"Error: {e}")

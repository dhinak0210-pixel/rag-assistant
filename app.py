try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
import os
import time
from dotenv import load_dotenv

load_dotenv(override=True)

from src.rag_chain import RAGChain
from src.ingest import load_folder
from src.chunker import chunk_documents
from src.guardrails import InputGuardrail

# Page Config
st.set_page_config(
    page_title="🤖 Free RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.gradient-text {
    background: linear-gradient(to right, #6366f1, #a855f7, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem;
    font-weight: 800;
}
.stChatMessage.user {
    background-color: rgba(59, 130, 246, 0.1);
    border-radius: 15px 15px 0px 15px;
}
.stChatMessage.assistant {
    background-color: rgba(168, 85, 247, 0.1);
    border-radius: 15px 15px 15px 0px;
}
.source-card {
    border-left: 4px solid #a855f7;
    background-color: rgba(168, 85, 247, 0.05);
    padding: 10px;
    margin-bottom: 10px;
    border-radius: 0 8px 8px 0;
}
.free-badge {
    background-color: #22c55e;
    color: white;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 0.8rem;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "docs_processed" not in st.session_state:
    st.session_state.docs_processed = False
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0

@st.cache_resource
def load_rag_chain():
    return RAGChain()

def check_groq_key():
    return bool(os.environ.get("GROQ_API_KEY"))

def process_files(uploaded_files):
    os.makedirs("./temp_docs", exist_ok=True)
    for f in uploaded_files:
        with open(os.path.join("./temp_docs", f.name), "wb") as out:
            out.write(f.getbuffer())
            
    docs = load_folder("./temp_docs")
    chunks = chunk_documents(docs)
    
    rag = load_rag_chain()
    total = rag.vector_store.store(chunks)
    
    st.session_state.total_chunks = total
    st.session_state.docs_processed = True
    
    # Cleanup temp
    for f in os.listdir("./temp_docs"):
        os.remove(os.path.join("./temp_docs", f))
    os.rmdir("./temp_docs")
    
    return True

# Sidebar
with st.sidebar:
    st.title("🤖 RAG Assistant")
    st.markdown('<span class="free-badge">💚 100% FREE</span>', unsafe_allow_html=True)
    
    if not check_groq_key():
        st.error("Add GROQ_API_KEY!")
        st.markdown("Get free key: [console.groq.com](https://console.groq.com)")
        st.stop()
        
    st.divider()
    
    st.subheader("📁 Upload Documents")
    uploaded_files = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt", "md"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Process & Index", type="primary"):
            with st.spinner("Processing documents..."):
                process_files(uploaded_files)
                st.success("Documents indexed successfully!")
                st.balloons()
                
    st.divider()
    
    st.subheader("📊 Knowledge Base")
    rag = load_rag_chain()
    stats = rag.vector_store.get_stats()
    st.session_state.total_chunks = stats["total"]
    
    st.metric("Total Chunks", st.session_state.total_chunks)
    st.write("Status: " + ("Ready ✅" if stats["ready"] else "Empty ❌"))
    
    st.divider()
    
    st.subheader("⚙️ Settings")
    model_choice = st.selectbox("Model", [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768"
    ])
    rag.model = model_choice
    top_k = st.slider("Top-K Sources", 1, 10, 5)
    st.caption("Lower temperature = more focused answers.")
    
    st.divider()
    
    if st.button("🗑️ Clear History"):
        st.session_state.messages = []
        rag.clear_history()
        st.rerun()
        
    st.divider()
    
    with st.expander("🛠️ Tech Stack"):
        st.markdown("""
        | Component | Tool | Cost |
        |---|---|---|
        | LLM | Groq LLaMA 3.1 | FREE |
        | Embed | MiniLM | FREE |
        | DB | ChromaDB | FREE |
        | UI | Streamlit | FREE |
        | Host | HuggingFace | FREE |
        | **TOTAL** | | **$0.00** |
        """)

# Main Area
st.markdown('<h1 class="gradient-text">🤖 Free RAG Assistant</h1>', unsafe_allow_html=True)
st.markdown("### Chat with your documents | 100% FREE")

stats = rag.vector_store.get_stats()
if not stats["ready"] and not uploaded_files:
    st.info("👈 Upload documents in sidebar to start!")
    with st.expander("How it works", expanded=True):
        st.markdown("""
        **Step 1:** Upload PDF/TXT in sidebar  
        **Step 2:** Click Process & Index  
        **Step 3:** Ask questions!
        """)
        col1, col2 = st.columns(2)
        with col1:
            st.button("What is AI?", disabled=True)
        with col2:
            st.button("Summarize the document", disabled=True)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 Sources ({len(msg['sources'])} chunks)"):
                for src in msg["sources"]:
                    st.markdown(f"""
                    <div class="source-card">
                        <strong>{src.get('filename', 'Unknown')}</strong> (Page {src.get('page', '1')}) 
                        <span class="free-badge">Score: {src.get('score', 0):.2f}</span><br>
                        <small>{src.get('content', '')[:300]}...</small>
                    </div>
                    """, unsafe_allow_html=True)

# Chat Input
guard = InputGuardrail()
if prompt := st.chat_input("Ask anything..."):
    is_safe, msg = guard.validate(prompt)
    if not is_safe:
        st.warning(msg)
        st.stop()
        
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        placeholder = st.empty()
        
        with st.spinner("🔍 Searching..."):
            sources = rag.retriever.retrieve(prompt, top_k=top_k)
            
        full_response = ""
        start_time = time.time()
        for token in rag.ask_stream(prompt, top_k=top_k, history=st.session_state.messages[:-1]):
            full_response += token
            placeholder.markdown(full_response + "▌")
            
        placeholder.markdown(full_response)
        latency = time.time() - start_time
        
        if sources:
            with st.expander(f"📚 Sources ({len(sources)} chunks)"):
                for src in sources:
                    st.markdown(f"""
                    <div class="source-card">
                        <strong>{src.get('filename', 'Unknown')}</strong> (Page {src.get('page', '1')}) 
                        <span class="free-badge">Score: {src.get('score', 0):.2f}</span><br>
                        <small>{src.get('content', '')[:300]}...</small>
                    </div>
                    """, unsafe_allow_html=True)
            st.caption(f"⚡ Latency: {latency:.2f}s | Model: {rag.model}")
            
    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response,
        "sources": sources
    })

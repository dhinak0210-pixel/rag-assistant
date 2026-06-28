"""
RAG Assistant — Streamlit UI.

An advanced document assistant with visual grounding, file management,
multi-mode synthesis, dynamic starter questions, and chat exporting.

Run:
  streamlit run app.py
"""

import os
import tempfile
import time
import traceback
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Set page configuration with a premium look
st.set_page_config(
    page_title="RAG Intelligence Assistant",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom color palette and CSS for visual excellence
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

/* Main application layout & background */
html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #08080d !important;
    color: #f3f4f6 !important;
}

/* Explicit heading styles */
h1, h2, h3, h4, h5, h6, .main-header {
    font-family: 'Outfit', sans-serif;
    color: #ffffff !important;
}

/* Ensure all normal text, paragraphs, labels, and markdown are fully visible */
p, span, label, li, ul, ol, small, [data-testid="stMarkdownContainer"] {
    color: #f3f4f6 !important;
}

/* Specific styling for code elements */
code {
    color: #f472b6 !important; /* Soft pink for inline code */
    background-color: rgba(244, 114, 182, 0.1) !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
}
pre code {
    color: #e5e7eb !important;
    background-color: transparent !important;
}

/* Button visibility and aesthetics - force dark glassmorphism and white text */
button, [data-testid^="stBaseButton"] button {
    background-color: rgba(255, 255, 255, 0.04) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.3s ease !important;
}
button:hover, [data-testid^="stBaseButton"] button:hover {
    background-color: rgba(99, 102, 241, 0.1) !important;
    border-color: rgba(99, 102, 241, 0.5) !important;
    color: #ffffff !important;
    transform: translateY(-1px) !important;
}
div[data-testid="stBaseButton-primary"] button {
    background: linear-gradient(135deg, #6366f1, #a855f7) !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
}
div[data-testid="stBaseButton-primary"] button:hover {
    background: linear-gradient(135deg, #5046e5, #9333ea) !important;
    color: #ffffff !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important;
    transform: translateY(-1px) !important;
}

/* Specific styling for sidebar widgets and their labels to be bright and readable */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"], 
[data-testid="stSidebar"] label, 
[data-testid="stSidebar"] span, 
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .stMarkdown {
    color: #e5e7eb !important;
}

/* Ensure form input labels and headers are bright and clear */
.stSlider label, .stSelectbox label, .stToggle label, .stFileUploader label, [data-testid="stSidebar"] .stSlider * {
    color: #e5e7eb !important;
    font-weight: 500 !important;
}

/* Ensure the selected value text in selectboxes inside sidebar is white */
[data-testid="stSidebar"] div[data-baseweb="select"] * {
    color: #ffffff !important;
}

/* Glassmorphism sidebar - target both section and content wrappers */
section[data-testid="stSidebar"], 
div[data-testid="stSidebarContent"], 
[data-testid="stSidebar"] {
    background-color: #0c0c14 !important;
    background: #0c0c14 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(20px) !important;
}

/* Premium gradient title */
.main-header {
    background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.85rem;
    font-weight: 800;
    margin-bottom: 2px;
    letter-spacing: -0.03em;
    display: inline-block;
}

.sub-header {
    color: #9ca3af !important;
    font-size: 1.05rem;
    margin-top: 0px;
    margin-bottom: 1.8rem;
    font-weight: 400;
}

/* Status Cards & Dashboard metrics */
.stat-box {
    background: rgba(18, 18, 28, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 16px;
    padding: 1.1rem;
    text-align: center;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(10px);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.stat-box:hover {
    transform: translateY(-2px);
    border-color: rgba(99, 102, 241, 0.3);
    box-shadow: 0 10px 20px rgba(99, 102, 241, 0.08);
}
.stat-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #818cf8 !important;
}
.stat-label {
    font-size: 0.72rem;
    color: #9ca3af !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}

/* Custom visual badges */
.model-badge {
    display: inline-block;
    background: rgba(99, 102, 241, 0.08);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 20px;
    padding: 4px 14px;
    margin-right: 8px;
    margin-bottom: 8px;
    font-size: 0.78rem;
    color: #a5b4fc !important;
    font-weight: 600;
}

/* Beautiful citations/sources */
.source-card {
    background: rgba(16, 185, 129, 0.04) !important;
    border: 1px solid rgba(16, 185, 129, 0.12) !important;
    border-left: 4px solid #10b981 !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
    margin: 14px 0 !important;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2) !important;
    transition: all 0.3s ease !important;
}
.source-card:hover {
    background: rgba(16, 185, 129, 0.08) !important;
    border-color: rgba(16, 185, 129, 0.3) !important;
    transform: translateY(-1px);
}
.source-card * {
    color: #e5e7eb !important;
}
.source-title, .source-title * {
    color: #34d399 !important;
    font-weight: 600 !important;
    font-size: 0.92rem;
    display: flex;
    justify-content: space-between;
}
.source-meta {
    color: #9ca3af !important;
    font-size: 0.75rem;
    margin-top: 4px;
    font-family: monospace;
}

/* Suggested prompts style */
.suggestion-btn {
    border: 1px solid rgba(255, 255, 255, 0.07) !important;
    background-color: rgba(255, 255, 255, 0.02) !important;
    border-radius: 12px !important;
    padding: 10px 15px !important;
    transition: all 0.2s ease-in-out !important;
    font-size: 0.85rem !important;
    text-align: left !important;
}
.suggestion-btn:hover {
    border-color: rgba(139, 92, 246, 0.4) !important;
    background-color: rgba(139, 92, 246, 0.04) !important;
    transform: scale(1.01);
}

/* Custom separator */
.sidebar-hr {
    border: 0;
    height: 1px;
    background: linear-gradient(to right, rgba(255, 255, 255, 0), rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
    margin: 1.5rem 0;
}

/* Chat Input Styling */
div[data-testid="stChatInput"] {
    border-radius: 20px !important;
    background-color: rgba(18, 18, 28, 0.7) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
    backdrop-filter: blur(12px) !important;
}

/* Chat Messages */
div[data-testid="stChatMessage"] {
    background: rgba(20, 20, 32, 0.45) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 18px !important;
    padding: 1.25rem !important;
    margin-bottom: 1.2rem !important;
    backdrop-filter: blur(10px) !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15) !important;
    transition: all 0.3s ease !important;
}
div[data-testid="stChatMessage"]:hover {
    border-color: rgba(99, 102, 241, 0.2) !important;
    transform: translateY(-1px);
}

/* Ensure text inside chat messages is fully white/readable */
div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] li,
div[data-testid="stChatMessage"] h1,
div[data-testid="stChatMessage"] h2,
div[data-testid="stChatMessage"] h3,
div[data-testid="stChatMessage"] h4,
div[data-testid="stChatMessage"] h5,
div[data-testid="stChatMessage"] h6 {
    color: #ffffff !important;
}

/* Streamlit styling overrides */
footer { visibility: hidden; }
div[data-testid="stForm"] {
    background-color: transparent !important;
    border: none !important;
}

/* Selectbox dropdown and popover menu visibility */
div[data-baseweb="select"] {
    color: #ffffff !important;
}
div[data-baseweb="popover"], div[role="listbox"] {
    background-color: #12121c !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}
div[role="option"] {
    color: #ffffff !important;
    background-color: transparent !important;
}
div[role="option"]:hover, div[role="option"][aria-selected="true"] {
    background-color: #6366f1 !important;
    color: #ffffff !important;
}
</style>
"""

ASSISTANT_MODES = {
    "Q&A Grounded": (
        "You are a precise, grounded document assistant.\n\n"
        "Rules:\n"
        "1. Answer ONLY using the CONTEXT provided. Do not use outside knowledge or make assumptions.\n"
        "2. If the context does not contain the answer, say: \"I don't have enough information in the provided documents.\"\n"
        "3. Cite source numbers like [1], [2], etc., when referring to details.\n"
        "4. Keep the answer concise and strictly factual."
    ),
    "Synthesizer / Summarizer": (
        "You are an expert document summarizer and synthesizer.\n\n"
        "Rules:\n"
        "1. Summarize the key concepts, entities, and takeaways from the provided CONTEXT.\n"
        "2. Structure your summary using clear bullet points and bold headings.\n"
        "3. If multiple documents/sources are present, compare and synthesize their findings.\n"
        "4. Keep it concise, structured, and easy to read."
    ),
    "Research Analyst": (
        "You are a professional research analyst.\n\n"
        "Rules:\n"
        "1. Provide a comprehensive, in-depth analysis of the user's query based on the CONTEXT.\n"
        "2. You may use external domain knowledge or general expert reasoning to add context, explanation, or structure, but you must clearly distinguish between facts directly in the documents and external synthesis.\n"
        "3. Structure your response with an Executive Summary, Key Findings, and Analysis details.\n"
        "4. Quote or reference specific source numbers [1], [2] for facts from the documents."
    )
}


def init_session_state():
    """Setup default session configuration values."""
    defaults = {
        "messages": [],
        "top_k": int(os.getenv("TOP_K", "3")),
        "chunk_size": 500,
        "chunk_overlap": 50,
        "use_stream": True,
        "mode": "Q&A Grounded",
        "active_prompt": None,
        "suggested_questions": [],
        "refreshed_suggestions": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


@st.cache_resource
def get_collection():
    """Retrieve ChromaDB collection instance."""
    from vector_store import get_or_create_store
    _, collection = get_or_create_store()
    return collection


def refresh_collection():
    """Clear cache and refresh vector store reference."""
    get_collection.clear()
    st.session_state.refreshed_suggestions = True


FILE_TYPE_ICONS = {
    ".pdf": "PDF", ".docx": "DOCX", ".xlsx": "XLSX", ".xls": "XLS",
    ".csv": "CSV", ".json": "JSON", ".txt": "TXT", ".md": "MD",
    ".py": "PY", ".js": "JS", ".ts": "TS", ".html": "HTML",
    ".xml": "XML", ".sql": "SQL", ".go": "GO", ".java": "JAVA",
    ".cpp": "CPP", ".c": "C", ".sh": "SH",
}

def get_file_icon(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    tag = FILE_TYPE_ICONS.get(ext, ext[1:].upper() if len(ext) > 1 else "FILE")
    return f'<span style="background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);border-radius:4px;padding:2px 6px;font-size:0.68rem;color:#9ca3af;font-weight:600;margin-right:6px;vertical-align:middle;display:inline-block;">{tag}</span>'


def get_indexed_files(collection) -> list[dict]:
    """Retrieve all stored documents grouped by source filename."""
    try:
        count = collection.count()
        if count == 0:
            return []
        res = collection.get(include=["metadatas"])
        if not res or not res.get("metadatas"):
            return []
        from collections import Counter
        source_counts = Counter()
        for meta in res["metadatas"]:
            if meta and "source" in meta:
                source_counts[meta["source"]] += 1
        files = []
        for src, cnt in source_counts.items():
            files.append({"path": src, "name": os.path.basename(src), "chunks": cnt})
        return sorted(files, key=lambda x: x["name"])
    except Exception as e:
        print(f"Error listing database files: {e}")
        return []


def delete_file_from_db(source_path, collection) -> bool:
    """Delete all chunks for a specific file path from vector database."""
    try:
        collection.delete(where={"source": source_path})
        refresh_collection()
        return True
    except Exception as e:
        st.sidebar.error(f"Failed to delete file: {e}")
        return False


def wipe_database(collection) -> bool:
    """Wipe the database collection completely."""
    try:
        from vector_store import COLLECTION_NAME, get_or_create_store
        client, _ = get_or_create_store()
        client.delete_collection(name=COLLECTION_NAME)
        refresh_collection()
        return True
    except Exception as e:
        st.sidebar.error(f"Failed to wipe database: {e}")
        return False


def generate_suggested_questions(collection) -> list[str]:
    """Generate suggested questions dynamically based on database contents."""
    try:
        count = collection.count()
        if count == 0:
            return [
                "What is retrieval-augmented generation?",
                "How do I adjust chunk size and overlap settings?",
                "What documents are currently indexed in this assistant?"
            ]
        
        res = collection.get(limit=3, include=["documents"])
        if not res or not res.get("documents"):
            return ["Summarize the uploaded files.", "What are the key terms in these documents?"]
            
        sample_text = "\n".join(res["documents"])
        
        from groq_chain import GROQ_API_KEY, GROQ_MODEL, _get_client
        if not GROQ_API_KEY:
            return ["Explain the main topics of my documents."]
            
        client = _get_client()
        prompt = f"""Based on the following text snippet from a user's uploaded document, generate exactly 3 short, engaging questions that a user might ask about this document. 
Return ONLY the 3 questions as a plain list, one per line. Do not include numbering or bullet points.

Text Snippet:
{sample_text[:1200]}
"""
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150,
        )
        questions_text = response.choices[0].message.content or ""
        questions = [
            q.strip().lstrip("1234567890.-*• ")
            for q in questions_text.strip().split("\n")
            if q.strip()
        ]
        return [q for q in questions if len(q) > 10][:3]
    except Exception:
        return [
            "Summarize the key points of the uploaded files.",
            "What are the major terms or concepts in these documents?",
            "What conclusions or recommendations do these documents outline?"
        ]


def index_uploaded_files(tmp_dir: str, progress_bar, status) -> int:
    """Index all files in temp directory with progress UI."""
    from rag_pipeline import index_path

    status.markdown("**Step 1/3** — Loading documents...")
    progress_bar.progress(20)
    time.sleep(0.1)

    status.markdown("**Step 2/3** — Chunking & embedding...")
    progress_bar.progress(55)

    try:
        count = index_path(
            tmp_dir,
            chunk_size=st.session_state.chunk_size,
            chunk_overlap=st.session_state.chunk_overlap,
            quiet=True,
        )
    except Exception as e:
        status.markdown(f"**Error:** {e}")
        progress_bar.progress(100)
        return 0

    progress_bar.progress(100)
    if count > 0:
        status.markdown(f"**Done!** Indexed **{count}** chunk(s).")
        refresh_collection()
    else:
        status.markdown("No content indexed. Check file formats (PDF/DOCX/Spreadsheet/Data/Code/Text).")
    return count


def render_source_chunks(chunks: list[dict]):
    """Render retrieved source chunks with relevance ratings."""
    if not chunks:
        return
    st.markdown("##### Sources & Relevance Grounding")
    for i, chunk in enumerate(chunks, start=1):
        source = os.path.basename(chunk.get("source", "unknown"))
        page = chunk.get("page", "?")
        score = chunk.get("score", 0)
        content = chunk.get("content", "").strip()
        
        # Display similarity score badge
        if score >= 0.75:
            score_badge = f'<span style="background-color:rgba(16,185,129,0.15);color:#34d399;padding:2px 8px;border-radius:10px;font-size:0.75rem;font-weight:600;">Match {score*100:.1f}%</span>'
        elif score >= 0.6:
            score_badge = f'<span style="background-color:rgba(245,158,11,0.15);color:#fbbf24;padding:2px 8px;border-radius:10px;font-size:0.75rem;font-weight:600;">Match {score*100:.1f}%</span>'
        else:
            score_badge = f'<span style="background-color:rgba(107,114,128,0.15);color:#9ca3af;padding:2px 8px;border-radius:10px;font-size:0.75rem;font-weight:600;">Match {score*100:.1f}%</span>'

        preview = content[:380] + ("..." if len(content) > 380 else "")
        safe = preview.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        
        st.markdown(
            f'<div class="source-card">'
            f'<div class="source-title"><span>[{i}] {source} (Page {page})</span> {score_badge}</div>'
            f'<div style="margin-top:8px; line-height: 1.5; color: #d1d5db;">{safe}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def main():
    init_session_state()
    st.markdown(THEME_CSS, unsafe_allow_html=True)

    try:
        collection = get_collection()
    except Exception as e:
        st.error(f"Failed to open ChromaDB: {e}")
        st.stop()

    # Sidebar Construction
    st.sidebar.markdown(
        '<p style="font-size:1.75rem;font-weight:700;margin-bottom:0px;background:linear-gradient(135deg, #818cf8, #a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">RAG Assistant</p>',
        unsafe_allow_html=True
    )
    st.sidebar.caption("Intelligent Knowledge Synthesis Engine")
    st.sidebar.markdown('<div class="sidebar-hr"></div>', unsafe_allow_html=True)

    # 1. Dashboard Metrics
    st.sidebar.markdown("### Metrics")
    c1, c2 = st.sidebar.columns(2)
    chunks_n = collection.count()
    turns_n = sum(1 for m in st.session_state.messages if m["role"] == "user")
    
    with c1:
        st.markdown(
            f'<div class="stat-box"><div class="stat-value">{chunks_n}</div><div class="stat-label">Chunks</div></div>',
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f'<div class="stat-box"><div class="stat-value">{turns_n}</div><div class="stat-label">Turns</div></div>',
            unsafe_allow_html=True
        )
    st.sidebar.markdown('<div class="sidebar-hr"></div>', unsafe_allow_html=True)

    # 2. Assistant Settings
    st.sidebar.markdown("### Engine Parameters")
    st.session_state.mode = st.sidebar.selectbox(
        "Assistant Mode / Persona",
        list(ASSISTANT_MODES.keys()),
        index=0,
        help="Change the persona to alter how the assistant structures and responds to queries."
    )
    
    st.session_state.top_k = st.sidebar.slider("Context Retrieve Count (Top-K)", 1, 10, st.session_state.top_k)
    st.session_state.use_stream = st.sidebar.toggle("Stream Generation", st.session_state.use_stream)

    with st.sidebar.expander("Advanced Indexing Settings"):
        st.session_state.chunk_size = st.slider("Chunk size", 100, 1000, st.session_state.chunk_size, 50)
        st.session_state.chunk_overlap = st.slider("Overlap", 0, 200, st.session_state.chunk_overlap, 10)
    st.sidebar.markdown('<div class="sidebar-hr"></div>', unsafe_allow_html=True)

    # 3. File Manager Dashboard
    st.sidebar.markdown("### Knowledge Base")
    db_files = get_indexed_files(collection)
    if not db_files:
        st.sidebar.markdown(
            '<div style="color:#6b7280;font-size:0.82rem;padding:8px 0;">No documents indexed yet.<br>Upload files below to begin.</div>',
            unsafe_allow_html=True
        )
    else:
        for i, file in enumerate(db_files):
            icon = get_file_icon(file['name'])
            file_col, del_col = st.sidebar.columns([4, 1])
            file_col.markdown(
                f"{icon} **{file['name']}**  \n`{file['chunks']} chunks`",
                unsafe_allow_html=True
            )
            if del_col.button("✕", key=f"del_{i}_{hash(file['path'])}", help=f"Remove {file['name']}"):
                if delete_file_from_db(file['path'], collection):
                    st.sidebar.success(f"Removed {file['name']}")
                    st.rerun()

    # 4. Upload & Process
    st.sidebar.markdown("### Upload Documents")
    st.sidebar.markdown(
        '<div style="color:#6b7280;font-size:0.75rem;margin-bottom:6px;line-height:1.4;">'
        'PDF · Word · Excel · CSV<br>'
        'JSON · HTML/XML · Markdown · TXT<br>'
        'Python · JS/TS · Java · C/C++ · SQL'
        '</div>',
        unsafe_allow_html=True
    )
    uploaded = st.sidebar.file_uploader(
        "Upload files",
        type=["pdf", "docx", "xlsx", "xls", "csv", "json", "txt", "md", "markdown",
              "html", "htm", "xml", "py", "js", "jsx", "ts", "tsx", "java", "cpp",
              "c", "h", "cs", "go", "sh", "sql"],
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    if uploaded:
        if st.sidebar.button("Process & Embed", type="primary", use_container_width=True):
            progress = st.sidebar.progress(0, text="Starting...")
            status = st.sidebar.empty()
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    for i, f in enumerate(uploaded):
                        path = Path(tmp) / f.name
                        path.write_bytes(f.getvalue())
                        progress.progress(int((i + 1) / len(uploaded) * 15), text=f"Processing {f.name}")

                    count = index_uploaded_files(tmp, progress, status)

                if count > 0:
                    st.session_state.refreshed_suggestions = True
                    time.sleep(0.3)
                    st.rerun()
            except Exception as e:
                st.sidebar.error(f"Indexing failed: {e}")

    # Index default documents folder
    docs_dir = Path(__file__).parent / "documents"
    if docs_dir.exists() and any(docs_dir.iterdir()):
        if st.sidebar.button("Index Local Documents Directory", use_container_width=True):
            with st.spinner("Indexing folder..."):
                from rag_pipeline import index_path
                count = index_path(str(docs_dir), quiet=True)
                refresh_collection()
            st.sidebar.success(f"Indexed {count} chunks")
            st.rerun()

    # Clear options
    st.sidebar.markdown('<div class="sidebar-hr"></div>', unsafe_allow_html=True)
    
    from groq_chain import GROQ_API_KEY, GROQ_MODEL, clear_chat_history
    from vector_store import CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL

    key_ok = "Connected" if GROQ_API_KEY else "API Key Missing"
    st.sidebar.info(
        f"**LLM Backend:** {GROQ_MODEL} ({key_ok})  \n"
        f"**Embedding Model:** `{EMBEDDING_MODEL}`  \n"
        f"**DB Path:** `{CHROMA_DB_PATH}`"
    )

    c_chat, c_db = st.sidebar.columns(2)
    if c_chat.button("Reset Chat", use_container_width=True):
        clear_chat_history()
        st.session_state.messages = []
        st.rerun()
    if c_db.button("Wipe DB", use_container_width=True):
        if wipe_database(collection):
            st.sidebar.warning("Database cleared!")
            st.rerun()

    # --- Main Chat Board ---
    st.markdown('<p class="main-header">RAG Intelligence Assistant</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Advanced context-aware knowledge retrieval & semantic analysis engine</p>',
        unsafe_allow_html=True,
    )

    # Badges row
    st.markdown(
        f'<span class="model-badge">Llama-3.3-70b-versatile</span>'
        f'<span class="model-badge">ChromaDB Vectorstore</span>'
        f'<span class="model-badge">MiniLM Embeddings</span>'
        f'<span class="model-badge">Mode: {st.session_state.mode}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # Dynamic Suggested Questions (Interactive Prompts)
    if not st.session_state.suggested_questions or st.session_state.refreshed_suggestions:
        st.session_state.suggested_questions = generate_suggested_questions(collection)
        st.session_state.refreshed_suggestions = False

    if not st.session_state.messages:
        if collection.count() == 0:
            st.markdown(
                '<div style="'
                'background:linear-gradient(135deg,rgba(99,102,241,0.06),rgba(168,85,247,0.06));'
                'border:1px solid rgba(99,102,241,0.15);border-radius:20px;padding:2rem 2rem;text-align:center;margin:1rem 0;">'
                '<h2 style="color:#a5b4fc;font-family:Outfit,sans-serif;margin-bottom:0.5rem;">No Documents Loaded</h2>'
                '<p style="color:#6b7280;max-width:500px;margin:0 auto 1.5rem;">'
                'Upload any document to get started. The assistant can read and analyse <strong style="color:#818cf8;">PDFs, Word docs, Excel sheets, CSV, JSON, HTML, XML, Markdown, plain text, and source code files</strong>.'
                '</p>'
                '<div style="display:flex;flex-wrap:wrap;justify-content:center;gap:8px;">'
                '<span style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.2);border-radius:20px;padding:4px 14px;font-size:0.78rem;color:#a5b4fc;">PDF</span>'
                '<span style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.2);border-radius:20px;padding:4px 14px;font-size:0.78rem;color:#a5b4fc;">DOCX</span>'
                '<span style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.2);border-radius:20px;padding:4px 14px;font-size:0.78rem;color:#a5b4fc;">Excel</span>'
                '<span style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.2);border-radius:20px;padding:4px 14px;font-size:0.78rem;color:#a5b4fc;">CSV</span>'
                '<span style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.2);border-radius:20px;padding:4px 14px;font-size:0.78rem;color:#a5b4fc;">JSON</span>'
                '<span style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.2);border-radius:20px;padding:4px 14px;font-size:0.78rem;color:#a5b4fc;">HTML</span>'
                '<span style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.2);border-radius:20px;padding:4px 14px;font-size:0.78rem;color:#a5b4fc;">Python</span>'
                '<span style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.2);border-radius:20px;padding:4px 14px;font-size:0.78rem;color:#a5b4fc;">TXT / MD</span>'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div style="background:rgba(16,185,129,0.05);border:1px solid rgba(16,185,129,0.15);'
                'border-radius:14px;padding:1rem 1.25rem;margin-bottom:1rem;">'
                '<span style="color:#10b981; font-weight: bold; margin-right: 8px;">●</span><strong style="color:#34d399;">Knowledge base ready!</strong> '
                '<span style="color:#9ca3af;">Ask anything about your documents below.</span></div>',
                unsafe_allow_html=True
            )
            st.markdown("##### Suggested Questions")
            cols = st.columns(len(st.session_state.suggested_questions))
            for i, q in enumerate(st.session_state.suggested_questions):
                if cols[i].button(q, key=f"suggest_{i}", use_container_width=True):
                    st.session_state.active_prompt = q
                    st.rerun()

    # Conversation History Rendering
    for idx, msg in enumerate(st.session_state.messages):
        avatar = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("source_chunks"):
                render_source_chunks(msg["source_chunks"])

    # Handle prompts (either from input or clicked suggestions)
    prompt = st.session_state.get("active_prompt")
    if not prompt:
        prompt = st.chat_input("Query the assistant on your documents...")
    else:
        st.session_state.active_prompt = None

    if prompt:
        # Add to history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Draw User message immediately
        with st.chat_message("user", avatar="user"):
            st.markdown(prompt)

        # Generate Assistant response
        with st.chat_message("assistant", avatar="assistant"):
            system_prompt = ASSISTANT_MODES[st.session_state.mode]

            if st.session_state.use_stream:
                placeholder = st.empty()
                answer_parts = []

                from groq_chain import GROQ_API_KEY, ask_stream, retrieve_context

                if not GROQ_API_KEY:
                    answer = "Groq API key missing. Add `GROQ_API_KEY` to your `.env` file."
                    source_chunks = []
                    placeholder.markdown(answer)
                else:
                    try:
                        ctx = retrieve_context(prompt, collection=collection, top_k=st.session_state.top_k)
                        source_chunks = ctx["chunks"]
                        if not source_chunks:
                            answer = "No relevant context found. Try uploading relevant files first."
                            placeholder.markdown(answer)
                        else:
                            for token in ask_stream(
                                prompt,
                                collection=collection,
                                top_k=st.session_state.top_k,
                                verbose=False,
                                chunks=source_chunks,
                                history=st.session_state.messages[:-1],
                                system_prompt=system_prompt,
                            ):
                                answer_parts.append(token)
                                placeholder.markdown("".join(answer_parts) + "▌")
                            answer = "".join(answer_parts)
                            placeholder.markdown(answer)
                    except Exception as e:
                        answer = f"Error encountered: {e}"
                        source_chunks = []
                        placeholder.markdown(answer)
            else:
                with st.spinner("Analyzing document context..."):
                    from groq_chain import ask, GROQ_API_KEY, retrieve_context
                    if not GROQ_API_KEY:
                        answer = "Groq API key missing. Add `GROQ_API_KEY` to your `.env` file."
                        source_chunks = []
                    else:
                        try:
                            ctx = retrieve_context(prompt, collection=collection, top_k=st.session_state.top_k)
                            source_chunks = ctx["chunks"]
                            if not source_chunks:
                                answer = "No relevant context found. Try uploading relevant files first."
                            else:
                                result = ask(
                                    prompt,
                                    collection=collection,
                                    top_k=st.session_state.top_k,
                                    verbose=False,
                                    chunks=source_chunks,
                                    history=st.session_state.messages[:-1],
                                    system_prompt=system_prompt,
                                )
                                answer = result["answer"]
                        except Exception as e:
                            answer = f"Error: {e}"
                            source_chunks = []
                st.markdown(answer)

            # Draw citations
            if source_chunks:
                render_source_chunks(source_chunks)

            # Session export trigger
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "source_chunks": source_chunks}
            )
            
            st.rerun()

    # Transcript Export Button
    if len(st.session_state.messages) > 0:
        st.markdown("---")
        transcript = ""
        for msg in st.session_state.messages:
            role_title = "User" if msg["role"] == "user" else "Assistant"
            transcript += f"### {role_title}\n{msg['content']}\n\n"
        
        st.download_button(
            label="Export Complete Chat Transcript",
            data=transcript,
            file_name=f"rag_assistant_transcript_{int(time.time())}.md",
            mime="text/markdown",
            use_container_width=True
        )


if __name__ == "__main__":
    main()

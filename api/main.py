from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import os
import shutil

from src.rag_chain import RAGChain
from src.guardrails import InputGuardrail, OutputGuardrail
import src.vectorstore as vs
from src.ingest import load_pdf, load_txt
from src.chunker import chunk_documents

app = FastAPI(title="Free RAG Assistant API", version="1.0.0")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
rag = None
input_guard = InputGuardrail()
output_guard = OutputGuardrail()

@app.on_event("startup")
async def startup_event():
    global rag
    print("Initializing RAG components...")
    rag = RAGChain()
    print("API started successfully.")

class AskRequest(BaseModel):
    question: str
    top_k: int = 5

class AskResponse(BaseModel):
    answer: str
    sources: list
    latency: float
    model: str
    safety_score: dict

@app.get("/")
def read_root():
    return {"message": "Welcome to the Free RAG API! Visit /docs for Swagger UI."}

@app.get("/health")
def health_check():
    stats = vs.get_stats()
    return {"status": "healthy", "vector_db": stats}

@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    # Guardrails - Input
    is_safe, msg = input_guard.validate(request.question)
    if not is_safe:
        raise HTTPException(status_code=400, detail=msg)
        
    result = rag.ask(request.question, top_k=request.top_k)
    
    # Guardrails - Output
    safety_score = output_guard.check(result["answer"], result["sources"])
    result["safety_score"] = safety_score
    
    return result

@app.post("/ask/stream")
def ask_question_stream(request: AskRequest):
    is_safe, msg = input_guard.validate(request.question)
    if not is_safe:
        raise HTTPException(status_code=400, detail=msg)
        
    def generate():
        for token in rag.ask_stream(request.question, top_k=request.top_k):
            yield token
            
    return StreamingResponse(generate(), media_type="text/plain")

@app.post("/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith(('.pdf', '.txt', '.md')):
        raise HTTPException(status_code=400, detail="Only PDF, TXT, and MD files are supported.")
        
    os.makedirs("data/documents", exist_ok=True)
    file_path = f"data/documents/{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    def process_file(path, filename):
        ext = filename.lower().split('.')[-1]
        docs = load_pdf(path) if ext == 'pdf' else load_txt(path)
        chunks = chunk_documents(docs)
        vs.store_documents(chunks)
        # Re-init retriever to update BM25
        global rag
        if rag:
            rag.retriever._initialize_bm25()
            
    background_tasks.add_task(process_file, file_path, file.filename)
    return {"message": f"File {file.filename} uploaded and processing started in background."}

@app.post("/clear")
def clear_history():
    if rag:
        rag.clear_history()
    return {"message": "Chat history cleared."}

@app.get("/stats")
def get_stats():
    return vs.get_stats()

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

import os
import sys
from dotenv import load_dotenv

def check_env():
    print("Checking environment...")
    if sys.version_info < (3, 9):
        print("Error: Python 3.9+ is required.")
        sys.exit(1)
        
    if not os.path.exists('.env'):
        print("Error: .env file not found. Please create it.")
        sys.exit(1)
        
    load_dotenv()
    if not os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") == "your_groq_key_here":
        print("Error: GROQ_API_KEY is not set in .env")
        sys.exit(1)
    print("Environment checks passed.")

def create_folders():
    print("Creating directories...")
    os.makedirs("data/documents", exist_ok=True)
    os.makedirs("chroma_db", exist_ok=True)

def create_sample_doc():
    if not os.listdir("data/documents"):
        print("Creating sample document...")
        with open("data/documents/sample.txt", "w") as f:
            f.write("RAG (Retrieval-Augmented Generation) is an AI framework that improves the quality of LLM-generated responses by grounding the model on external sources of knowledge. The process involves retrieving relevant information from a vector database and appending it to the user's prompt as context.")

def run_tests():
    print("Running Pipeline Tests...")
    from pipeline import run_pipeline, run_evaluation
    run_pipeline("data/documents")
    run_evaluation()

if __name__ == "__main__":
    print("=== RAG App Setup ===")
    check_env()
    create_folders()
    create_sample_doc()
    run_tests()
    print("\n=== Setup Complete! ===")
    print("To run the Streamlit UI:")
    print("  streamlit run app.py")
    print("To run the API:")
    print("  uvicorn api.main:app --reload")

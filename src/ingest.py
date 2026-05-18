import os
import PyPDF2
import requests
from bs4 import BeautifulSoup

def load_pdf(path):
    docs = []
    try:
        reader = PyPDF2.PdfReader(path)
        filename = os.path.basename(path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and len(text) >= 20:
                docs.append({
                    "content": text,
                    "source": path,
                    "filename": filename,
                    "page": i + 1,
                    "doc_type": "pdf"
                })
    except Exception as e:
        print(f"Error loading PDF {path}: {e}")
    return docs

def load_txt(path):
    docs = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
            if len(text) >= 20:
                docs.append({
                    "content": text,
                    "source": path,
                    "filename": os.path.basename(path),
                    "page": 1,
                    "doc_type": "txt"
                })
    except Exception as e:
        print(f"Error loading TXT {path}: {e}")
    return docs

def load_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for script in soup(["script", "style"]):
            script.decompose()
            
        text = soup.get_text(separator=' ', strip=True)
        if len(text) >= 20:
            return {
                "content": text,
                "source": url,
                "filename": url,
                "page": 1,
                "doc_type": "url"
            }
    except Exception as e:
        print(f"Error loading URL {url}: {e}")
    return None

def load_folder(folder_path):
    all_docs = []
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    for root, _, files in os.walk(folder_path):
        for file in files:
            path = os.path.join(root, file)
            print(f"Loading {file}...")
            if file.lower().endswith('.pdf'):
                all_docs.extend(load_pdf(path))
            elif file.lower().endswith(('.txt', '.md')):
                all_docs.extend(load_txt(path))
    return all_docs

if __name__ == "__main__":
    os.makedirs("./data", exist_ok=True)
    with open("./data/sample.txt", "w", encoding="utf-8") as f:
        f.write("This is a sample document about AI. " * 20)
    
    docs = load_folder("./data")
    print(f"Loaded {len(docs)} documents.")

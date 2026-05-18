import os
import glob
from bs4 import BeautifulSoup
import requests
import PyPDF2

try:
    from pdf2image import convert_from_path
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

def load_pdf(file_path):
    """Reads a PDF file page by page. Uses OCR for scanned pages if available."""
    docs = []
    try:
        filename = os.path.basename(file_path)
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            
            # Cache images if OCR is needed
            images = None
            
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                
                # Fallback to OCR if page has no extractable text
                if (not text or not text.strip()) and OCR_AVAILABLE:
                    try:
                        if images is None:
                            images = convert_from_path(file_path)
                        if i < len(images):
                            print(f"Applying OCR to {filename} page {i+1}...")
                            text = pytesseract.image_to_string(images[i])
                    except Exception as ocr_e:
                        print(f"OCR failed for page {i+1}: {ocr_e}")
                        
                if text and text.strip():
                    docs.append({
                        "content": text.strip(),
                        "source": file_path,
                        "filename": filename,
                        "page": i + 1,
                        "type": "pdf"
                    })
        print(f"Loaded PDF: {filename} ({len(docs)} pages)")
    except Exception as e:
        print(f"Error loading PDF {file_path}: {e}")
    return docs

def load_txt(file_path):
    """Reads a plain text or markdown file."""
    docs = []
    try:
        filename = os.path.basename(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            if text:
                docs.append({
                    "content": text.strip(),
                    "source": file_path,
                    "filename": filename,
                    "page": 1,
                    "type": "txt" if file_path.endswith('.txt') else "md"
                })
        print(f"Loaded Text File: {filename}")
    except Exception as e:
        print(f"Error loading text file {file_path}: {e}")
    return docs

def load_url(url):
    """Scrapes content from a website."""
    docs = []
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
            
        text = soup.get_text(separator=' ')
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        if text:
            docs.append({
                "content": text.strip(),
                "source": url,
                "filename": url,
                "page": 1,
                "type": "url"
            })
        print(f"Loaded URL: {url}")
    except Exception as e:
        print(f"Error loading URL {url}: {e}")
    return docs

def load_folder(folder_path):
    """Loads all supported files from a directory."""
    all_docs = []
    try:
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            return all_docs
            
        files = []
        for ext in ('*.pdf', '*.txt', '*.md'):
            files.extend(glob.glob(os.path.join(folder_path, '**', ext), recursive=True))
            
        if not files:
            print(f"No supported files found in {folder_path}")
            return all_docs
            
        for file_path in files:
            ext = file_path.lower().split('.')[-1]
            if ext == 'pdf':
                all_docs.extend(load_pdf(file_path))
            elif ext in ('txt', 'md'):
                all_docs.extend(load_txt(file_path))
                
        print(f"Successfully loaded {len(files)} files from {folder_path}")
    except Exception as e:
        print(f"Error loading folder {folder_path}: {e}")
    return all_docs

if __name__ == "__main__":
    # Test loading
    print("Testing ingest.py")
    docs = load_txt("README.md") if os.path.exists("README.md") else []
    print(f"Loaded {len(docs)} documents.")

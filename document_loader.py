"""
Document loader for RAG (Retrieval-Augmented Generation) applications.

Loads PDF, DOCX, XLSX, CSV, JSON, TXT, MD, and source code files from a single file or a directory.
Each loaded chunk is returned as a dictionary with keys: content, source, page.
"""

import os
from pathlib import Path

# External dependencies
from PyPDF2 import PdfReader

# File extensions we support
SUPPORTED_EXTENSIONS = {
    # Documents
    ".pdf", ".docx", ".txt", ".md", ".markdown", ".html", ".htm", ".xml",
    # Data & Config
    ".csv", ".xlsx", ".xls", ".json", ".yml", ".yaml", ".ini", ".toml",
    # Source Code files
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".cpp", ".c", ".h", ".cs", ".go", ".sh", ".sql"
}


def load_pdf(file_path: str) -> list[dict]:
    """Load a PDF file and return one document dict per page."""
    documents = []
    source = os.path.abspath(file_path)

    try:
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)
        print(f"  Reading PDF: {os.path.basename(file_path)} ({total_pages} page(s))")

        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
                text = text.strip()

                documents.append(
                    {
                        "content": text,
                        "source": source,
                        "page": page_num,
                    }
                )
                print(f"    Page {page_num}/{total_pages} loaded ({len(text)} characters)")
            except Exception as e:
                print(f"    Warning: Could not read page {page_num}: {e}")
                documents.append(
                    {
                        "content": "",
                        "source": source,
                        "page": page_num,
                    }
                )

    except Exception as e:
        print(f"  Error: Failed to load PDF '{file_path}': {e}")

    return documents


def load_docx(file_path: str) -> list[dict]:
    """Load a Word Document (.docx) file."""
    import zipfile
    import xml.etree.ElementTree as ET
    
    documents = []
    source = os.path.abspath(file_path)
    try:
        with zipfile.ZipFile(file_path) as docx:
            xml_content = docx.read('word/document.xml')
            root = ET.fromstring(xml_content)
            namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            paragraphs = []
            for p in root.findall('.//w:p', namespaces):
                texts = [t.text for t in p.findall('.//w:t', namespaces) if t.text]
                if texts:
                    paragraphs.append("".join(texts))
            content = "\n\n".join(paragraphs).strip()
            documents.append({
                "content": content,
                "source": source,
                "page": 1
            })
            print(f"  Loaded Word file: {os.path.basename(file_path)} ({len(content)} characters)")
    except Exception as e:
        print(f"  Error loading Word document '{file_path}': {e}")
    return documents


def load_csv(file_path: str) -> list[dict]:
    """Load a CSV file, formatting each row into a key-value description."""
    import csv
    documents = []
    source = os.path.abspath(file_path)
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)
            rows = list(reader)
            if not rows:
                return []
            header = rows[0]
            content_lines = []
            for row_idx, row in enumerate(rows[1:], start=1):
                row_desc = []
                for col_idx, val in enumerate(row):
                    col_name = str(header[col_idx]).strip() if col_idx < len(header) and header[col_idx] else f"Column {col_idx+1}"
                    row_desc.append(f"{col_name}: {val}")
                content_lines.append(f"Row {row_idx}: " + " | ".join(row_desc))
            content = "\n".join(content_lines)
            documents.append({
                "content": content,
                "source": source,
                "page": 1
            })
            print(f"  Loaded CSV file: {os.path.basename(file_path)} ({len(content)} characters)")
    except Exception as e:
        print(f"  Error loading CSV '{file_path}': {e}")
    return documents


def load_json(file_path: str) -> list[dict]:
    """Load a JSON file, formatting the output structured text."""
    import json
    documents = []
    source = os.path.abspath(file_path)
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            data = json.load(f)
        content = json.dumps(data, indent=2)
        documents.append({
            "content": content,
            "source": source,
            "page": 1
        })
        print(f"  Loaded JSON file: {os.path.basename(file_path)} ({len(content)} characters)")
    except Exception as e:
        print(f"  Error loading JSON '{file_path}': {e}")
    return documents


def load_xlsx(file_path: str) -> list[dict]:
    """Load an Excel spreadsheet file sheet-by-sheet."""
    documents = []
    source = os.path.abspath(file_path)
    try:
        import openpyxl
        wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
        content_parts = []
        for sheet_name in wb.sheetnames:
            sheet = wb[sheet_name]
            content_parts.append(f"--- Sheet: {sheet_name} ---")
            
            rows = list(sheet.iter_rows(values_only=True))
            if not rows:
                continue
            header = rows[0]
            for r_idx, row in enumerate(rows[1:], start=1):
                if not any(val is not None for val in row):
                    continue
                row_desc = []
                for c_idx, val in enumerate(row):
                    if val is None:
                        val = ""
                    col_name = str(header[c_idx]).strip() if c_idx < len(header) and header[c_idx] is not None else f"Column {c_idx+1}"
                    row_desc.append(f"{col_name}: {val}")
                content_parts.append(f"Row {r_idx}: " + " | ".join(row_desc))
        content = "\n".join(content_parts)
        documents.append({
            "content": content,
            "source": source,
            "page": 1
        })
        print(f"  Loaded Excel spreadsheet: {os.path.basename(file_path)} ({len(content)} characters)")
    except Exception as e:
        print(f"  Error loading spreadsheet '{file_path}': {e}")
    return documents


def load_text_file(file_path: str) -> list[dict]:
    """Load a plain text, markdown, configuration, or source code file."""
    documents = []
    source = os.path.abspath(file_path)

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read().strip()

        documents.append(
            {
                "content": content,
                "source": source,
                "page": 1,
            }
        )
        print(f"  Loaded file: {os.path.basename(file_path)} ({len(content)} characters)")

    except Exception as e:
        print(f"  Error: Failed to load text file '{file_path}': {e}")

    return documents


def load_file(file_path: str) -> list[dict]:
    """Load one file based on its extension."""
    path = Path(file_path)
    extension = path.suffix.lower()

    if not path.is_file():
        print(f"  Skipping (not a file): {file_path}")
        return []

    if extension not in SUPPORTED_EXTENSIONS:
        print(f"  Skipping (unsupported type {extension}): {file_path}")
        return []

    print(f"Loading: {path.name}")

    if extension == ".pdf":
        return load_pdf(file_path)
    elif extension == ".docx":
        return load_docx(file_path)
    elif extension in (".xlsx", ".xls"):
        return load_xlsx(file_path)
    elif extension == ".csv":
        return load_csv(file_path)
    elif extension == ".json":
        return load_json(file_path)

    # Plain text, markdown, source code
    return load_text_file(file_path)


def load_directory(directory_path: str) -> list[dict]:
    """Load all supported files from a directory (non-recursive)."""
    all_documents = []
    directory = Path(directory_path)

    if not directory.is_dir():
        print(f"Error: '{directory_path}' is not a valid directory.")
        return all_documents

    files = sorted(
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        print(f"No supported files found in: {directory_path}")
        print(f"  Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        return all_documents

    print(f"\nFound {len(files)} file(s) in '{directory_path}'\n")

    for i, file_path in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] ", end="")
        try:
            docs = load_file(str(file_path))
            all_documents.extend(docs)
        except Exception as e:
            print(f"  Error: Unexpected failure for '{file_path}': {e}")

    print(f"\nDone. Loaded {len(all_documents)} document chunk(s) total.\n")
    return all_documents


def load_documents(path: str) -> list[dict]:
    """Main entry point: load from a single file or a directory."""
    path_obj = Path(path)

    if not path_obj.exists():
        print(f"Error: Path does not exist: {path}")
        return []

    if path_obj.is_file():
        print(f"\nLoading single file: {path}\n")
        return load_file(str(path_obj))

    if path_obj.is_dir():
        return load_directory(str(path_obj))

    print(f"Error: Path is neither a file nor a directory: {path}")
    return []


if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("Document Loader — self-test")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp_dir:
        sample_txt = Path(tmp_dir) / "notes.txt"
        sample_md = Path(tmp_dir) / "readme.md"

        sample_txt.write_text(
            "Hello from a text file.\nThis is sample content for RAG testing.",
            encoding="utf-8",
        )
        sample_md.write_text(
            "# Sample Markdown\n\n- Item one\n- Item two\n\nEnd of document.",
            encoding="utf-8",
        )

        print(f"\nTest directory: {tmp_dir}\n")
        results = load_documents(tmp_dir)

        print("Sample output:")
        print("-" * 40)
        for doc in results[:2]:
            preview = doc["content"][:80].replace("\n", " ")
            if len(doc["content"]) > 80:
                preview += "..."
            print(f"  source: {doc['source']}")
            print(f"  page:   {doc['page']}")
            print(f"  content: {preview}")
            print()

        print(f"Total chunks loaded: {len(results)}")
        print("=" * 60)

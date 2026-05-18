import uuid
import re

def chunk_documents(docs, strategy="recursive", chunk_size=500, overlap=50):
    """
    Chunks documents into smaller pieces.
    Strategies: 'recursive' (default) or 'markdown' (splits by headers).
    """
    all_chunks = []
    
    for doc in docs:
        content = doc.get('content', '')
        if not content:
            continue
            
        if strategy == "markdown":
            # Split by Markdown headers (e.g. ## )
            sections = re.split(r'\n(?=#{1,6}\s)', content)
            chunk_num = 1
            for section in sections:
                if len(section.strip()) >= 30:
                    all_chunks.append(_create_chunk_dict(section, doc, chunk_num))
                    chunk_num += 1
        else:
            # 1. Paragraph split
            paragraphs = re.split(r'\n\s*\n', content)
        
        current_chunk = ""
        chunk_num = 1
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # If a single paragraph is too large, split by sentences
            if len(para) > chunk_size:
                sentences = re.split(r'(?<=[.!?]) +', para)
                for sentence in sentences:
                    # If single sentence is too large, split by words
                    if len(sentence) > chunk_size:
                        words = sentence.split()
                        temp_sentence = ""
                        for word in words:
                            if len(temp_sentence) + len(word) + 1 <= chunk_size:
                                temp_sentence += word + " "
                            else:
                                if len(current_chunk) + len(temp_sentence) <= chunk_size:
                                    current_chunk += temp_sentence
                                else:
                                    if len(current_chunk) >= 30:
                                        all_chunks.append(_create_chunk_dict(current_chunk, doc, chunk_num))
                                        chunk_num += 1
                                        # Overlap
                                        current_chunk = current_chunk[-overlap:] if overlap > 0 else ""
                                    current_chunk += temp_sentence
                                temp_sentence = word + " "
                        
                        if temp_sentence:
                            if len(current_chunk) + len(temp_sentence) <= chunk_size:
                                current_chunk += temp_sentence
                            else:
                                if len(current_chunk) >= 30:
                                    all_chunks.append(_create_chunk_dict(current_chunk, doc, chunk_num))
                                    chunk_num += 1
                                    current_chunk = current_chunk[-overlap:] if overlap > 0 else ""
                                current_chunk += temp_sentence
                    else:
                        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                            current_chunk += sentence + " "
                        else:
                            if len(current_chunk) >= 30:
                                all_chunks.append(_create_chunk_dict(current_chunk, doc, chunk_num))
                                chunk_num += 1
                                current_chunk = current_chunk[-overlap:] if overlap > 0 else ""
                            current_chunk += sentence + " "
            else:
                if len(current_chunk) + len(para) + 2 <= chunk_size:
                    current_chunk += para + "\n\n"
                else:
                    if len(current_chunk) >= 30:
                        all_chunks.append(_create_chunk_dict(current_chunk, doc, chunk_num))
                        chunk_num += 1
                        current_chunk = current_chunk[-overlap:] if overlap > 0 else ""
                    current_chunk += para + "\n\n"
        
        if strategy != "markdown" and current_chunk.strip() and len(current_chunk.strip()) >= 30:
            all_chunks.append(_create_chunk_dict(current_chunk, doc, chunk_num))
            
    print(f"Created {len(all_chunks)} total chunks from {len(docs)} documents using '{strategy}' strategy.")
    return all_chunks

def _create_chunk_dict(content, original_doc, chunk_num):
    """Helper to format the chunk dictionary."""
    return {
        "content": content.strip(),
        "chunk_id": str(uuid.uuid4()),
        "source": original_doc.get("source", ""),
        "filename": original_doc.get("filename", ""),
        "page": original_doc.get("page", 1),
        "chunk_num": chunk_num
    }

if __name__ == "__main__":
    print("Testing chunker.py")
    test_docs = [{"content": "This is a very long text. " * 50, "source": "test.txt", "filename": "test.txt", "page": 1}]
    chunks = chunk_documents(test_docs, chunk_size=100, overlap=20)
    print(f"Generated {len(chunks)} chunks for testing.")

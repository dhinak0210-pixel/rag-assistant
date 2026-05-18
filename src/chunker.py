import os

def chunk_documents(docs, size=500, overlap=50):
    chunks = []
    for doc in docs:
        content = doc.get("content", "")
        # Find natural break points loosely; for simplicity just split and group
        words = content.split()
        current_chunk = []
        current_length = 0
        chunk_num = 0
        
        i = 0
        while i < len(words):
            word = words[i]
            if current_length + len(word) + 1 > size and current_chunk:
                chunk_text = " ".join(current_chunk)
                if len(chunk_text) >= 30:
                    chunks.append({
                        "content": chunk_text,
                        "chunk_id": f"{doc['filename']}_p{doc.get('page', 1)}_c{chunk_num}",
                        "source": doc["source"],
                        "filename": doc["filename"],
                        "page": doc.get("page", 1),
                        "chunk_num": chunk_num
                    })
                    chunk_num += 1
                
                # Backtrack for overlap
                overlap_words = []
                overlap_length = 0
                for w in reversed(current_chunk):
                    if overlap_length + len(w) + 1 > overlap:
                        break
                    overlap_words.insert(0, w)
                    overlap_length += len(w) + 1
                    
                current_chunk = overlap_words
                current_length = overlap_length
                
            current_chunk.append(word)
            current_length += len(word) + 1
            i += 1
            
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) >= 30:
                chunks.append({
                    "content": chunk_text,
                    "chunk_id": f"{doc['filename']}_p{doc.get('page', 1)}_c{chunk_num}",
                    "source": doc["source"],
                    "filename": doc["filename"],
                    "page": doc.get("page", 1),
                    "chunk_num": chunk_num
                })

    print(f"Created {len(chunks)} chunks from {len(docs)} docs")
    return chunks

if __name__ == "__main__":
    sample_docs = [{
        "content": "What is AI? Artificial Intelligence is fascinating. " * 50,
        "source": "sample.txt",
        "filename": "sample.txt",
        "page": 1
    }]
    chunks = chunk_documents(sample_docs, size=200, overlap=50)
    for c in chunks[:2]:
        print(c["chunk_id"], len(c["content"]))

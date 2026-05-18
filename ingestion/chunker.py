import os
import re
from typing import List, Dict

DATA_PATH = "data/processed"


def clean_text(text: str) -> str:
    
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def split_into_chunks(text: str, chunk_size=500, overlap=50):
    
    words = text.split()
    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]

        chunks.append(" ".join(chunk))

        start = end - overlap  

    return chunks


def load_and_chunk():
    
    all_chunks = []

    for file_name in os.listdir(DATA_PATH):
        if file_name.endswith(".txt"):
            file_path = os.path.join(DATA_PATH, file_name)

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

                text = clean_text(text)
                chunks = split_into_chunks(text)

                for i, chunk in enumerate(chunks):
                    all_chunks.append({
                        "file_name": file_name,
                        "chunk_id": i,
                        "text": chunk
                    })

    return all_chunks


if __name__ == "__main__":
    chunks = load_and_chunk()

    print(f"Total chunks created: {len(chunks)}\n")

    print("Sample chunk:\n")
    print(chunks[0]["text"])
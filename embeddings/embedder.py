from sentence_transformers import SentenceTransformer
import pickle
import os

#  embeddings model.
model = SentenceTransformer('all-MiniLM-L6-v2')


def generate_embeddings(chunks):
    
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(texts, show_progress_bar=True)

    return embeddings


def build_embedding_store(chunks):
    
    embeddings = generate_embeddings(chunks)

    vector_store = []

    for i, chunk in enumerate(chunks):
        vector_store.append({
            "file_name": chunk["file_name"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "embedding": embeddings[i]
        })

    return vector_store


def save_embeddings(vector_store, path="data/embeddings.pkl"):
    
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(vector_store, f)


if __name__ == "__main__":
    
    from ingestion.chunker import load_and_chunk

    chunks = load_and_chunk()

    print(f"Loaded {len(chunks)} chunks")

    vector_store = build_embedding_store(chunks)

    print(f"Generated embeddings for {len(vector_store)} chunks")

    save_embeddings(vector_store)

    print("Embeddings saved successfully ✔")
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')


def load_embeddings(path="data/embeddings.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data



def build_faiss_index(embeddings_data):
    vectors = np.array(
        [item["embedding"] for item in embeddings_data]
    ).astype("float32")

    dimension = vectors.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    return index, vectors



def search(query, index, embeddings_data, k=3):
    query_vector = model.encode([query]).astype("float32")

    distances, indices = index.search(query_vector, k)

    results = []

    for i in indices[0]:
        results.append(embeddings_data[i])

    return results


def test_search(query, index, embeddings_data, k=3):
    query_vector = model.encode([query]).astype("float32")

    distances, indices = index.search(query_vector, k)

    print("\n" + "="*60)
    print(f" QUERY: {query}")
    print("="*60)

    for rank, idx in enumerate(indices[0]):
        print(f"\nRank {rank+1}")
        print(f"File: {embeddings_data[idx]['file_name']}")
        print(f"Chunk ID: {embeddings_data[idx]['chunk_id']}")
        print("\nContent Preview:\n")
        print(embeddings_data[idx]["text"][:500])
        print("-"*60)



if __name__ == "__main__":
    print("Loading embeddings...")

    data = load_embeddings()

    print(f"Loaded {len(data)} embeddings")

    index, vectors = build_faiss_index(data)

    print("FAISS index built successfully ✔")
    print("Vector dimension:", vectors.shape[1])

  
    test_query = "What is ReLU activation function?"
    test_search(test_query, index, data)
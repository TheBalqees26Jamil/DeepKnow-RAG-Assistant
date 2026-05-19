from retrieval.retriever import load_embeddings, build_faiss_index, search
from llm.llm_client import generate_answer

def rag_pipeline(query, index, data, k=3):

    contexts = search(query, index, data, k=k)

    answer = generate_answer(query, contexts)

    return answer, contexts


if __name__ == "__main__":

    data = load_embeddings()
    index, _ = build_faiss_index(data)

    query = "What is ReLU?"

    answer, contexts = rag_pipeline(query, index, data)

    print("\nANSWER:\n")
    print(answer)

from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity



model = SentenceTransformer('all-MiniLM-L6-v2')

def answer_length_score(answer: str) -> float:
    
    if not answer or len(answer.strip()) < 20:
        return 0.0
    if len(answer.strip()) < 100:
        return 0.5
    return 1.0




def semantic_embed(text: str):
    """
    Real semantic embeddings using sentence-transformers
    """

    embedding = model.encode(text)

    return embedding.reshape(1, -1)



def groundedness_score(answer: str, contexts: List[str]) -> float:
    
    

    if not answer or not contexts:
        return 0.0

    answer_vec = semantic_embed(answer)

    context_text = " ".join(contexts)
    context_vec = semantic_embed(context_text)

    score = cosine_similarity(answer_vec, context_vec)[0][0]

    return float(max(0.0, min(score, 1.0)))




def relevance_score(query: str, answer: str) -> float:
    
    if not query or not answer:
        return 0.0

    query_vec = semantic_embed(query)
    answer_vec = semantic_embed(answer)

    score = cosine_similarity(query_vec, answer_vec)[0][0]

    return float(max(0.0, min(score, 1.0)))




def precision_at_k(retrieved_chunks: List[str], relevant_chunks: List[str]) -> float:
    """
    Precision@K
    """
    if not retrieved_chunks:
        return 0.0

    retrieved_set = set(retrieved_chunks)
    relevant_set = set(relevant_chunks)

    hits = retrieved_set & relevant_set

    return len(hits) / len(retrieved_set)


def recall_at_k(retrieved_chunks: List[str], relevant_chunks: List[str]) -> float:
    """
    Recall@K
    """
    if not relevant_chunks:
        return 0.0

    retrieved_set = set(retrieved_chunks)
    relevant_set = set(relevant_chunks)

    hits = retrieved_set & relevant_set

    return len(hits) / len(relevant_set)




def evaluate_rag(
    query: str,
    retrieved_chunks: List[str],
    answer: str,
    relevant_chunks: List[str] = None
) -> Dict[str, float]:
    

    results = {
        "answer_length_score": answer_length_score(answer),
        "groundedness_score": groundedness_score(answer, retrieved_chunks),
        "relevance_score": relevance_score(query, answer),
    }

    
    if relevant_chunks:
        results["retrieval_precision"] = precision_at_k(
            retrieved_chunks, relevant_chunks
        )
        results["retrieval_recall"] = recall_at_k(
            retrieved_chunks, relevant_chunks
        )

    #  Overall Score (Average Safe)
    results["overall_score"] = sum(results.values()) / len(results)

    return results
from fastapi import APIRouter , HTTPException

from retrieval.retriever import (
    load_embeddings,
    build_faiss_index,
    search
)

from llm.llm_client import generate_answer
from evaluation.metrics import evaluate_rag
from safety.guardrails import is_safe_query

from api.schemas import AskRequest, AskResponse


router = APIRouter()


# Load system once
print("Loading embeddings and FAISS index...")

data = load_embeddings()
index, _ = build_faiss_index(data)

print("System loaded successfully.")


@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "DeepKnow RAG API"
    }


@router.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):

    query = request.query

    # Safety check
    if not is_safe_query(query):
        raise HTTPException(
            status_code=400,
            detail="Blocked unsafe query."
        )

    try:

        # Retrieve contexts
        contexts = search(query, index, data, k=3)

        # Generate answer
        answer = generate_answer(query, contexts)

        # Evaluation
        evaluation = evaluate_rag(
            query=query,
            retrieved_chunks=[c["text"] for c in contexts],
            answer=answer
        )

        return {
            "answer": answer,
            "retrieved_chunks": contexts,
            "evaluation": evaluation
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
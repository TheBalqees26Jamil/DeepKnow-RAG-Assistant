from fastapi import APIRouter , HTTPException
from monitoring.drift_tracker import log_metrics
from monitoring.drift_detector import log_drift , check_drift

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

    if not is_safe_query(query):
        raise HTTPException(status_code=400, detail="Blocked unsafe query.")

    try:
        contexts = search(query, index, data, k=3)
        answer = generate_answer(query, contexts)

        if answer is None or answer == "":
            answer = "Unable to generate answer right now."

        evaluation = evaluate_rag(
            query=query,
            retrieved_chunks=[c["text"] for c in contexts],
            answer=answer
        )

        #
        try:
            log_drift(query, answer, contexts, evaluation)
            if check_drift(evaluation):
                print("⚠️ DRIFT ALERT: Model quality dropped!")
            
        except Exception as drift_error:
            print(f"Drift logging error (non-critical): {drift_error}")

        return {
            "answer": answer,
            "retrieved_chunks": contexts,
            "evaluation": evaluation
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
from fastapi import FastAPI
from api.routes import router


app = FastAPI(
    title="DeepKnow RAG API",
    description="Productionized RAG Backend Service",
    version="1.0.0"
)


app.include_router(router)


@app.get("/")
def root():
    return {
        "message": "DeepKnow RAG API is running"
    }
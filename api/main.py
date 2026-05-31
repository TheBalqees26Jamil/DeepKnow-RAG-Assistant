from fastapi import FastAPI
from api.routes import router
from prometheus_fastapi_instrumentator import Instrumentator


app = FastAPI(
    title="DeepKnow RAG API",
    description="Productionized RAG Backend Service",
    version="1.0.0"
)

app.include_router(router)

#  Monitoring (Prometheus)

instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
)

instrumentator.instrument(app).expose(app)


@app.get("/")
def root():
    return {
        "message": "DeepKnow RAG API is running"
    }
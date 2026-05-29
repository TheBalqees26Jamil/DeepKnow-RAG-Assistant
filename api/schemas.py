from pydantic import BaseModel
from typing import List, Dict


class AskRequest(BaseModel):
    query: str
    show_chunks: bool = False


class RetrievedChunk(BaseModel):
    file_name: str
    text: str


class AskResponse(BaseModel):
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    evaluation: Dict[str, float]
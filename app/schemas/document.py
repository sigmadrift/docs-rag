import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    status: str
    error: str | None
    created_at: datetime


class SearchHit(BaseModel):
    document_id: uuid.UUID
    filename: str
    seq: int
    content: str
    score: float


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


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


# top_k 상한: 청크마다 리랭커 추론과 LLM 컨텍스트 비용이 붙으므로 낮게 잡는다.
# (문서 목록 조회의 상한 200과 달리 검색은 청크당 연산이 무겁다)
_TOP_K = Field(5, ge=1, le=50)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = _TOP_K


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = _TOP_K

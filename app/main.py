from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import documents, search
from app.services.embedding import _model


@asynccontextmanager
async def lifespan(app: FastAPI):
    _model()  # 임베딩 모델 미리 로드 (첫 요청 지연 방지)
    yield


app = FastAPI(title="docs-rag", version="0.1.0", lifespan=lifespan)
app.include_router(documents.router, prefix="/api")
app.include_router(search.router, prefix="/api")


@app.get("/health", tags=["meta"])
async def health():
    return {"status": "ok"}

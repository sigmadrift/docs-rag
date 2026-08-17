from contextlib import asynccontextmanager

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import FastAPI

from app.api import ask, documents, search
from app.core.config import get_settings
from app.services import embedding


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedding.preload_and_check()  # 모델 미리 로드 + 차원 검증 (첫 요청 지연/저장 시점 오류 방지)
    # ingest 작업 큐. 워커(app/worker.py)가 같은 redis에서 작업을 소비한다.
    app.state.arq = await create_pool(RedisSettings.from_dsn(get_settings().redis_url))
    yield
    await app.state.arq.aclose()


app = FastAPI(title="docs-rag", version="0.1.0", lifespan=lifespan)
app.include_router(documents.router, prefix="/api")
app.include_router(search.router, prefix="/api")
app.include_router(ask.router, prefix="/api")


@app.get("/health", tags=["meta"])
async def health():
    return {"status": "ok"}

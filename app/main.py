from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import documents, search
from app.core.config import get_settings
from app.services.embedding import _model


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = _model()  # 임베딩 모델 미리 로드 (첫 요청 지연 방지)
    # DB 스키마(Vector 차원)는 settings.embedding_dim으로 만들어지므로, 모델 실제 출력과
    # 어긋나면 저장 시점에야 터진다. 시작 시점에 바로 실패시킨다.
    actual = model.get_sentence_embedding_dimension()
    expected = get_settings().embedding_dim
    if actual != expected:
        raise RuntimeError(
            f"임베딩 차원 불일치: 모델={actual}, 설정/DB={expected}. "
            f"EMBEDDING_DIM을 맞추고 마이그레이션을 다시 확인하세요."
        )
    yield


app = FastAPI(title="docs-rag", version="0.1.0", lifespan=lifespan)
app.include_router(documents.router, prefix="/api")
app.include_router(search.router, prefix="/api")


@app.get("/health", tags=["meta"])
async def health():
    return {"status": "ok"}

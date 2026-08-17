"""임베딩 모델 래퍼. 모델 로딩은 무겁기 때문에 프로세스당 1회만 수행."""
import asyncio
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.core.config import get_settings


@lru_cache
def _model() -> SentenceTransformer:
    return SentenceTransformer(get_settings().embedding_model)


async def embed(texts: list[str]) -> list[list[float]]:
    # CPU/GPU 바운드 작업이라 이벤트 루프를 막지 않도록 스레드로 넘김
    loop = asyncio.get_running_loop()
    vectors = await loop.run_in_executor(
        None, lambda: _model().encode(texts, normalize_embeddings=True)
    )
    return vectors.tolist()

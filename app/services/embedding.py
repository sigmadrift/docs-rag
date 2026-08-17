"""임베딩 모델 래퍼. 모델 로딩은 무겁기 때문에 프로세스당 1회만 수행."""
import asyncio
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.core.config import get_settings


@lru_cache
def _model() -> SentenceTransformer:
    return SentenceTransformer(get_settings().embedding_model)


def preload_and_check() -> None:
    """모델을 미리 로드하고 출력 차원이 설정/DB 스키마와 맞는지 검증한다.

    DB 스키마(Vector 차원)는 settings.embedding_dim으로 만들어지므로, 모델 실제 출력과
    어긋나면 저장 시점에야 터진다. 프로세스 시작 시점에 바로 실패시킨다. (API·워커 공용)
    """
    actual = _model().get_sentence_embedding_dimension()
    expected = get_settings().embedding_dim
    if actual != expected:
        raise RuntimeError(
            f"임베딩 차원 불일치: 모델={actual}, 설정/DB={expected}. "
            f"EMBEDDING_DIM을 맞추고 마이그레이션을 다시 확인하세요."
        )


async def embed(texts: list[str]) -> list[list[float]]:
    # CPU/GPU 바운드 작업이라 이벤트 루프를 막지 않도록 스레드로 넘김
    loop = asyncio.get_running_loop()
    vectors = await loop.run_in_executor(
        None, lambda: _model().encode(texts, normalize_embeddings=True)
    )
    return vectors.tolist()

"""Cross-encoder 리랭커. 벡터 검색이 뽑은 후보를 질의와 직접 대조해 재정렬한다.

임베딩(bi-encoder)은 질의와 청크를 따로 벡터화하므로 빠르지만 둘의 상호작용을 보지 못한다.
cross-encoder는 (질의, 청크) 쌍을 한 번에 넣어 정확한 대신 느리므로 상위 후보에만 적용한다.
"""
import asyncio
from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.core.config import get_settings
from app.schemas.document import SearchHit


@lru_cache
def _model() -> CrossEncoder:
    return CrossEncoder(get_settings().reranker_model)


def preload() -> None:
    """모델을 미리 로드한다 (첫 검색 요청이 모델 로딩 때문에 수십 초 걸리는 것을 방지)."""
    _model()


async def rerank(query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
    """후보를 재정렬해 상위 top_k만 돌려준다.

    반환된 hit의 score는 코사인 유사도가 아니라 리랭커 점수(0~1 관련도)로 교체된다.
    num_labels=1 모델이면 CrossEncoder가 sigmoid를 적용하므로 이미 0~1 범위다.
    """
    if not hits:
        return []
    loop = asyncio.get_running_loop()
    # 임베딩과 마찬가지로 CPU/GPU 바운드라 이벤트 루프를 막지 않도록 스레드로 넘김
    scores = await loop.run_in_executor(
        None, lambda: _model().predict([(query, h.content) for h in hits])
    )
    ranked = sorted(
        (h.model_copy(update={"score": float(s)}) for h, s in zip(hits, scores, strict=True)),
        key=lambda h: h.score,
        reverse=True,
    )
    return ranked[:top_k]

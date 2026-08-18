"""리랭커 재정렬 로직 — 실제 cross-encoder 모델은 로드하지 않고 점수만 가짜로 대체한다."""
import uuid

import pytest

from app.schemas.document import SearchHit
from app.services import reranker


class _FakeModel:
    """predict가 (query, content) 쌍을 받아 미리 정한 점수를 돌려주는 대역."""

    def __init__(self, scores: dict[str, float]):
        self.scores = scores

    def predict(self, pairs):
        return [self.scores[content] for _, content in pairs]


def _hit(content: str, score: float) -> SearchHit:
    return SearchHit(
        document_id=uuid.uuid4(), filename="f.txt", seq=0, content=content, score=score
    )


@pytest.fixture
def fake_model(monkeypatch):
    def _install(scores: dict[str, float]):
        monkeypatch.setattr(reranker, "_model", lambda: _FakeModel(scores))

    return _install


async def test_rerank_reorders_by_reranker_score(fake_model):
    # 벡터 점수 순서(a > b > c)와 리랭커 점수 순서(c > a > b)가 어긋나게 둔다
    hits = [_hit("a", 0.9), _hit("b", 0.8), _hit("c", 0.7)]
    fake_model({"a": 0.5, "b": 0.1, "c": 0.99})

    out = await reranker.rerank("질의", hits, top_k=3)

    assert [h.content for h in out] == ["c", "a", "b"]


async def test_rerank_replaces_score_and_truncates(fake_model):
    hits = [_hit("a", 0.9), _hit("b", 0.8), _hit("c", 0.7)]
    fake_model({"a": 0.5, "b": 0.1, "c": 0.99})

    out = await reranker.rerank("질의", hits, top_k=2)

    assert len(out) == 2
    # score는 코사인 유사도가 아니라 리랭커 점수로 바뀌어야 한다
    assert out[0].score == pytest.approx(0.99)
    assert out[1].score == pytest.approx(0.5)
    # 나머지 필드는 그대로 보존
    assert out[0].filename == "f.txt"


async def test_rerank_empty_hits_skips_model():
    # 모델을 대체하지 않았으므로, 빈 입력에서 모델을 건드리면 실제 로딩이 일어나 실패한다
    assert await reranker.rerank("질의", [], top_k=5) == []

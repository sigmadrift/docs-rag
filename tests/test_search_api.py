"""/api/search 통합 테스트 — 실제 postgres(pgvector)가 필요하며, 없으면 skip.

임베딩 모델은 로드하지 않고 가짜 벡터로 대체한다 (검색 SQL·필터·인증 경로 검증이 목적).
"""
import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core import config
from app.db.base import Base
from app.models import Chunk, Document
from app.services import reranker

DIM = 1024


def _fake_vec(t: str) -> list[float]:
    v = [0.0] * DIM
    v[0 if "압착" in t else 1] = 1.0
    return v


async def _fake_embed(texts: list[str]) -> list[list[float]]:
    return [_fake_vec(t) for t in texts]


@pytest.fixture
async def db(monkeypatch):
    """실제 .env의 DB로 연결. 접속 불가면 skip. 테스트 데이터는 끝나고 정리."""
    monkeypatch.setenv("API_KEY", "test-key")
    config.get_settings.cache_clear()
    settings = config.get_settings()

    engine = create_async_engine(settings.database_url)
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)
    except Exception as e:  # noqa: BLE001
        await engine.dispose()
        pytest.skip(f"DB 연결 불가 ({type(e).__name__}) — docker compose up -d 후 실행")

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    doc_ids = [uuid.uuid4() for _ in range(3)]
    async with session_factory() as s:
        # done 문서(정답), processing 문서(필터로 제외돼야 함), done이지만 무관한 문서
        s.add_all(
            [
                Document(id=doc_ids[0], filename="압착기준.txt", content_type="text/plain", status="done"),
                Document(id=doc_ids[1], filename="처리중.txt", content_type="text/plain", status="processing"),
                Document(id=doc_ids[2], filename="무관.txt", content_type="text/plain", status="done"),
            ]
        )
        await s.flush()
        s.add_all(
            [
                Chunk(document_id=doc_ids[0], seq=0, content="압착 불량 판정 기준", embedding=_fake_vec("압착")),
                Chunk(document_id=doc_ids[1], seq=0, content="압착 관련이지만 처리중", embedding=_fake_vec("압착")),
                Chunk(document_id=doc_ids[2], seq=0, content="전혀 다른 내용", embedding=_fake_vec("기타")),
            ]
        )
        await s.commit()

    yield session_factory

    async with session_factory() as s:
        await s.execute(delete(Document).where(Document.id.in_(doc_ids)))
        await s.commit()
    await engine.dispose()
    config.get_settings.cache_clear()


@pytest.fixture
async def client(db, monkeypatch):
    monkeypatch.setattr("app.services.embedding.embed", _fake_embed)

    from app.db.session import get_session
    from app.main import app

    async def _test_session():
        async with db() as s:
            yield s

    app.dependency_overrides[get_session] = _test_session
    try:
        # transport만 사용 → lifespan(모델 로딩, arq 연결) 건너뜀
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
            yield c
    finally:
        app.dependency_overrides.clear()


async def test_search_requires_api_key(client):
    r = await client.post("/api/search", json={"query": "압착", "top_k": 3})
    assert r.status_code == 401


async def test_search_returns_done_docs_only(client):
    r = await client.post(
        "/api/search",
        json={"query": "압착 불량", "top_k": 5},
        headers={"X-API-Key": "test-key"},
    )
    assert r.status_code == 200
    hits = r.json()
    assert hits, "검색 결과가 비어 있음"
    # 최상위 히트는 done 상태의 압착 문서여야 하고, processing 문서는 결과에 없어야 함
    assert hits[0]["filename"] == "압착기준.txt"
    assert hits[0]["score"] == pytest.approx(1.0)
    assert all(h["filename"] != "처리중.txt" for h in hits)


class _FakeReranker:
    """content별 점수를 미리 정해둔 cross-encoder 대역 (모르는 청크는 0점)."""

    def __init__(self, scores: dict[str, float]):
        self.scores = scores

    def predict(self, pairs):
        return [self.scores.get(content, 0.0) for _, content in pairs]


async def test_search_reranks_candidates(client, monkeypatch):
    """리랭킹을 켜면 top_k보다 넓게 후보를 뽑아 재정렬한다.

    top_k=1이어도 후보를 rerank_candidates만큼 가져오기 때문에, 벡터 1위가 아니었던
    청크가 리랭커 점수로 1위가 될 수 있다.
    """
    settings = config.get_settings()
    monkeypatch.setattr(settings, "rerank_enabled", True)
    monkeypatch.setattr(settings, "rerank_candidates", 10)
    # 벡터 검색 1위("압착 불량 판정 기준")를 리랭커가 아래로 끌어내리도록 점수를 뒤집는다
    monkeypatch.setattr(
        reranker,
        "_model",
        lambda: _FakeReranker({"압착 불량 판정 기준": 0.1, "전혀 다른 내용": 0.9}),
    )

    r = await client.post(
        "/api/search",
        json={"query": "압착 불량", "top_k": 1},
        headers={"X-API-Key": "test-key"},
    )

    assert r.status_code == 200
    hits = r.json()
    assert len(hits) == 1
    assert hits[0]["filename"] == "무관.txt"
    assert hits[0]["score"] == pytest.approx(0.9)

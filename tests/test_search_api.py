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
from app.services import rag, reranker

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
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
            await conn.run_sync(Base.metadata.create_all)
    except Exception as e:  # noqa: BLE001
        await engine.dispose()
        pytest.skip(f"DB 연결 불가 ({type(e).__name__}) — docker compose up -d 후 실행")

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    doc_ids = [uuid.uuid4() for _ in range(4)]
    async with session_factory() as s:
        # done 문서(정답), processing 문서(필터로 제외돼야 함), done이지만 무관한 문서
        s.add_all(
            [
                Document(id=doc_ids[0], filename="압착기준.txt", content_type="text/plain", status="done"),
                Document(id=doc_ids[1], filename="처리중.txt", content_type="text/plain", status="processing"),
                Document(id=doc_ids[2], filename="무관.txt", content_type="text/plain", status="done"),
                Document(id=doc_ids[3], filename="품번규격.txt", content_type="text/plain", status="done"),
            ]
        )
        await s.flush()
        s.add_all(
            [
                Chunk(document_id=doc_ids[0], seq=0, content="압착 불량 판정 기준", embedding=_fake_vec("압착")),
                Chunk(document_id=doc_ids[1], seq=0, content="압착 관련이지만 처리중", embedding=_fake_vec("압착")),
                Chunk(document_id=doc_ids[2], seq=0, content="전혀 다른 내용", embedding=_fake_vec("기타")),
                # 임베딩은 질의와 먼 방향(=벡터로는 안 잡힘)이지만 질의어를 문자열로 포함한다.
                # 조사가 붙어 있어 어절 단위 전문검색이라면 놓쳤을 케이스이기도 하다.
                Chunk(document_id=doc_ids[3], seq=0,
                      content="품번 XK-2201 압착을 규격대로 확인한다", embedding=_fake_vec("기타")),
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


@pytest.fixture
def stages(db, monkeypatch):
    """검색 단계 on/off를 테스트마다 명시한다 (설정 기본값이 바뀌어도 테스트가 흔들리지 않게).

    기본은 벡터 단독. 하이브리드/리랭커가 필요한 테스트에서 각자 켠다.
    """
    s = config.get_settings()
    monkeypatch.setattr(s, "hybrid_enabled", False)
    monkeypatch.setattr(s, "rerank_enabled", False)
    return s


async def test_search_requires_api_key(client):
    r = await client.post("/api/search", json={"query": "압착", "top_k": 3})
    assert r.status_code == 401


async def test_search_returns_done_docs_only(client, stages):
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


async def test_search_reranks_candidates(client, stages, monkeypatch):
    """리랭킹을 켜면 top_k보다 넓게 후보를 뽑아 재정렬한다.

    top_k=1이어도 후보를 rerank_candidates만큼 가져오기 때문에, 벡터 1위가 아니었던
    청크가 리랭커 점수로 1위가 될 수 있다.
    """
    monkeypatch.setattr(stages, "rerank_enabled", True)
    monkeypatch.setattr(stages, "candidate_k", 10)
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


async def test_keyword_search_matches_across_particles(db, stages):
    """트라이그램 키워드 검색은 조사가 붙어 있어도 매칭한다.

    "압착 규격"으로 "…압착을 규격대로…"를 찾는다. 어절 단위 전문검색이라면 조사 때문에
    놓쳤을 케이스다. (word_similarity 0.5 — pg_trgm 기본 임계값 0.6이면 탈락하므로
    keyword_threshold를 낮춰 잡은 이유이기도 하다.)
    """
    async with db() as session:
        hits = await rag._keyword_search(session, "압착 규격", limit=10)

    assert any(h.filename == "품번규격.txt" for h in hits)
    # 관련 없는 청크(유사도 0.2대)는 임계값에서 걸러진다
    assert all(h.filename != "무관.txt" for h in hits)


async def test_hybrid_adds_keyword_only_hit(client, stages, monkeypatch):
    """하이브리드는 벡터가 후보로 못 올린 청크를 키워드 쪽에서 끌어올린다."""
    monkeypatch.setattr(stages, "candidate_k", 1)
    headers = {"X-API-Key": "test-key"}
    query = "품번 XK-2201 압착"

    # 벡터 단독: 품번규격.txt는 임베딩 방향이 질의와 달라 후보 1개 안에 들지 못한다
    r = await client.post("/api/search", json={"query": query, "top_k": 1}, headers=headers)
    assert [h["filename"] for h in r.json()] == ["압착기준.txt"]

    # 하이브리드: 문자열이 거의 그대로 일치(0.93)해 키워드 1위로 올라오고, RRF로 합류한다
    monkeypatch.setattr(stages, "hybrid_enabled", True)
    r = await client.post("/api/search", json={"query": query, "top_k": 2}, headers=headers)
    assert "품번규격.txt" in [h["filename"] for h in r.json()]

from sqlalchemy import func, literal, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models import Chunk, Document
from app.schemas.document import SearchHit
from app.services import embedding, reranker

# RRF 상수. 클수록 상위 순위의 우위가 약해진다. 60은 원 논문(Cormack 2009)의 기본값.
_RRF_K = 60


def _to_hits(rows, score_fn) -> list[SearchHit]:
    return [
        SearchHit(
            document_id=chunk.document_id,
            filename=filename,
            seq=chunk.seq,
            content=chunk.content,
            score=score_fn(raw),
        )
        for chunk, filename, raw in rows
    ]


async def _vector_search(session: AsyncSession, query: str, limit: int) -> list[SearchHit]:
    # score = 1 - cosine_distance = 코사인 유사도. 정규화 임베딩 기준 이론 범위 [-1, 1].
    # 주의: status='done' 필터 + HNSW 인덱스 조합은 후보를 스캔 후 걸러내므로, done 비율이
    # 낮아지면 limit보다 적게 반환될 수 있다 (pgvector 0.8+의 hnsw.iterative_scan으로 완화 가능).
    [qvec] = await embedding.embed([query])
    distance = Chunk.embedding.cosine_distance(qvec)
    stmt = (
        select(Chunk, Document.filename, distance.label("distance"))
        .join(Document, Chunk.document_id == Document.id)
        .where(Document.status == "done")
        .order_by(distance)
        .limit(limit)
    )
    return _to_hits((await session.execute(stmt)).all(), lambda d: 1.0 - d)


async def _keyword_search(session: AsyncSession, query: str, limit: int) -> list[SearchHit]:
    """트라이그램 키워드 검색. 벡터가 놓치는 정확한 용어·품번·수치를 잡아준다.

    한국어는 조사가 붙어 어절 단위 전문검색이 자주 빗나가므로, 3글자 단위로 쪼개
    부분 매칭하는 word_similarity(<%)를 쓴다. gin_trgm_ops 인덱스를 탄다.
    """
    # <% 연산자의 임계값은 GUC라서 쿼리 단위로 정해야 한다. is_local=True라 이 트랜잭션에만 적용.
    await session.execute(
        select(func.set_config("pg_trgm.word_similarity_threshold",
                               str(get_settings().keyword_threshold), True))
    )
    similarity = func.word_similarity(literal(query), Chunk.content)
    stmt = (
        select(Chunk, Document.filename, similarity.label("similarity"))
        .join(Document, Chunk.document_id == Document.id)
        .where(Document.status == "done", literal(query).op("<%")(Chunk.content))
        .order_by(similarity.desc())
        .limit(limit)
    )
    return _to_hits((await session.execute(stmt)).all(), float)


def fuse(rankings: list[list[SearchHit]], k: int = _RRF_K) -> list[SearchHit]:
    """여러 순위 목록을 RRF(Reciprocal Rank Fusion)로 합친다.

    점수가 아니라 순위만 쓰기 때문에, 스케일이 전혀 다른 검색(코사인 유사도 vs 문자열
    유사도)을 정규화 없이 합칠 수 있다. 반환된 hit의 score는 RRF 점수로 교체된다.
    """
    fused: dict[tuple, list] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking):
            key = (hit.document_id, hit.seq)  # 청크를 유일하게 식별
            entry = fused.setdefault(key, [0.0, hit])
            entry[0] += 1.0 / (k + rank + 1)
    return sorted(
        (hit.model_copy(update={"score": score}) for score, hit in fused.values()),
        key=lambda h: h.score,
        reverse=True,
    )


async def search(session: AsyncSession, query: str, top_k: int = 5) -> list[SearchHit]:
    """벡터(+키워드) 검색 → 융합 → 재정렬 → 상위 top_k.

    응답의 score는 켜진 단계에 따라 의미가 다르다: 기본은 코사인 유사도,
    하이브리드를 켜면 RRF 점수, 리랭커까지 켜면 리랭커 관련도(0~1).
    """
    settings = get_settings()
    # 융합·재정렬이 있으면 벡터 검색은 후보만 넓게 뽑고, 최종 순위는 뒷단계가 정한다.
    widen = settings.hybrid_enabled or settings.rerank_enabled
    limit = max(top_k, settings.candidate_k) if widen else top_k

    hits = await _vector_search(session, query, limit)
    if settings.hybrid_enabled:
        hits = fuse([hits, await _keyword_search(session, query, limit)])
    if settings.rerank_enabled:
        return await reranker.rerank(query, hits, top_k)
    return hits[:top_k]


def build_messages(question: str, hits: list[SearchHit]) -> list[dict]:
    """검색 결과를 근거로 답하게 하는 chat 메시지 구성."""
    context = "\n\n".join(f"[{h.filename} #{h.seq}]\n{h.content}" for h in hits)
    system = (
        "너는 사내 문서 기반 질의응답 어시스턴트다. 아래 문서 발췌만을 근거로 한국어로 답하라. "
        "발췌에 없는 내용은 추측하지 말고 모른다고 답하라. "
        "답변에 사용한 근거는 [파일명 #번호] 형식으로 인용하라."
    )
    user = f"문서 발췌:\n{context}\n\n질문: {question}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

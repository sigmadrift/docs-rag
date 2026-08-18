from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models import Chunk, Document
from app.schemas.document import SearchHit
from app.services import embedding, reranker


async def search(session: AsyncSession, query: str, top_k: int = 5) -> list[SearchHit]:
    # score = 1 - cosine_distance = 코사인 유사도. 정규화 임베딩 기준 이론 범위 [-1, 1].
    # 주의: status='done' 필터 + HNSW 인덱스 조합은 후보를 스캔 후 걸러내므로, done 비율이
    # 낮아지면 top_k보다 적게 반환될 수 있다 (pgvector 0.8+의 hnsw.iterative_scan으로 완화 가능).
    settings = get_settings()
    # 리랭킹을 켜면 벡터 검색은 후보만 넓게 뽑고, 최종 순위는 cross-encoder가 정한다.
    limit = max(top_k, settings.rerank_candidates) if settings.rerank_enabled else top_k
    [qvec] = await embedding.embed([query])
    distance = Chunk.embedding.cosine_distance(qvec)
    stmt = (
        select(Chunk, Document.filename, distance.label("distance"))
        .join(Document, Chunk.document_id == Document.id)
        .where(Document.status == "done")
        .order_by(distance)
        .limit(limit)
    )
    rows = (await session.execute(stmt)).all()
    hits = [
        SearchHit(
            document_id=chunk.document_id,
            filename=filename,
            seq=chunk.seq,
            content=chunk.content,
            score=1.0 - dist,
        )
        for chunk, filename, dist in rows
    ]
    if not settings.rerank_enabled:
        return hits
    return await reranker.rerank(query, hits, top_k)


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

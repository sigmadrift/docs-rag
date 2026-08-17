from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk, Document
from app.schemas.document import SearchHit
from app.services import embedding


async def search(session: AsyncSession, query: str, top_k: int = 5) -> list[SearchHit]:
    [qvec] = await embedding.embed([query])
    distance = Chunk.embedding.cosine_distance(qvec)
    stmt = (
        select(Chunk, Document.filename, distance.label("distance"))
        .join(Document, Chunk.document_id == Document.id)
        .where(Document.status == "done")
        .order_by(distance)
        .limit(top_k)
    )
    rows = (await session.execute(stmt)).all()
    return [
        SearchHit(
            document_id=chunk.document_id,
            filename=filename,
            seq=chunk.seq,
            content=chunk.content,
            score=1.0 - dist,
        )
        for chunk, filename, dist in rows
    ]

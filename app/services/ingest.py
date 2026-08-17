"""업로드된 파일을 파싱 → 청킹 → 임베딩 → 저장. BackgroundTasks에서 호출됨."""
import uuid
from pathlib import Path

from sqlalchemy import select

from app.db.session import SessionLocal
from app.models import Chunk, Document
from app.services import chunking, embedding, parser


async def process_document(document_id: uuid.UUID, path: Path) -> None:
    async with SessionLocal() as session:
        doc = await session.scalar(select(Document).where(Document.id == document_id))
        if doc is None:
            return
        doc.status = "processing"
        await session.commit()

        try:
            text = parser.extract_text(path, doc.content_type)
            pieces = chunking.split_text(text)
            vectors = await embedding.embed(pieces) if pieces else []
            session.add_all(
                Chunk(document_id=doc.id, seq=i, content=c, embedding=v)
                for i, (c, v) in enumerate(zip(pieces, vectors))
            )
            doc.status = "done"
        except Exception as e:  # noqa: BLE001
            doc.status = "failed"
            doc.error = str(e)[:2000]
        await session.commit()

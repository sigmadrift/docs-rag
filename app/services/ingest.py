"""업로드된 파일을 파싱 → 청킹 → 임베딩 → 저장. arq 워커(app/worker.py)에서 호출됨."""
import uuid
from pathlib import Path

from sqlalchemy import delete

from app.db.session import SessionLocal
from app.models import Chunk, Document
from app.services import chunking, embedding, parser


async def process_document(document_id: uuid.UUID, path: Path) -> None:
    async with SessionLocal() as session:
        doc = await session.get(Document, document_id)
        if doc is None:
            return
        doc.status = "processing"
        doc.error = None
        await session.commit()

        try:
            text = parser.extract_text(path, doc.content_type)
            pieces = chunking.split_text(text)
            vectors = await embedding.embed(pieces) if pieces else []
            # 재시도 대비: 이전 시도가 남긴 청크가 있으면 지우고 새로 쓴다 (중복 방지)
            await session.execute(delete(Chunk).where(Chunk.document_id == doc.id))
            session.add_all(
                Chunk(document_id=doc.id, seq=i, content=c, embedding=v)
                for i, (c, v) in enumerate(zip(pieces, vectors, strict=True))
            )
            doc.status = "done"
            await session.commit()
        except Exception as e:
            # DB 오류로 세션이 오염됐을 수 있으므로 rollback 후 문서를 다시 읽어 기록
            await session.rollback()
            doc = await session.get(Document, document_id)
            if doc is not None:
                doc.status = "failed"
                doc.error = str(e)[:2000]
                await session.commit()
            raise  # arq가 재시도(max_tries)할 수 있게 다시 던진다. 최종 실패 시 상태는 failed로 남음

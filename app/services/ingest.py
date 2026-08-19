"""업로드된 파일을 파싱 → 청킹 → 임베딩 → 저장. arq 워커(app/worker.py)에서 호출됨."""
import asyncio
import logging
import uuid
from pathlib import Path

from sqlalchemy import delete

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.models import Chunk, Document
from app.services import chunking, embedding, parser, table

log = logging.getLogger(__name__)


async def _mark_failed(document_id: uuid.UUID, error: str) -> None:
    """실패 상태만 기록한다. 호출한 쪽 세션이 취소·오염됐을 수 있어 새 세션을 쓴다."""
    async with SessionLocal() as session:
        doc = await session.get(Document, document_id)
        if doc is not None:
            doc.status = "failed"
            doc.error = error[:2000]
            await session.commit()


async def process_document(document_id: uuid.UUID, path: Path) -> None:
    async with SessionLocal() as session:
        doc = await session.get(Document, document_id)
        if doc is None:
            return
        doc.status = "processing"
        doc.error = None
        await session.commit()

        try:
            if parser.is_spreadsheet(path, doc.content_type):
                # 표는 행마다 의미가 완결되므로 행을 그대로 청크로 쓴다
                pieces = table.extract_rows(path)
            else:
                pieces = chunking.split_text(parser.extract_text(path, doc.content_type))
            # 한 번에 다 넣지 않고 나눠 부른다. 수백 개를 한 통에 넘기면 진행 상황이
            # 전혀 보이지 않고, 작업 타임아웃에 걸려도 어디까지 됐는지 알 수 없다.
            batch = get_settings().embed_batch_size
            vectors: list[list[float]] = []
            for start in range(0, len(pieces), batch):
                vectors.extend(await embedding.embed(pieces[start : start + batch]))
                log.info("임베딩 %d/%d (%s)", len(vectors), len(pieces), doc.filename)
            # 재시도 대비: 이전 시도가 남긴 청크가 있으면 지우고 새로 쓴다 (중복 방지)
            await session.execute(delete(Chunk).where(Chunk.document_id == doc.id))
            session.add_all(
                Chunk(document_id=doc.id, seq=i, content=c, embedding=v)
                for i, (c, v) in enumerate(zip(pieces, vectors, strict=True))
            )
            doc.status = "done"
            await session.commit()
        except (Exception, asyncio.CancelledError) as e:
            # 작업 타임아웃은 CancelledError로 오는데 이건 Exception이 아니라서, 놓치면
            # 문서가 processing 상태에 영원히 갇힌다(워커 재시작 전까지 복구되지 않는다).
            # 취소된 태스크에서는 기존 세션의 후속 await도 즉시 취소되므로 shield로 감싼다.
            await asyncio.shield(_mark_failed(document_id, str(e) or type(e).__name__))
            raise  # arq가 재시도(max_tries)할 수 있게 다시 던진다. 최종 실패 시 상태는 failed로 남음

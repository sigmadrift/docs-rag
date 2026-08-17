import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.security import require_api_key
from app.db.session import get_session
from app.models import Document
from app.schemas.document import DocumentOut
from app.services.ingest import process_document

router = APIRouter(prefix="/documents", tags=["documents"], dependencies=[Depends(require_api_key)])


@router.post("", response_model=DocumentOut, status_code=202)
async def upload(
    file: UploadFile,
    background: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
):
    # 파일명은 basename만 사용 (경로 구분자가 섞인 이름으로 uploads 밖에 쓰는 것 방지)
    safe_name = Path(file.filename or "unnamed").name or "unnamed"

    # 파일 저장을 DB 커밋보다 먼저: 저장이 실패하면 pending 상태로 남는 문서가 생기지 않는다
    doc_id = uuid.uuid4()
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / f"{doc_id}_{safe_name}"
    dest.write_bytes(await file.read())

    doc = Document(id=doc_id, filename=safe_name, content_type=file.content_type or "text/plain")
    session.add(doc)
    await session.commit()

    background.add_task(process_document, doc.id, dest)
    return doc


@router.get("", response_model=list[DocumentOut])
async def list_documents(session: AsyncSession = Depends(get_session)):
    return (await session.scalars(select(Document).order_by(Document.created_at.desc()))).all()


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(document_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    doc = await session.get(Document, document_id)
    if doc is None:
        raise HTTPException(404, "document not found")
    return doc


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
):
    doc = await session.get(Document, document_id)
    if doc is None:
        raise HTTPException(404, "document not found")
    await session.delete(doc)
    await session.commit()
    for f in Path(settings.upload_dir).glob(f"{document_id}_*"):
        f.unlink(missing_ok=True)

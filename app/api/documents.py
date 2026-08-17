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
    doc = Document(filename=file.filename or "unnamed", content_type=file.content_type or "text/plain")
    session.add(doc)
    await session.commit()

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(exist_ok=True)
    dest = upload_dir / f"{doc.id}_{doc.filename}"
    dest.write_bytes(await file.read())

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
async def delete_document(document_id: uuid.UUID, session: AsyncSession = Depends(get_session)):
    doc = await session.get(Document, document_id)
    if doc is None:
        raise HTTPException(404, "document not found")
    await session.delete(doc)
    await session.commit()

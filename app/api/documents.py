import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.security import require_api_key
from app.db.session import get_session
from app.models import Document
from app.schemas.document import DocumentOut

router = APIRouter(prefix="/documents", tags=["documents"], dependencies=[Depends(require_api_key)])


async def _save_upload(file: UploadFile, dest: Path, max_bytes: int) -> None:
    """업로드를 청크 단위로 디스크에 저장. 전체를 메모리에 올리지 않고, 크기 초과 시 413."""
    size = 0
    with dest.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"파일이 너무 큼 (최대 {max_bytes // (1024 * 1024)}MB)")
            out.write(chunk)


@router.post("", response_model=DocumentOut, status_code=202)
async def upload(
    file: UploadFile,
    request: Request,
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
    await _save_upload(file, dest, settings.max_upload_mb * 1024 * 1024)

    doc = Document(id=doc_id, filename=safe_name, content_type=file.content_type or "text/plain")
    session.add(doc)
    try:
        await session.commit()
    except Exception:
        dest.unlink(missing_ok=True)  # DB에 없는 문서의 파일이 디스크에 남지 않게
        raise

    # job_id=문서 id: 워커 재시작 복구가 재등록해도 큐에 남아 있는 작업과 중복되지 않는다
    try:
        await request.app.state.arq.enqueue_job(
            "process_document", str(doc.id), str(dest), _job_id=str(doc.id)
        )
    except Exception:  # noqa: BLE001  redis 오류 종류가 다양해 전부 실패 상태로 강등
        doc.status = "failed"
        doc.error = "작업 큐 등록 실패 (redis 연결 확인)"
        await session.commit()
        raise HTTPException(503, "작업 큐에 연결할 수 없음")
    return doc


@router.get("", response_model=list[DocumentOut])
async def list_documents(
    session: AsyncSession = Depends(get_session),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    stmt = select(Document).order_by(Document.created_at.desc()).limit(limit).offset(offset)
    return (await session.scalars(stmt)).all()


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

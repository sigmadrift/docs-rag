"""arq 인제스트 워커. API 프로세스와 분리되어 재시작·재시도에 안전하다.

실행: uv run arq app.worker.WorkerSettings
"""
import logging
import uuid
from pathlib import Path
from typing import ClassVar

from arq.connections import RedisSettings
from arq.constants import result_key_prefix
from sqlalchemy import select

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.models import Document
from app.services import embedding, ingest

log = logging.getLogger(__name__)

# arq는 자기 로거만 설정하므로, 앱 모듈(app.*)의 INFO 로그가 보이도록 루트를 잡아준다.
# 인제스트 진행률(임베딩 N/M)이 여기로 나온다.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


async def process_document(ctx: dict, document_id: str, path: str) -> None:
    await ingest.process_document(uuid.UUID(document_id), Path(path))


async def startup(ctx: dict) -> None:
    embedding.preload_and_check()

    # 재시작 복구: 워커가 죽어 pending/processing으로 남은 문서를 다시 큐에 넣는다.
    # job_id=문서 id라서 큐에 아직 남아 있는 작업과는 중복 등록되지 않는다.
    async with SessionLocal() as session:
        stuck = (
            await session.scalars(
                select(Document).where(Document.status.in_(("pending", "processing")))
            )
        ).all()
        for doc in stuck:
            files = sorted(Path(get_settings().upload_dir).glob(f"{doc.id}_*"))
            if not files:
                doc.status = "failed"
                doc.error = "복구 실패: 업로드 원본 파일이 없음"
                continue
            # 이전 시도의 결과가 남아 있으면 arq가 같은 job_id를 중복으로 보고 조용히 무시한다.
            # (실패한 문서가 keep_result 기간 내내 재처리되지 못하는 것을 막는다)
            await ctx["redis"].delete(result_key_prefix + str(doc.id))
            job = await ctx["redis"].enqueue_job(
                "process_document", str(doc.id), str(files[0]), _job_id=str(doc.id)
            )
            log.info(
                "미완료 문서 재등록: %s (%s)%s",
                doc.id, doc.filename, "" if job else " — 이미 큐에 있음",
            )
        await session.commit()


class WorkerSettings:
    functions: ClassVar = [process_document]
    on_startup = startup
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    job_timeout = get_settings().ingest_job_timeout
    max_tries = 3

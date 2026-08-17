"""검색 → LLM 답변 생성 → SSE 스트리밍.

이벤트 순서: sources(검색 근거 JSON) → delta(토큰 조각들) → done
"""
import json

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.core.security import require_api_key
from app.db.session import get_session
from app.schemas.document import AskRequest
from app.services import llm, rag

router = APIRouter(prefix="/ask", tags=["ask"], dependencies=[Depends(require_api_key)])


@router.post("")
async def ask(body: AskRequest, session: AsyncSession = Depends(get_session)):
    # 검색은 스트리밍 시작 전에 끝낸다 (세션은 응답 시작 후 닫히므로 생성기 안에서 쓰지 않음)
    hits = await rag.search(session, body.question, body.top_k)

    async def gen():
        yield {
            "event": "sources",
            "data": json.dumps([h.model_dump(mode="json") for h in hits], ensure_ascii=False),
        }
        try:
            async for delta in llm.stream_chat(rag.build_messages(body.question, hits)):
                yield {"event": "delta", "data": delta}
        except Exception as e:  # noqa: BLE001  스트림 도중엔 상태코드를 못 바꾸므로 이벤트로 전달
            yield {"event": "error", "data": str(e)[:500]}
            return
        yield {"event": "done", "data": ""}

    return EventSourceResponse(gen())

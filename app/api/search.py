from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import require_api_key
from app.db.session import get_session
from app.schemas.document import SearchHit, SearchRequest
from app.services import rag

router = APIRouter(prefix="/search", tags=["search"], dependencies=[Depends(require_api_key)])


@router.post("", response_model=list[SearchHit])
async def search(body: SearchRequest, session: AsyncSession = Depends(get_session)):
    return await rag.search(session, body.query, body.top_k)

"""MCP 서버. REST와 동일한 서비스 레이어를 재사용한다.

실행: uv run python -m app.mcp_server.server  (Streamable HTTP, :8001/mcp)
"""
import uuid

from mcp.server import MCPServer
from sqlalchemy import select

from app.db.session import SessionLocal
from app.models import Chunk, Document
from app.services import rag

mcp = MCPServer("docs-rag")


@mcp.tool()
async def search_documents(query: str, top_k: int = 5) -> list[dict]:
    """사내 문서(작업표준서, 불량이력, 도면 메타데이터 등)에서 질의와 의미적으로 유사한
    구절을 검색한다. 결과는 유사도 순이며 각 항목에 document_id, filename, seq, content, score를 포함한다.
    답변할 때는 반드시 filename과 seq를 근거로 인용할 것."""
    async with SessionLocal() as session:
        hits = await rag.search(session, query, top_k)
    return [h.model_dump(mode="json") for h in hits]


@mcp.tool()
async def list_documents() -> list[dict]:
    """인덱싱된 문서 목록과 처리 상태(pending/processing/done/failed)를 반환한다."""
    async with SessionLocal() as session:
        docs = (await session.scalars(select(Document).order_by(Document.created_at.desc()))).all()
    return [{"id": str(d.id), "filename": d.filename, "status": d.status} for d in docs]


@mcp.resource("doc://{document_id}")
async def get_document_text(document_id: str) -> str:
    """문서 전체 텍스트(청크를 순서대로 이어붙인 것)."""
    async with SessionLocal() as session:
        chunks = (
            await session.scalars(
                select(Chunk).where(Chunk.document_id == uuid.UUID(document_id)).order_by(Chunk.seq)
            )
        ).all()
    return "\n".join(c.content for c in chunks)


@mcp.prompt()
def answer_from_docs(question: str) -> str:
    """문서 근거 기반으로만 답하도록 유도하는 프롬프트."""
    return (
        f"다음 질문에 답하기 전에 search_documents 도구로 관련 문서를 검색하세요. "
        f"검색 결과에 없는 내용은 모른다고 답하고, 답변에는 반드시 출처(filename, seq)를 명시하세요.\n\n"
        f"질문: {question}"
    )


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8001, stateless_http=True)

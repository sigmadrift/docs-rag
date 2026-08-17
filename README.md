# docs-rag

사내 문서 RAG 검색 API + MCP 서버. FastAPI / async SQLAlchemy 2.0 / pgvector / BGE-M3 학습 프로젝트.

## 구조

```
app/
  main.py            FastAPI 앱 (REST, /api)
  mcp_server/        MCP 서버 (Streamable HTTP, :8001/mcp) — 서비스 레이어 재사용
  api/               라우터 (documents, search)
  services/          parser → chunking → embedding → ingest / rag(search)
  models/            SQLAlchemy 모델 (Document, Chunk[Vector])
  schemas/           Pydantic 입출력
  db/                engine/session
  core/              settings, api key
alembic/             마이그레이션 (0001: vector ext + 테이블 + HNSW 인덱스)
tests/
```

## 시작

```bash
cp .env.example .env
docker compose up -d db
uv sync
uv run alembic upgrade head
uv run uvicorn app.main:app --reload            # REST  → http://localhost:8000/docs
uv run python -m app.mcp_server.server          # MCP   → http://localhost:8001/mcp
```

첫 실행 시 BGE-M3(약 2.2GB) 다운로드됨.

## 사용

```bash
curl -X POST localhost:8000/api/documents -H "X-API-Key: change-me" -F file=@작업표준서.pdf
curl -X POST localhost:8000/api/search -H "X-API-Key: change-me" \
  -H "Content-Type: application/json" -d '{"query":"압착 불량 판정 기준","top_k":3}'

# LLM 답변 (SSE 스트리밍: sources → delta... → done). Ollama 등 OpenAI 호환 서버 필요(LLM_* 설정)
curl -N -X POST localhost:8000/api/ask -H "X-API-Key: change-me" \
  -H "Content-Type: application/json" -d '{"question":"0.5sq 와이어의 최소 인장강도는?","top_k":3}'
```

Claude Desktop 연결: `scripts/claude_desktop_config.example.json` 참고.

## 로드맵

- [x] 1주차: 업로드/청킹/임베딩/검색 REST, Alembic, 기본 테스트
- [x] 2주차: MCP 서버(SDK v2) 구동·프로토콜 검증, Bearer 인증(`MCP_BEARER_TOKEN`)
- [x] 3주차(일부): SSE 스트리밍 답변 `/api/ask` (OpenAI 호환 LLM 연동)
- [ ] 3주차(남은 것): 리랭커, ingest를 arq 워커로 분리
- [ ] 이후: 문장 단위 청킹, 하이브리드 검색(BM25+벡터), MCP OAuth, 평가(RAGAS, BGE-M3 vs KURE-v1 비교)

배포 형태: 사용자 직접 호출이 아니라 사내 정문 API(ASP.NET Core)가 REST(:8000)를 호출하고,
사내 LLM 챗 클라이언트가 MCP(:8001)로 붙는 내부 서비스.

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
```

Claude Desktop 연결: `scripts/claude_desktop_config.example.json` 참고.

## 로드맵

- [ ] 1주차: 업로드/청킹/임베딩/검색 REST, Alembic, 기본 테스트
- [ ] 2주차: MCP 서버 연결, Claude Desktop에서 실제 질의, 도구 설명 튜닝
- [ ] 3주차: SSE 스트리밍 답변(LLM 연동), 리랭커, ingest를 arq 워커로 분리, Docker Compose 배포(OCI)
- [ ] 이후: 문장 단위 청킹, 하이브리드 검색(BM25+벡터), MCP 인증(Bearer → OAuth), 평가(RAGAS)

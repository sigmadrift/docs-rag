# docs-rag

사내 문서 RAG 검색 API + MCP 서버. FastAPI / async SQLAlchemy 2.0 / pgvector / BGE-M3 학습 프로젝트.

## 구조

```
app/
  main.py            FastAPI 앱 (REST, /api)
  worker.py          arq 인제스트 워커 (redis 큐 소비, 재시작 복구)
  mcp_server/        MCP 서버 (Streamable HTTP, :8001/mcp) — 서비스 레이어 재사용
  api/               라우터 (documents, search, ask)
  services/          parser → chunking(문장 단위) | table(표는 행 단위) → embedding → ingest
                     rag(하이브리드 검색) / reranker
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
docker compose up -d                            # postgres(pgvector) + redis
uv sync
uv run alembic upgrade head
uv run uvicorn app.main:app --reload            # REST  → http://localhost:8000/docs
uv run arq app.worker.WorkerSettings            # 인제스트 워커 (업로드 처리)
uv run python -m app.mcp_server.server          # MCP   → http://localhost:8001/mcp
```

첫 실행 시 BGE-M3(약 2.2GB) 다운로드됨.

지원 형식은 PDF, 텍스트(txt/md), 엑셀(xlsx/xlsm)이다. 엑셀은 문장 단위로 자르면 한 청크에
여러 행이 섞여 LLM이 다른 행의 수치를 인용할 수 있으므로, **행 하나를 청크 하나로** 만든다.
문서 메타(제목·문서번호)는 각 행 앞에 붙이고, 다단 헤더는 병합을 풀어 열별로 합성한다.

## 사용

```bash
curl -X POST localhost:8000/api/documents -H "X-API-Key: change-me" -F file=@작업표준서.pdf
curl -X POST localhost:8000/api/search -H "X-API-Key: change-me" \
  -H "Content-Type: application/json" -d '{"query":"압착 불량 판정 기준","top_k":3}'

# LLM 답변 (SSE 스트리밍: sources → delta... → done). Ollama 등 OpenAI 호환 서버 필요(LLM_* 설정)
curl -N -X POST localhost:8000/api/ask -H "X-API-Key: change-me" \
  -H "Content-Type: application/json" -d '{"question":"0.5sq 와이어의 최소 인장강도는?","top_k":3}'
```

### 검색 파이프라인

`/api/search`와 MCP `search_documents`는 같은 경로를 탄다.

1. **벡터 검색** — 질의 임베딩과의 코사인 거리로 `CANDIDATE_K`개 후보를 뽑는다.
2. **키워드 검색** (`HYBRID_ENABLED=true`, 기본 켜짐) — pg_trgm 트라이그램으로 같은 수만큼 뽑아
   RRF(Reciprocal Rank Fusion)로 합친다. 한국어는 조사가 붙어 어절 단위 전문검색이 자주 빗나가는데,
   3글자 단위 매칭은 "압착을"과 "압착"을 함께 잡는다. RRF는 점수 대신 순위만 쓰므로 스케일이 다른
   두 검색을 정규화 없이 융합할 수 있다.
3. **리랭킹** (`RERANK_ENABLED=true`, 기본 꺼짐) — cross-encoder(BGE-reranker-v2-m3)가
   (질의, 청크) 쌍을 직접 채점해 재정렬한다. 모델 약 2.2GB가 추가로 내려받아진다.

응답의 `score`는 마지막에 적용된 단계의 점수다 — 벡터 단독이면 코사인 유사도, 하이브리드면 RRF 점수,
리랭커까지 켜면 리랭커 관련도(0~1).

> 트라이그램은 질의가 길수록 유사도가 낮아진다. 문장형 질의가 많아 키워드 쪽이 자주 비면
> `KEYWORD_THRESHOLD`를 낮추거나, 형태소 분석 기반(pgroonga)으로의 교체를 검토할 것.


Claude Desktop 연결: `scripts/claude_desktop_config.example.json` 참고.

## 로드맵

- [x] 1주차: 업로드/청킹/임베딩/검색 REST, Alembic, 기본 테스트
- [x] 2주차: MCP 서버(SDK v2) 구동·프로토콜 검증, Bearer 인증(`MCP_BEARER_TOKEN`)
- [x] 3주차(일부): SSE 스트리밍 답변 `/api/ask` (OpenAI 호환 LLM 연동)
- [x] 3주차: ingest를 arq 워커로 분리(재시작 복구·재시도), 문장 단위 청킹, 검색 통합 테스트
- [x] 3주차: 리랭커(cross-encoder 재정렬, `RERANK_ENABLED=true`로 활성화)
- [x] 4주차: 하이브리드 검색(벡터 + pg_trgm 트라이그램 키워드, RRF 융합)
- [ ] 이후: MCP OAuth, 평가(RAGAS, BGE-M3 vs KURE-v1 비교)

배포 형태: 사용자 직접 호출이 아니라 사내 정문 API(ASP.NET Core)가 REST(:8000)를 호출하고,
사내 LLM 챗 클라이언트가 MCP(:8001)로 붙는 내부 서비스.

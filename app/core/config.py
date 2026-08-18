from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    chunk_size: int = 500
    chunk_overlap: int = 50
    api_key: str = "change-me"
    upload_dir: str = "uploads"
    max_upload_mb: int = 50
    # arq 작업 큐 (ingest 워커). API와 워커가 같은 redis를 바라봐야 한다.
    redis_url: str = "redis://localhost:6379"
    # MCP 서버 Bearer 토큰. 빈 값이면 인증 비활성화(로컬 개발용), 사내 배포 시 반드시 설정.
    mcp_bearer_token: str = ""
    # MCP 서버가 외부에 알리는 자기 주소 (AuthSettings의 issuer/resource URL)
    mcp_public_url: str = "http://localhost:8001"
    # OpenAI 호환 LLM 서버 (Ollama/vLLM/사내 서버 공용 — URL과 모델명만 바꾸면 교체됨)
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"  # Ollama는 아무 값이나 허용, vLLM/사내 서버는 실제 키
    llm_model: str = "mistral"
    # 검색 후보 수. 벡터/키워드 검색이 각각 이만큼 뽑고, 융합·재정렬을 거쳐 top_k만 남는다.
    candidate_k: int = 30
    # 하이브리드 검색: 벡터 결과와 트라이그램 키워드 결과를 RRF로 융합한다.
    # pg_trgm 확장과 인덱스가 필요하므로 alembic upgrade head를 먼저 돌려야 한다.
    hybrid_enabled: bool = True
    # 키워드 매칭 최소 유사도(word_similarity). pg_trgm 기본값 0.6은 질의가 조금만 길어져도
    # 아무것도 걸리지 않아 하이브리드가 무의미해지므로 낮춰 잡는다.
    # 올리면 정확 매칭만, 내리면 잡음이 늘어난다.
    keyword_threshold: float = 0.3
    # 리랭커(cross-encoder). 켜면 후보를 재정렬해 top_k를 고른다.
    # 모델이 별도로 약 2.2GB 내려받아지고 검색이 느려지므로 기본은 비활성화.
    rerank_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"


@lru_cache
def get_settings() -> Settings:
    return Settings()

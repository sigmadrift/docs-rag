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


@lru_cache
def get_settings() -> Settings:
    return Settings()

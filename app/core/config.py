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
    # MCP 서버 Bearer 토큰. 빈 값이면 인증 비활성화(로컬 개발용), 사내 배포 시 반드시 설정.
    mcp_bearer_token: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()

import secrets

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.core.config import Settings, get_settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(
    key: str | None = Security(api_key_header),
    settings: Settings = Depends(get_settings),
) -> None:
    # 타이밍 공격 방지를 위해 상수 시간 비교
    if not secrets.compare_digest(key or "", settings.api_key):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid api key")

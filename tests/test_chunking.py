import pytest

from app.services.chunking import split_text


@pytest.fixture
def chunk_settings(monkeypatch):
    """청킹 설정을 덮어쓰고, 테스트 후 설정 캐시를 비워 다른 테스트 오염 방지."""
    from app.core import config

    def _set(size: int, overlap: int):
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://x:x@localhost/x")
        monkeypatch.setenv("CHUNK_SIZE", str(size))
        monkeypatch.setenv("CHUNK_OVERLAP", str(overlap))
        config.get_settings.cache_clear()

    yield _set
    config.get_settings.cache_clear()


def test_empty_returns_nothing():
    assert split_text("   ") == []


def test_overlap_applied(chunk_settings):
    chunk_settings(size=10, overlap=3)
    chunks = split_text("a" * 25)
    assert chunks[0] == "a" * 10
    assert len(chunks) == 4  # 0-10, 7-17, 14-24, 21-25


def test_overlap_ge_size_raises(chunk_settings):
    chunk_settings(size=10, overlap=10)
    with pytest.raises(ValueError):
        split_text("hello world")

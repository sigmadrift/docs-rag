from app.services.chunking import split_text


def test_empty_returns_nothing():
    assert split_text("   ") == []


def test_overlap_applied(monkeypatch):
    from app.core import config
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://x:x@localhost/x")
    monkeypatch.setenv("CHUNK_SIZE", "10")
    monkeypatch.setenv("CHUNK_OVERLAP", "3")
    config.get_settings.cache_clear()

    chunks = split_text("a" * 25)
    assert chunks[0] == "a" * 10
    assert len(chunks) == 4  # 0-10, 7-17, 14-24, 21-25

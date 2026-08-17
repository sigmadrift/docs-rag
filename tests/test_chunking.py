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


def test_short_text_single_chunk(chunk_settings):
    chunk_settings(size=100, overlap=10)
    assert split_text("짧은 문장 하나.") == ["짧은 문장 하나."]


def test_sentences_packed_without_splitting(chunk_settings):
    """문장 중간이 잘리지 않고, 각 청크가 size 이내여야 한다."""
    chunk_settings(size=30, overlap=0)
    text = "첫 번째 문장이다. 두 번째 문장이다. 세 번째 문장이다. 네 번째 문장이다."
    chunks = split_text(text)
    assert all(len(c) <= 30 for c in chunks)
    # 모든 청크 경계가 문장 경계와 일치 (문장이 통째로 보존됨)
    sentences = ["첫 번째 문장이다.", "두 번째 문장이다.", "세 번째 문장이다.", "네 번째 문장이다."]
    for sent in sentences:
        assert any(sent in c for c in chunks)


def test_overlap_carries_last_sentence(chunk_settings):
    chunk_settings(size=25, overlap=12)
    text = "가나다라마바사아.\n자차카타파하가나.\n마바사아자차카타."
    chunks = split_text(text)
    assert len(chunks) >= 2
    # 앞 청크의 마지막 문장이 다음 청크 머리에 겹쳐야 함
    assert chunks[1].startswith("자차카타파하가나.")


def test_long_sentence_hard_split(chunk_settings):
    """size를 넘는 단일 문장은 문자 단위로 강제 분할되고 문자 오버랩이 적용된다."""
    chunk_settings(size=10, overlap=3)
    chunks = split_text("a" * 25)
    assert chunks[0] == "a" * 10
    assert all(len(c) <= 10 for c in chunks)
    assert "".join([chunks[0]] + [c[3:] for c in chunks[1:]]) == "a" * 25


def test_overlap_ge_size_raises(chunk_settings):
    chunk_settings(size=10, overlap=10)
    with pytest.raises(ValueError):
        split_text("hello world")

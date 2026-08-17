from app.core.config import get_settings


def split_text(text: str) -> list[str]:
    """단순 문자 기반 슬라이딩 윈도우. 나중에 문장/문단 단위로 개선 여지 있음."""
    s = get_settings()
    size, overlap = s.chunk_size, s.chunk_overlap
    if overlap >= size:
        raise ValueError(f"chunk_overlap({overlap})은 chunk_size({size})보다 작아야 함 (무한 루프)")
    text = text.strip()
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return chunks

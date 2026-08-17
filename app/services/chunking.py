import re

from app.core.config import get_settings

# 문장 경계: 종결부호(./!/?/。) 뒤 공백, 또는 줄바꿈. 한국어 평서문(…다.)도 이 규칙에 걸린다.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。])\s+|\n+")


def _sentences(text: str) -> list[str]:
    return [p for p in (s.strip() for s in _SENTENCE_BOUNDARY.split(text)) if p]


def _tail_within(sentences: list[str], budget: int) -> tuple[list[str], int]:
    """뒤에서부터 budget 문자 이내로 담기는 문장들과 그 길이(연결 공백 포함)를 반환."""
    kept: list[str] = []
    kept_len = 0
    for sent in reversed(sentences):
        added = len(sent) + (1 if kept else 0)
        if kept_len + added > budget:
            break
        kept.insert(0, sent)
        kept_len += added
    return kept, kept_len


def split_text(text: str) -> list[str]:
    """문장 단위로 모아 chunk_size(문자)를 넘지 않게 청크를 만든다.

    - 청크 사이에는 앞 청크의 마지막 문장들을 chunk_overlap 문자 이내로 겹쳐 문맥을 잇는다.
    - chunk_size를 넘는 단일 문장(표, 긴 나열 등)은 문자 단위로 강제 분할한다.
    """
    s = get_settings()
    size, overlap = s.chunk_size, s.chunk_overlap
    if overlap >= size:
        raise ValueError(f"chunk_overlap({overlap})은 chunk_size({size})보다 작아야 함 (무한 루프)")
    text = text.strip()
    if not text:
        return []

    # 1) 문장 목록으로 분해. 너무 긴 문장은 여기서 미리 size 이하 조각으로 쪼갠다.
    pieces: list[str] = []
    for sent in _sentences(text):
        while len(sent) > size:
            pieces.append(sent[:size])
            sent = sent[size - overlap :]
        pieces.append(sent)

    # 2) 문장들을 size를 넘지 않게 청크로 패킹 (연결은 공백 1칸)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for piece in pieces:
        added = len(piece) + (1 if current else 0)
        if current and current_len + added > size:
            chunks.append(" ".join(current))
            current, current_len = _tail_within(current, overlap)
            # 오버랩을 이어 붙이면 size를 넘는 경우엔 오버랩을 포기 (청크가 size를 넘지 않게)
            if current and current_len + 1 + len(piece) > size:
                current, current_len = [], 0
            added = len(piece) + (1 if current else 0)
        current.append(piece)
        current_len += added
    if current:
        chunks.append(" ".join(current))
    return chunks

"""RRF(Reciprocal Rank Fusion) 융합 로직 — DB 없이 순위 목록만으로 검증한다."""
import uuid

import pytest

from app.schemas.document import SearchHit
from app.services.rag import _RRF_K, fuse

DOC = uuid.uuid4()


def _hit(seq: int, score: float = 0.0) -> SearchHit:
    return SearchHit(document_id=DOC, filename="f.txt", seq=seq, content=f"c{seq}", score=score)


def _rrf(*ranks: int) -> float:
    """0-based 순위들의 RRF 점수 합."""
    return sum(1.0 / (_RRF_K + r + 1) for r in ranks)


def test_item_in_both_rankings_wins():
    # 2번은 어느 목록에서도 1위가 아니지만, 양쪽에 모두 들어 있어 합산 점수가 가장 높다
    vector = [_hit(1), _hit(2)]
    keyword = [_hit(3), _hit(2)]

    out = fuse([vector, keyword])

    assert [h.seq for h in out] == [2, 1, 3]
    assert out[0].score == pytest.approx(_rrf(1, 1))


def test_keyword_only_hit_is_kept():
    # 한쪽 목록에만 있는 항목도 결과에서 빠지지 않는다 (하이브리드의 존재 이유)
    out = fuse([[_hit(1)], [_hit(9)]])

    assert {h.seq for h in out} == {1, 9}
    assert all(h.score == pytest.approx(_rrf(0)) for h in out)


def test_same_chunk_is_deduped_by_document_and_seq():
    out = fuse([[_hit(1, score=0.9)], [_hit(1, score=0.2)]])

    assert len(out) == 1
    # score는 원래 유사도가 아니라 RRF 합으로 교체된다
    assert out[0].score == pytest.approx(_rrf(0, 0))


def test_empty_rankings():
    assert fuse([[], []]) == []

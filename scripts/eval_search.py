"""검색 설정별 성능 비교. 인제스트가 끝난 DB에 대고 돌린다.

    uv run python -m scripts.eval_search    (프로젝트 루트에서. app 패키지를 임포트한다)

각 질문의 정답 청크가 상위 몇 번째로 나오는지 재서 Hit@K와 MRR을 계산한다.
정답 판정은 청크 ID가 아니라 내용에 포함돼야 할 문자열(must_contain)로 하므로,
재인제스트해서 청크 ID가 바뀌어도 그대로 쓸 수 있다.
"""
import asyncio
import json
from pathlib import Path

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.services import rag

TOP_K = 10
CASES = json.loads((Path(__file__).parent / "eval_queries.json").read_text(encoding="utf-8"))

# (이름, hybrid, rerank) — 단계를 하나씩 켜가며 기여도를 본다
SETUPS = [
    ("벡터 단독", False, False),
    ("+ 하이브리드", True, False),
    ("+ 리랭커", True, True),
]


async def rank_of_answer(session, case: dict) -> int | None:
    """정답 청크의 순위(1부터). 상위 TOP_K 안에 없으면 None."""
    hits = await rag.search(session, case["question"], TOP_K)
    return next(
        (i for i, h in enumerate(hits, 1) if all(t in h.content for t in case["must_contain"])),
        None,
    )


def score(ranks: list[int | None]) -> dict:
    n = len(ranks)
    found = [r for r in ranks if r]
    return {
        "Hit@1": sum(r == 1 for r in found) / n,
        "Hit@3": sum(r <= 3 for r in found) / n,
        "Hit@5": sum(r <= 5 for r in found) / n,
        # MRR: 정답 순위의 역수 평균. 상위에 놓을수록 1에 가깝다
        "MRR": sum(1 / r for r in found) / n,
    }


async def main() -> None:
    settings = get_settings()
    print(f"질문 {len(CASES)}개 / top_k={TOP_K}\n")
    results = {}
    for name, hybrid, rerank in SETUPS:
        settings.hybrid_enabled, settings.rerank_enabled = hybrid, rerank
        async with SessionLocal() as session:
            ranks = [await rank_of_answer(session, c) for c in CASES]
        results[name] = score(ranks)
        misses = [c["question"][:40] for c, r in zip(CASES, ranks, strict=True) if r is None]
        print(f"{name:14} {results[name]}")
        if misses:
            print(f"{'':14} 놓친 질문 {len(misses)}개: {misses[:3]}")

    print(f"\n{'설정':14} {'Hit@1':>7} {'Hit@3':>7} {'Hit@5':>7} {'MRR':>7}")
    for name, s in results.items():
        print(f"{name:14} {s['Hit@1']:>7.2f} {s['Hit@3']:>7.2f} {s['Hit@5']:>7.2f} {s['MRR']:>7.3f}")


if __name__ == "__main__":
    asyncio.run(main())

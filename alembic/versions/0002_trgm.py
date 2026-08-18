"""hybrid search: pg_trgm extension + trigram index on chunks.content

Revision ID: 0002
Revises: 0001
"""
from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    # 트라이그램 GIN 인덱스. 한국어는 조사가 붙어 어절 단위 전문검색이 자주 빗나가는데,
    # 3글자 단위로 쪼개는 트라이그램은 "압착을"과 "압착"을 함께 매칭한다.
    # word_similarity(<%) 연산자도 이 인덱스를 탄다.
    op.execute(
        "CREATE INDEX ix_chunks_content_trgm ON chunks USING gin (content gin_trgm_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunks_content_trgm")
    # pg_trgm 확장은 다른 곳에서 쓸 수 있으므로 남겨둔다

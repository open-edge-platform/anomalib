"""add app state table

Revision ID: 1f4f5b1d9f6f
Revises: 7a213a27d666
Create Date: 2026-03-19 10:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "1f4f5b1d9f6f"
down_revision: str | Sequence[str] | None = "7a213a27d666"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "app_state",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("last_used_project_id", sa.String(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.ForeignKeyConstraint(["last_used_project_id"], ["projects.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("app_state")

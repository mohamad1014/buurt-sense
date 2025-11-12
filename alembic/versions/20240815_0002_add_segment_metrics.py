"""Add optional metadata columns for recorded segments."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20240815_0002"
down_revision = "20240808_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add frame_count, audio_duration_ms, checksum, and size_bytes columns."""

    op.add_column(
        "segments", sa.Column("frame_count", sa.Integer(), nullable=True)
    )
    op.add_column(
        "segments", sa.Column("audio_duration_ms", sa.Integer(), nullable=True)
    )
    op.add_column(
        "segments", sa.Column("checksum", sa.String(length=128), nullable=True)
    )
    op.add_column(
        "segments", sa.Column("size_bytes", sa.Integer(), nullable=True)
    )


def downgrade() -> None:
    """Remove segment metadata columns."""

    op.drop_column("segments", "size_bytes")
    op.drop_column("segments", "checksum")
    op.drop_column("segments", "audio_duration_ms")
    op.drop_column("segments", "frame_count")

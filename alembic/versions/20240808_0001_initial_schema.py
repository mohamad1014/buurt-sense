"""Initial database schema."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "20240808_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create tables for sessions, segments, and detections."""

    op.create_table(
        "recording_sessions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("device_id", sa.String(length=36), nullable=False),
        sa.Column("operator_alias", sa.String(length=64), nullable=True),
        sa.Column("notes", sa.String(length=500), nullable=True),
        sa.Column("timezone", sa.String(length=64), nullable=False, server_default="UTC"),
        sa.Column("app_version", sa.String(length=64), nullable=True),
        sa.Column("model_bundle_version", sa.String(length=64), nullable=True),
        sa.Column("device_info", sa.JSON(), nullable=True),
        sa.Column("gps_origin", sa.JSON(), nullable=True),
        sa.Column("orientation_origin", sa.JSON(), nullable=True),
        sa.Column("config_snapshot", sa.JSON(), nullable=True),
        sa.Column(
            "detection_summary",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column(
            "redact_location",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "segments",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("index", sa.Integer(), nullable=False),
        sa.Column("start_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("file_path", sa.String(length=512), nullable=False),
        sa.Column("gps_trace", sa.JSON(), nullable=True),
        sa.Column("orientation_trace", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["session_id"], ["recording_sessions.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("session_id", "index", name="uq_segment_session_index"),
    )

    op.create_table(
        "detections",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("segment_id", sa.String(length=36), nullable=False),
        sa.Column("label", sa.String(length=128), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("gps_point", sa.JSON(), nullable=True),
        sa.Column("orientation", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["segment_id"], ["segments.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("segment_id", "timestamp", "label", name="uq_detection_event"),
    )


def downgrade() -> None:
    """Drop all Buurt Sense tables."""

    op.drop_table("detections")
    op.drop_table("segments")
    op.drop_table("recording_sessions")

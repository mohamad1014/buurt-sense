"""Database models for the durable session storage."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.ext.mutable import MutableDict, MutableList
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class RecordingSession(Base):
    """Root entity representing a single recording session."""

    __tablename__ = "recording_sessions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    device_id: Mapped[str] = mapped_column(
        String(36), nullable=False, default=lambda: str(uuid4())
    )
    operator_alias: Mapped[str | None] = mapped_column(String(64))
    notes: Mapped[str | None] = mapped_column(String(500))
    timezone: Mapped[str] = mapped_column(
        String(64), nullable=False, default="UTC", server_default="UTC"
    )
    app_version: Mapped[str | None] = mapped_column(String(64))
    model_bundle_version: Mapped[str | None] = mapped_column(String(64))
    device_info: Mapped[dict[str, Any] | None] = mapped_column(
        MutableDict.as_mutable(JSON)
    )
    gps_origin: Mapped[dict[str, Any] | None] = mapped_column(
        MutableDict.as_mutable(JSON)
    )
    orientation_origin: Mapped[dict[str, Any] | None] = mapped_column(
        MutableDict.as_mutable(JSON)
    )
    config_snapshot: Mapped[dict[str, Any] | None] = mapped_column(
        MutableDict.as_mutable(JSON)
    )
    detection_summary: Mapped[dict[str, Any]] = mapped_column(
        MutableDict.as_mutable(JSON),
        nullable=False,
        default=dict,
        server_default="{}",
    )
    redact_location: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="0"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    segments: Mapped[list["Segment"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Segment.index",
    )


class Segment(Base):
    """A time-bounded media chunk within a session."""

    __tablename__ = "segments"
    __table_args__ = (
        UniqueConstraint("session_id", "index", name="uq_segment_session_index"),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    session_id: Mapped[str] = mapped_column(
        ForeignKey("recording_sessions.id", ondelete="CASCADE"), nullable=False
    )
    index: Mapped[int] = mapped_column(Integer, nullable=False)
    start_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    frame_count: Mapped[int | None] = mapped_column(Integer)
    audio_duration_ms: Mapped[int | None] = mapped_column(Integer)
    checksum: Mapped[str | None] = mapped_column(String(128))
    size_bytes: Mapped[int | None] = mapped_column(Integer)
    gps_trace: Mapped[list[dict[str, Any]]] = mapped_column(
        MutableList.as_mutable(JSON), default=list
    )
    orientation_trace: Mapped[list[dict[str, Any]]] = mapped_column(
        MutableList.as_mutable(JSON), default=list
    )

    session: Mapped["RecordingSession"] = relationship(back_populates="segments")
    detections: Mapped[list["Detection"]] = relationship(
        back_populates="segment",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Detection.timestamp",
    )


class Detection(Base):
    """A single detection result tied to a segment."""

    __tablename__ = "detections"
    __table_args__ = (
        UniqueConstraint("segment_id", "timestamp", "label", name="uq_detection_event"),
    )

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    segment_id: Mapped[str] = mapped_column(
        ForeignKey("segments.id", ondelete="CASCADE"), nullable=False
    )
    label: Mapped[str] = mapped_column(String(128), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    gps_point: Mapped[dict[str, Any] | None] = mapped_column(
        MutableDict.as_mutable(JSON)
    )
    orientation: Mapped[dict[str, Any] | None] = mapped_column(
        MutableDict.as_mutable(JSON)
    )

    segment: Mapped["Segment"] = relationship(back_populates="detections")

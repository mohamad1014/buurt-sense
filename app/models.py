"""Data models used by the Buurt Sense API."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Optional
from uuid import UUID, uuid4


@dataclass(frozen=True, slots=True)
class Session:
    """Representation of a recording session."""

    id: UUID
    started_at: datetime
    ended_at: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
        }

    def model_copy(self, *, update: Mapping[str, Any] | None = None) -> "Session":
        data = {
            "id": self.id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }
        if update:
            data.update(update)
        return Session(**data)

    @classmethod
    def _parse_datetime(cls, value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        raise TypeError(f"Unsupported datetime value: {value!r}")

    @classmethod
    def _parse_uuid(cls, value: Any) -> UUID:
        if isinstance(value, UUID):
            return value
        if isinstance(value, str):
            return UUID(value)
        raise TypeError(f"Unsupported UUID value: {value!r}")

    @classmethod
    def model_validate(cls, payload: Mapping[str, Any]) -> "Session":
        data = dict(payload)
        return cls(
            id=cls._parse_uuid(data["id"]),
            started_at=cls._parse_datetime(data["started_at"]),
            ended_at=cls._parse_datetime(data["ended_at"]) if data.get("ended_at") else None,
        )

    @classmethod
    def new(cls) -> "Session":
        from datetime import timezone

        return cls(id=uuid4(), started_at=datetime.now(timezone.utc))

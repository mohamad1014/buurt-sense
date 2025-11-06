from __future__ import annotations

from datetime import datetime
from typing import Optional

from timezonefinder import TimezoneFinder

from .schemas import DetectionSummary, GPSOrigin

_tf = TimezoneFinder(in_memory=True)


def derive_timezone(lat: float, lon: float) -> str:
    timezone = _tf.timezone_at(lat=lat, lng=lon)
    return timezone or "UTC"


def round_gps_origin(gps_origin: GPSOrigin, digits: int = 5) -> GPSOrigin:
    return GPSOrigin(
        lat=round(gps_origin.lat, digits),
        lon=round(gps_origin.lon, digits),
        accuracy_m=gps_origin.accuracy_m,
        captured_at=gps_origin.captured_at,
    )


def session_status(ended_at: Optional[datetime]) -> str:
    return "completed" if ended_at else "active"


def ensure_detection_summary(summary: Optional[DetectionSummary]) -> DetectionSummary:
    return summary or DetectionSummary()

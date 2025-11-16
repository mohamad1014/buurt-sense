"""Lightweight inference adapters for audio and video detections."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional
from uuid import UUID

from .schemas import DetectionCreate

LOG = logging.getLogger(__name__)


@dataclass(slots=True)
class InferenceConfig:
    """Runtime configuration that governs inference thresholds and behaviour."""

    global_threshold: float = 0.6
    class_thresholds: Mapping[str, float] = field(default_factory=dict)
    class_cooldown_sec: int = 30
    enable_audio: bool = True
    enable_video: bool = True
    audio_model_id: str = "yamnet-small"
    video_model_id: str = "yolo-v8n-quant"

    def threshold_for(self, detection_class: str, fallback: float | None = None) -> float:
        """Return the effective threshold for a class."""

        if detection_class in self.class_thresholds:
            return float(self.class_thresholds[detection_class])
        if fallback is not None:
            return fallback
        return self.global_threshold


@dataclass(slots=True)
class SegmentInferenceInput:
    """Description of a media segment to analyse."""

    session_id: UUID
    segment_id: UUID
    file_path: Path
    start_ts: datetime
    end_ts: datetime
    frame_count: int | None = None
    audio_duration_ms: int | None = None
    size_bytes: int | None = None


@dataclass(slots=True)
class DetectionProposal:
    """Intermediate detection payload emitted by a detector."""

    detection_class: str
    confidence: float
    timestamp: datetime
    model_id: str
    inference_latency_ms: int
    orientation_heading_deg: float | None = None

    def to_schema(self) -> DetectionCreate:
        """Convert the proposal to the API schema."""

        return DetectionCreate(
            detection_class=self.detection_class,
            confidence=self.confidence,
            timestamp=self.timestamp,
            model_id=self.model_id,
            inference_latency_ms=self.inference_latency_ms,
            orientation_heading_deg=self.orientation_heading_deg,
        )


@dataclass(slots=True)
class DetectionState:
    """Stateful tracker to enforce cooldowns across segments."""

    last_seen: MutableMapping[str, datetime] = field(default_factory=dict)


class AudioDetector:
    """Audio detector stub that can run on constrained devices."""

    def __init__(self, *, model_id: str) -> None:
        self._model_id = model_id

    def detect(self, request: SegmentInferenceInput) -> list[DetectionProposal]:
        """
        Produce detections from audio content.

        This implementation is a lightweight placeholder; it hashes the segment
        size into a stable confidence value while preserving realistic latency.
        """

        start = time.perf_counter()
        size_bytes = request.size_bytes or 0
        confidence = min(0.9, 0.6 + (size_bytes % 10_000) / 50_000)

        latency_ms = max(int((time.perf_counter() - start) * 1000), 1)
        return [
            DetectionProposal(
                detection_class="ambient_noise",
                confidence=confidence,
                timestamp=request.end_ts,
                model_id=self._model_id,
                inference_latency_ms=latency_ms,
            )
        ]


class VideoDetector:
    """Video detector stub for devices that can sample frames."""

    def __init__(self, *, model_id: str) -> None:
        self._model_id = model_id

    def detect(self, request: SegmentInferenceInput) -> list[DetectionProposal]:
        """
        Emit a lightweight detection using frame metadata.

        The placeholder uses frame count to derive a stable confidence while
        simulating modest processing latency.
        """

        start = time.perf_counter()
        frame_count = request.frame_count or 0
        confidence = min(0.9, 0.55 + (frame_count % 120) / 300)

        latency_ms = max(int((time.perf_counter() - start) * 1000), 1)
        return [
            DetectionProposal(
                detection_class="ambient_noise",
                confidence=confidence,
                timestamp=request.end_ts,
                model_id=self._model_id,
                inference_latency_ms=latency_ms,
            )
        ]


class InferenceEngine:
    """Coordinate audio and video detectors with thresholding and cooldowns."""

    def __init__(self, config: InferenceConfig | None = None) -> None:
        self._config = config or InferenceConfig()
        self._audio = AudioDetector(model_id=self._config.audio_model_id)
        self._video = VideoDetector(model_id=self._config.video_model_id)
        self._stats: Dict[str, int] = {
            "runs": 0,
            "proposals": 0,
            "accepted": 0,
            "filtered_threshold": 0,
            "filtered_cooldown": 0,
        }

    def detect(
        self,
        request: SegmentInferenceInput,
        *,
        session_config: Mapping[str, object] | None = None,
        state: DetectionState | None = None,
    ) -> list[DetectionCreate]:
        """
        Run inference for a segment, applying thresholds and cooldowns.

        The returned list is safe to persist directly via :class:`DetectionCreate`.
        """

        self._stats["runs"] += 1
        proposals = list(self._run_detectors(request))
        self._stats["proposals"] += len(proposals)

        effective_threshold = self._config.global_threshold
        cooldown_sec = self._config.class_cooldown_sec
        if session_config:
            session_threshold = session_config.get("confidence_threshold")
            if isinstance(session_threshold, (float, int)):
                effective_threshold = float(session_threshold)
            session_cooldown = session_config.get("class_cooldown_sec")
            if isinstance(session_cooldown, (float, int)):
                cooldown_sec = int(session_cooldown)

        tracker = state or DetectionState()
        accepted: list[DetectionCreate] = []
        for proposal in proposals:
            threshold = self._config.threshold_for(
                proposal.detection_class, fallback=effective_threshold
            )
            if proposal.confidence < threshold:
                self._stats["filtered_threshold"] += 1
                LOG.debug(
                    "Dropping detection below threshold",
                    extra={
                        "class": proposal.detection_class,
                        "confidence": proposal.confidence,
                        "threshold": threshold,
                    },
                )
                continue

            if self._is_within_cooldown(
                tracker.last_seen,
                proposal.detection_class,
                proposal.timestamp,
                cooldown_sec,
            ):
                self._stats["filtered_cooldown"] += 1
                continue

            tracker.last_seen[proposal.detection_class] = proposal.timestamp
            accepted.append(proposal.to_schema())

        self._stats["accepted"] += len(accepted)
        return accepted

    def stats(self) -> Mapping[str, int]:
        """Return a snapshot of basic counters for observability."""

        return dict(self._stats)

    def _run_detectors(self, request: SegmentInferenceInput) -> Iterable[DetectionProposal]:
        """Yield proposals from active detectors."""

        if self._config.enable_audio:
            yield from self._audio.detect(request)
        if self._config.enable_video:
            yield from self._video.detect(request)

    @staticmethod
    def _is_within_cooldown(
        last_seen: Mapping[str, datetime],
        detection_class: str,
        timestamp: datetime,
        cooldown_sec: int,
    ) -> bool:
        """Return True if detection_class was seen within the cooldown window."""

        if cooldown_sec <= 0:
            return False
        previous = last_seen.get(detection_class)
        if previous is None:
            return False
        return timestamp - previous < timedelta(seconds=cooldown_sec)

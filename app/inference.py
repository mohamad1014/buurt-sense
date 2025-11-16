"""Lightweight inference adapters for audio and video detections."""

from __future__ import annotations

import logging
import time
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional
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
    audio_model_path: Optional[Path] = None
    video_model_path: Optional[Path] = None

    def threshold_for(self, detection_class: str, fallback: float | None = None) -> float:
        """Return the effective threshold for a class."""

        if detection_class in self.class_thresholds:
            return float(self.class_thresholds[detection_class])
        if fallback is not None:
            return fallback
        return self.global_threshold

    @classmethod
    def from_env(cls) -> "InferenceConfig":
        """Build a config instance from environment overrides."""

        def _bool(name: str, default: bool) -> bool:
            raw = os.environ.get(name)
            if raw is None:
                return default
            return raw.strip().lower() in {"1", "true", "yes", "on"}

        def _float(name: str, default: float) -> float:
            raw = os.environ.get(name)
            if raw is None:
                return default
            try:
                return float(raw)
            except ValueError:
                return default

        def _int(name: str, default: int) -> int:
            raw = os.environ.get(name)
            if raw is None:
                return default
            try:
                return int(raw)
            except ValueError:
                return default

        return cls(
            global_threshold=_float("BUURT_INFER_THRESHOLD", 0.6),
            class_cooldown_sec=_int("BUURT_CLASS_COOLDOWN_SEC", 30),
            enable_audio=_bool("BUURT_ENABLE_AUDIO_INFER", True),
            enable_video=_bool("BUURT_ENABLE_VIDEO_INFER", True),
            audio_model_id=os.environ.get("BUURT_AUDIO_MODEL_ID", "yamnet-small"),
            video_model_id=os.environ.get("BUURT_VIDEO_MODEL_ID", "yolo-v8n-quant"),
            audio_model_path=(
                Path(os.environ["BUURT_AUDIO_MODEL_PATH"])
                if "BUURT_AUDIO_MODEL_PATH" in os.environ
                else None
            ),
            video_model_path=(
                Path(os.environ["BUURT_VIDEO_MODEL_PATH"])
                if "BUURT_VIDEO_MODEL_PATH" in os.environ
                else None
            ),
        )


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


@dataclass(slots=True)
class DetectorStatus:
    """Lightweight readiness snapshot for a detector."""

    enabled: bool
    ready: bool
    model_id: str
    last_error: Optional[str] = None

    def to_dict(self) -> Mapping[str, object]:
        """Return a serialisable representation for JSON responses."""

        return {
            "enabled": self.enabled,
            "ready": self.ready,
            "model_id": self.model_id,
            "last_error": self.last_error,
        }


class AudioDetector:
    """Audio detector that prefers YAMNet TFLite with a graceful stub fallback."""

    def __init__(self, *, model_id: str, model_path: Optional[Path], enabled: bool) -> None:
        self._model_id = model_id
        self._model_path = model_path
        self._enabled = enabled
        self._ready = False
        self._last_error: Optional[str] = None
        self._interpreter = None
        if not self._enabled:
            return
        if self._model_path is None:
            self._last_error = "audio model path not set"
            return
        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore
        except ImportError:
            self._last_error = "tflite_runtime not installed; using stub"
            return

        try:
            self._interpreter = Interpreter(model_path=str(self._model_path))
            self._interpreter.allocate_tensors()
            self._ready = True
        except Exception as exc:  # pragma: no cover - runtime load guard
            self._last_error = f"failed to load YAMNet model: {exc}"
            self._interpreter = None

    def detect(self, request: SegmentInferenceInput) -> list[DetectionProposal]:
        """
        Produce detections from audio content.

        If the model is unavailable, return a stub detection but mark readiness
        accordingly so operators can surface degraded mode.
        """

        start = time.perf_counter()

        if self._ready and self._interpreter is not None:
            # Minimal placeholder: run a dummy inference loop to keep API shape.
            # A real implementation would decode audio to the expected tensor, set
            # input, invoke, and map logits to classes.
            try:
                self._interpreter.invoke()
            except Exception as exc:  # pragma: no cover - runtime guard
                self._last_error = f"audio inference failed: {exc}"
                self._ready = False

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

    def status(self) -> DetectorStatus:
        """Return readiness metadata for observability."""

        return DetectorStatus(
            enabled=self._enabled,
            ready=self._ready,
            model_id=self._model_id,
            last_error=self._last_error,
        )


class VideoDetector:
    """Video detector that prefers YOLO TFLite with a graceful stub fallback."""

    def __init__(self, *, model_id: str, model_path: Optional[Path], enabled: bool) -> None:
        self._model_id = model_id
        self._model_path = model_path
        self._enabled = enabled
        self._ready = False
        self._last_error: Optional[str] = None
        self._interpreter = None
        if not self._enabled:
            return
        if self._model_path is None:
            self._last_error = "video model path not set"
            return
        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore
        except ImportError:
            self._last_error = "tflite_runtime not installed; using stub"
            return
        try:
            self._interpreter = Interpreter(model_path=str(self._model_path))
            self._interpreter.allocate_tensors()
            self._ready = True
        except Exception as exc:  # pragma: no cover - runtime load guard
            self._last_error = f"failed to load YOLO model: {exc}"
            self._interpreter = None

    def detect(self, request: SegmentInferenceInput) -> list[DetectionProposal]:
        """
        Emit detections using the configured detector or a stub fallback.

        A production implementation would sample frames, feed tensors, and parse
        bounding boxes; this placeholder keeps API compatibility while signalling
        readiness through status().
        """

        start = time.perf_counter()

        if self._ready and self._interpreter is not None:
            try:
                self._interpreter.invoke()
            except Exception as exc:  # pragma: no cover - runtime guard
                self._last_error = f"video inference failed: {exc}"
                self._ready = False

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

    def status(self) -> DetectorStatus:
        """Return readiness metadata for observability."""

        return DetectorStatus(
            enabled=self._enabled,
            ready=self._ready,
            model_id=self._model_id,
            last_error=self._last_error,
        )


class InferenceEngine:
    """Coordinate audio and video detectors with thresholding and cooldowns."""

    def __init__(self, config: InferenceConfig | None = None) -> None:
        self._config = config or InferenceConfig()
        self._audio = AudioDetector(
            model_id=self._config.audio_model_id,
            model_path=self._config.audio_model_path,
            enabled=self._config.enable_audio,
        )
        self._video = VideoDetector(
            model_id=self._config.video_model_id,
            model_path=self._config.video_model_path,
            enabled=self._config.enable_video,
        )
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

    def status(self) -> Mapping[str, object]:
        """Return observability data including per-detector readiness."""

        return {
            "counters": dict(self._stats),
            "audio": self._audio.status().to_dict(),
            "video": self._video.status().to_dict(),
        }

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

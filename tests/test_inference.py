"""Unit tests for the inference pipeline primitives."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from app.inference import (
    DetectionState,
    InferenceConfig,
    InferenceEngine,
    SegmentInferenceInput,
)


def _make_request(
    tmp_path,
    *,
    size_bytes: int = 6000,
    frame_count: int = 60,
) -> SegmentInferenceInput:
    """Create a realistic inference request backed by a temporary file."""

    file_path = tmp_path / "sample.pcm"
    file_path.write_bytes(b"a" * size_bytes)
    start = datetime.now(UTC)
    end = start + timedelta(seconds=5)
    return SegmentInferenceInput(
        session_id=uuid4(),
        segment_id=uuid4(),
        file_path=file_path,
        start_ts=start,
        end_ts=end,
        frame_count=frame_count,
        audio_duration_ms=5000,
        size_bytes=size_bytes,
    )


def test_detection_respects_threshold(tmp_path) -> None:
    """Detections below the configured threshold should be dropped."""

    engine = InferenceEngine(InferenceConfig(global_threshold=0.9))
    request = _make_request(tmp_path, size_bytes=100)  # yields low confidence

    detections = engine.detect(request)

    assert detections == []


def test_cooldown_filters_back_to_back_segments(tmp_path) -> None:
    """Cooldown prevents duplicate class emissions within the window."""

    engine = InferenceEngine(InferenceConfig(class_cooldown_sec=30))
    state = DetectionState()
    first = _make_request(tmp_path)
    second = _make_request(tmp_path)
    second = SegmentInferenceInput(
        session_id=first.session_id,
        segment_id=second.segment_id,
        file_path=second.file_path,
        start_ts=first.end_ts,
        end_ts=first.end_ts + timedelta(seconds=1),
        frame_count=second.frame_count,
        audio_duration_ms=second.audio_duration_ms,
        size_bytes=second.size_bytes,
    )

    accepted_first = engine.detect(first, state=state)
    accepted_second = engine.detect(second, state=state)

    assert len(accepted_first) >= 1
    assert accepted_second == []


def test_session_config_overrides_threshold_and_cooldown(tmp_path) -> None:
    """Session-provided thresholds and cooldowns should influence filtering."""

    engine = InferenceEngine()
    session_config = {"confidence_threshold": 0.65, "class_cooldown_sec": 0}
    request = _make_request(tmp_path, size_bytes=8000, frame_count=90)

    detections = engine.detect(request, session_config=session_config)

    assert detections, "at least one detection should be emitted"
    model_ids = {item.model_id for item in detections}
    assert model_ids <= {"yamnet-small", "yolo-v8n-quant"}

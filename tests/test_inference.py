"""Unit tests for the inference pipeline primitives."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import sys
import types
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


def test_status_exposes_readiness() -> None:
    """Inference status should report detector readiness even when stubbed."""

    engine = InferenceEngine()
    status = engine.status()
    assert "audio" in status and "video" in status
    assert isinstance(status["audio"].get("enabled"), bool)
    assert "ready" in status["audio"]
    assert "model_id" in status["audio"]


def _install_fake_tflite(monkeypatch, invoked_flag: dict[str, bool]) -> None:
    """Register a fake tflite_runtime Interpeter to avoid the real dependency."""

    class FakeInterpreter:
        def __init__(self, *args, **kwargs):
            invoked_flag["constructed"] = True

        def allocate_tensors(self):
            invoked_flag["allocated"] = True

        def invoke(self):
            invoked_flag["invoked"] = True

    fake_interpreter_module = types.SimpleNamespace(Interpreter=FakeInterpreter)
    fake_runtime = types.ModuleType("tflite_runtime")
    fake_runtime.interpreter = fake_interpreter_module
    monkeypatch.setitem(sys.modules, "tflite_runtime", fake_runtime)
    monkeypatch.setitem(sys.modules, "tflite_runtime.interpreter", fake_interpreter_module)


def test_audio_detector_invokes_tflite_when_available(monkeypatch, tmp_path) -> None:
    """Audio detector should load and invoke the TFLite interpreter when present."""

    invoked: dict[str, bool] = {}
    _install_fake_tflite(monkeypatch, invoked)

    dummy_model = tmp_path / "yamnet.tflite"
    dummy_model.write_bytes(b"fake")

    config = InferenceConfig(
        audio_model_path=dummy_model,
        video_model_path=None,
        enable_audio=True,
        enable_video=False,
    )
    engine = InferenceEngine(config)
    request = _make_request(tmp_path)

    detections = engine.detect(request)

    assert detections, "audio detector should emit a detection when model loads"
    assert invoked.get("constructed") and invoked.get("allocated")
    assert invoked.get("invoked")
    status = engine.status()["audio"]
    assert status["ready"] is True
    assert status["last_error"] is None


def test_video_detector_invokes_tflite_when_available(monkeypatch, tmp_path) -> None:
    """Video detector should load and invoke the TFLite interpreter when present."""

    invoked: dict[str, bool] = {}
    _install_fake_tflite(monkeypatch, invoked)

    dummy_model = tmp_path / "yolo.tflite"
    dummy_model.write_bytes(b"fake")

    config = InferenceConfig(
        video_model_path=dummy_model,
        audio_model_path=None,
        enable_audio=False,
        enable_video=True,
    )
    engine = InferenceEngine(config)
    request = _make_request(tmp_path, frame_count=32)

    detections = engine.detect(request)

    assert detections, "video detector should emit a detection when model loads"
    assert invoked.get("constructed") and invoked.get("allocated")
    assert invoked.get("invoked")
    status = engine.status()["video"]
    assert status["ready"] is True
    assert status["last_error"] is None

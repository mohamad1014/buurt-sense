const statusMessage = document.getElementById("status-message");
const startButton = document.getElementById("start-session");
const stopButton = document.getElementById("stop-session");
const refreshButton = document.getElementById("refresh-sessions");
const activeSessionDetails = document.getElementById("active-session-details");
const sessionList = document.getElementById("session-list");
const captureSettingsForm = document.getElementById("capture-settings-form");
const audioInferenceToggle = document.getElementById("config-audio-inference");
const videoInferenceToggle = document.getElementById("config-video-inference");
const segmentLengthInput = document.getElementById("config-segment-length");
const overlapInput = document.getElementById("config-overlap");
const confidenceInput = document.getElementById("config-confidence");
const previewVideo = document.getElementById("capture-preview");
const previewHelp = document.getElementById("preview-help");
const detectionList = document.getElementById("detection-feed");
const detectionEmptyState = document.getElementById("detection-empty");
const clearDetectionsButton = document.getElementById("clear-detections");
const resetConfigButton = document.getElementById("reset-config");
const recordingBanner = document.getElementById("recording-banner");
const previewToggleButton = document.getElementById("toggle-preview");
const segmentLengthDisplay = document.getElementById("segment-length-display");
const overlapDisplay = document.getElementById("overlap-display");
const confidenceDisplay = document.getElementById("confidence-display");
let recordingBannerInterval = null;

let activeSession = null;
let liveWebSocket = null;
let pollingTimer = null;
let longPollController = null;
let lastRevision = null;
let shuttingDown = false;
const CONFIG_STORAGE_KEY = "buurt-capture-config";
const detectionLog = [];
const captureState = {
  context: null,
  uploads: [],
  orientationPermissionRequested: false,
  inference: null,
};

const inferenceState = {
  audioModel: null,
  audioModelPromise: null,
  videoModel: null,
  videoModelPromise: null,
  runtimeReady: false,
  loading: false,
  audioModelUrl:
    window.BUURT_AUDIO_MODEL_URL || "/static/tflite/YamNet_float.tflite",
  videoModelUrl:
    window.BUURT_VIDEO_MODEL_URL ||
    "https://huggingface.co/qualcomm/ResNet-Mixed-Convolution/resolve/main/ResNet-Mixed-Convolution_float.tflite?download=true",
  audioModelId: "yamnet-client-tfjs",
  videoModelId: "resnet-mc-client-tflite",
  wasmBaseUrl: "/static/tflite",
  classMap: [],
  classMapUrl:
    window.BUURT_YAMNET_CLASS_MAP_URL ||
    "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv",
  videoClassMap: [],
  videoClassMapUrl:
    window.BUURT_VIDEO_CLASS_MAP_URL ||
    "/static/labels/kinetics400_label_map.txt",
  videoFrameCount: 16,
  videoInputSize: 112,
};

class InferenceWorkerBridge {
  constructor(config) {
    this.config = config;
    this.worker = null;
    this.requests = new Map();
    this.nextId = 1;
    this.initPromise = null;
  }

  ensureWorker() {
    if (this.worker) {
      return this.initPromise;
    }
    try {
      this.worker = new Worker("/static/inference.worker.js");
    } catch (error) {
      console.warn("Inference worker unavailable; falling back to no-op", error);
      return Promise.reject(error);
    }
    this.worker.onmessage = (event) => {
      const message = event.data || {};
      const pending = this.requests.get(message.id);
      if (!pending) {
        return;
      }
      this.requests.delete(message.id);
      if (message.success) {
        pending.resolve(message.result);
      } else {
        pending.reject(new Error(message.error || "Inference worker error"));
      }
    };
    this.worker.onerror = (event) => {
      console.error("Inference worker error", event);
    };
    this.initPromise = this.postMessage("INIT", this.config);
    return this.initPromise;
  }

  async postMessage(type, payload = null, transfer = []) {
    if (!this.worker) {
      await this.ensureWorker();
    }
    await this.initPromise;
    const id = this.nextId++;
    const promise = new Promise((resolve, reject) => {
      this.requests.set(id, { resolve, reject });
    });
    try {
      this.worker.postMessage({ id, type, payload }, transfer);
    } catch (error) {
      this.requests.delete(id);
      throw error;
    }
    return promise;
  }

  preload() {
    return this.postMessage("PRELOAD");
  }

  runDetections(payload, transfer = []) {
    return this.postMessage("RUN_DETECTIONS", payload, transfer);
  }

  terminate() {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
    this.requests.clear();
    this.nextId = 1;
    this.initPromise = null;
  }
}

let inferenceWorkerBridge = null;

function getInferenceWorkerBridge() {
  if (!inferenceWorkerBridge) {
    inferenceWorkerBridge = new InferenceWorkerBridge({
      audioModelUrl: inferenceState.audioModelUrl,
      videoModelUrl: inferenceState.videoModelUrl,
      audioModelId: inferenceState.audioModelId,
      videoModelId: inferenceState.videoModelId,
      wasmBaseUrl: inferenceState.wasmBaseUrl,
      classMapUrl: inferenceState.classMapUrl,
      videoClassMapUrl: inferenceState.videoClassMapUrl,
      videoFrameCount: inferenceState.videoFrameCount,
      videoInputSize: inferenceState.videoInputSize,
    });
  }
  return inferenceWorkerBridge;
}

function preloadInferenceResources() {
  try {
    const worker = getInferenceWorkerBridge();
    worker.preload().catch((error) => {
      console.warn("Inference worker preload failed", error);
    });
  } catch (error) {
    console.warn("Inference worker unavailable", error);
  }
}

async function ensureInferenceRuntime() {
  try {
    await getInferenceWorkerBridge().preload();
  } catch (error) {
    console.warn("Inference runtime preload failed", error);
  }
}

async function prepareAudioInputForWorker(audioInput, sampleRate) {
  if (!audioInput) {
    return null;
  }
  if (audioInput instanceof Float32Array) {
    return { pcm: audioInput, sampleRate: sampleRate || 48000 };
  }
  if (Array.isArray(audioInput)) {
    return { pcm: Float32Array.from(audioInput), sampleRate: sampleRate || 48000 };
  }
  if (audioInput?.pcm instanceof Float32Array) {
    return {
      pcm: audioInput.pcm,
      sampleRate: audioInput.sampleRate || sampleRate || 48000,
    };
  }
  if (audioInput instanceof Blob) {
    const arrayBuffer = await audioInput.arrayBuffer();
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) {
      return null;
    }
    const audioCtx = new AudioCtx();
    try {
      const decoded = await audioCtx.decodeAudioData(arrayBuffer.slice(0));
      const channel = decoded.getChannelData(0);
      const pcm = new Float32Array(channel.length);
      pcm.set(channel);
      return { pcm, sampleRate: decoded.sampleRate };
    } finally {
      audioCtx.close().catch(() => {});
    }
  }
  return null;
}

async function runAudioInference(audioInput, sampleRate, startTs, endTs) {
  const prepared = await prepareAudioInputForWorker(audioInput, sampleRate);
  if (!prepared) {
    return [];
  }
  const start = startTs instanceof Date ? startTs : new Date(startTs || Date.now());
  const end = endTs instanceof Date ? endTs : new Date(endTs || Date.now());
  const payload = {
    audio: {
      pcm: prepared.pcm.buffer.slice(0),
      sampleRate: prepared.sampleRate,
      startTs: start.toISOString(),
      endTs: end.toISOString(),
      segmentDurationMs: computeSegmentDurationMs(start, end),
    },
  };
  const detections = await getInferenceWorkerBridge().runDetections(payload, [payload.audio.pcm]);
  return Array.isArray(detections) ? detections : [];
}

async function runVideoInference(videoSource, startTs, endTs) {
  if (!videoSource?.data) {
    return [];
  }
  const frames = videoSource.data instanceof Float32Array
    ? videoSource.data
    : new Float32Array(videoSource.data);
  const start = startTs instanceof Date ? startTs : new Date(startTs || Date.now());
  const end = endTs instanceof Date ? endTs : new Date(endTs || Date.now());
  const payload = {
    video: {
      frames: frames.buffer.slice(0),
      frameCount: videoSource.frameCount || inferenceState.videoFrameCount,
      inputSize: inferenceState.videoInputSize,
      startTs: start.toISOString(),
      endTs: end.toISOString(),
      segmentDurationMs: computeSegmentDurationMs(start, end),
    },
  };
  const detections = await getInferenceWorkerBridge().runDetections(payload, [payload.video.frames]);
  return Array.isArray(detections) ? detections : [];
}

const DEFAULT_CONFIG = {
  segment_length_sec: 10,
  overlap_sec: 5,
  confidence_threshold: 0.1,
  preview_enabled: false,
  audio_inference_enabled: true,
  video_inference_enabled: true,
};

function loadStoredConfig() {
  try {
    const raw = window.localStorage?.getItem(CONFIG_STORAGE_KEY);
    if (!raw) {
      return { ...DEFAULT_CONFIG };
    }
    const parsed = JSON.parse(raw);
    return { ...DEFAULT_CONFIG, ...parsed };
  } catch (error) {
    console.warn("Failed to parse saved capture config", error);
    return { ...DEFAULT_CONFIG };
  }
}

const userConfig = loadStoredConfig();

function persistConfig() {
  try {
    window.localStorage?.setItem(CONFIG_STORAGE_KEY, JSON.stringify(userConfig));
  } catch (error) {
    console.warn("Failed to persist capture config", error);
  }
}

function clamp(value, min, max) {
  let result = value;
  if (typeof min === "number") {
    result = Math.max(result, min);
  }
  if (typeof max === "number") {
    result = Math.min(result, max);
  }
  return result;
}

function toTimestampMs(value) {
  if (value instanceof Date) {
    return value.getTime();
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value) {
    const parsed = Date.parse(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return null;
}

function computeSegmentDurationMs(startTs, endTs) {
  const start = toTimestampMs(startTs);
  const end = toTimestampMs(endTs);
  if (typeof start === "number" && typeof end === "number") {
    return Math.max(Math.round(end - start), 0);
  }
  return null;
}

function parseNumberInput(element, fallback, options = {}) {
  const raw = element?.value?.trim();
  const parsed = Number.parseFloat(raw);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return clamp(parsed, options.min, options.max);
}

function getEffectiveConfig() {
  const segment = clamp(
    Number(userConfig.segment_length_sec) || DEFAULT_CONFIG.segment_length_sec,
    5,
    60,
  );
  const overlap = clamp(
    Number(userConfig.overlap_sec) ?? DEFAULT_CONFIG.overlap_sec,
    0,
    Math.max(segment - 1, 0),
  );
  const threshold = clamp(
    Number(userConfig.confidence_threshold) ?? DEFAULT_CONFIG.confidence_threshold,
    0,
    1,
  );
  const preview = Boolean(userConfig.preview_enabled);
  const audioInference = userConfig.audio_inference_enabled ?? DEFAULT_CONFIG.audio_inference_enabled;
  const videoInference = userConfig.video_inference_enabled ?? DEFAULT_CONFIG.video_inference_enabled;
  return {
    segment_length_sec: segment,
    overlap_sec: overlap,
    confidence_threshold: threshold,
    preview_enabled: preview,
    audio_inference_enabled: Boolean(audioInference),
    video_inference_enabled: Boolean(videoInference),
  };
}

function updateConfigFromInputs() {
  if (!captureSettingsForm) {
    return getEffectiveConfig();
  }
  const segment = parseNumberInput(segmentLengthInput, DEFAULT_CONFIG.segment_length_sec, {
    min: 5,
    max: 60,
  });
  const overlapMax = Math.max(segment - 1, 0);
  if (overlapInput) {
    overlapInput.max = String(overlapMax);
  }
  const overlap = parseNumberInput(overlapInput, DEFAULT_CONFIG.overlap_sec, {
    min: 0,
    max: overlapMax,
  });
  const threshold = parseNumberInput(
    confidenceInput,
    DEFAULT_CONFIG.confidence_threshold,
    {
      min: 0,
      max: 1,
    },
  );

  userConfig.segment_length_sec = segment;
  userConfig.overlap_sec = overlap;
  userConfig.confidence_threshold = threshold;
  if (audioInferenceToggle) {
    userConfig.audio_inference_enabled = Boolean(audioInferenceToggle.checked);
  }
  if (videoInferenceToggle) {
    userConfig.video_inference_enabled = Boolean(videoInferenceToggle.checked);
  }
  persistConfig();
  return getEffectiveConfig();
}

function applyConfigToInputs() {
  if (!captureSettingsForm) {
    return;
  }
  const config = getEffectiveConfig();
  if (segmentLengthInput) {
    segmentLengthInput.value = String(config.segment_length_sec);
  }
  if (overlapInput) {
    overlapInput.max = String(Math.max(config.segment_length_sec - 1, 0));
    overlapInput.value = String(config.overlap_sec);
  }
  if (confidenceInput) {
    confidenceInput.value = config.confidence_threshold.toFixed(2);
  }
  if (audioInferenceToggle) {
    audioInferenceToggle.checked = Boolean(config.audio_inference_enabled);
  }
  if (videoInferenceToggle) {
    videoInferenceToggle.checked = Boolean(config.video_inference_enabled);
  }
  updateSliderDisplays(config);
  updatePreviewToggleButton();
  updatePreviewVisibility();
}

function updateSliderDisplays(config = getEffectiveConfig()) {
  if (segmentLengthDisplay) {
    segmentLengthDisplay.textContent = `${config.segment_length_sec}s`;
  }
  if (overlapDisplay) {
    overlapDisplay.textContent = `${config.overlap_sec}s`;
  }
  if (confidenceDisplay) {
    confidenceDisplay.textContent = config.confidence_threshold.toFixed(2);
  }
}

function updatePreviewVisibility(stream) {
  if (!previewVideo) {
    return;
  }
  const shouldShow = Boolean(userConfig.preview_enabled) && Boolean(stream || previewVideo.srcObject);
  if (shouldShow && stream) {
    previewVideo.srcObject = stream;
    previewVideo.muted = true;
    previewVideo
      .play()
      .catch(() => {});
  } else if (!shouldShow && previewVideo.srcObject) {
    try {
      previewVideo.pause();
    } catch {
      // ignore
    }
    previewVideo.srcObject = null;
  }
  previewVideo.classList.toggle("is-visible", shouldShow);
  if (previewHelp) {
    previewHelp.hidden = shouldShow;
  }
}

function clearPreview() {
  if (!previewVideo) {
    return;
  }
  try {
    previewVideo.pause();
  } catch {
    // ignore
  }
  previewVideo.srcObject = null;
  previewVideo.classList.remove("is-visible");
  if (previewHelp) {
    previewHelp.hidden = false;
  }
}

function updatePreviewToggleButton() {
  if (!previewToggleButton) {
    return;
  }
  previewToggleButton.textContent = userConfig.preview_enabled
    ? "Disable preview"
    : "Enable preview";
  previewToggleButton.dataset.active = userConfig.preview_enabled ? "true" : "false";
}

function startRecordingBanner() {
  if (!recordingBanner) {
    return;
  }
  const startTime = new Date();
  const update = () => {
    if (!captureState.context) {
      return;
    }
    const now = new Date();
    const diffMs = now - startTime;
    const minutes = String(Math.floor(diffMs / 60000)).padStart(2, "0");
    const seconds = String(Math.floor((diffMs % 60000) / 1000)).padStart(2, "0");
    const segments = captureState.context.segmentIndex;
    recordingBanner.textContent = `Recording · ${minutes}:${seconds} elapsed · ${segments} segment${segments === 1 ? "" : "s"} uploaded`;
    recordingBanner.dataset.active = "true";
  };
  update();
  if (recordingBannerInterval) {
    window.clearInterval(recordingBannerInterval);
  }
  recordingBannerInterval = window.setInterval(update, 1000);
}

function stopRecordingBanner() {
  if (recordingBannerInterval) {
    window.clearInterval(recordingBannerInterval);
    recordingBannerInterval = null;
  }
  if (recordingBanner) {
    recordingBanner.dataset.active = "false";
    recordingBanner.textContent = "Recording…";
  }
}

function formatDurationLabel(valueMs, suffix) {
  const numeric = typeof valueMs === "number" ? valueMs : Number(valueMs);
  if (!Number.isFinite(numeric) || numeric <= 0) {
    return null;
  }
  if (numeric >= 1000) {
    const seconds = numeric / 1000;
    const decimals = seconds >= 10 ? 0 : 1;
    return `${seconds.toFixed(decimals)}s ${suffix}`;
  }
  return `${Math.max(Math.round(numeric), 1)}ms ${suffix}`;
}

function detectionSegmentDurationMs(det) {
  if (typeof det?.segment_duration_ms === "number") {
    return det.segment_duration_ms;
  }
  if (typeof det?.segment_length_sec === "number") {
    return det.segment_length_sec * 1000;
  }
  return null;
}

function renderDetectionLog() {
  if (!detectionList) {
    return;
  }
  detectionList.innerHTML = "";
  if (!detectionLog.length) {
    if (detectionEmptyState) {
      detectionEmptyState.hidden = false;
    }
    return;
  }
  if (detectionEmptyState) {
    detectionEmptyState.hidden = true;
  }
  for (const entry of detectionLog) {
    const item = document.createElement("li");
    const header = document.createElement("div");
    header.className = "det-header";
    const label = document.createElement("span");
    label.textContent = entry.label;
    const confidence = document.createElement("span");
    confidence.textContent = entry.confidence;
    header.append(label, confidence);
    const meta = document.createElement("div");
    meta.className = "det-meta";
    const metaParts = [entry.time, entry.model];
    if (entry.segment) {
      metaParts.push(entry.segment);
    }
    if (entry.latency) {
      metaParts.push(entry.latency);
    }
    meta.textContent = metaParts.join(" · ");
    item.append(header, meta);
    detectionList.append(item);
  }
}

function appendDetectionsToLog(detections) {
  if (!detections || !detections.length) {
    return;
  }
  const mapped = detections.map((det) => {
    const label =
      det.detection_class ||
      det.class ||
      det.model_id ||
      "Detection";
    const confidence = typeof det.confidence === "number"
      ? `${Math.round(det.confidence * 100)}%`
      : "—";
    const ts = det.timestamp ? new Date(det.timestamp) : new Date();
    const segmentLabel = formatDurationLabel(
      detectionSegmentDurationMs(det),
      "segment",
    );
    const latencyLabel = formatDurationLabel(
      det.inference_latency_ms,
      "inference",
    );
    return {
      label,
      confidence,
      model: det.model_id || "client",
      time: ts.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
      segment: segmentLabel,
      latency: latencyLabel,
    };
  });
  detectionLog.unshift(...mapped);
  if (detectionLog.length > 25) {
    detectionLog.length = 25;
  }
  renderDetectionLog();
}

function clearDetectionLog() {
  detectionLog.length = 0;
  renderDetectionLog();
}

const kineticsNormalization = {
  mean: [0.43216, 0.394666, 0.37645],
  std: [0.22803, 0.22145, 0.216989],
};

function formatTimestamp(timestamp) {
  if (!timestamp) {
    return "--";
  }
  try {
    const date = new Date(timestamp);
    return date.toLocaleString(undefined, {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      year: "numeric",
      month: "short",
      day: "2-digit",
    });
  } catch (error) {
    console.error("Failed to format timestamp", error);
    return timestamp;
  }
}

function setStatus(message, tone = "info") {
  statusMessage.textContent = message;
  statusMessage.dataset.tone = tone;
}

function toggleButtons(isRecording) {
  startButton.disabled = isRecording;
  stopButton.disabled = !isRecording;
}

function handleSessionSnapshot(sessions) {
  const ordered = [...sessions].sort((a, b) =>
    a.started_at < b.started_at ? 1 : -1,
  );

  renderSessions(ordered);

  const running = ordered.find((session) => !session.ended_at);
  if (running) {
    activeSession = running;
  } else if (activeSession) {
    const latest = ordered.find((session) => session.id === activeSession.id);
    activeSession = latest || null;
  } else {
    activeSession = null;
  }

  renderActiveSession(activeSession);
  toggleButtons(Boolean(activeSession && !activeSession.ended_at));
}

function renderActiveSession(session) {
  activeSessionDetails.innerHTML = "";

  if (!session) {
    const statusTerm = document.createElement("dt");
    statusTerm.textContent = "Status";
    const statusValue = document.createElement("dd");
    statusValue.textContent = "No session running";
    activeSessionDetails.append(statusTerm, statusValue);
    return;
  }

  const data = [
    ["Status", session.ended_at ? "Stopped" : "Recording"],
    ["Session ID", session.id],
    ["Started", formatTimestamp(session.started_at)],
    ["Ended", session.ended_at ? formatTimestamp(session.ended_at) : "--"],
  ];

  for (const [term, value] of data) {
    const dt = document.createElement("dt");
    dt.textContent = term;
    const dd = document.createElement("dd");
    dd.textContent = value;
    activeSessionDetails.append(dt, dd);
  }
}

function renderSessions(sessions) {
  sessionList.innerHTML = "";

  if (!sessions.length) {
    const emptyState = document.createElement("li");
    emptyState.textContent = "No sessions yet.";
    sessionList.append(emptyState);
    return;
  }

  for (const session of sessions) {
    const item = document.createElement("li");
    const title = document.createElement("strong");
    title.textContent = session.ended_at ? "Completed session" : "Active session";
    const started = document.createElement("span");
    started.textContent = `Started: ${formatTimestamp(session.started_at)}`;
    const ended = document.createElement("span");
    ended.textContent = session.ended_at
      ? `Ended: ${formatTimestamp(session.ended_at)}`
      : "In progress";

    item.append(title, started, ended);
    sessionList.append(item);
  }
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || response.statusText);
  }
  return response.json();
}

async function getMobileDetections(blob, startTs, endTs, index) {
  try {
    const inference = captureState.inference;
    if (!inference) {
      console.warn("Inference pipeline missing; skipping detections");
      return [];
    }
    const config = getEffectiveConfig();
    const segmentAudio = config.audio_inference_enabled
      ? inference.consumeAudioSegment()
      : null;
    const segmentVideo = config.video_inference_enabled
      ? inference.consumeVideoFrames()
      : null;
    const worker = getInferenceWorkerBridge();
    const payload = {
      audio: segmentAudio
        ? {
            pcm: segmentAudio.pcm.buffer,
            sampleRate: segmentAudio.sampleRate,
            startTs: startTs.toISOString(),
            endTs: endTs.toISOString(),
            segmentDurationMs: computeSegmentDurationMs(startTs, endTs),
          }
        : null,
      video: segmentVideo
        ? {
            frames: segmentVideo.data.buffer,
            frameCount: segmentVideo.frameCount,
            inputSize: inferenceState.videoInputSize,
            startTs: startTs.toISOString(),
            endTs: endTs.toISOString(),
            segmentDurationMs: computeSegmentDurationMs(startTs, endTs),
          }
        : null,
      index,
    };
    const transfer = [];
    if (payload.audio?.pcm) {
      transfer.push(payload.audio.pcm);
    }
    if (payload.video?.frames) {
      transfer.push(payload.video.frames);
    }
    const detections = await worker.runDetections(payload, transfer);
    inference.resetSegment();
    const combined = Array.isArray(detections) ? detections : [];
    appendDetectionsToLog(combined);
    return combined;
  } catch (error) {
    console.warn("Mobile inference unavailable, skipping detections", error);
    if (captureState.inference) {
      captureState.inference.resetSegment();
    }
    return [];
  }
}

async function loadSessions(options = {}) {
  const announce = options.announce ?? true;
  try {
    const payload = await fetchJson("/sessions/updates");
    lastRevision = payload.revision;
    handleSessionSnapshot(payload.sessions);
    if (announce) {
      setStatus("Session list updated.");
    }
  } catch (error) {
    console.error("Failed to load sessions", error);
    setStatus(`Unable to load session history: ${error.message}`, "error");
  }
}

async function startSession() {
  try {
    setStatus("Starting session…");
    startButton.disabled = true;
    const sessionPayload = await buildSessionPayload();
    const session = await fetchJson("/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(sessionPayload),
    });
    activeSession = session;
    renderActiveSession(session);
    clearDetectionLog();
    startRecordingBanner();
    toggleButtons(true);
    let captureError = null;
    try {
      await startMediaCapture(session);
    } catch (error) {
      captureError = error;
      console.warn("Media capture unavailable", error);
    }
    await loadSessions({ announce: false });
    if (captureError) {
      setStatus(
        `Session started, but media capture failed: ${captureError.message}`,
        "warning",
      );
    } else {
      setStatus("Session started. Recording in progress.", "success");
    }
  } catch (error) {
    console.error("Failed to start session", error);
    setStatus(`Unable to start session: ${error.message}`, "error");
    startButton.disabled = false;
  }
}

async function buildSessionPayload() {
  const now = new Date();
  const defaults = {
    lat: 52.3676,
    lon: 4.9041,
    accuracy_m: 5,
    orientation_heading_deg: 0,
  };

  const gpsOrigin = await resolveGpsOrigin(now, defaults);
  const orientationOrigin = await resolveOrientationOrigin(now, defaults);
  const config = updateConfigFromInputs();

  return {
    started_at: now.toISOString(),
    operator_alias: "Browser Operator",
    notes: "Started from local UI",
    app_version: "web-ui",
    model_bundle_version: "demo",
    gps_origin: gpsOrigin,
    orientation_origin: orientationOrigin,
    config_snapshot: {
      segment_length_sec: config.segment_length_sec,
      overlap_sec: config.overlap_sec,
      confidence_threshold: config.confidence_threshold,
    },
    detection_summary: {
      total_detections: 0,
      by_class: {},
    },
    redact_location: false,
    skip_backend_capture: true,
  };
}

function supportsGeolocation() {
  return Boolean(navigator?.geolocation);
}

async function resolveGpsOrigin(now, defaults) {
  if (!supportsGeolocation()) {
    return {
      lat: defaults.lat,
      lon: defaults.lon,
      accuracy_m: defaults.accuracy_m,
      captured_at: now.toISOString(),
    };
  }

  try {
    const position = await new Promise((resolve, reject) => {
      navigator.geolocation.getCurrentPosition(resolve, reject, {
        enableHighAccuracy: true,
        timeout: 8000,
        maximumAge: 10000,
      });
    });

    return {
      lat: position.coords.latitude,
      lon: position.coords.longitude,
      accuracy_m: position.coords.accuracy,
      captured_at: new Date(position.timestamp || Date.now()).toISOString(),
    };
  } catch (error) {
    console.warn("Geolocation unavailable", error);
    return {
      lat: defaults.lat,
      lon: defaults.lon,
      accuracy_m: defaults.accuracy_m,
      captured_at: now.toISOString(),
    };
  }
}

async function resolveOrientationOrigin(now, defaults) {
  if (typeof window === "undefined" || !("DeviceOrientationEvent" in window)) {
    return {
      heading_deg: defaults.orientation_heading_deg,
      captured_at: now.toISOString(),
    };
  }

  if (
    typeof DeviceOrientationEvent.requestPermission === "function" &&
    !captureState.orientationPermissionRequested
  ) {
    captureState.orientationPermissionRequested = true;
    try {
      const permission = await DeviceOrientationEvent.requestPermission();
      if (permission !== "granted") {
        return {
          heading_deg: defaults.orientation_heading_deg,
          captured_at: now.toISOString(),
        };
      }
    } catch (error) {
      console.warn("Orientation permission denied", error);
      return {
        heading_deg: defaults.orientation_heading_deg,
        captured_at: now.toISOString(),
      };
    }
  }

  const reading = await new Promise((resolve) => {
    const timeout = setTimeout(() => {
      window.removeEventListener("deviceorientation", handler);
      resolve(null);
    }, 2000);

    function handler(event) {
      clearTimeout(timeout);
      window.removeEventListener("deviceorientation", handler);
      resolve(event);
    }

    window.addEventListener("deviceorientation", handler, { once: true });
  });

  if (!reading) {
    return {
      heading_deg: defaults.orientation_heading_deg,
      captured_at: now.toISOString(),
    };
  }

  return {
    heading_deg: reading.alpha ?? defaults.orientation_heading_deg,
    pitch_deg: reading.beta ?? null,
    roll_deg: reading.gamma ?? null,
    captured_at: new Date().toISOString(),
  };
}

function supportsMediaCapture() {
  return Boolean(navigator?.mediaDevices?.getUserMedia && window.MediaRecorder);
}

async function startMediaCapture(session) {
  if (!supportsMediaCapture()) {
    throw new Error(
      "Media capture is not supported by this browser; uploads disabled.",
    );
  }

  if (captureState.context) {
    await stopMediaCapture();
  }

  const segmentLengthSec =
    session?.config_snapshot?.segment_length_sec ?? 10;
  const segmentLengthMs = Math.max(segmentLengthSec * 1000, 1000);
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: true,
    video: {
      facingMode: "environment",
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
  });
  updatePreviewVisibility(stream);
  const options = resolveRecorderOptions(stream);
  const recorder = new MediaRecorder(stream, options);
  try {
    captureState.inference = await startLiveInference(stream, segmentLengthMs);
  } catch (error) {
    console.warn("Live inference pipeline failed to start; continuing without client detections", error);
    captureState.inference = null;
  }
  const context = {
    sessionId: session.id,
    recorder,
    stream,
    segmentIndex: 0,
    segmentLengthMs,
    segmentStart: null,
  };

  recorder.addEventListener("start", () => {
    context.segmentStart = new Date();
    if (captureState.inference) {
      captureState.inference.resetSegment();
    }
  });

  recorder.addEventListener("dataavailable", (event) => {
    if (!event.data || event.data.size === 0) {
      context.segmentStart = new Date();
      return;
    }
    const start = context.segmentStart ?? new Date();
    const end = new Date();
    context.segmentStart = end;
    const segmentIndex = context.segmentIndex;
    const uploadPromise = (async () => {
      const detections = await getMobileDetections(
        event.data,
        start,
        end,
        segmentIndex,
      );
      return uploadSegmentBlob(
        context.sessionId,
        segmentIndex,
        start,
        end,
        event.data,
        detections,
      );
    })();
    captureState.uploads.push(uploadPromise);
    uploadPromise
      .catch((error) => {
        console.error("Segment upload failed", error);
        setStatus(`Segment upload failed: ${error.message}`, "error");
      })
      .finally(() => {
        captureState.uploads = captureState.uploads.filter(
          (pending) => pending !== uploadPromise,
        );
      });
    context.segmentIndex += 1;
  });

  recorder.addEventListener("error", (event) => {
    const message = event.error?.message || "Recorder error";
    console.error("Recorder error", event.error);
    setStatus(`Recorder error: ${message}`, "error");
  });

  try {
    recorder.start(segmentLengthMs);
  } catch (error) {
    if (captureState.inference) {
      try {
        captureState.inference.stop();
      } catch (stopError) {
        console.warn("Failed to stop inference after recorder error", stopError);
      }
      captureState.inference = null;
    }
    stream.getTracks().forEach((track) => track.stop());
    throw error;
  }
  captureState.context = context;
}

function resolveRecorderOptions(stream) {
  if (typeof MediaRecorder === "undefined") {
    return {};
  }

  const userAgent = navigator.userAgent?.toLowerCase() || "";
  const isSafari =
    userAgent.includes("safari") &&
    !userAgent.includes("chrome") &&
    !userAgent.includes("android");
  const hasVideo = stream.getVideoTracks().length > 0;
  const candidates = hasVideo
    ? isSafari
      ? [
          "video/mp4;codecs=avc1.42E01E,mp4a.40.2",
          "video/mp4",
          "video/webm;codecs=vp8,opus",
        ]
      : ["video/webm;codecs=vp8,opus", "video/webm", "video/mp4"]
    : ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"];

  const supported = candidates.find((type) =>
    MediaRecorder.isTypeSupported?.(type),
  );

  return supported ? { mimeType: supported } : {};
}

function createInferenceBuffers() {
  let audioChunks = [];
  let audioSampleRate = null;
  let videoFrames = [];
  const maxVideoFrames = Math.max(inferenceState.videoFrameCount * 3, 24);

  return {
    setAudioSampleRate(rate) {
      audioSampleRate = rate;
    },
    pushAudioChunk(chunk) {
      if (!chunk || !chunk.length) {
        return;
      }
      const copy = new Float32Array(chunk.length);
      copy.set(chunk);
      audioChunks.push(copy);
    },
    pushVideoFrame(frameData) {
      if (!frameData || !frameData.length) {
        return;
      }
      videoFrames.push(frameData);
      if (videoFrames.length > maxVideoFrames) {
        videoFrames.shift();
      }
    },
    consumeAudioSegment() {
      if (!audioChunks.length || !audioSampleRate) {
        return null;
      }
      const totalLength = audioChunks.reduce((sum, chunk) => sum + chunk.length, 0);
      const merged = new Float32Array(totalLength);
      let offset = 0;
      for (const chunk of audioChunks) {
        merged.set(chunk, offset);
        offset += chunk.length;
      }
      audioChunks = [];
      return { pcm: merged, sampleRate: audioSampleRate };
    },
    consumeVideoFrames() {
      if (!videoFrames.length) {
        return null;
      }
      const frameCount = Math.min(
        videoFrames.length,
        Math.max(inferenceState.videoFrameCount, 1),
      );
      const startIndex = Math.max(videoFrames.length - frameCount, 0);
      const selected = videoFrames.slice(startIndex);
      const targetSize = Math.max(inferenceState.videoInputSize, 1);
      const data = new Float32Array(frameCount * targetSize * targetSize * 3);
      for (let i = 0; i < selected.length; i++) {
        data.set(selected[i], i * selected[i].length);
      }
      videoFrames = [];
      return { data, frameCount };
    },
    resetSegment() {
      audioChunks = [];
      videoFrames = [];
    },
    hasAudioData() {
      return audioChunks.length > 0;
    },
    hasVideoData() {
      return videoFrames.length > 0;
    },
  };
}

async function startLiveInference(stream, segmentLengthMs) {
  const buffers = createInferenceBuffers();
  const videoTrack = stream.getVideoTracks()[0] || null;
  const audioTrack = stream.getAudioTracks()[0] || null;
  const stopFns = [];
  let stopped = false;
  let audioEnabled = Boolean(audioTrack);
  let videoEnabled = Boolean(videoTrack);

  const inferenceConfig = getEffectiveConfig();
  const allowAudioInference = Boolean(inferenceConfig.audio_inference_enabled);
  const allowVideoInference = Boolean(inferenceConfig.video_inference_enabled);

  if (audioTrack && allowAudioInference) {
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    buffers.setAudioSampleRate(audioCtx.sampleRate);
    const source = audioCtx.createMediaStreamSource(new MediaStream([audioTrack]));
    if (audioCtx.audioWorklet) {
      const workletCode = `
class CaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs?.[0]?.[0];
    if (input) {
      this.port.postMessage(input);
    }
    return true;
  }
}
registerProcessor("capture-processor", CaptureProcessor);
`;
      const workletUrl = URL.createObjectURL(
        new Blob([workletCode], { type: "application/javascript" }),
      );
      try {
        await audioCtx.audioWorklet.addModule(workletUrl);
        const node = new AudioWorkletNode(audioCtx, "capture-processor", {
          numberOfInputs: 1,
          numberOfOutputs: 0,
          channelCount: 1,
        });
        node.port.onmessage = (event) => {
          buffers.pushAudioChunk(event.data);
        };
        source.connect(node);
        stopFns.push(() => {
          node.port.onmessage = null;
          node.disconnect();
          source.disconnect();
          audioCtx.close().catch(() => {});
          URL.revokeObjectURL(workletUrl);
        });
      } catch (error) {
        console.warn("AudioWorklet unavailable, falling back to ScriptProcessor", error);
        URL.revokeObjectURL(workletUrl);
        const processor = audioCtx.createScriptProcessor(4096, 1, 1);
        processor.onaudioprocess = (event) => {
          buffers.pushAudioChunk(event.inputBuffer.getChannelData(0));
        };
        source.connect(processor);
        processor.connect(audioCtx.destination);
        stopFns.push(() => {
          processor.disconnect();
          source.disconnect();
          audioCtx.close().catch(() => {});
        });
      }
    } else {
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      processor.onaudioprocess = (event) => {
        buffers.pushAudioChunk(event.inputBuffer.getChannelData(0));
      };
      source.connect(processor);
      processor.connect(audioCtx.destination);
      stopFns.push(() => {
        processor.disconnect();
        source.disconnect();
        audioCtx.close().catch(() => {});
      });
    }
  }

  if (videoTrack && allowVideoInference) {
    const targetSize = Math.max(inferenceState.videoInputSize, 1);
    const frameIntervalMs = Math.max(
      segmentLengthMs / Math.max(inferenceState.videoFrameCount, 1),
      50,
    );
    let lastCapture = 0;

    const canvas =
      typeof OffscreenCanvas !== "undefined"
        ? new OffscreenCanvas(targetSize, targetSize)
        : (() => {
            const element = document.createElement("canvas");
            element.width = targetSize;
            element.height = targetSize;
            return element;
          })();
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) {
      videoEnabled = false;
    }

    const captureFrame = async (source) => {
      if (stopped || !ctx) {
        return;
      }
      const now = performance.now();
      if (now - lastCapture < frameIntervalMs) {
        return;
      }
      lastCapture = now;
      try {
        drawFrameToCanvas(ctx, source, targetSize);
        const pixels = ctx.getImageData(0, 0, targetSize, targetSize).data;
        const frameData = new Float32Array(targetSize * targetSize * 3);
        normalizeFrame(pixels, frameData, 0);
        buffers.pushVideoFrame(frameData);
      } catch (error) {
        console.warn("Video frame capture failed", error);
      }
    };

    if (window.MediaStreamTrackProcessor && window.VideoFrame) {
      try {
        const processor = new MediaStreamTrackProcessor({ track: videoTrack });
        const reader = processor.readable.getReader();
        const readLoop = async () => {
          if (stopped) {
            return;
          }
          const { value, done } = await reader.read();
          if (done || stopped) {
            return;
          }
          const frame = value;
          await captureFrame(frame);
          frame.close();
          readLoop();
        };
        readLoop();
        stopFns.push(() => {
          reader.cancel().catch(() => {});
        });
      } catch (error) {
        console.warn("MediaStreamTrackProcessor unavailable; falling back to video element", error);
        const video = document.createElement("video");
        video.srcObject = new MediaStream([videoTrack]);
        video.muted = true;
        video.playsInline = true;
        video.preload = "auto";
        video.addEventListener("loadedmetadata", () => {
          video.play().catch(() => {});
        }, { once: true });
        let rafId = null;
        const loop = () => {
          captureFrame(video);
          if (!stopped) {
            rafId = requestAnimationFrame(loop);
          }
        };
        rafId = requestAnimationFrame(loop);
        stopFns.push(() => {
          if (rafId) {
            cancelAnimationFrame(rafId);
          }
          video.srcObject = null;
        });
      }
    } else {
      const video = document.createElement("video");
      video.srcObject = new MediaStream([videoTrack]);
      video.muted = true;
      video.playsInline = true;
      video.preload = "auto";
      video.addEventListener("loadedmetadata", () => {
        video.play().catch(() => {});
      }, { once: true });
      let rafId = null;
      const loop = () => {
        captureFrame(video);
        if (!stopped) {
          rafId = requestAnimationFrame(loop);
        }
      };
      rafId = requestAnimationFrame(loop);
      stopFns.push(() => {
        if (rafId) {
          cancelAnimationFrame(rafId);
        }
        video.srcObject = null;
      });
    }
  }

  window.setTimeout(() => {
    if (audioTrack && !buffers.hasAudioData()) {
      audioEnabled = false;
      console.warn("Audio inference disabled: no PCM captured from live stream");
    }
    if (videoTrack && !buffers.hasVideoData()) {
      videoEnabled = false;
      console.warn("Video inference disabled: unable to read frames from live stream");
    }
  }, 1000);

  return {
    consumeAudioSegment() {
      return audioEnabled ? buffers.consumeAudioSegment() : null;
    },
    consumeVideoFrames() {
      return videoEnabled ? buffers.consumeVideoFrames() : null;
    },
    resetSegment() {
      buffers.resetSegment();
    },
    stop() {
      stopped = true;
      stopFns.forEach((fn) => {
        try {
          fn();
        } catch (error) {
          console.warn("Failed to stop inference component", error);
        }
      });
    },
  };
}

async function stopMediaCapture() {
  const context = captureState.context;
  if (!context) {
    return;
  }

  if (captureState.inference) {
    try {
      captureState.inference.stop();
    } catch (error) {
      console.warn("Failed to stop inference pipelines", error);
    }
    captureState.inference = null;
  }

  captureState.context = null;
  const { recorder, stream } = context;
  const stopPromise = new Promise((resolve) => {
    if (recorder.state === "inactive") {
      resolve();
      return;
    }
    recorder.addEventListener("stop", resolve, { once: true });
  });
  if (recorder.state !== "inactive") {
    recorder.stop();
  }
  stream.getTracks().forEach((track) => track.stop());
  await stopPromise;
  clearPreview();
  stopRecordingBanner();

  if (captureState.uploads.length) {
    const pending = [...captureState.uploads];
    captureState.uploads = [];
    await Promise.allSettled(pending);
  }
}

async function uploadSegmentBlob(
  sessionId,
  index,
  startTs,
  endTs,
  blob,
  detections = null,
) {
  const formData = new FormData();
  formData.append("index", String(index));
  formData.append("start_ts", startTs.toISOString());
  formData.append("end_ts", endTs.toISOString());
  formData.append(
    "file",
    blob,
    `segment-${String(index).padStart(4, "0")}.webm`,
  );

  if (detections && detections.length) {
    formData.append("detections", JSON.stringify(detections));
  }

  const response = await fetch(`/sessions/${sessionId}/segments/upload`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || response.statusText);
  }
  return response.json();
}

async function stopSession() {
  if (!activeSession) {
    setStatus("No active session to stop.", "error");
    return;
  }

  try {
    setStatus("Stopping session…");
    stopButton.disabled = true;
    await stopMediaCapture();
    const session = await fetchJson(`/sessions/${activeSession.id}/stop`, {
      method: "POST",
    });
    activeSession = session;
    renderActiveSession(session);
    toggleButtons(false);
    setStatus("Session stopped.", "success");
    stopRecordingBanner();
    await loadSessions({ announce: false });
  } catch (error) {
    console.error("Failed to stop session", error);
    setStatus(`Unable to stop session: ${error.message}`, "error");
    stopButton.disabled = false;
  }
}

function stopPolling() {
  if (pollingTimer) {
    window.clearInterval(pollingTimer);
    pollingTimer = null;
  }
}

function startPolling() {
  if (pollingTimer) {
    return;
  }
  stopLongPolling();
  pollingTimer = window.setInterval(() => {
    loadSessions({ announce: false }).catch((error) => {
      console.error("Polling update failed", error);
    });
  }, 10000);
}


function drawFrameToCanvas(ctx, source, targetSize) {
  const videoWidth =
    source?.videoWidth ||
    source?.displayWidth ||
    source?.width ||
    source?.codedWidth ||
    targetSize;
  const videoHeight =
    source?.videoHeight ||
    source?.displayHeight ||
    source?.height ||
    source?.codedHeight ||
    targetSize;
  const videoRatio = videoWidth / videoHeight;
  let sx = 0;
  let sy = 0;
  let sw = videoWidth;
  let sh = videoHeight;

  if (videoRatio > 1) {
    sh = videoHeight;
    sw = sh;
    sx = (videoWidth - sw) / 2;
  } else if (videoRatio < 1) {
    sw = videoWidth;
    sh = sw;
    sy = (videoHeight - sh) / 2;
  }

  if (typeof ctx.drawImage === "function") {
    ctx.drawImage(source, sx, sy, sw, sh, 0, 0, targetSize, targetSize);
  }
}

function normalizeFrame(pixels, target, offset) {
  const mean = kineticsNormalization.mean;
  const std = kineticsNormalization.std;
  let writeIndex = offset;
  for (let i = 0; i < pixels.length; i += 4) {
    const r = pixels[i] / 255;
    const g = pixels[i + 1] / 255;
    const b = pixels[i + 2] / 255;
    target[writeIndex++] = (r - mean[0]) / std[0];
    target[writeIndex++] = (g - mean[1]) / std[1];
    target[writeIndex++] = (b - mean[2]) / std[2];
  }
}

async function fetchModelFromCache(url) {
  if (window.caches) {
    const cache = await caches.open("buurt-models");
    const cached = await cache.match(url);
    if (cached) {
      return cached.arrayBuffer();
    }
    const response = await fetch(url, { cache: "force-cache" });
    await cache.put(url, response.clone());
    return response.arrayBuffer();
  }
  const response = await fetch(url, { cache: "force-cache" });
  return response.arrayBuffer();
}

async function loadScriptWithFallback(urls) {
  for (const url of urls) {
    try {
      // eslint-disable-next-line no-await-in-loop
      await new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = url;
        script.onload = () => resolve();
        script.onerror = () => reject(new Error(`Failed to load ${url}`));
        document.head.append(script);
      });
      return;
    } catch (error) {
      console.warn("Script load failed, trying next URL", url, error);
    }
  }
  throw new Error("All script sources failed to load");
}

function softmaxTop1(values) {
  let maxLogit = -Infinity;
  for (let i = 0; i < values.length; i++) {
    if (values[i] > maxLogit) {
      maxLogit = values[i];
    }
  }
  let sumExp = 0;
  let maxIdx = 0;
  let maxProb = 0;
  for (let i = 0; i < values.length; i++) {
    const expVal = Math.exp(values[i] - maxLogit);
    sumExp += expVal;
    if (expVal > maxProb) {
      maxProb = expVal;
      maxIdx = i;
    }
  }
  if (sumExp > 0) {
    maxProb /= sumExp;
  }
  return { maxIdx, maxProb };
}

async function loadClassMap() {
  try {
    const response = await fetch(inferenceState.classMapUrl, { cache: "force-cache" });
    if (!response.ok) {
      throw new Error(`Failed to fetch class map: ${response.statusText}`);
    }
    const text = await response.text();
    const lines = text
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);
    const headers = lines.shift(); // drop header
    const labels = [];
    for (const line of lines) {
      const parts = line.trim().split(",");
      // CSV format: index,mid,display_name
      if (parts.length >= 3) {
        labels.push(parts[2].replace(/(^\"|\"$)/g, "").trim());
      }
    }
    return labels;
  } catch (error) {
    console.warn("Failed to load class map", error);
    return [];
  }
}

async function loadVideoClassMap() {
  try {
    const response = await fetch(inferenceState.videoClassMapUrl, { cache: "force-cache" });
    if (!response.ok) {
      throw new Error(`Failed to fetch video class map: ${response.statusText}`);
    }
    const text = await response.text();
    return text
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);
  } catch (error) {
    console.warn("Failed to load video class map", error);
    return [];
  }
}

function buildMelFilterbank(
  numMelBins,
  numSpectrogramBins,
  sampleRate,
  lowerEdgeHz,
  upperEdgeHz,
) {
  const hzToMel = (hz) => 1127 * Math.log(1 + hz / 700);
  const melToHz = (mel) => 700 * (Math.exp(mel / 1127) - 1);

  const lowerMel = hzToMel(lowerEdgeHz);
  const upperMel = hzToMel(upperEdgeHz);
  const melPoints = new Float32Array(numMelBins + 2);
  for (let i = 0; i < melPoints.length; i++) {
    melPoints[i] = lowerMel + ((upperMel - lowerMel) * i) / (numMelBins + 1);
  }
  const hzPoints = melPoints.map(melToHz);
  const binFrequencies = new Float32Array(numSpectrogramBins);
  const linearFreqs = (sampleRate / 2) / (numSpectrogramBins - 1);
  for (let i = 0; i < numSpectrogramBins; i++) {
    binFrequencies[i] = i * linearFreqs;
  }

  const filterbank = [];
  for (let m = 0; m < numMelBins; m++) {
    const lower = hzPoints[m];
    const center = hzPoints[m + 1];
    const upper = hzPoints[m + 2];
    const weights = new Float32Array(numSpectrogramBins);
    for (let k = 0; k < numSpectrogramBins; k++) {
      const freq = binFrequencies[k];
      let w = 0;
      if (freq >= lower && freq <= center) {
        w = (freq - lower) / (center - lower + 1e-8);
      } else if (freq > center && freq <= upper) {
        w = (upper - freq) / (upper - center + 1e-8);
      }
      weights[k] = w;
    }
    filterbank.push(weights);
  }

  return window.tf.tensor2d(filterbank, [numMelBins, numSpectrogramBins]);
}

function stopLongPolling() {
  if (longPollController) {
    longPollController.aborted = true;
    longPollController = null;
  }
}

async function startLongPolling() {
  if (longPollController) {
    return;
  }

  stopPolling();

  const controller = { aborted: false };
  longPollController = controller;

  const loop = async () => {
    while (!controller.aborted) {
      try {
        const params = new URLSearchParams();
        if (lastRevision !== null && lastRevision !== undefined) {
          params.set("cursor", String(lastRevision));
        }
        const url = params.toString()
          ? `/sessions/updates?${params.toString()}`
          : "/sessions/updates";
        const payload = await fetchJson(url);
        lastRevision = payload.revision;
        handleSessionSnapshot(payload.sessions);
      } catch (error) {
        if (controller.aborted) {
          return;
        }
        console.error("Long-polling update failed", error);
        await new Promise((resolve) => setTimeout(resolve, 2000));
      }
    }
  };

  loop().catch((error) => {
    if (!controller.aborted) {
      console.error("Long-polling loop terminated unexpectedly", error);
      startPolling();
    }
  });
}

function connectWebSocket() {
  if (!("WebSocket" in window)) {
    return false;
  }

  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  liveWebSocket = new WebSocket(`${protocol}://${window.location.host}/ws/sessions`);

  liveWebSocket.onmessage = (event) => {
    try {
      const sessions = JSON.parse(event.data);
      handleSessionSnapshot(sessions);
    } catch (error) {
      console.error("Failed to parse websocket payload", error);
    }
  };

  liveWebSocket.onclose = () => {
    if (shuttingDown) {
      return;
    }
    liveWebSocket = null;
    startLongPolling();
  };

  liveWebSocket.onerror = () => {
    if (liveWebSocket) {
      liveWebSocket.close();
    }
  };

  stopPolling();
  stopLongPolling();
  return true;
}

function setupLiveUpdates() {
  if (connectWebSocket()) {
    return;
  }
  if ("fetch" in window) {
    startLongPolling();
    return;
  }
  startPolling();
}

if (captureSettingsForm) {
  captureSettingsForm.addEventListener("submit", (event) => {
    event.preventDefault();
  });
  captureSettingsForm.addEventListener("input", () => {
    const config = updateConfigFromInputs();
    updateSliderDisplays(config);
  });
}

if (clearDetectionsButton) {
  clearDetectionsButton.addEventListener("click", () => {
    clearDetectionLog();
  });
}

if (resetConfigButton) {
  resetConfigButton.addEventListener("click", () => {
    Object.assign(userConfig, { ...DEFAULT_CONFIG });
    persistConfig();
    applyConfigToInputs();
    setStatus("Capture settings reset to defaults.", "info");
  });
}

if (previewToggleButton) {
  previewToggleButton.addEventListener("click", () => {
    userConfig.preview_enabled = !userConfig.preview_enabled;
    persistConfig();
    updatePreviewToggleButton();
    if (!userConfig.preview_enabled) {
      clearPreview();
    } else if (captureState.context?.stream) {
      updatePreviewVisibility(captureState.context.stream);
    }
  });
}

function cleanupLiveUpdates() {
  shuttingDown = true;
  stopMediaCapture().catch((error) => {
    console.error("Failed to stop media capture", error);
  });
  stopPolling();
  stopLongPolling();
  if (liveWebSocket) {
    const socket = liveWebSocket;
    liveWebSocket = null;
    socket.onclose = null;
    socket.close();
  }
  if (inferenceWorkerBridge) {
    inferenceWorkerBridge.terminate();
    inferenceWorkerBridge = null;
  }
}

startButton.addEventListener("click", startSession);
stopButton.addEventListener("click", stopSession);
refreshButton.addEventListener("click", loadSessions);

document.addEventListener("DOMContentLoaded", async () => {
  preloadInferenceResources();
  applyConfigToInputs();
  renderDetectionLog();
  setStatus("Loading sessions…");
  await loadSessions();
  setupLiveUpdates();
});

window.addEventListener("beforeunload", cleanupLiveUpdates);

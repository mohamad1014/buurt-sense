const statusMessage = document.getElementById("status-message");
const startButton = document.getElementById("start-session");
const stopButton = document.getElementById("stop-session");
const refreshButton = document.getElementById("refresh-sessions");
const activeSessionDetails = document.getElementById("active-session-details");
const sessionList = document.getElementById("session-list");
const captureSettingsForm = document.getElementById("capture-settings-form");
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
const CAPTURE_VIDEO_FRAME_COUNT = 16;
const CAPTURE_VIDEO_INPUT_SIZE = 112;
const INFERENCE_INITIAL_TIMEOUT_MS = 20000;
const INFERENCE_TIMEOUT_MS = 4000;

const VIDEO_MODEL_DISABLE_STORAGE_KEY = "buurt-disable-w8a16";
const VIDEO_MODEL_OPT_IN_STORAGE_KEY = "buurt-enable-w8a16";
const MODEL_WARMUP_STORAGE_KEY = "buurt-enable-model-warmup";
let inferenceWorker = null;
let inferenceWarm = false;
let inferenceWarmupPromise = null;

class InferenceWorkerClient {
  constructor(config) {
    this.config = config;
    const workerBaseUrl =
      (typeof import.meta !== "undefined" && import.meta.url) ||
      window.location.href;
    const workerUrl = new URL("./inference.worker.js", workerBaseUrl);
    this.worker = new Worker(workerUrl, { name: "buurt-inference" });
    this.messageId = 0;
    this.pending = new Map();
    this.worker.onmessage = (event) => {
      const { id, success, result, error } = event.data || {};
      if (!id) {
        return;
      }
      const resolver = this.pending.get(id);
      if (!resolver) {
        return;
      }
      this.pending.delete(id);
      if (success) {
        resolver.resolve(result);
      } else {
        const err = new Error(error?.message || "Inference worker error");
        err.stack = error?.stack;
        resolver.reject(err);
      }
    };
    this.worker.onerror = (event) => {
      const error = new Error(event?.message || "Inference worker script error");
      this.rejectAll(error);
    };
  }

  async init() {
    await this.call("init", this.config);
  }

  async processSegment(payload) {
    const transfer = [];
    if (payload.audio?.pcm instanceof Float32Array) {
      transfer.push(payload.audio.pcm.buffer);
    }
    if (payload.video?.data instanceof Float32Array) {
      transfer.push(payload.video.data.buffer);
    }
    return this.call("processSegment", payload, transfer);
  }

  async warmup() {
    return this.call("warmup", {});
  }

  async call(type, payload, transfer = []) {
    const id = `msg_${Date.now()}_${this.messageId++}`;
    const promise = new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
    });
    this.worker.postMessage({ id, type, payload }, transfer);
    return promise;
  }

  terminate() {
    this.worker.terminate();
    this.rejectAll(new Error("Inference worker terminated"));
  }

  rejectAll(error) {
    for (const { reject } of this.pending.values()) {
      reject(error);
    }
    this.pending.clear();
  }
}

async function ensureInferenceWorker() {
  if (!inferenceWorker) {
    const config = getInferenceWorkerConfig();
    const client = new InferenceWorkerClient(config);
    await client.init();
    inferenceWorker = client;
  }
  return inferenceWorker;
}

async function resetInferenceWorker() {
  if (inferenceWorker) {
    inferenceWorker.terminate();
    inferenceWorker = null;
  }
  inferenceWarm = false;
}

function getInferenceWorkerConfig() {
  const audioModelUrl =
    window.BUURT_AUDIO_MODEL_URL || "/static/tflite/YamNet_float.tflite";
  const videoModelSources = buildVideoModelSources();
  const config = {
    audioModelUrl,
    videoModelSources,
    audioModelId: "yamnet-client-tfjs",
    videoModelId: shouldAttemptW8A16()
      ? "resnet-mc-client-tflite-w8a16"
      : "resnet-mc-client-tflite",
    wasmBaseUrl: "/static/tflite",
    classMapUrl:
      window.BUURT_YAMNET_CLASS_MAP_URL ||
      "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv",
    videoClassMapUrl:
      window.BUURT_VIDEO_CLASS_MAP_URL ||
      "/static/labels/kinetics400_label_map.txt",
    videoFrameCount: CAPTURE_VIDEO_FRAME_COUNT,
    videoInputSize: CAPTURE_VIDEO_INPUT_SIZE,
  };
  return config;
}

function buildVideoModelSources() {
  const configured = window.BUURT_VIDEO_MODEL_URL;
  const sources = [];
  if (configured) {
    sources.push(configured);
  }
  if (shouldAttemptW8A16()) {
    sources.push("/static/tflite/ResNet-Mixed-Convolution_w8a16.tflite");
  }
  sources.push("/static/tflite/ResNet-Mixed-Convolution_float.tflite");
  sources.push(
    "https://huggingface.co/qualcomm/ResNet-Mixed-Convolution/resolve/main/ResNet-Mixed-Convolution_float.tflite?download=true",
  );
  const deduped = [];
  for (const url of sources) {
    if (url && !deduped.includes(url)) {
      deduped.push(url);
    }
  }
  return deduped;
}

function shouldAttemptW8A16() {
  if (window.BUURT_ENABLE_W8A16 !== undefined) {
    return Boolean(window.BUURT_ENABLE_W8A16);
  }
  if (isW8A16Disabled()) {
    return false;
  }
  try {
    return window.localStorage?.getItem(VIDEO_MODEL_OPT_IN_STORAGE_KEY) === "1";
  } catch {
    return false;
  }
}

function isW8A16Disabled() {
  try {
    return window.localStorage?.getItem(VIDEO_MODEL_DISABLE_STORAGE_KEY) === "1";
  } catch {
    return false;
  }
}

function disableW8A16ForFuture() {
  try {
    window.localStorage?.setItem(VIDEO_MODEL_DISABLE_STORAGE_KEY, "1");
  } catch {
    // ignore
  }
}

function shouldDisableW8A16AfterError(error) {
  const message = String(error?.message || "");
  return shouldAttemptW8A16() && /dlopen|dynamic linking/i.test(message);
}

async function processSegmentThroughWorker(payload) {
  let worker = await ensureInferenceWorker();
  const inferencePromise = worker.processSegment(payload);
  try {
    const timeout = inferenceWarm
      ? INFERENCE_TIMEOUT_MS
      : INFERENCE_INITIAL_TIMEOUT_MS;
    const result = await resolveWithTimeout(inferencePromise, timeout);
    inferenceWarm = true;
    return result;
  } catch (error) {
    if (shouldDisableW8A16AfterError(error)) {
      console.warn(
        "Disabling client w8a16 model after runtime failure, falling back to float weights",
        error,
      );
      disableW8A16ForFuture();
      await resetInferenceWorker();
      worker = await ensureInferenceWorker();
      return worker.processSegment(payload);
    }
    if (error?.name === "TimeoutError") {
      console.warn("Inference timed out; uploading segment without detections");
      return [];
    }
    throw error;
  }
}

function resolveWithTimeout(promise, timeoutMs) {
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
    return promise;
  }
  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(() => {
      const timeoutError = new Error("Inference timeout");
      timeoutError.name = "TimeoutError";
      promise.catch(() => {});
      reject(timeoutError);
    }, timeoutMs);
    promise
      .then((value) => {
        window.clearTimeout(timer);
        resolve(value);
      })
      .catch((error) => {
        window.clearTimeout(timer);
        reject(error);
      });
  });
}

function shouldWarmupModels() {
  if (window.BUURT_ENABLE_MODEL_WARMUP !== undefined) {
    return Boolean(window.BUURT_ENABLE_MODEL_WARMUP);
  }
  return true;
}

async function warmupInferenceModels() {
  if (inferenceWarmupPromise) {
    return inferenceWarmupPromise;
  }
  inferenceWarmupPromise = (async () => {
    try {
      const worker = await ensureInferenceWorker();
      await worker.warmup();
      inferenceWarm = true;
    } catch (error) {
      console.warn("Inference warm-up failed", error);
    } finally {
      inferenceWarmupPromise = null;
    }
  })();
  return inferenceWarmupPromise;
}

const DEFAULT_CONFIG = {
  segment_length_sec: 10,
  overlap_sec: 5,
  confidence_threshold: 0.1,
  preview_enabled: false,
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
  return {
    segment_length_sec: segment,
    overlap_sec: overlap,
    confidence_threshold: threshold,
    preview_enabled: preview,
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
    meta.textContent = `${entry.time} · ${entry.model}`;
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
    return {
      label,
      confidence,
      model: det.model_id || "client",
      time: ts.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
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
  if (!captureState.inference && captureState.context?.stream) {
    try {
      captureState.inference = await startLiveInference(
        captureState.context.stream,
        captureState.context.segmentLengthMs,
      );
    } catch (restartError) {
      console.warn("Unable to restart inference pipeline", restartError);
    }
  }

  const inference = captureState.inference;
  if (!inference) {
    console.warn("Inference pipeline missing; skipping detections");
    return [];
  }
  const segmentAudio = inference.consumeAudioSegment();
  const segmentVideo = inference.consumeVideoFrames();
  inference.resetSegment();
  if (!segmentAudio && !segmentVideo) {
    return [];
  }

  const payload = {
    audio: segmentAudio
      ? { pcm: segmentAudio.pcm, sampleRate: segmentAudio.sampleRate }
      : null,
    video: segmentVideo
      ? { data: segmentVideo.data, frameCount: segmentVideo.frameCount }
      : null,
    startTs: startTs.getTime(),
    endTs: endTs.getTime(),
  };

  try {
    const detections = await processSegmentThroughWorker(payload);
    appendDetectionsToLog(detections);
    return detections;
  } catch (error) {
    console.warn("Mobile inference unavailable, skipping detections", error);
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

function createInferenceBuffers() {
  let audioChunks = [];
  let audioSampleRate = null;
  let videoFrames = [];
  const maxVideoFrames = Math.max(CAPTURE_VIDEO_FRAME_COUNT * 3, 24);

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
        Math.max(CAPTURE_VIDEO_FRAME_COUNT, 1),
      );
      const startIndex = Math.max(videoFrames.length - frameCount, 0);
      const selected = videoFrames.slice(startIndex);
      const targetSize = Math.max(CAPTURE_VIDEO_INPUT_SIZE, 1);
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

  if (audioTrack) {
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

  if (videoTrack) {
    const targetSize = Math.max(CAPTURE_VIDEO_INPUT_SIZE, 1);
    const frameIntervalMs = Math.max(
      segmentLengthMs / Math.max(CAPTURE_VIDEO_FRAME_COUNT, 1),
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
        video.addEventListener(
          "loadedmetadata",
          () => {
            video.play().catch(() => {});
          },
          { once: true },
        );
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
      video.addEventListener(
        "loadedmetadata",
        () => {
          video.play().catch(() => {});
        },
        { once: true },
      );
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

function supportsMediaCapture() {
  return Boolean(navigator?.mediaDevices?.getUserMedia && window.MediaRecorder);
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
  const stream = await navigator.mediaDevices.getUserMedia(
    resolveMediaConstraints(),
  );
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

function resolveMediaConstraints() {
  const userAgent = navigator.userAgent?.toLowerCase() || "";
  const isMobile =
    /android|iphone|ipad|ipod/.test(userAgent) || navigator.maxTouchPoints > 1;
  const cores = navigator.hardwareConcurrency || 4;
  const lowPower = isMobile || cores <= 4;
  if (lowPower) {
    return {
      audio: true,
      video: {
        facingMode: "environment",
        width: { ideal: 640 },
        height: { ideal: 360 },
        frameRate: { max: 15 },
      },
    };
  }
  return {
    audio: true,
    video: {
      facingMode: "environment",
      width: { ideal: 1280 },
      height: { ideal: 720 },
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
  inferenceWarm = false;
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
  await resetInferenceWorker();
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
}

startButton.addEventListener("click", startSession);
stopButton.addEventListener("click", stopSession);
refreshButton.addEventListener("click", loadSessions);

document.addEventListener("DOMContentLoaded", async () => {
  if (shouldWarmupModels()) {
    warmupInferenceModels().catch((error) => {
      console.warn("Model warm-up scheduling failed", error);
    });
  }
  applyConfigToInputs();
  renderDetectionLog();
  setStatus("Loading sessions…");
  await loadSessions();
  setupLiveUpdates();
});

window.addEventListener("beforeunload", cleanupLiveUpdates);

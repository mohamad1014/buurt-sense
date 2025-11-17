const statusMessage = document.getElementById("status-message");
const startButton = document.getElementById("start-session");
const stopButton = document.getElementById("stop-session");
const refreshButton = document.getElementById("refresh-sessions");
const activeSessionDetails = document.getElementById("active-session-details");
const sessionList = document.getElementById("session-list");

let activeSession = null;
let liveWebSocket = null;
let pollingTimer = null;
let longPollController = null;
let lastRevision = null;
let shuttingDown = false;
const captureState = {
  context: null,
  uploads: [],
  orientationPermissionRequested: false,
};

const inferenceState = {
  audioModel: null,
  videoModel: null,
  runtimeReady: false,
  loading: false,
  audioModelUrl:
    window.BUURT_AUDIO_MODEL_URL || "/static/tflite/YamNet_float.tflite",
  videoModelUrl:
    window.BUURT_VIDEO_MODEL_URL ||
    "https://huggingface.co/qualcomm/Yolo-X/resolve/main/Yolo-X_float.tflite?download=true",
  audioModelId: "yamnet-client-tfjs",
  videoModelId: "yolox-client-tfjs",
  wasmBaseUrl: "/static/tflite",
  classMap: [],
  classMapUrl:
    window.BUURT_YAMNET_CLASS_MAP_URL ||
    "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv",
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
    await ensureInferenceRuntime();
    const audioDetections = await runAudioInference(blob, endTs);
    return audioDetections;
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

  return {
    started_at: now.toISOString(),
    operator_alias: "Browser Operator",
    notes: "Started from local UI",
    app_version: "web-ui",
    model_bundle_version: "demo",
    gps_origin: gpsOrigin,
    orientation_origin: orientationOrigin,
    config_snapshot: {
      segment_length_sec: 30,
      overlap_sec: 5,
      confidence_threshold: 0.1,
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
    session?.config_snapshot?.segment_length_sec ?? 30;
  const segmentLengthMs = Math.max(segmentLengthSec * 1000, 1000);
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: true,
    video: {
      facingMode: "environment",
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
  });
  const options = resolveRecorderOptions(stream);
  const recorder = new MediaRecorder(stream, options);
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
  });

  recorder.addEventListener("dataavailable", (event) => {
    if (!event.data || event.data.size === 0) {
      context.segmentStart = new Date();
      return;
    }
    const start = context.segmentStart ?? new Date();
    const end = new Date();
    context.segmentStart = end;
    const uploadPromise = uploadSegmentBlob(
      context.sessionId,
      context.segmentIndex,
      start,
      end,
      event.data,
    );
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
    stream.getTracks().forEach((track) => track.stop());
    throw error;
  }
  captureState.context = context;
}

function resolveRecorderOptions(stream) {
  if (typeof MediaRecorder === "undefined") {
    return {};
  }

  const hasVideo = stream.getVideoTracks().length > 0;
  const candidates = hasVideo
    ? [
        "video/webm;codecs=vp9,opus",
        "video/webm;codecs=vp8,opus",
        "video/webm",
        "video/mp4",
      ]
    : ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"];

  const supported = candidates.find((type) =>
    MediaRecorder.isTypeSupported?.(type),
  );

  return supported ? { mimeType: supported } : {};
}

async function stopMediaCapture() {
  const context = captureState.context;
  if (!context) {
    return;
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

  if (captureState.uploads.length) {
    const pending = [...captureState.uploads];
    captureState.uploads = [];
    await Promise.allSettled(pending);
  }
}

async function uploadSegmentBlob(sessionId, index, startTs, endTs, blob) {
  const formData = new FormData();
  formData.append("index", String(index));
  formData.append("start_ts", startTs.toISOString());
  formData.append("end_ts", endTs.toISOString());
  formData.append(
    "file",
    blob,
    `segment-${String(index).padStart(4, "0")}.webm`,
  );

  const detections = await getMobileDetections(blob, startTs, endTs, index);
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

async function ensureInferenceRuntime() {
  if (inferenceState.runtimeReady) {
    return;
  }
  if (inferenceState.loading) {
    // Wait for another caller to finish loading.
    while (inferenceState.loading) {
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
    return;
  }

  inferenceState.loading = true;
  try {
    // Load full TensorFlow.js bundle (includes CPU backend)
    if (!window.tf) {
      const tfUrls = [
        "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.20.0/dist/tf.min.js",
        "https://unpkg.com/@tensorflow/tfjs@4.20.0/dist/tf.min.js",
      ];
      await loadScriptWithFallback(tfUrls);
    }

    if (window.tf?.setBackend) {
      await window.tf.setBackend("cpu");
      await window.tf.ready();
    }

    // Load TFLite plugin from self-hosted assets to avoid CDN missing WASM glue.
    if (!window.tflite) {
      const base = inferenceState.wasmBaseUrl;
      await loadScriptWithFallback([`${base}/tf-tflite.min.js`]);
      await loadScriptWithFallback([`${base}/tflite_web_api_cc_simd.js`]);
      if (window.tflite && window.tflite.setWasmPath) {
        window.tflite.setWasmPath(`${base}/`);
      }
    }

    if (!inferenceState.classMap.length) {
      inferenceState.classMap = await loadClassMap();
    }

    inferenceState.runtimeReady = Boolean(window.tf && window.tflite);
    if (!inferenceState.runtimeReady) {
      throw new Error("Inference runtime failed to initialize. TF: " + Boolean(window.tf) + ", TFLite: " + Boolean(window.tflite));
    }
  } finally {
    inferenceState.loading = false;
  }
}

async function loadAudioModel() {
  if (inferenceState.audioModel) {
    return inferenceState.audioModel;
  }
  if (!inferenceState.runtimeReady || !window.tflite) {
    throw new Error("Inference runtime not ready");
  }
  if (!inferenceState.audioModelUrl) {
    throw new Error("Audio model URL not configured");
  }

  try {
    // Load TFLite model from URL with options
    inferenceState.audioModel = await window.tflite.loadTFLiteModel(
      inferenceState.audioModelUrl,
      {
        numThreads: Math.max(navigator.hardwareConcurrency / 2 || 1, 1),
      }
    );
    return inferenceState.audioModel;
  } catch (error) {
    console.error("Failed to load audio model from URL", error);
    // Fallback: try loading from cache
    try {
      const buffer = await fetchModelFromCache(inferenceState.audioModelUrl);
      inferenceState.audioModel = await window.tflite.loadTFLiteModel(buffer, {
        numThreads: Math.max(navigator.hardwareConcurrency / 2 || 1, 1),
      });
      return inferenceState.audioModel;
    } catch (fallbackError) {
      console.error("Failed to load audio model from cache", fallbackError);
      throw error;
    }
  }
}

function downsampleToMono(audioBuffer, targetRate = 16000) {
  const channelData = audioBuffer.getChannelData(0);
  if (audioBuffer.sampleRate === targetRate) {
    return channelData;
  }
  const ratio = audioBuffer.sampleRate / targetRate;
  const newLength = Math.floor(channelData.length / ratio);
  const result = new Float32Array(newLength);
  for (let i = 0; i < newLength; i++) {
    result[i] = channelData[Math.floor(i * ratio)];
  }
  return result;
}

async function runAudioInference(blob, endTs) {
  try {
    const model = await loadAudioModel();
    const arrayBuffer = await blob.arrayBuffer();
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const decoded = await audioCtx.decodeAudioData(arrayBuffer.slice(0));
    const mono = downsampleToMono(decoded, 16000);

    const start = performance.now();
    // Compute log-mel spectrogram patches
    const waveform = window.tf.tensor1d(mono);
    const frameLength = 400; // 25ms
    const frameStep = 160; // 10ms
    const fftLength = 512;
    const numMelBins = 64;
    const patchFrames = 96; // ~0.96s
    const patchHop = 48; // ~0.48s
    const lowerEdgeHz = 125;
    const upperEdgeHz = 7500;

    const stft = window.tf.signal.stft(
      waveform,
      frameLength,
      frameStep,
      fftLength,
      () => window.tf.signal.hannWindow(frameLength)
    );
    const magnitude = window.tf.abs(stft);
    const powerSpec = window.tf.square(magnitude);
    const melMatrix = buildMelFilterbank(
      numMelBins,
      fftLength / 2 + 1,
      16000,
      lowerEdgeHz,
      upperEdgeHz
    );
    const melSpec = window.tf.matMul(powerSpec, melMatrix, false, true);
    const logMelSpec = window.tf.log(melSpec.add(1e-6));

    // Frame into patches; pad time axis if needed to meet patchFrames length.
    const timeSteps = logMelSpec.shape[0] || 0;
    const padFrames = Math.max(patchFrames - timeSteps, 0);
    let framedInput = logMelSpec;
    let disposeFramedInput = false;
    if (padFrames > 0) {
      framedInput = window.tf.pad(logMelSpec, [
        [0, padFrames],
        [0, 0],
      ]);
      disposeFramedInput = true;
    }

    const safeHop = Math.max(patchHop, 1);
    const slices = [];
    for (let start = 0; start + patchFrames <= framedInput.shape[0]; start += safeHop) {
      slices.push(framedInput.slice([start, 0], [patchFrames, numMelBins]));
    }
    if (!slices.length) {
      if (disposeFramedInput) {
        framedInput.dispose();
      }
      waveform.dispose();
      stft.dispose();
      magnitude.dispose();
      powerSpec.dispose();
      melMatrix.dispose();
      melSpec.dispose();
      logMelSpec.dispose();
      return [];
    }

    const patches = window.tf.stack(slices); // [numPatches, patchFrames, numMelBins]
    const numPatches = patches.shape[0];
    // YamNet expects batch=1; average patches to a single patch.
    const mergedPatch = patches.mean(0); // [patchFrames, numMelBins]
    const inputTensor = mergedPatch.reshape([1, 1, patchFrames, numMelBins]);

    let output;
    try {
      output = model.predict(inputTensor);
    } finally {
      inputTensor.dispose();
      waveform.dispose();
      stft.dispose();
      magnitude.dispose();
      powerSpec.dispose();
      melMatrix.dispose();
      melSpec.dispose();
      logMelSpec.dispose();
      if (disposeFramedInput) {
        framedInput.dispose();
      }
      patches.dispose();
      mergedPatch.dispose();
      slices.forEach((s) => s.dispose());
    }

    const logits = await output.data();
    if (output.dispose) {
      output.dispose();
    }
    if (!logits || logits.length === 0) {
      return [];
    }

    // Average logits across patches, then softmax
    const numClasses = logits.length / numPatches;
    const classTotals = new Float32Array(numClasses);
    for (let i = 0; i < logits.length; i++) {
      classTotals[i % numClasses] += logits[i];
    }
    const avgLogits = Array.from(classTotals).map((v) => v / numPatches);
    const { maxIdx, maxProb } = softmaxTop1(avgLogits);

    const latencyMs = Math.max(Math.round(performance.now() - start), 1);
    const confidence = Math.max(Math.min(maxProb, 1), 0);
    const label =
      inferenceState.classMap[maxIdx] ||
      `audio_class_${maxIdx}`;

    return [
      {
        class: label,
        confidence,
        timestamp: endTs.toISOString(),
        model_id: inferenceState.audioModelId,
        inference_latency_ms: latencyMs,
        origin: "client",
      },
    ];
  } catch (error) {
    console.error("Audio inference failed", error);
    return [];
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
    const lines = text.trim().split("\n");
    const headers = lines.shift(); // drop header
    const labels = [];
    for (const line of lines) {
      const parts = line.split(",");
      // CSV format: index,mid,display_name
      if (parts.length >= 3) {
        labels.push(parts[2].replace(/(^\"|\"$)/g, ""));
      }
    }
    return labels;
  } catch (error) {
    console.warn("Failed to load class map", error);
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
  setStatus("Loading sessions…");
  await loadSessions();
  setupLiveUpdates();
});

window.addEventListener("beforeunload", cleanupLiveUpdates);

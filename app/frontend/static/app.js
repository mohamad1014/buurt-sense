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
    const session = await fetchJson("/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildSessionPayload()),
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

function buildSessionPayload() {
  const now = new Date().toISOString();
  return {
    started_at: now,
    operator_alias: "Browser Operator",
    notes: "Started from local UI",
    app_version: "web-ui",
    model_bundle_version: "demo",
    gps_origin: {
      lat: 52.3676,
      lon: 4.9041,
      accuracy_m: 5,
      captured_at: now,
    },
    orientation_origin: {
      heading_deg: 0,
      captured_at: now,
    },
    config_snapshot: {
      segment_length_sec: 30,
      overlap_sec: 5,
      confidence_threshold: 0.6,
    },
    detection_summary: {
      total_detections: 0,
      by_class: {},
    },
    redact_location: false,
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
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const options = {};
  if (
    typeof MediaRecorder !== "undefined" &&
    MediaRecorder.isTypeSupported?.("audio/webm;codecs=opus")
  ) {
    options.mimeType = "audio/webm;codecs=opus";
  }
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

  recorder.start(segmentLengthMs);
  captureState.context = context;
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

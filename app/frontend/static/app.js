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
    const session = await fetchJson("/sessions", { method: "POST" });
    activeSession = session;
    renderActiveSession(session);
    toggleButtons(true);
    setStatus("Session started. Recording in progress.", "success");
    await loadSessions({ announce: false });
  } catch (error) {
    console.error("Failed to start session", error);
    setStatus(`Unable to start session: ${error.message}`, "error");
    startButton.disabled = false;
  }
}

async function stopSession() {
  if (!activeSession) {
    setStatus("No active session to stop.", "error");
    return;
  }

  try {
    setStatus("Stopping session…");
    stopButton.disabled = true;
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

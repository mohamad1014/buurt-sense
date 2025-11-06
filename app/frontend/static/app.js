const statusMessage = document.getElementById("status-message");
const startButton = document.getElementById("start-session");
const stopButton = document.getElementById("stop-session");
const refreshButton = document.getElementById("refresh-sessions");
const activeSessionDetails = document.getElementById("active-session-details");
const sessionList = document.getElementById("session-list");

let activeSession = null;

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

async function loadSessions() {
  try {
    const sessions = await fetchJson("/sessions");
    renderSessions(sessions.sort((a, b) => (a.started_at < b.started_at ? 1 : -1)));
    if (activeSession) {
      const latest = sessions.find((session) => session.id === activeSession.id);
      if (latest) {
        activeSession = latest;
        renderActiveSession(activeSession);
        toggleButtons(!Boolean(activeSession.ended_at));
      }
    }
    setStatus("Session list updated.");
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
    await loadSessions();
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
    await loadSessions();
  } catch (error) {
    console.error("Failed to stop session", error);
    setStatus(`Unable to stop session: ${error.message}`, "error");
    stopButton.disabled = false;
  }
}

startButton.addEventListener("click", startSession);
stopButton.addEventListener("click", stopSession);
refreshButton.addEventListener("click", loadSessions);

document.addEventListener("DOMContentLoaded", async () => {
  setStatus("Loading sessions…");
  await loadSessions();
});

# Buurt Sense

AI-powered neighbourhood awareness platform. Empowers communities to record and analyze surroundings in real time — detecting incidents like fights, robberies, fireworks, gunshots, screams, glass breaks, and vehicle crashes through lightweight on-device intelligence with instant local alerts.

## Current Code Status

- ✅ **FastAPI backend skeleton** with endpoints to create, stop, retrieve, and list recording sessions backed by an in-memory store (`app/main.py`, `app/storage.py`).
- ✅ **Browser control panel** to start/stop sessions and view history, served at the root path (`app/frontend`).
- ✅ **Pydantic session model** that enforces timezone-aware timestamps (`app/models.py`).
- ✅ **Comprehensive pytest suite** covering the session lifecycle and error handling (`tests/test_sessions.py`).
- ⚠️ **No production persistence or frontend yet**—all data is lost when the process stops, and the roadmap items below remain planned work.

## Running the Local API

The FastAPI service is created via an application factory called `create_app`. When running the development server, make sure to reference the callable instead of a module-level `app` (which is why `uvicorn app.main:app` fails).

```bash
uv run uvicorn --factory app.main:create_app --reload --host 0.0.0.0 --port 8000
```

### Interactive Docs

Once the server is running, visit `http://localhost:8000/` for the recording control panel or `http://localhost:8000/docs` for the automatically generated Swagger UI.

### Browser Session Defaults

The "Start session" button in the bundled UI now sends a complete JSON payload with placeholder metadata (demo operator alias, Amsterdam GPS, and default config snapshot). This keeps the UX friction-free while still satisfying the backend schema requirements. If you want the button to emit different defaults, edit `buildSessionPayload()` in `app/frontend/static/app.js` (for example to plug in real device info or GPS coordinates) or replace the UI with your own client that submits customized payloads.

When the browser performs the actual media capture, the payload also includes `skip_backend_capture: true`. This flag tells the server to disable its synthetic capture backend for that session so your uploads do not collide with the demo generator. Remove it (or set to `false`) if you want the backend to fabricate placeholder segments and detections automatically.

### Recording Backend Configuration

The default recording backend now captures media segments continuously while sessions are active and stores them on disk. Configure its behaviour with the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `BUURT_CAPTURE_ROOT` | Root directory where segment files are written. | `recordings` (relative to the working directory) |
| `BUURT_SEGMENT_LENGTH_SEC` | Target duration, in seconds, for each generated segment. | `5.0` |
| `BUURT_SEGMENT_BYTES_PER_SEC` | Approximate bytes written per second of captured media (controls file size). | `32000` |

Segments are streamed into `<BUURT_CAPTURE_ROOT>/<session-id>/segment-XXXX.pcm` and detections are emitted in real time while the session is running.

## Running Tests

Install dependencies with [uv](https://docs.astral.sh/uv/) and execute the pytest suite to validate the API behaviour:

```bash
# Install runtime + development dependencies
uv sync --extra dev

# Run all tests
uv run --extra dev pytest
```

`uv sync --extra dev` creates a project-local virtual environment automatically (default: `.venv`) with both runtime and development dependencies. You do not need to activate it manually; `uv run` handles environment resolution for each command.

## Database Migrations

The project uses [Alembic](https://alembic.sqlalchemy.org/) for schema management. Set `BUURTSENSE_DB_URL` to point at your database (defaults to the SQLite file `sqlite+aiosqlite:///./buurtsense.db`). Run migrations with either the helper script or Alembic CLI:

```bash
# Apply all pending migrations using the helper script
uv run python scripts/init_db.py

# Or run Alembic directly
uv run alembic upgrade head
```

You can inspect the generated SQL with `uv run alembic history` and create new revisions via `uv run alembic revision -m "message"` once your models change.

### Test Coverage

The test suite covers:
- ✅ Frontend serving (HTML and static assets)
- ✅ Session lifecycle and storage
- ✅ API endpoints for session management
- ✅ Error handling for unknown/invalid sessions


## 1. Mission & Vision
Buurt Sense is a civic and community-focused safety tool for Dutch neighbourhoods (initial launch scope: Netherlands). It helps residents, watch volunteers, and local authorities capture contextual evidence and increase situational awareness in areas where traditional reporting is slow or incomplete.

## 2. Problem
- Unsafe areas often lack timely, trusted, contextual evidence.
- Existing reporting channels (e.g., manual phone calls) are reactive and fragmented.
- High-friction data collection discourages continuous monitoring.

## 3. MVP Scope (Phase 0)
Focus: Local (on-device) inference, continuous recording, offline-first storage, basic detection feedback to the user. No cloud sync, no authentication, no background recording yet.

Included:
- Mobile web interface (PWA target) to start/stop a recording session.
- Continuous audio + video capture segmented into reasonable chunks (configurable from UI; default TBD).
- On-device inference using lightweight pre-trained models (video action recognition, audio event classification, YOLO-like object/action detectors when feasible).
- Local storage of segments and detection metadata.
- Real-time display of detected incident classes + confidence.
- GPS coordinates + orientation (street-facing direction) captured per segment.
- Simple local alert (visual UI indicator) when a detection surpasses threshold.

Excluded (Future Phases):
- Cloud upload & centralized dashboards.
- Multi-user alert distribution.
- Authentication & roles.
- Background / minimized recording.
- Advanced false positive mitigation pipelines.
- Model retraining and continuous improvement loop.

## 4. Core Personas
- Resident user (records and views detections).
- Neighbourhood watch volunteer (future: aggregate + share).
- Local authority liaison (future: dashboard consumption).

## 5. Key Features (MVP)
| Category | Feature | Status | Notes |
|----------|---------|--------|-------|
| Capture | Start/stop session | Implemented | FastAPI endpoints plus the browser control panel handle create/stop flows (`app/main.py`, `app/frontend/static/app.js`). |
| Capture | Continuous AV segmentation | Partial (browser audio) | The control panel now streams live microphone audio via `MediaRecorder` and uploads segments to `/sessions/{id}/segments/upload`; the backend synthesiser remains as a fallback when capture is unavailable. |
| Sensors | GPS + orientation capture | Partial (browser APIs) | Control panel now requests geolocation + device orientation on session start and falls back to Amsterdam defaults when permissions are denied. |
| Inference | Local audio event classification | Stubbed | Backend fabricates a single `ambient_noise` detection per segment without model execution. |
| Inference | Local video action recognition (lightweight) | Not implemented | No video processing or model loading logic exists. |
| Detection | Fight / Robbery / Fireworks / Gunshot / Scream / Glass Break / Vehicle Crash | Not implemented | Only the placeholder `ambient_noise` class is produced; planned incident classes are absent in API/UI. |
| UX | Real-time detection overlay + confidence | Not implemented | Control panel shows session metadata only and lacks detection visualisation. |
| Storage | Local file + metadata store (IndexedDB / File System Access API) | Not implemented on client | Persistence currently happens server-side (SQLite + `recordings/`), with no browser-side IndexedDB/File System Access usage. |
| Alerts | Local UI alert only | Not implemented | No detection-triggered notifications beyond generic status text. |
| Config | Segment length / confidence threshold UI controls | Not implemented | Payload uses fixed defaults in `buildSessionPayload`; there are no configurable inputs. |
| Offline | Full offline recording & later sync (future) | Deferred | Still a later-phase deliverable once client-side capture exists. |

## 6. Architecture (MVP Concept)
```
┌──────────────┐    start/stop    ┌────────────────┐
│  Web Client  │ ───────────────▶ │  Capture Layer │
│ (PWA React?) │ ◀─────────────── │  (MediaStream) │
└─────┬────────┘                  └──────┬─────────┘
			│ video/audio frames               │ sensor (GPS, orientation)
			▼                                   ▼
┌─────────────┐  frames      ┌──────────────────┐
│ Inference   │────────────▶ │ Detection Router │
│ (WASM/Py?)  │ confidences  └─────────┬────────┘
└────┬────────┘                        │
		 │ metadata                        │
		 ▼                                 ▼
┌──────────────┐   persist    ┌────────────────┐
│ Local Store  │◀──────────── │  UI Overlay    │
│ (IndexedDB)  │────────────▶ │  (Status/Log)  │
└──────────────┘   query      └────────────────┘
```
Notes:
- Preference for a single language: Python. (Open question: front-end build system & integration pattern.)
- Option A: Pure web client + WASM-compiled models (if Python avoided client-side).
- Option B: Local lightweight Python backend (FastAPI) serving inference endpoints; client streams frames.
TBD after tech stack decision.

## 7. Technical Architecture (MVP)

### Core Stack
- **Frontend**: React PWA with offline capabilities
- **Local Processing**: Python FastAPI service for inference
- **Models**: Hybrid approach - lightweight local + cloud validation
- **Storage**: IndexedDB (metadata) + Local File System (media)
- **Target Performance**: Real-time processing on mid-tier mobile devices (2GB+ RAM)

### Durable Session Storage
- **Engine**: Async SQLite via SQLAlchemy 2.0 (`sqlite+aiosqlite`)
- **Pragmas**: WAL + foreign-key enforcement for resilience and concurrency
- **Tables**: `recording_sessions`, `segments`, `detections`
- **Access Layer**: `SessionStorage` facade for session lifecycle, segments, and detections
- **Initialization**: `uv run python scripts/init_db.py` (applies Alembic migrations)
- **Configuration**: `BUURTSENSE_DB_URL` points at the runtime database (default: `sqlite+aiosqlite:///./buurtsense.db`)

### Model Strategy (Revised)
- **Primary Detection**: Lightweight local models for initial screening
- **Validation**: Cloud-based models for complex incident classification
- **Fallback**: Full local processing when offline

## 8. Data Flow (MVP)
1. User opens React PWA and presses Start.
2. MediaStream captures audio/video; segments of 30s length with 5s overlap for contextual continuity.
3. Fixed sampling: every 2nd video frame + sliding 1000ms audio windows sent to TFLite inference.
4. Detections above threshold logged with timestamp, GPS, orientation.
5. Segment media + metadata persisted locally (IndexedDB + filesystem).
6. UI overlay updates live classifications until user stops session.

## 9. AI / ML Strategy (MVP)
- All inference local; cloud reserved for heavier composite models later.
- Pre-trained models only; no fine-tuning during MVP.
- Confidence threshold (global, adjustable) (TBD default, e.g., 0.6).
- False positive mitigation (Phase 1): simple cooldown per class (e.g., ignore repeat of same class within X seconds).
- Future: dynamic thresholding, ensemble voting, model calibration.

## 10. Configuration (Initial Set)
| Setting | Description | Default |
|---------|-------------|---------|
| segment_length_sec | Duration of each persisted media chunk | 30 |
| segment_overlap_sec | Overlap between consecutive segments | 5 |
| frame_sampling_interval | Process every Nth frame (N) | 2 |
| audio_window_ms | Sliding window for audio classification | 1000 |
| confidence_threshold | Minimum confidence to surface detection (global) | 0.60 |
| class_cooldown_sec | Minimum time between identical alerts | 30 |
| video_resolution | Target processed resolution | 640x480 |
| audio_sample_rate | Target processed audio sample rate | 16000 |

## 11. Privacy & Safety (MVP Position)
- Data retained only locally (no automatic upload).
- User-initiated sessions; explicit Start/Stop clarity.
- Visual recording indicator (UI banner / pulsing icon) (to implement).
- Orientation + GPS collected strictly for contextual incident mapping.
- No anonymization yet (faces/voices still present) — flagged as Phase 1 upgrade.
- Regulatory scope: anticipate GDPR; actual compliance measures (DPIA, consent flows) deferred.
Outstanding Questions:
- Local encryption of stored segments? (At rest strategy TBD.)
- Retention auto-delete policy (e.g., purge after X days)?

## 12. Alerting (MVP)
- Local UI notifications only.
- No push/SMS/email yet.
- Latency target: Detection surfaced within a few seconds of occurrence.

## 13. Session & Data Model (Draft)
Entities (draft):
```
User (future) {
	id, role, created_at
}
RecordingSession {
	id, started_at, ended_at?, device_info, gps_origin, orientation_origin
}
Segment {
	id, session_id, index, start_ts, end_ts, file_uri, gps_trace[], orientation_trace[]
}
Detection {
	id, segment_id, class, confidence, timestamp, gps_point, orientation
}
ConfigProfile (future) {
	id, user_id, thresholds, segment_length_sec, cooldowns
}
```

## 14. Roadmap (High-Level)
Phase 0 (MVP): Local capture + inference + storage + UI overlay.
Phase 1: Basic auth, background recording, privacy features (face blur, voice dampening), false positive improvements.
Phase 2: Cloud sync, shared neighbourhood dashboards, alert distribution to trusted circle.
Phase 3: Advanced analytics (heatmaps, temporal trends, predictive modeling), model feedback & retraining pipeline.

### Near-Term Backend Roadmap
- **Imminent validation**: flesh out backend unit and integration tests that cover session lifecycle endpoints and health reporting.
- **Upcoming refactor**: extract a storage abstraction layer to support swapping local development storage for cloud backends without touching route logic.
- **Future persistence**: design durable session archival (object storage + metadata DB) once on-device retention limits are defined.

## 15. Installation (Draft Placeholder)
Prerequisites:
- Python >= 3.11 (if choosing Python backend route)
- Node.js >= 20 (if choosing React/Svelte front-end)

Setup (TBD pending architecture decision):
```
# Clone
git clone https://github.com/mohamad1014/buurt-sense.git
cd buurt-sense

# Install backend dependencies with uv
uv sync

# Run the minimal FastAPI service
uv run uvicorn backend.main:app --reload

# (Option B) Frontend
npm install  # after package.json added
```

## 16. Backend Quickstart
Follow these steps to get the FastAPI backend running locally.

### 1. Install Dependencies
```bash
uv sync --extra dev
```

### 2. Launch the Development Server
```bash
uv run uvicorn --factory app.main:create_app --reload --host 0.0.0.0 --port 8000
```

### 3. Exercise the API
Use HTTPie or curl to interact with core endpoints once the server is running.

#### Health Check (`GET /health`)
```bash
http GET :8000/health
# or
curl -X GET http://localhost:8000/health
```

#### Start a Session (`POST /sessions`)
```bash
http POST :8000/sessions device_id="demo-device" gps_origin:='{"lat":52.36,"lon":4.88}'
# or
curl -X POST http://localhost:8000/sessions \
  -H "Content-Type: application/json" \
  -d '{"device_id":"demo-device","gps_origin":{"lat":52.36,"lon":4.88}}'
```

#### Stop a Session (`POST /sessions/{session_id}/stop`)
```bash
http POST :8000/sessions/{session_id}/stop
# or
curl -X POST http://localhost:8000/sessions/{session_id}/stop
```

#### Retrieve Session Data (`GET /sessions/{session_id}`)
```bash
http GET :8000/sessions/{session_id}
# or
curl -X GET http://localhost:8000/sessions/{session_id}
```

## 17. Development Tasks (Next)
- Implement Python FastAPI inference service skeleton.
- Integrate React PWA media capture (video+audio) + segmentation (30s w/5s overlap).
- Frame & audio sampling pipeline (every 2nd frame, sliding audio window).
- TFLite model loading (YOLOv8n tiny + YAMNet + action classifier placeholder).
- Detection overlay UI with confidence & cooldown handling.
- Local persistence (IndexedDB + file storage) of segments & metadata.
- Configuration panel (segment length, overlap, confidence threshold, cooldown).
- GPS + orientation capture integration.
- Logging format for Detection records (JSON schema).

## 18. Testing Strategy (Planned)
- Unit: segmentation logic, threshold filtering.
- Integration: end-to-end mock session -> stored detections.
- Model sanity: sample clips produce expected class probabilities.
- Future: confusion matrix generation, performance benchmarks on low-end devices.

### Running Tests

```bash
# Ensure dependencies are installed (runtime + dev)
uv sync --extra dev

# Run all tests
uv run --extra dev pytest

# Run tests with verbose output
uv run --extra dev pytest -v

# Run specific test file
uv run --extra dev pytest tests/test_frontend.py
```

The suite exercises the FastAPI session endpoints end-to-end, covering happy-path start/stop flows and error scenarios such as
unknown or double-stopped sessions. It also validates frontend serving and static asset delivery.

## 19. Code Style & Tooling

Python source in this repository is formatted with [Black](https://black.readthedocs.io/) and linted with [Ruff](https://docs.astral.sh/ruff/). Run them locally before opening a pull request:

```
uv run --extra dev black backend tests
uv run --extra dev ruff check backend tests
```

Convenience targets are available via `make` and automatically execute through `uv`:

```
make format
make lint
```

## 20. Contribution
Not open for external contributions at MVP; will define guidelines (linting, conventional commits, test coverage) later.

## 21. License
MIT License (see `LICENSE`).

## 22. Outstanding Questions (Need Clarification)
1. Max adjustable segment length range (current default 30s; propose 10–60s?) & overlap configurability.
2. Local encryption of stored segments (implement now or defer Phase 1?).
3. Retention policy (none vs purge after X days vs user selectable?).
4. Per-class confidence thresholds (keep global 0.60 or e.g., gunshot=0.75?).
5. Orientation acquisition method (compass only vs fused sensors?).
6. Tagline final wording (any tweak or keep current?).
7. False positive feedback UX (dismiss button logging vs future report flow?).
8. File naming convention confirm: session-{id}/segment-{index}.mp4 & segment-{index}.json.
9. Future distribution: remain simple Python or add Docker earlier?
10. Minimum device spec baseline (proposed mid-tier Android 2022+, 2GB RAM) – confirm.

## 23. Disclaimer (Draft)
Buurt Sense is an assistive awareness tool and does not replace contacting emergency services. Accuracy of detections is probabilistic; users should exercise judgment.

---
Please provide answers or preferences for the Outstanding Questions so the README can be finalized.

## 22. Local API Prototype

An initial FastAPI service lives in `app/` and persists data to a local SQLite database (`buurt_sense.db`). Install dependencies and run the server locally:

```
uv sync
uv run uvicorn --factory app.main:create_app --reload
```

### Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /sessions` | Create a recording session. Derives timezone from the provided origin and enforces metadata validation (segment length 10–60s, overlap below segment length, operator alias ≤64 chars, notes ≤500 chars). Defaults to a pseudonymous device UUID when missing. |
| `GET /sessions` | List sessions with rounded origin coordinates, config snapshot, and detection summary counts. Includes a `status` flag (`active` / `completed`). |
| `GET /sessions/{id}` | Retrieve full session metadata including segments. Use `expand=traces` to include GPS & orientation traces and `include=full_detections` to inline detections. |
| `POST /sessions/{id}/segments` | Attach a media segment to a session along with traces, checksum, and sizing info. |
| `POST /sessions/{id}/segments/upload` | Upload an audio/video blob directly from the browser; the backend stores it under the capture root and persists the derived metadata. |
| `GET /sessions/{id}/detections` | Paginate detections associated with a session. |
| `POST /segments/{id}/detections` | Add a detection and automatically refresh the session’s aggregated detection summary (totals, per-class counts, first/last timestamps, highest-confidence detection). |

### Stored Metadata Highlights

- Sessions persist structured `gps_origin`, `orientation_origin`, `config_snapshot`, `detection_summary`, and the `redact_location` toggle.
- Segments store simplified traces, frame/audio metrics, and file integrity metadata (size + checksum).
- Detections capture model provenance, latency, GPS/orientation context, and update session aggregates immediately on insert.

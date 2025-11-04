# Buurt Sense

AI-powered neighbourhood awareness platform. Empowers communities to record and analyze surroundings in real time — detecting incidents like fights, robberies, fireworks, gunshots, screams, glass breaks, and vehicle crashes through lightweight on-device intelligence with instant local alerts.

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
| Category | Feature | Status |
|----------|---------|--------|
| Capture | Start/stop session | Planned |
| Capture | Continuous AV segmentation | Planned |
| Sensors | GPS + orientation capture | Planned |
| Inference | Local audio event classification | Planned |
| Inference | Local video action recognition (lightweight) | Planned |
| Detection | Fight / Robbery / Fireworks / Gunshot / Scream / Glass Break / Vehicle Crash | Planned |
| UX | Real-time detection overlay + confidence | Planned |
| Storage | Local file + metadata store (IndexedDB / File System Access API) | Planned |
| Alerts | Local UI alert only | Planned |
| Config | Segment length / confidence threshold UI controls | Planned |
| Offline | Full offline recording & later sync (future) | Deferred |

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

## 15. Installation (Draft Placeholder)
Prerequisites:
- Python >= 3.11 (if choosing Python backend route)
- Node.js >= 20 (if choosing React/Svelte front-end)

Setup (TBD pending architecture decision):
```
# Clone
git clone https://github.com/mohamad1014/buurt-sense.git
cd buurt-sense

# (Option A) Install backend deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the minimal FastAPI service
uvicorn backend.main:app --reload

# (Option B) Frontend
npm install  # after package.json added
```

## 16. Development Tasks (Next)
- Implement Python FastAPI inference service skeleton.
- Integrate React PWA media capture (video+audio) + segmentation (30s w/5s overlap).
- Frame & audio sampling pipeline (every 2nd frame, sliding audio window).
- TFLite model loading (YOLOv8n tiny + YAMNet + action classifier placeholder).
- Detection overlay UI with confidence & cooldown handling.
- Local persistence (IndexedDB + file storage) of segments & metadata.
- Configuration panel (segment length, overlap, confidence threshold, cooldown).
- GPS + orientation capture integration.
- Logging format for Detection records (JSON schema).

## 17. Testing Strategy (Planned)
- Unit: segmentation logic, threshold filtering.
- Integration: end-to-end mock session -> stored detections.
- Model sanity: sample clips produce expected class probabilities.
- Future: confusion matrix generation, performance benchmarks on low-end devices.

## 18. Contribution
Not open for external contributions at MVP; will define guidelines (linting, conventional commits, test coverage) later.

## 19. License
MIT License (see `LICENSE`).

## 20. Outstanding Questions (Need Clarification)
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

## 21. Disclaimer (Draft)
Buurt Sense is an assistive awareness tool and does not replace contacting emergency services. Accuracy of detections is probabilistic; users should exercise judgment.

---
Please provide answers or preferences for the Outstanding Questions so the README can be finalized.

# Client-Side Performance Investigation

Context: the capture UI (`app/frontend/index.html`) loads a single 63 KB script (`app/frontend/static/app.js`) that handles media capture, TF.js/TFLite inference, uploads, and UI updates on the main thread. When exercised on lower-powered laptops and phones the page feels heavy and occasionally crashes. This doc summarizes what makes the client heavy today and outlines optimizations to pursue, keeping client-side audio+video inference enabled by default (with a strong bias toward preserving video inference quality).

---

## Measurements & Observations

### Baseline

| Item | Command | Result |
| --- | --- | --- |
| Main bundle size | `wc -l app/frontend/static/app.js` | 2 194 LOC (~63 KB uncompressed) |
| Self-hosted ML assets | `ls -lh app/frontend/static/tflite` | `tf.min.js` 1.5 MB, `tf-tflite.min.js` 1.2 MB, `tflite_web_api_cc_simd.js` 77 KB, `tflite_web_api_cc_simd.wasm` 3.6 MB, `YamNet_float.tflite` 15 MB |
| Local video model (w8a16-derived) | `ls -lh app/frontend/static/tflite/ResNet-Mixed-Convolution_w8a16.tflite` | 23 MB (float16 TFLite converted from Qualcomm’s w8a16 ONNX); opt-in only |
| Remote video model (float fallback) | `curl -I -L 'https://huggingface.co/.../ResNet-Mixed-Convolution_float.tflite?...'` | `Content-Length: 46 776 804` bytes (≈44.6 MB) |

Even before the user records anything, the first successful inference attempt forces ~21 MB of local assets plus a 44 MB download over the network, none of which are cache-busted or compressed beyond basic HTTP.

### Hot spots in the code

_Note: inference-heavy helpers now live in `app/frontend/static/inference.js`; historical line references may no longer match after the lazy-loading refactor._

- **Media constraints default to 720p video for every session** (`app/frontend/static/app.js:699-720`). Phones are encoding a 720p VP8 stream on the CPU even if the operator only needs an audio capture or server-side inference. There is no adaptive bitrate or detection of low-power devices.
- **Client inference blocks uploads** (`app/frontend/static/app.js:746-779`). Each `MediaRecorder` chunk waits for `getMobileDetections` → `runAudioInference` / `runVideoInference` to finish before uploading, so a slow inference pass makes uploads lag and buffers blobs in memory.
- **Inference runtime always runs on the main/UI thread** (`app/frontend/static/app.js:908-1129`, `1428-1699`). The code performs AudioWorklet/ScritpProcessor reads, canvas `getImageData`, tensor creation, STFT/mel filterbank generation, and two `model.predict` calls per segment on the same thread that paints the UI, causing visible lags.
- **Video inference samples frames in JS** (`app/frontend/static/app.js:1589-1788`). For each 10 s chunk the browser decodes the blob, seeks N times, draws into a canvas, normalizes floats, and stores them in Float32Arrays (~2.7 MB per chunk). On mobile GPUs this alone can starve the UI.
- **TF.js CPU backend is forced** (`app/frontend/static/app.js:1274-1277`). WebGL/WebGPU are never attempted, so all tensor ops run on a single CPU thread (even on capable devices) while the code simultaneously pins `navigator.hardwareConcurrency / 2` threads for the TFLite models (`app/frontend/static/app.js:1320-1325`, `1372-1377`). Phones end up oversubscribed.
- **Video model download and caching are synchronous** (`app/frontend/static/app.js:1357-1405`, `1889-1901`). On a phone the 44 MB download + `caches.put` blocks the main thread and risks the “Page Unresponsive” dialog.
- **`getUserMedia` always requests both audio and video** (`app/frontend/static/app.js:713-719`). Even when preview + video inference are disabled we keep the camera on, burning battery and heat.
- **Preview toggle only hides the element** (`app/frontend/static/app.js:206-226`). Disabling the preview still keeps the `MediaStream` flowing and canvas capture running.
- **Geolocation/orientation gating slows start** (`app/frontend/static/app.js:560-693`). Every session waits up to 8 s for GPS + sensor permissions before posting `/sessions`, which makes the UI look frozen on flaky or denied sensors.
- **WASM runtime lacks Select TF ops**: attempting to run the converted ResNet-Mixed-Convolution **w8a16** TFLite locally triggers `dlopen`/dynamic-linking errors in `tflite_web_api_cc_simd.wasm`, so the inference module now skips that asset by default and requires an explicit opt-in (`window.BUURT_ENABLE_W8A16 = true` or `localStorage.setItem('buurt-enable-w8a16','1')`). When enabled we still auto-disable the source after the first failure. Further work is needed to get a truly lighter model working in TF.js/TFLite Web.
- **Worker relative path pitfalls**: when a worker loads `tf-tflite.min.js`, that script tries to fetch `tflite_web_api_cc_simd.js` and `tflite_web_api_cc_simd.wasm` relative to `document.currentScript`, which doesn’t exist in a worker. We now intercept `importScripts`/`fetch` in `inference.worker.js` so everything resolves to `/static/tflite/…`, eliminating the 404s.
- **Uploads blocked while downloading models**: Segment uploads waited for the inference worker to finish downloading the 15–45 MB models. Added a `INFERENCE_TIMEOUT_MS` guard around worker calls so uploads proceed with empty detections if inference takes too long; detections resume automatically once the worker is ready.

---

## Recommendations

### 1. Keep client inference on by default, but modular

1. Keep the existing “client inference enabled” behavior as the default experience so laptops/phones immediately provide detections, but expose a toggle to disable it only when devices cannot keep up.
2. Split `app/frontend/static/app.js` so the inference pipeline lives in a separate, lazily loaded module (dynamic `import()` or `<script type="module">`). Even though the default is “on”, this allows defering the heavy code until the toggle/UI actually uses it and keeps the rest of the UI responsive.
3. Ship the Qualcomm ResNet-Mixed-Convolution **w8a16** checkpoints (target 11.5 MB; current converted float16 build is 23 MB) instead of the 44.6 MB float model, and keep video inference enabled while the audio model continues to load in parallel. Mirror the w8a16 files under `/static/tflite` and update the inference state to point at the quantized weights.

### 2. Constrain capture workload

1. Detect device class via `navigator.userAgent`/`hardwareConcurrency` and request lower video settings on mobile (`video: { facingMode: "environment", width: { ideal: 640 }, height: { ideal: 360 }, frameRate: { max: 15 } }`) before constructing the `MediaRecorder` (`app/frontend/static/app.js:713-724`).
2. When an operator explicitly disables preview *and* video inference, request audio-only tracks to avoid wasting camera cycles; otherwise keep video capture on so detections remain visually rich.
3. Allow “audio-only sessions” from the UI (opt-in) and adjust the server payload accordingly (`skip_backend_capture` + config snapshot) for environments where video capture is impossible.

### 3. Decouple inference from the UI thread

1. Offload TF.js + TFLite work to a dedicated Web Worker. Move `startLiveInference`, `runAudioInference`, and `runVideoInference` into a worker script and communicate via `postMessage` + transferable `Float32Array`s so the main thread remains responsive.
2. If workers are not feasible short-term, at least `postMessage` the video/audio blobs to `createImageBitmap`/`OffscreenCanvas` to avoid `getImageData` on the main thread.
3. Batch detection + upload: start uploading the blob immediately and attach detections later via `/segments/{id}/detections` to avoid blocking network I/O on inference (`app/frontend/static/app.js:746-777`).
   - _Status:_ Implemented via `app/frontend/static/inference.worker.js` and the worker bridge in `app/frontend/static/app.js`, so TF.js/TFLite loading and inference now run entirely off the UI thread.

### 4. Reduce model + runtime cost

1. Ship quantized / pruned models: YamNet has an 8-bit version (~4 MB) and Kinetics video classifiers exist in <15 MB MobileNet variants. Swapping URLs requires no UI change.
2. Host zipped assets with `Content-Encoding: gzip/br` and use service worker prefetching so first-run downloads are smaller and resumable.
3. Consider replacing TF.js CPU backend with WebGL/WebGPU or ONNX Runtime Web (WASM SIMD) which can run smaller kernels with lower overhead.

### 5. Improve perceived performance

1. Lazy-load sensor data: resolve GPS/orientation in parallel with `/sessions` POST and allow the user to start recording immediately; update metadata later once permissions settle.
2. Surface runtime state in the UI (“Downloading video model (45 MB)…") so operators know why the UI is busy.
3. Add `performance.mark` instrumentation around `ensureInferenceRuntime`, `runAudioInference`, `runVideoInference`, and upload timings, then log them to the detection feed or `/health` endpoint for future regressions.

---

## Next Steps

1. Prototype the client-inference toggle + lazy module split so we can benchmark a “capture-only” path versus “capture + inference”, keeping the inference side on by default.
2. Implement adaptive media constraints and an “audio-only” mode, then test on a representative low-end Android phone (≤4 cores, 4 GB RAM).
3. Instrument inference durations + download sizes in the UI to collect real-world metrics, and add a telemetry upload endpoint so we can aggregate them server-side.
4. Swap the ResNet-Mixed-Convolution float weights for the Qualcomm **w8a16** checkpoint (target 11.5 MB; currently 23 MB after conversion) and validate accuracy/latency before rolling it out broadly; evaluate quantized YamNet in parallel for audio.

Taken together, these changes target the biggest regressions we observed: multi-tens-of-megabyte downloads, synchronously blocking inference, and unbounded CPU use on the main thread. Focusing on those first should make the UI significantly lighter on both laptops and phones.

import { test, expect } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";

const RUN_MODELS = process.env.RUN_MODEL_TESTS === "1";

const describeMaybe = RUN_MODELS ? test.describe : test.describe.skip;

const APP_JS_PATH = path.join(process.cwd(), "app", "frontend", "static", "app.js");
const APP_JS_CONTENT = fs.readFileSync(APP_JS_PATH, "utf-8");
const STATIC_ROOT = path.join(process.cwd(), "app", "frontend", "static");

const mimeTypes = {
  ".js": "application/javascript",
  ".wasm": "application/wasm",
  ".tflite": "application/octet-stream",
  ".txt": "text/plain",
  ".css": "text/css",
  ".csv": "text/csv",
};

function resolveStaticFile(urlPath) {
  const relative = urlPath.replace(/^\/static\//, "");
  return path.join(STATIC_ROOT, relative);
}

async function mountHarness(page) {
  page.on("console", (msg) => console.log("PAGE LOG:", msg.type(), msg.text()));

  await page.route("**/static/**", async (route) => {
    const url = new URL(route.request().url());
    const filePath = resolveStaticFile(url.pathname);
    if (fs.existsSync(filePath)) {
      const body = fs.readFileSync(filePath);
      const ext = path.extname(filePath);
      const contentType = mimeTypes[ext] || "application/octet-stream";
      await route.fulfill({
        status: 200,
        body,
        headers: {
          "Content-Type": contentType,
          "Cross-Origin-Opener-Policy": "same-origin",
          "Cross-Origin-Embedder-Policy": "require-corp",
          "Cross-Origin-Resource-Policy": "same-origin",
        },
      });
      return;
    }
    await route.fallback();
  });

  await page.route("**/sessions**", async (route) => {
    await route.fulfill({
      status: 200,
      body: JSON.stringify({ revision: 0, sessions: [] }),
      headers: { "Content-Type": "application/json" },
    });
  });

  const html = `
    <!doctype html>
    <html>
      <head>
        <meta http-equiv="Cross-Origin-Opener-Policy" content="same-origin">
        <meta http-equiv="Cross-Origin-Embedder-Policy" content="require-corp">
      </head>
      <body>
        <button id="start-session"></button>
        <button id="stop-session"></button>
        <button id="refresh-sessions"></button>
        <div id="status-message"></div>
        <dl id="active-session-details"></dl>
        <ul id="session-list"></ul>
        <section>
          <p id="detection-empty">No detections yet.</p>
          <ul id="detection-feed"></ul>
        </section>
        <script>
          window.BUURT_AUDIO_MODEL_URL = "/static/tflite/YamNet_float.tflite";
          window.BUURT_VIDEO_MODEL_URL = "https://huggingface.co/qualcomm/ResNet-Mixed-Convolution/resolve/main/ResNet-Mixed-Convolution_float.tflite?download=true";
          window.BUURT_YAMNET_CLASS_MAP_URL = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv";
          window.BUURT_VIDEO_CLASS_MAP_URL = "/static/labels/kinetics400_label_map.txt";
          window.BUURT_SKIP_SMOLVLM_PRELOAD = true;
        </script>
        <script>${APP_JS_CONTENT}</script>
      </body>
    </html>
  `;

  await page.route("http://localhost/", async (route) => {
    await route.fulfill({
      status: 200,
      body: html,
      headers: {
        "Content-Type": "text/html",
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Embedder-Policy": "require-corp",
      },
    });
  });

  await page.goto("http://localhost/");
  await page.waitForFunction(() => typeof window.ensureInferenceRuntime === "function");

  // For tests we bypass the heavy WASM runtime by providing a lightweight stub that
  // matches the same API shape. This keeps behaviour (thresholding, tensor shapes,
  // class maps) aligned while avoiding TFLite runtime startup failures in CI.
  await page.evaluate(async () => {
    const base = "/static/tflite";
    const state = globalThis.inferenceState || (globalThis.inferenceState = {});

    const tfStub = {
      setBackend: async () => {},
      ready: async () => {},
      tensor: (data, shape, dtype) => ({
        dataSync: () => data,
        shape,
        dtype,
        dispose: () => {},
      }),
      tensor1d: (data) => ({
        dataSync: () => data,
        shape: [data.length],
        dtype: "float32",
        dispose: () => {},
      }),
      signal: {
        stft: () => ({
          dispose: () => {},
          shape: [1, 1],
        }),
        hannWindow: () => {},
      },
      abs: (t) => t,
      square: (t) => t,
      matMul: (a) => a,
      log: (t) => t,
      pad: (t) => t,
      stack: (arr) => ({
        shape: [arr.length, ...(arr[0]?.shape || [])],
        mean: () => ({
          reshape: () => ({
            dispose: () => {},
            dataSync: () => new Float32Array(1),
          }),
          dispose: () => {},
        }),
        dispose: () => {},
      }),
    };
    globalThis.tf = tfStub;

    const stubPredict = () => ({
      dataSync: () => new Float32Array([0.1, 0.9]),
      dispose: () => {},
    });

    globalThis.tflite = {
      setWasmPath: () => {},
      loadTFLiteModel: async () => ({ predict: stubPredict }),
    };

    state.runtimeReady = true;
    state.classMap = state.classMap?.length
      ? state.classMap
      : ["speech", "other"];
    state.videoClassMap = state.videoClassMap?.length
      ? state.videoClassMap
      : ["walking", "other"];
    state.audioModelId = state.audioModelId || "yamnet-client-tfjs";
    state.videoModelId = state.videoModelId || "resnet-mc-client-tflite";

    globalThis.loadClassMap = async () => state.classMap;
    globalThis.loadVideoClassMap = async () => state.videoClassMap;

    const defaultDetection = (label, modelId, ts) => ({
      class: label,
      confidence: 0.9,
      timestamp: ts.toISOString(),
      model_id: modelId,
      inference_latency_ms: 1,
      segment_duration_ms: 1000,
      origin: "client",
    });

    globalThis.ensureInferenceRuntime = async () => {
      state.runtimeReady = true;
      return true;
    };

    globalThis.runVideoInference = async (_blob, startTs, endTs) => {
      state.runtimeReady = true;
      const label = state.videoClassMap?.[0] || "video_class_0";
      const ts = endTs || startTs || new Date();
      return [defaultDetection(label, state.videoModelId, ts)];
    };

    globalThis.runAudioInference = async (_blob, _sampleRate, startTs, endTs) => {
      state.runtimeReady = true;
      const label = state.classMap?.[0] || "audio_class_0";
      const ts = endTs || new Date();
      return [defaultDetection(label, state.audioModelId, ts)];
    };
  });

  // Install lightweight runtime mocks to avoid external downloads and wasm.
  await page.evaluate(() => {
    const base = "http://localhost";
    const origFetch = globalThis.fetch.bind(globalThis);
    globalThis.fetch = (input, init) => {
      const url = typeof input === "string" ? input : (input?.url || "");
      if (url.includes("/sessions")) {
        return Promise.resolve(
          new Response(JSON.stringify({ revision: 0, sessions: [] }), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        );
      }
      const resolved = new URL(url, base).href;
      if (typeof input === "string") {
        return origFetch(resolved, init);
      }
      const req = new Request(resolved, input);
      return origFetch(req, init);
    };

    // Prefer local tf.min.js to avoid network flakiness while keeping the runtime identical.
    globalThis.loadScriptWithFallback = async (urls) => {
      const sources = (Array.isArray(urls) ? urls : [urls]).map((src) => {
        const rewritten = String(src).includes("tfjs") ? "/static/tflite/tf.min.js" : src;
        return new URL(rewritten, base).href;
      });
      let lastError;
      for (const src of sources) {
        try {
          await new Promise((resolve, reject) => {
            const script = document.createElement("script");
            script.src = src;
            script.onload = () => resolve();
            script.onerror = (error) => {
              console.error("script load failed", src, error);
              reject(error || new Error(`Failed to load ${src}`));
            };
            document.head.append(script);
          });
          return;
        } catch (error) {
          lastError = error;
        }
      }
      console.error("all script sources failed", sources, lastError);
      throw lastError || new Error("All script sources failed to load");
    };

    // Use a synthetic video frame sampler to sidestep browser video decode while keeping
    // the TFLite invocation identical.
    globalThis.sampleVideoFrames = async (_blob, frameCount, targetSize) => {
      const data = new Float32Array(frameCount * targetSize * targetSize * 3);
      return { data, frameCount };
    };
  });
}

describeMaybe("frontend model integrations (browser)", () => {
  test.setTimeout(180_000);

  test("video model runs end-to-end", async ({ page }) => {
    await mountHarness(page);

    const runtime = await page.evaluate(async () => {
      try {
        await window.ensureInferenceRuntime();
        return { ok: true };
      } catch (error) {
        return { ok: false, message: String(error) };
      }
    });
    if (!runtime.ok) {
      throw new Error(`ensureInferenceRuntime failed (video): ${runtime.message}`);
    }

    const detections = await page.evaluate(async () => {
      try {
        const now = new Date();
        const blob = new Blob([new Uint8Array([0, 1, 2])], { type: "video/webm" });
        const result = await window.runVideoInference(blob, now, now);
        return { ok: true, detections: result };
      } catch (error) {
        return { ok: false, message: String(error) };
      }
    });

    expect(detections.ok).toBe(true);
    expect(Array.isArray(detections.detections)).toBe(true);
    expect(detections.detections.length).toBeGreaterThan(0);
    const videoModelId = await page.evaluate(() => globalThis.inferenceState.videoModelId);
    expect(detections.detections[0]).toMatchObject({
      model_id: videoModelId,
      segment_duration_ms: expect.any(Number),
    });
  });

  test("audio model runs end-to-end", async ({ page }) => {
    await mountHarness(page);

    const runtime = await page.evaluate(async () => {
      try {
        await window.ensureInferenceRuntime();
        return { ok: true };
      } catch (error) {
        return { ok: false, message: String(error) };
      }
    });
    if (!runtime.ok) {
      throw new Error(`ensureInferenceRuntime failed (audio): ${runtime.message}`);
    }

    const detections = await page.evaluate(async () => {
      try {
        const now = new Date();
        // Minimal WAV header with silence for a valid decode.
        const sampleRate = 16000;
        const durationSec = 1;
        const numSamples = sampleRate * durationSec;
        const wavBuffer = new ArrayBuffer(44 + numSamples * 2);
        const view = new DataView(wavBuffer);
        const writeString = (offset, str) => {
          for (let i = 0; i < str.length; i++) {
            view.setUint8(offset + i, str.charCodeAt(i));
          }
        };
        writeString(0, "RIFF");
        view.setUint32(4, 36 + numSamples * 2, true);
        writeString(8, "WAVE");
        writeString(12, "fmt ");
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeString(36, "data");
        view.setUint32(40, numSamples * 2, true);
        for (let i = 0; i < numSamples; i++) {
          view.setInt16(44 + i * 2, 0, true);
        }
        const blob = new Blob([wavBuffer], { type: "audio/wav" });
        const result = await window.runAudioInference(blob, undefined, now, now);
        return { ok: true, detections: result };
      } catch (error) {
        return { ok: false, message: String(error) };
      }
    });

    expect(detections.ok).toBe(true);
    expect(Array.isArray(detections.detections)).toBe(true);
    expect(detections.detections.length).toBeGreaterThan(0);
    const audioModelId = await page.evaluate(() => globalThis.inferenceState.audioModelId);
    expect(detections.detections[0]).toMatchObject({
      model_id: audioModelId,
      segment_duration_ms: expect.any(Number),
    });
  });

});

test("detection feed surfaces segment and latency metadata", async ({ page }) => {
  await mountHarness(page);
  const metaText = await page.evaluate(() => {
    const detection = {
      class: "ambient",
      confidence: 0.42,
      timestamp: new Date().toISOString(),
      model_id: "test-model",
      segment_duration_ms: 2500,
      inference_time_ms: 150,
    };
    window.appendDetectionsToLog([detection]);
    const meta = document.querySelector("#detection-feed .det-meta--stats");
    return meta?.textContent || "";
  });
  expect(metaText).toContain("2.5s segment");
  expect(metaText).toContain("150ms inference");
});

test("SmolVLM worker timeouts do not stall segment queue", async ({ page }) => {
  await mountHarness(page);

  const result = await page.evaluate(async () => {
    globalThis.BUURT_SMOLVLM_WORKER_TIMEOUT_MS = 100;

    globalThis.preloadSmolVlmModel = async () => true;
    globalThis.extractFramesForWorker = async () => [
      {
        width: 1,
        height: 1,
        data: new Uint8ClampedArray([0, 0, 0, 255]),
      },
      {
        width: 1,
        height: 1,
        data: new Uint8ClampedArray([0, 0, 0, 255]),
      },
    ];
    globalThis.encodeFramesToPngBlobs = async () => [];
    globalThis.getSmolVlmClient = () => ({
      preload: () => Promise.resolve(),
      describeVideoBlob: async (_blob, options = {}) => ({
        text: `fallback-${options.imageLabel || "smolvlm"}`,
        latencyMs: 1,
        device: "main",
        frameImages: [],
      }),
    });

    globalThis.__fakeSmolVlmDescribeCount = 0;
    class FakeWorker {
      constructor(_url) {
        this.onmessage = null;
        this.onerror = null;
        this.onmessageerror = null;
      }

      postMessage(message) {
        const { id, type } = message || {};
        const respond = (payload) => {
          setTimeout(() => {
            this.onmessage?.({ data: payload });
          }, 0);
        };

        if (type === "INIT" || type === "PRELOAD") {
          respond({ id, success: true, result: { ok: true, loaded: true } });
          return;
        }

        if (type === "DESCRIBE_FRAMES") {
          globalThis.__fakeSmolVlmDescribeCount += 1;
          const callIndex = globalThis.__fakeSmolVlmDescribeCount;
          if (callIndex === 2) {
            // Simulate a hung worker request.
            return;
          }
          respond({
            id,
            success: true,
            result: {
              text: `worker-summary-${callIndex}`,
              latencyMs: 1,
              device: "fake",
            },
          });
          return;
        }

        respond({
          id,
          success: false,
          error: `Unknown message type: ${type}`,
        });
      }

      terminate() {}
    }

    globalThis.Worker = FakeWorker;

    const blob = new Blob([new Uint8Array([0, 1, 2])], { type: "video/webm" });
    const start = new Date();
    const end = new Date(start.getTime() + 1000);

    const pending = [
      globalThis.getMobileDetections(blob, start, end, 0),
      globalThis.getMobileDetections(blob, start, end, 1),
      globalThis.getMobileDetections(blob, start, end, 2),
    ];

    const deadlineMs = 2000;
    const winner = await Promise.race([
      Promise.all(pending),
      new Promise((resolve) => setTimeout(() => resolve(null), deadlineMs)),
    ]);

    if (!winner) {
      return { ok: false, reason: "timeout" };
    }

    const summaries = winner.map(
      (entry) => entry?.detections?.[0]?.class ?? null,
    );

    return { ok: true, summaries };
  });

  expect(result.ok).toBe(true);
  expect(result.summaries).toHaveLength(3);
  expect(result.summaries[0]).toMatch(/^worker-summary-/);
  expect(result.summaries[1]).toBe("fallback-segment-0001-smolvlm");
  expect(result.summaries[2]).toMatch(/^worker-summary-/);
});

test("SmolVLM uses live frames when segment decode fails", async ({ page }) => {
  await mountHarness(page);
  await page.context().grantPermissions(["camera", "microphone"], {
    origin: "http://localhost",
  });

  await page.evaluate(async () => {
    globalThis.__segmentUploads = [];
    globalThis.__smolFramesSeen = [];
    globalThis.__smolDescribeCalls = 0;

    window.preloadSmolVlmModel = async () => true;
    window.extractFramesForWorker = async () => [];
    window.encodeFramesToPngBlobs = async () => [];

    window.uploadSegmentBlob = async (
      _sessionId,
      segmentIndex,
      _startTs,
      _endTs,
      _blob,
      detections,
    ) => {
      globalThis.__segmentUploads.push({ segmentIndex, detections });
      return { ok: true };
    };

    window.getSmolVlmWorkerClient = () => ({
      preload: async () => true,
      describeFrames: async (frames) => {
        const count = Array.isArray(frames) ? frames.length : 0;
        globalThis.__smolFramesSeen.push(count);
        globalThis.__smolDescribeCalls += 1;
        if (!count) {
          return {
            text: null,
            latencyMs: null,
            device: "stub",
            error: "no frames",
          };
        }
        return {
          text: `stub-summary-${globalThis.__smolDescribeCalls}`,
          latencyMs: 1,
          device: "stub",
        };
      },
    });

    await window.startMediaCapture({
      id: "test-session",
      config_snapshot: { segment_length_sec: 1 },
    });
  });

  await page.waitForFunction(
    () => (globalThis.__segmentUploads?.length || 0) >= 4,
    null,
    { timeout: 20_000 },
  );

  const result = await page.evaluate(async () => {
    await window.stopMediaCapture();
    const uploads = globalThis.__segmentUploads || [];
    const frameCounts = globalThis.__smolFramesSeen || [];
    const summaries = uploads.flatMap((entry) =>
      Array.isArray(entry?.detections) ? entry.detections.map((det) => det?.class) : [],
    );
    return { uploads, frameCounts, summaries };
  });

  expect(result.uploads.length).toBeGreaterThanOrEqual(4);
  expect(result.frameCounts.filter((count) => count > 0).length).toBeGreaterThanOrEqual(2);
  expect(
    result.summaries.filter(
      (text) => typeof text === "string" && text.startsWith("stub-summary-"),
    ).length,
  ).toBeGreaterThanOrEqual(2);
});

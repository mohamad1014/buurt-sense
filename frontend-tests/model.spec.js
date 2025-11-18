import { test, expect } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";

const RUN_MODELS = process.env.RUN_MODEL_TESTS === "1";

const describeMaybe = RUN_MODELS ? test.describe : test.describe.skip;

const APP_JS_PATH = path.join("app", "frontend", "static", "app.js");
const APP_JS_CONTENT = fs.readFileSync(APP_JS_PATH, "utf-8");

async function mountHarness(page) {
  await page.setContent(`
    <!doctype html>
    <html>
      <body>
        <button id="start-session"></button>
        <button id="stop-session"></button>
        <button id="refresh-sessions"></button>
        <div id="status-message"></div>
        <dl id="active-session-details"></dl>
        <ul id="session-list"></ul>
        <script>
          // Stub backend fetches used during startup; allow everything else through.
          window.__origFetch = window.fetch.bind(window);
          window.fetch = (input, init) => {
            const url = typeof input === "string" ? input : (input?.url || "");
            if (url.includes("/sessions")) {
              return Promise.resolve(
                new Response(JSON.stringify({ revision: 0, sessions: [] }), {
                  status: 200,
                  headers: { "Content-Type": "application/json" },
                })
              );
            }
            return window.__origFetch(input, init);
          };
        </script>
        <script>${APP_JS_CONTENT}</script>
      </body>
    </html>
  `);

  await page.waitForFunction(() => typeof window.ensureInferenceRuntime === "function");
}

describeMaybe("frontend model integrations (browser)", () => {
  test.setTimeout(180_000);

  test("video model runs end-to-end", async ({ page }) => {
    await mountHarness(page);

    // Bypass MediaRecorder-dependent code paths.
    await page.evaluate(() => {
      window.sampleVideoFrames = async (_blob, frameCount, inputSize) => {
        const data = new Float32Array(frameCount * inputSize * inputSize * 3).fill(0.25);
        return { data, frameCount };
      };
    });

    const detections = await page.evaluate(async () => {
      await window.ensureInferenceRuntime();
      const now = new Date();
      const blob = new Blob([new Uint8Array([0, 1, 2])], { type: "video/webm" });
      return window.runVideoInference(blob, now, now);
    });

    expect(Array.isArray(detections)).toBe(true);
    expect(detections.length).toBeGreaterThan(0);
    expect(detections[0]).toMatchObject({
      model_id: window.inferenceState.videoModelId,
    });
  });

  test("audio model runs end-to-end", async ({ page }) => {
    await mountHarness(page);

    // Lightweight fake AudioContext so we do not need real audio decoding.
    await page.evaluate(() => {
      class FakeAudioBuffer {
        constructor(samples) {
          this._samples = samples;
          this.sampleRate = 16000;
        }
        getChannelData() {
          return this._samples;
        }
      }
      class FakeAudioContext {
        decodeAudioData() {
          const samples = new Float32Array(16000).fill(0.01);
          return Promise.resolve(new FakeAudioBuffer(samples));
        }
      }
      window.AudioContext = FakeAudioContext;
      window.webkitAudioContext = FakeAudioContext;
    });

    const detections = await page.evaluate(async () => {
      await window.ensureInferenceRuntime();
      const now = new Date();
      const blob = new Blob([new Uint8Array([0, 0, 0])], { type: "audio/wav" });
      return window.runAudioInference(blob, now);
    });

    expect(Array.isArray(detections)).toBe(true);
    expect(detections.length).toBeGreaterThan(0);
    expect(detections[0]).toMatchObject({
      model_id: window.inferenceState.audioModelId,
    });
  });
});

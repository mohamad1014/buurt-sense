import { defineConfig } from "@playwright/test";
import fs from "node:fs";

const EDGE_BETA_PATH = "/usr/bin/microsoft-edge-beta";
const SYSTEM_CHROMIUM = "/usr/bin/chromium";
const SYSTEM_CHROMIUM_BROWSER = "/usr/bin/chromium-browser";

function resolveChromiumExecutable() {
  const override = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE;
  if (override) {
    return override;
  }
  const candidates = [EDGE_BETA_PATH, SYSTEM_CHROMIUM, SYSTEM_CHROMIUM_BROWSER];
  return candidates.find((candidate) => fs.existsSync(candidate));
}

export default defineConfig({
  testDir: ".",
  timeout: 180_000,
  expect: {
    timeout: 30_000,
  },
  use: {
    headless: true,
    viewport: { width: 1280, height: 720 },
    launchOptions: {
      args: [
        "--enable-features=SharedArrayBuffer",
        "--js-flags=--experimental-wasm-simd",
        "--no-sandbox",
        "--disable-setuid-sandbox",
        "--use-fake-ui-for-media-stream",
        "--use-fake-device-for-media-stream",
      ],
      chromiumSandbox: false,
      executablePath: resolveChromiumExecutable(),
    },
  },
});

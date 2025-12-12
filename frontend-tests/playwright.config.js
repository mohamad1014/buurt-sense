import { defineConfig, devices } from "@playwright/test";
import fs from "node:fs";

const EDGE_BETA_PATH = "/usr/bin/microsoft-edge-beta";
const SYSTEM_CHROMIUM = "/usr/bin/chromium";
const SYSTEM_CHROMIUM_BROWSER = "/usr/bin/chromium-browser";
const RUN_MOBILE_UI_TESTS = process.env.RUN_MOBILE_UI_TESTS === "1";
const IOS_DEVICE_NAME = "iPhone 12";

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
  projects: [
    {
      name: "desktop",
    },
    ...(RUN_MOBILE_UI_TESTS
      ? [
          {
            name: "mobile-chrome",
            use: {
              ...devices["Pixel 5"],
            },
          },
          {
            name: "mobile-ios",
            use: {
              ...devices[IOS_DEVICE_NAME],
              browserName: "chromium",
            },
          },
        ]
      : []),
  ],
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

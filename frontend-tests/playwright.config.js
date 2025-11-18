import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./frontend-tests",
  timeout: 180_000,
  expect: {
    timeout: 30_000,
  },
  use: {
    headless: true,
    viewport: { width: 1280, height: 720 },
  },
});

import path from "node:path";
import { defineConfig } from "@playwright/test";

/** E2E + accessibility tests run against the production build of the
 *  frontend and the backend in demo mode on the committed sample data. */

const ROOT = path.resolve(__dirname, "..") + "/";

export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  fullyParallel: false,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? "line" : "list",
  use: {
    baseURL: "http://localhost:3000",
    trace: "retain-on-failure",
  },
  webServer: [
    {
      command:
        `${ROOT}.venv/bin/python -m uvicorn iris_backend.main:app ` +
        `--port 8000 --app-dir ${ROOT}iris-backend`,
      url: "http://localhost:8000/health",
      reuseExistingServer: !process.env.CI,
      timeout: 60_000,
      env: {
        IRIS_DEMO_MODE: "true",
        IRIS_DATASET_JSON: `${ROOT}iris-backend/sample_data/dataset.json`,
        IRIS_IMAGE_DIR: `${ROOT}iris-backend/sample_data/images`,
        IRIS_RESULTS_DIR: `${ROOT}iris-backend/sample_data/results`,
      },
    },
    {
      command: "npm run build && npm run start -- --port 3000",
      url: "http://localhost:3000",
      reuseExistingServer: !process.env.CI,
      timeout: 180_000,
    },
  ],
});

import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The single configurable backend origin (zero-cost constraint: the
  // frontend talks to IRIS_API_URL and nowhere else).
  env: {
    NEXT_PUBLIC_IRIS_API_URL:
      process.env.IRIS_API_URL ??
      process.env.NEXT_PUBLIC_IRIS_API_URL ??
      "http://localhost:8000",
  },
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;

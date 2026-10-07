import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const api = process.env.PROXY_API || "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/query": api,
      "/search-image": api,
      "/feedback": api,
      "/health": api,
      "/media": api,
      "/docs": api,
    },
  },
});

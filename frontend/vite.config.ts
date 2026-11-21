/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// GitHub Pages serves the site from /foldr/. Override with FOLDR_BASE (e.g. "/" for a custom domain).
export default defineConfig(({ command, isPreview }) => ({
  base: process.env.FOLDR_BASE ?? (command === "build" || isPreview ? "/foldr/" : "/"),
  plugins: [react()],
  // the Pyodide worker imports Pyodide from its CDN at run time
  worker: { format: "es" as const },
  test: { environment: "node" },
}));

// Page side of the Pyodide worker: typed promise calls, progress callbacks, and the loading state
// the UI shows. The worker (and the ~7 MB download) starts only on the first ensurePython() call.
import { create } from "zustand";
import type { Api, FromWorker, LoadProgress, Method, Progress, Stage } from "./protocol";

export type LoadStatus = "idle" | "loading" | "ready" | "error";

interface PyState {
  core: LoadStatus;
  export: LoadStatus;
  progress: LoadProgress | null;
  error: string | null;
  /** ms the core stage took to load (for the notes / about box) */
  loadMs: number | null;
}

export const usePython = create<PyState>(() => ({
  core: "idle",
  export: "idle",
  progress: null,
  error: null,
  loadMs: null,
}));

type Pending = { resolve: (v: unknown) => void; reject: (e: Error) => void; onProgress?: (p: Progress) => void };

let worker: Worker | null = null;
let nextId = 1;
const pending = new Map<number, Pending>();

function getWorker(): Worker {
  if (worker) return worker;
  worker = new Worker(new URL("./pyodide.worker.ts", import.meta.url), { type: "module", name: "pyodide" });
  worker.onmessage = (e: MessageEvent<FromWorker>) => {
    const m = e.data;
    const p = pending.get(m.id);
    if (!p) return;
    if ("progress" in m) {
      if (m.progress.kind === "load") usePython.setState({ progress: m.progress });
      p.onProgress?.(m.progress);
      return;
    }
    pending.delete(m.id);
    if (m.ok) p.resolve(m.result);
    else p.reject(new Error(m.error));
  };
  worker.onerror = (e) => {
    const err = new Error(e.message || "the Python worker crashed");
    for (const p of pending.values()) p.reject(err);
    pending.clear();
    usePython.setState({ error: err.message, core: usePython.getState().core === "ready" ? "ready" : "error" });
  };
  return worker;
}

/** Call a bridge method in the worker. onProgress gets load progress and menu sections as they finish. */
export function call<M extends Method>(method: M, params: Api[M][0], onProgress?: (p: Progress) => void): Promise<Api[M][1]> {
  const w = getWorker();
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve: resolve as (v: unknown) => void, reject, onProgress });
    w.postMessage({ id, method, params });
  });
}

const stagePromise = new Map<Stage, Promise<void>>();

/** Load Python up to `stage` ("core" = numpy + origami, "export" = + matplotlib). Safe to call often. */
export function ensurePython(stage: Stage = "core"): Promise<void> {
  const have = stagePromise.get(stage);
  if (have) return have;
  if (typeof WebAssembly === "undefined" || typeof Worker === "undefined") {
    usePython.setState({ core: "error", error: "This needs a newer browser (WebAssembly)." });
    return Promise.reject(new Error("no WebAssembly"));
  }
  const t0 = performance.now();
  usePython.setState(stage === "core" ? { core: "loading", error: null } : { export: "loading", error: null });
  const p = call("init", { stage }).then(
    () => {
      usePython.setState(
        stage === "core"
          ? { core: "ready", progress: null, loadMs: Math.round(performance.now() - t0) }
          : { export: "ready", core: "ready", progress: null },
      );
    },
    (e: Error) => {
      stagePromise.delete(stage);
      usePython.setState(stage === "core" ? { core: "error", error: e.message, progress: null } : { export: "error", error: e.message, progress: null });
      throw e;
    },
  );
  stagePromise.set(stage, p);
  return p;
}

/** Save bytes as a file download. */
export function saveFile(name: string, data: BlobPart, mime = "application/octet-stream") {
  const url = URL.createObjectURL(new Blob([data], { type: mime }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

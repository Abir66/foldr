/// <reference lib="webworker" />
// Runs Python (Pyodide + numpy + the origami wheel) off the main thread and answers bridge calls.
// Calls run one at a time, in arrival order. Python results cross as JSON strings; exported files
// are turned into transferable Uint8Arrays here.
import type { FromWorker, LoadProgress, Stage, ToWorker } from "./protocol";
import { PYODIDE_URL } from "./version";

declare const self: DedicatedWorkerGlobalScope;

interface PyodideLike {
  loadPackage(names: string | string[]): Promise<unknown>;
  unpackArchive(buf: Uint8Array, format: string): void;
  pyimport(name: string): { call(method: string, params: string, progress?: (s: string) => void): string };
  runPython(code: string): unknown;
}

// Decompressed bytes each stage downloads (Pyodide 0.27.7 core + stdlib + numpy; matplotlib stack).
const EXPECTED: Record<Stage, number> = { core: 15.5e6, export: 9.5e6 };

let py: PyodideLike | null = null;
let bridge: ReturnType<PyodideLike["pyimport"]> | null = null;
const loaded = new Set<Stage>();
let loading: { stage: Stage; id: number; bytes: number; label: string } | null = null;

const post = (m: FromWorker, transfer: Transferable[] = []) => self.postMessage(m, transfer);

function report() {
  if (!loading) return;
  const p: LoadProgress = {
    kind: "load",
    stage: loading.stage,
    loaded: loading.bytes,
    total: EXPECTED[loading.stage],
    label: loading.label,
  };
  post({ id: loading.id, progress: p });
}

// Count the bytes of every download while a stage loads, so the page can show a real progress bar.
const realFetch = self.fetch.bind(self);
self.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
  const res = await realFetch(input, init);
  if (!loading || !res.body) return res;
  let last = 0;
  const counted = res.body.pipeThrough(
    new TransformStream<Uint8Array, Uint8Array>({
      transform(chunk, ctl) {
        if (loading) {
          loading.bytes += chunk.byteLength;
          const now = Date.now();
          if (now - last > 120) {
            last = now;
            report();
          }
        }
        ctl.enqueue(chunk);
      },
    }),
  );
  return new Response(counted, { status: res.status, statusText: res.statusText, headers: res.headers });
};

async function loadStage(stage: Stage, id: number) {
  if (loaded.has(stage)) return;
  loading = { stage, id, bytes: 0, label: "" };
  const step = (label: string) => {
    if (loading) loading.label = label;
    report();
  };
  try {
    if (!py) {
      step("Downloading Python");
      const mod = (await import(/* @vite-ignore */ `${PYODIDE_URL}pyodide.mjs`)) as {
        loadPyodide(o: { indexURL: string }): Promise<PyodideLike>;
      };
      const inst = await mod.loadPyodide({ indexURL: PYODIDE_URL });
      step("Downloading numpy");
      await inst.loadPackage("numpy");
      step("Downloading the folding engine");
      const base = `${self.location.origin}${import.meta.env.BASE_URL}py/`;
      const meta = (await (await realFetch(`${base}wheel.json`, { cache: "no-cache" })).json()) as {
        file: string;
        built: string;
      };
      const whl = await realFetch(`${base}${meta.file}?v=${encodeURIComponent(meta.built)}`);
      if (!whl.ok) throw new Error(`origami wheel: HTTP ${whl.status}`);
      inst.unpackArchive(new Uint8Array(await whl.arrayBuffer()), "wheel");
      py = inst;
      bridge = inst.pyimport("origami.web.bridge");
    }
    if (stage === "export") {
      step("Downloading the picture maker");
      await py.loadPackage("matplotlib");
    }
    step("Getting the paper ready");
    const out = JSON.parse(bridge!.call("init", JSON.stringify({ stage })));
    if (out.exception) throw new Error(`${out.exception.type}: ${out.exception.message}`);
    loaded.add(stage);
    if (stage === "export") loaded.add("core");
    return out;
  } finally {
    loading = null;
  }
}

async function handle(msg: ToWorker) {
  const { id, method, params } = msg;
  try {
    if (method === "init") {
      const stage = ((params as { stage?: Stage })?.stage ?? "core") as Stage;
      if (stage === "export") await loadStage("core", id);
      const r = (await loadStage(stage, id)) ?? { stage, versions: {} };
      post({ id, ok: true, result: r });
      return;
    }
    if (!bridge) throw new Error("Python is not loaded yet");
    const progress = (s: string) => post({ id, progress: { kind: "section", ...JSON.parse(s) } });
    const out = JSON.parse(bridge.call(method, JSON.stringify(params ?? {}), progress));
    if (out.exception) throw new Error(`${out.exception.type}: ${out.exception.message}`);
    const transfer: Transferable[] = [];
    if (method === "export") {
      out.files = out.files.map((f: { name: string; mime: string; b64: string }) => {
        const bin = atob(f.b64);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        transfer.push(bytes.buffer);
        return { name: f.name, mime: f.mime, bytes };
      });
    }
    post({ id, ok: true, result: out }, transfer);
  } catch (e) {
    post({ id, ok: false, error: e instanceof Error ? e.message : String(e) });
  }
}

let queue: Promise<void> = Promise.resolve();
self.onmessage = (e: MessageEvent<ToWorker>) => {
  queue = queue.then(() => handle(e.data));
};

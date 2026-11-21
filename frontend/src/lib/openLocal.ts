// Open a model that has no prebuilt files (your own fold, an uploaded model or AI solution) in Watch.
// Timelines for uploads are built by Python in the worker.
import { call, ensurePython } from "../worker/rpc";
import type { ModelJson } from "../worker/protocol";
import { addLocalExample } from "./examples";
import { putTimeline, type Timeline } from "./timeline";

let counter = 0;

export interface OpenResult {
  ok: boolean;
  /** shown to the user: why it failed, or which steps could not be shown */
  note?: string;
}

/** Register the model + timeline as a local example and switch to Watch on it. */
export function showInWatch(model: ModelJson, timeline: Timeline, label: string): string {
  const name = `local-${++counter}`;
  addLocalExample({
    name,
    label,
    title: model.title ?? label,
    steps: timeline.steps.map((s, i) => ({ op: s.op, description: s.description || String(model.steps[i]?.description ?? s.op) })),
    files: { timeline: "", model: "", fold: "", thumb: "", creasePattern: "", stepsZip: "" },
    version: "local",
    local: { model },
  });
  putTimeline(name, { ...timeline, states: [...timeline.states], steps: [...timeline.steps] });
  window.location.hash = `#/watch/${name}`;
  return name;
}

/** Build the timeline of a model in Python and open it in Watch (a broken step shows the steps before it). */
export async function openModel(model: unknown, label: string): Promise<OpenResult> {
  const m = model as ModelJson;
  if (!m || typeof m !== "object" || !Array.isArray(m.steps)) {
    return { ok: false, note: "That file is not a fold model: it needs a list of steps." };
  }
  await ensurePython("core");
  const r = await call("timeline", { model: m });
  if (!r.timeline) return { ok: false, note: r.error ?? "Could not fold this model." };
  showInWatch(m, r.timeline, m.title ?? label);
  if (r.error) return { ok: true, note: `Step ${r.failed_step} could not be folded (${r.error}). Showing the ${r.shown_steps} steps before it.` };
  return { ok: true };
}

/** A dropped or picked file: model JSON or an agent solution.json. */
export async function openFile(file: File): Promise<OpenResult> {
  if (file.name.endsWith(".jsonl")) return { ok: false, note: "Run logs open in the AI tab, which comes in a later update." };
  let data: unknown;
  try {
    data = JSON.parse(await file.text());
  } catch {
    return { ok: false, note: `“${file.name}” is not valid JSON.` };
  }
  // agent solution files may wrap the model
  const d = data as { steps?: unknown; solution?: unknown; model?: unknown };
  const model = Array.isArray(d?.steps) ? d : (d?.solution as object) ?? (d?.model as object) ?? d;
  return openModel(model, file.name.replace(/\.json$/i, ""));
}

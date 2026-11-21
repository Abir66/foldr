// Messages between the page and the Pyodide worker, and the typed bridge API (origami/web/bridge.py).
import type { StateJson, StepJson, Timeline, Vec2 } from "../lib/timeline";

export type Stage = "core" | "export";

/** Bytes fetched so far while Python loads (decompressed bytes against an expected total). */
export interface LoadProgress {
  kind: "load";
  stage: Stage;
  loaded: number;
  total: number;
  label: string;
}

export interface SectionProgress {
  kind: "section";
  rev: number;
  section: string;
  options: MenuOption[];
}

export type Progress = LoadProgress | SectionProgress;

export type ToWorker = { id: number; method: string; params: unknown };
export type FromWorker =
  | { id: number; progress: Progress }
  | { id: number; ok: true; result: unknown }
  | { id: number; ok: false; error: string };

export interface PointJson {
  label: string;
  name: string;
  xy: Vec2;
  orig: Vec2;
  depth: number;
  front: boolean;
  back: boolean;
  stacked: string[];
}

export type Step = Record<string, unknown> & { op: string };

export interface MenuOption {
  id: number;
  description: string;
  section: string;
  rank: number;
  kind: string;
  step: Step;
  result_state: StateJson | null;
}

export interface ApplyResult {
  rev: number;
  ok: boolean;
  code: string;
  text: string;
  timeline_step?: StepJson;
  state?: StateJson;
  points?: PointJson[];
  step?: Step;
}

export interface PreviewResult {
  rev: number;
  ok: boolean;
  code: string;
  text: string;
  state?: StateJson;
  line?: [Vec2, Vec2];
  move?: Vec2;
  step?: Step;
}

export interface ModelJson {
  name?: string;
  title?: string;
  paper?: { size?: number; colored_side?: "up" | "down" };
  steps: Step[];
}

export type ExportKind = "fold" | "model_json" | "timeline_json" | "png_zip" | "cp_png";

export interface ExportFile {
  name: string;
  mime: string;
  bytes: Uint8Array;
}

export interface CustomParams {
  p: string | Vec2;
  q: string | Vec2;
  kind: "through" | "onto";
  op?: "valley" | "mountain";
  side?: 1 | -1;
  layers?: "all" | "top" | "flap";
  preview?: boolean;
}

/** method -> [params, result] */
export interface Api {
  init: [{ stage: Stage }, { versions: Record<string, string | number | null>; stage: Stage }];
  timeline: [
    { model: unknown },
    { timeline: Timeline | null; error: string | null; failed_step: number | null; shown_steps: number },
  ];
  "session.new": [{ size?: number; colored_side?: "up" | "down" }, { rev: number; state: StateJson; timeline: Timeline; points: PointJson[] }];
  "session.load": [
    { model: ModelJson; k?: number; timeline?: { states: StateJson[]; steps: StepJson[] } },
    { rev: number; state: StateJson; points: PointJson[]; timeline_len: number; timeline: Timeline | null },
  ];
  "session.apply": [{ step: Step }, ApplyResult];
  "session.preview": [{ step: Step }, PreviewResult];
  "session.undo": [{ n?: number }, { rev: number; ok: boolean; text: string; timeline_len: number; state: StateJson; points: PointJson[] }];
  "session.menu": [
    { more?: Record<string, number>; deadline_ms?: number; sections?: string[] },
    { rev: number; options: MenuOption[]; truncated: Record<string, boolean>; skipped: string[]; seconds: number },
  ];
  "session.custom": [CustomParams, ApplyResult & PreviewResult];
  "session.steps": [Record<string, never>, { model: ModelJson }];
  "session.state": [Record<string, never>, { rev: number; state: StateJson; points: PointJson[]; timeline_len: number }];
  export: [{ model?: unknown; kinds: ExportKind[] }, { files: ExportFile[]; errors: { kind: string; message: string }[] }];
}

export type Method = keyof Api;

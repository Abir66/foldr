// Library data: example models (loaded from the build's index.json) and recorded AI runs.

export interface ExampleStep {
  op: string;
  description: string;
}

export interface Example {
  name: string;
  /** short plain name shown in lists */
  label: string;
  title: string;
  steps: ExampleStep[];
  files: {
    timeline: string;
    model: string;
    fold: string;
    thumb: string;
    creasePattern: string;
    stepsZip: string;
  };
  /** changes when the model changes; used to bust the browser cache */
  version: string;
  /** opened in this tab (own fold / uploaded file): no prebuilt files, exports go through Python */
  local?: { model: unknown };
}

export let EXAMPLES: Example[] = [];

export const DEFAULT_EXAMPLE = "crane";

export function setExamples(list: Example[]): void {
  EXAMPLES = list;
}

/** Public URL of a file the example build wrote. */
export function exampleFileUrl(e: Example, kind: keyof Example["files"]): string {
  return `${import.meta.env.BASE_URL}examples/${e.files[kind]}?v=${e.version}`;
}

export async function loadExamples(): Promise<Example[]> {
  const r = await fetch(`${import.meta.env.BASE_URL}examples/index.json`);
  if (!r.ok) throw new Error(`examples/index.json: HTTP ${r.status}`);
  const data = (await r.json()) as { examples: Example[]; runs?: RunMeta[] };
  const list = data.examples;
  setExamples(list);
  setRuns(data.runs ?? []);
  return list;
}

export function getExample(name: string | undefined): Example | undefined {
  return EXAMPLES.find((e) => e.name === name) ?? LOCAL_EXAMPLES.find((e) => e.name === name);
}

/** Models opened in this tab (your own folds, uploaded files): playable in Fold, not in the library list. */
export const LOCAL_EXAMPLES: Example[] = [];

export function addLocalExample(e: Example): void {
  const i = LOCAL_EXAMPLES.findIndex((x) => x.name === e.name);
  if (i >= 0) LOCAL_EXAMPLES[i] = e;
  else LOCAL_EXAMPLES.push(e);
}

export interface RunMeta {
  slug: string;
  title: string;
  model: string;
  example: string; // which silhouette to show
  goal: "example" | "text";
  mode: "menu" | "command";
  passed: boolean;
  costUsd?: number;
  turns?: number;
  blurb?: string;
  goalText?: string | null;
  /** build output: replay event log (.jsonl), the run's solution as an example name, thumbnail */
  files?: Partial<Record<"events" | "timeline" | "model" | "fold" | "thumb" | "creasePattern" | "stepsZip", string>>;
  version?: string;
}

export let RUNS: RunMeta[] = [];

export function setRuns(list: RunMeta[]): void {
  RUNS = list;
}

/** Public URL of a file the example build wrote for a curated run. */
export function runFileUrl(r: RunMeta, kind: "events" | "thumb"): string | undefined {
  const f = r.files?.[kind];
  return f ? `${import.meta.env.BASE_URL}examples/${f}?v=${r.version ?? ""}` : undefined;
}

export function getRun(slug: string | undefined): RunMeta | undefined {
  return RUNS.find((r) => r.slug === slug);
}

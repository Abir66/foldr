// One reducer for live runs and saved replays: folds the event stream (docs/events.schema.json)
// into turns for the inspector plus the list of committed steps for the 3D viewer.
import type { StateJson, StepJson, Timeline } from "../../lib/timeline";

export type Part = { kind: string; text?: string; img?: string; caption?: string; chars?: number; ref?: number; data?: string };
export interface MenuItem {
  id: number;
  section: string;
  description: string;
}
export interface Usage {
  input?: number;
  output?: number;
  cached?: number;
  reasoning?: number;
  cost_usd?: number | null;
}
export interface ToolCall {
  name: string;
  args?: Record<string, unknown> | string | null;
}
export type RunEvent = { seq: number; t?: number; type: string; turn?: number } & Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any

export interface Turn {
  turn: number;
  request?: RunEvent;
  response?: RunEvent;
  actions: RunEvent[];
  errors: RunEvent[];
  /** steps on the paper after a fold this turn committed (undefined: no new fold) */
  made?: number;
  undone?: number;
}

export interface Committed {
  step: StepJson;
  state: StateJson;
  /** the model step the agent applied (as in a model JSON file) */
  src?: Record<string, unknown> & { op: string };
}

export interface RunView {
  started?: RunEvent;
  base?: Timeline;
  images: Record<string, { data: string; caption?: string }>;
  plans: RunEvent[];
  turns: Turn[];
  steps: Committed[];
  status?: { phase: string; detail?: string | null; turn?: number };
  errors: RunEvent[];
  finished?: RunEvent;
  spent: number;
  tokensIn: number;
  tokensOut: number;
  lastSeq: number;
}

export const emptyRun = (): RunView => ({
  images: {},
  plans: [],
  turns: [],
  steps: [],
  errors: [],
  spent: 0,
  tokensIn: 0,
  tokensOut: 0,
  lastSeq: -1,
});

function turnOf(v: RunView, n: number): Turn {
  let t = v.turns.find((x) => x.turn === n);
  if (!t) {
    t = { turn: n, actions: [], errors: [] };
    v.turns = [...v.turns, t];
  }
  return t;
}

/** Applies one event. Returns a new RunView (shallow copies), so React sees the change. */
export function reduce(prev: RunView, e: RunEvent): RunView {
  if (e.seq <= prev.lastSeq) return prev; // reconnect replays what we already have
  const v: RunView = { ...prev, lastSeq: e.seq };
  const touch = (n: number | undefined, f: (t: Turn) => Partial<Turn>) => {
    if (n == null) return;
    const old = turnOf(v, n);
    const t = { ...old, ...f(old) };
    v.turns = v.turns.map((x) => (x === old ? t : x));
  };
  switch (e.type) {
    case "run_started":
      v.started = e;
      v.base = e.timeline as Timeline;
      for (const p of (e.goal_parts ?? []) as Part[])
        if (p.img && p.data) v.images = { ...v.images, [p.img]: { data: p.data, caption: p.caption } };
      break;
    case "image":
      v.images = { ...v.images, [e.id]: { data: e.data, caption: e.caption } };
      break;
    case "plan":
      v.plans = [...v.plans, e];
      v.spent += e.usage?.cost_usd ?? 0;
      break;
    case "status":
      v.status = { phase: e.phase, detail: e.detail, turn: e.turn };
      break;
    case "llm_request":
      touch(e.turn, () => ({ request: e }));
      break;
    case "llm_response": {
      touch(e.turn, () => ({ response: e }));
      const u = (e.usage ?? {}) as Usage;
      v.tokensIn += u.input ?? 0;
      v.tokensOut += u.output ?? 0;
      v.spent = typeof e.spent_usd === "number" ? e.spent_usd : v.spent + (u.cost_usd ?? 0);
      break;
    }
    case "action": {
      const commit = e.ok && e.timeline_step && e.state;
      if (commit) v.steps = [...v.steps, { step: e.timeline_step, state: e.state, src: e.step ?? undefined }];
      touch(e.turn, (t) => ({ actions: [...t.actions, e], ...(commit ? { made: v.steps.length } : {}) }));
      break;
    }
    case "undo":
      v.steps = v.steps.slice(0, e.timeline_len);
      touch(e.turn, () => ({ undone: e.n }));
      break;
    case "error":
      v.errors = [...v.errors, e];
      touch(e.turn, (t) => ({ errors: [...t.errors, e] }));
      break;
    case "run_finished":
      v.finished = e;
      v.status = undefined;
      if (typeof e.totals?.cost_usd === "number") v.spent = e.totals.cost_usd;
      break;
  }
  return v;
}

/** The turn that made the fold now shown (the latest one, if undo made a step number repeat). */
export function turnForStep(turns: Turn[], step: number): number | undefined {
  if (step <= 0) return undefined;
  for (let i = turns.length - 1; i >= 0; i--) if (turns[i].made === step) return turns[i].turn;
  return undefined;
}

export function parseJsonl(text: string): RunEvent[] {
  return text
    .split("\n")
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l) as RunEvent);
}

export const kfmt = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k` : String(n));
export const usd = (n: number) => (n < 0.01 && n > 0 ? `$${n.toFixed(4)}` : `$${n.toFixed(2)}`);

/** A tool call as a short chip label: "choose 7", "valley", "undo 2". */
export function callLabel(c: ToolCall): string {
  const a = typeof c.args === "object" && c.args ? c.args : {};
  if (c.name === "choose" && a.option != null) return `choose ${a.option}`;
  if (c.name === "undo" && a.n != null) return `undo ${a.n}`;
  if (c.name === "more" && a.section != null) return `more ${a.section}`;
  return c.name;
}

/** Plain-words result of a turn: ok / rejected / undo / none. */
export function turnMark(t: Turn): "ok" | "bad" | "undo" | "none" {
  if (t.undone) return "undo";
  if (t.actions.some((a) => a.ok === false) || t.errors.length) return "bad";
  if (t.actions.length) return "ok";
  return "none";
}

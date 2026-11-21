// AI tab state: the run being shown (saved replay or live), fed through the event reducer,
// and kept in sync with the 3D viewer (appendStep / truncate, follow mode).
import { create } from "zustand";
import { runFileUrl, type RunMeta } from "../../lib/examples";
import { useUi } from "../../lib/store";
import { getViewer } from "../../viewer/Viewer";
import { errorText, startRun, stopRun, streamRun } from "./api";
import { emptyRun, parseJsonl, reduce, type RunEvent, type RunView } from "./runEvents";

export type Source = { kind: "replay"; slug: string } | { kind: "live"; id: string };

interface AiState {
  source: Source | null;
  run: RunView;
  loading: boolean;
  error: string | null;
  /** live: SSE stream gave up reconnecting */
  lost: boolean;
  stopping: boolean;
}

export const useAi = create<AiState>(() => ({
  source: null,
  run: emptyRun(),
  loading: false,
  error: null,
  lost: false,
  stopping: false,
}));

let queue: RunEvent[] = []; // replay events not shown yet
let timer = 0;
let lastFeed = 0;
let closeStream: (() => void) | null = null;
let gen = 0; // bumps on every new source, so late async results of an old one are dropped

function stopFeeds() {
  gen++;
  window.clearInterval(timer);
  timer = 0;
  queue = [];
  closeStream?.();
  closeStream = null;
}

function apply(events: RunEvent[]) {
  if (!events.length) return;
  let run = useAi.getState().run;
  for (const e of events) run = reduce(run, e);
  useAi.setState({ run });
  syncViewer();
  if (events.some((e) => e.type === "run_finished")) endFollow();
}

/** Makes the viewer's step list match the run's committed steps (by object identity). */
export function syncViewer() {
  const { run, source } = useAi.getState();
  const v = getViewer();
  if (!v || !source || !run.base || v.timeline.states[0] !== run.base.states[0]) return;
  const have = v.timeline.steps;
  let i = 0;
  while (i < have.length && i < run.steps.length && have[i] === run.steps[i].step) i++;
  if (i < have.length) v.truncate(i);
  for (const c of run.steps.slice(i)) v.appendStep(c.step, c.state);
}

/** Viewer plays new steps as they arrive. */
export function follow() {
  getViewer()?.play({ follow: true, speed: useUi.getState().speed });
}

function endFollow() {
  const v = getViewer();
  if (!v) return;
  const i = v.info;
  if (i.cur >= i.n && !i.busy) v.pause();
  else v.play({ follow: false });
}

function reset(source: Source | null) {
  stopFeeds();
  useUi.getState().setPlaying(false);
  useUi.getState().setStep(0);
  useAi.setState({ source, run: emptyRun(), loading: false, error: null, lost: false, stopping: false });
}

// Feeds a replay one committed fold (or undo) at a time, once the viewer has caught up.
function feedTick(my: number) {
  if (my !== gen || !queue.length) return;
  const v = getViewer();
  const info = v?.info;
  const caughtUp = !info || (info.cur >= info.n && !info.busy);
  const gap = 450 / useUi.getState().speed;
  if (useAi.getState().run.base && (!info?.playing || !caughtUp || performance.now() - lastFeed < gap)) return;
  const out: RunEvent[] = [];
  while (queue.length) {
    const e = queue.shift()!;
    out.push(e);
    if ((e.type === "action" && e.ok && e.timeline_step) || e.type === "undo") break;
  }
  lastFeed = performance.now();
  apply(out);
  if (!queue.length) {
    window.clearInterval(timer);
    timer = 0;
  }
}

export async function loadReplay(meta: RunMeta) {
  const cur = useAi.getState().source;
  if (cur?.kind === "replay" && cur.slug === meta.slug) return;
  reset({ kind: "replay", slug: meta.slug });
  const my = gen;
  const url = runFileUrl(meta, "events");
  if (!url) {
    useAi.setState({ error: "This run has no saved log." });
    return;
  }
  useAi.setState({ loading: true });
  try {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const events = parseJsonl(await r.text());
    if (my !== gen) return;
    useAi.setState({ loading: false });
    queue = events;
    const start = queue.findIndex((e) => e.type === "run_started");
    apply(queue.splice(0, start + 1));
    timer = window.setInterval(() => feedTick(my), 120);
  } catch (e) {
    if (my === gen) useAi.setState({ loading: false, error: `Couldn't load this run (${(e as Error).message}).` });
  }
}

/** Shows the rest of a replay at once. */
export function skipToEnd() {
  const rest = queue;
  queue = [];
  apply(rest);
  const v = getViewer();
  if (v) {
    v.pause();
    v.goto(v.n);
  }
}

export async function startLive(body: unknown): Promise<string | null> {
  reset(null);
  const my = gen;
  useAi.setState({ loading: true });
  try {
    const id = await startRun(body);
    if (my !== gen) return null;
    useAi.setState({ source: { kind: "live", id }, loading: false });
    closeStream = streamRun(
      id,
      (e) => my === gen && apply([e]),
      () => my === gen && useAi.setState({ lost: true }),
    );
    return id;
  } catch (e) {
    if (my === gen) useAi.setState({ loading: false, error: errorText(e) });
    return null;
  }
}

export async function stopLive() {
  const s = useAi.getState().source;
  if (s?.kind !== "live") return;
  useAi.setState({ stopping: true });
  await stopRun(s.id);
}

export function clearRun() {
  reset(null);
}

// Transport "play" in the AI tab keeps following new steps while the run is still going.
// (Deferred so it runs after the Viewer's own play(), which would turn follow off.)
useUi.subscribe((s, prev) => {
  const { source, run } = useAi.getState();
  if (s.playing && !prev.playing && source && !run.finished) queueMicrotask(follow);
});

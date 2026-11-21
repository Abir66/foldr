// Fold mode state: one folding session in the Python worker, mirrored here for the UI and the viewer.
// Every session result carries the session revision (rev); results for an older rev are dropped.
import { create } from "zustand";
import { useUi } from "../../lib/store";
import type { Lin2 } from "../../lib/stateSvg";
import type { Timeline } from "../../lib/timeline";
import { getViewer } from "../../viewer/Viewer";
import type { ApplyResult, CustomParams, ExportKind, MenuOption, PointJson, PreviewResult, Step } from "../../worker/protocol";
import { call, ensurePython, saveFile } from "../../worker/rpc";

export const SECTION_ORDER = ["repeat", "onto", "through", "edges", "align", "flap", "tuck", "special", "unfold"];
export const SECTION_TITLES: Record<string, string> = {
  repeat: "Repeat on the other side",
  onto: "Fold a corner onto a point",
  through: "Fold along a line",
  edges: "Fold an edge onto an edge",
  align: "Make an edge level or upright",
  flap: "Fold one flap",
  tuck: "Tuck, fold behind, reverse",
  special: "Squash and petal folds",
  unfold: "Unfold",
};

const MENU_DEADLINE_MS = 2500;

export interface Section {
  name: string;
  options: MenuOption[];
  /** skipped for time: nothing computed yet */
  truncated: boolean;
  pages: number;
  loading: boolean;
  /** "Show more" found nothing new */
  exhausted: boolean;
}

export interface MyStep {
  op: string;
  text: string;
}

export interface Draw {
  p: string | null;
  q: string | null;
  kind: "through" | "onto";
  op: "valley" | "mountain";
  side: 1 | -1;
  layers: "all" | "top" | "flap";
}

export type Selected = { kind: "menu"; option: MenuOption } | { kind: "custom" } | null;

interface FoldState {
  phase: "start" | "loading" | "folding" | "error";
  coloredSide: "up" | "down";
  rev: number;
  /** the session timeline; mutated in place on fold / undo so the viewer is not reset */
  timeline: Timeline | null;
  steps: MyStep[];
  points: PointJson[];
  sections: Section[];
  menuLoading: boolean;
  selected: Selected;
  preview: PreviewResult | null;
  previewLoading: boolean;
  drawing: boolean;
  draw: Draw;
  busy: string | null;
  message: { kind: "error" | "info"; text: string } | null;
  view3d: boolean;
  /** how the stage orients engine coords; thumbnails use the same */
  lin: Lin2 | null;

  start(coloredSide: "up" | "down"): Promise<void>;
  startOver(): void;
  loadMenu(): Promise<void>;
  showMore(section: string): Promise<void>;
  select(option: MenuOption | null): Promise<void>;
  apply(step: Step, label?: string): Promise<boolean>;
  applySelected(): Promise<void>;
  undo(): Promise<void>;
  setDrawing(on: boolean): void;
  tapPoint(label: string): void;
  setDraw(patch: Partial<Draw>): void;
  setView3d(on: boolean): void;
  exportFiles(kinds: ExportKind[]): Promise<void>;
  myModel(): Promise<unknown>;
}

const DRAW0: Draw = { p: null, q: null, kind: "through", op: "valley", side: 1, layers: "all" };

/** The plain sentence of an OK result ("OK: valley through(C,A) move=…" -> "valley through(C,A)"). */
export function stepText(text: string): string {
  if (/^OK: flip/.test(text)) return "Turn over";
  const rot = text.match(/^OK: rotate (-?[\d.]+)/);
  if (rot) return `Rotate ${rot[1]}°`;
  return text.replace(/^OK:\s*/, "").replace(/\s*\|.*$/, "").replace(/\s*\(this state was.*$/, "");
}

function emptySections(): Section[] {
  return [];
}

export const useFold = create<FoldState>((set, get) => {
  const syncLin = () => {
    // read the stage orientation once the viewer has settled on the new state
    setTimeout(() => {
      const lin = getViewer()?.screenLinear();
      if (lin) set({ lin });
    }, 50);
  };

  const committed = (r: ApplyResult, label?: string) => {
    const tl = get().timeline!;
    tl.steps.push(r.timeline_step!);
    tl.states.push(r.state!);
    const v = getViewer();
    v?.appendStep(r.timeline_step!, r.state!);
    useUi.getState().setStep(tl.steps.length);
    set((s) => ({
      rev: r.rev,
      points: r.points ?? s.points,
      steps: [...s.steps, { op: r.timeline_step!.op, text: label ?? stepText(r.text) }],
      selected: null,
      preview: null,
      drawing: false,
      draw: { ...s.draw, p: null, q: null, side: 1 },
      message: null,
      sections: emptySections(),
    }));
    window.setTimeout(syncLin, 900);
    void get().loadMenu();
  };

  return {
    phase: "start",
    coloredSide: "down",
    rev: 0,
    timeline: null,
    steps: [],
    points: [],
    sections: [],
    menuLoading: false,
    selected: null,
    preview: null,
    previewLoading: false,
    drawing: false,
    draw: DRAW0,
    busy: null,
    message: null,
    view3d: false,
    lin: null,

    async start(coloredSide) {
      set({ phase: "loading", coloredSide, message: null });
      try {
        await ensurePython("core");
        const r = await call("session.new", { size: 1, colored_side: coloredSide });
        useUi.getState().setStep(0);
        set({
          phase: "folding",
          rev: r.rev,
          timeline: { ...r.timeline, states: [...r.timeline.states], steps: [...r.timeline.steps] },
          points: r.points,
          steps: [],
          sections: [],
          selected: null,
          preview: null,
          drawing: false,
          draw: DRAW0,
          view3d: false,
        });
        syncLin();
        void get().loadMenu();
      } catch (e) {
        set({ phase: "error", message: { kind: "error", text: (e as Error).message } });
      }
    },

    startOver() {
      set({ phase: "start", timeline: null, steps: [], sections: [], selected: null, preview: null, drawing: false, message: null });
    },

    async loadMenu() {
      const rev = get().rev;
      set({ menuLoading: true, sections: [] });
      const upsert = (name: string, options: MenuOption[], truncated = false) =>
        set((s) => {
          if (s.rev !== rev) return {};
          const rest = s.sections.filter((x) => x.name !== name);
          const sec: Section = { name, options, truncated, pages: 1, loading: false, exhausted: false };
          const all = [...rest, sec].sort((a, b) => SECTION_ORDER.indexOf(a.name) - SECTION_ORDER.indexOf(b.name));
          return { sections: all };
        });
      try {
        const r = await call("session.menu", { deadline_ms: MENU_DEADLINE_MS }, (p) => {
          if (p.kind === "section" && p.rev === rev && p.options.length) upsert(p.section, p.options);
        });
        if (r.rev !== get().rev) return;
        const by = new Map<string, MenuOption[]>();
        for (const o of r.options) by.set(o.section, [...(by.get(o.section) ?? []), o]);
        for (const [name, opts] of by) upsert(name, opts);
        for (const [name, t] of Object.entries(r.truncated)) if (t && !by.has(name)) upsert(name, [], true);
      } catch (e) {
        if (get().rev === rev) set({ message: { kind: "error", text: `Suggestions failed: ${(e as Error).message}` } });
      } finally {
        if (get().rev === rev) set({ menuLoading: false });
      }
    },

    async showMore(name) {
      const rev = get().rev;
      const sec = get().sections.find((s) => s.name === name);
      if (!sec) return;
      const pages = sec.truncated ? 1 : sec.pages + 1;
      const patch = (p: Partial<Section>) =>
        set((s) => (s.rev !== rev ? {} : { sections: s.sections.map((x) => (x.name === name ? { ...x, ...p } : x)) }));
      patch({ loading: true });
      try {
        const r = await call("session.menu", { sections: [name], more: pages > 1 ? { [name]: pages } : undefined });
        if (r.rev !== get().rev) return;
        const got = r.options.filter((o) => o.section === name);
        const seen = new Set(sec.options.map((o) => o.description));
        const fresh = got.filter((o) => !seen.has(o.description));
        patch({ options: [...sec.options, ...fresh], truncated: false, pages, loading: false, exhausted: fresh.length === 0 });
      } catch (e) {
        patch({ loading: false });
        set({ message: { kind: "error", text: (e as Error).message } });
      }
    },

    async select(option) {
      if (!option) {
        set({ selected: null, preview: null });
        return;
      }
      const rev = get().rev;
      set({ selected: { kind: "menu", option }, drawing: false, preview: null, previewLoading: true, message: null });
      try {
        const r = await call("session.preview", { step: option.step });
        const s = get();
        if (r.rev === rev && s.selected?.kind === "menu" && s.selected.option === option) set({ preview: r });
      } finally {
        set({ previewLoading: false });
      }
    },

    async apply(step, label) {
      set({ busy: "Folding…", message: null });
      try {
        const r = await call("session.apply", { step });
        if (!r.ok) {
          set({ message: { kind: "error", text: r.text } });
          return false;
        }
        committed(r, label);
        return true;
      } catch (e) {
        set({ message: { kind: "error", text: (e as Error).message } });
        return false;
      } finally {
        set({ busy: null });
      }
    },

    async applySelected() {
      const s = get();
      if (s.selected?.kind === "menu") {
        await get().apply(s.selected.option.step, s.selected.option.description);
        return;
      }
      if (s.selected?.kind === "custom" && s.preview?.step) {
        const d = s.draw;
        const what = d.kind === "through" ? `through ${d.p} and ${d.q}` : d.side === 1 ? `${d.p} onto ${d.q}` : `${d.q} onto ${d.p}`;
        await get().apply(s.preview.step, `${d.op === "valley" ? "Valley" : "Mountain"} fold ${what}`);
      }
    },

    async undo() {
      const tl = get().timeline;
      if (!tl || !tl.steps.length) return;
      set({ busy: "Undoing…", message: null });
      try {
        const r = await call("session.undo", { n: 1 });
        if (!r.ok) {
          set({ message: { kind: "error", text: r.text } });
          return;
        }
        tl.steps.length = r.timeline_len;
        tl.states.length = r.timeline_len + 1;
        getViewer()?.truncate(r.timeline_len);
        useUi.getState().setStep(r.timeline_len);
        set((s) => ({
          rev: r.rev,
          points: r.points,
          steps: s.steps.slice(0, r.timeline_len),
          selected: null,
          preview: null,
          draw: { ...s.draw, p: null, q: null },
          sections: [],
        }));
        syncLin();
        void get().loadMenu();
      } finally {
        set({ busy: null });
      }
    },

    setDrawing(on) {
      set((s) => ({ drawing: on, draw: { ...s.draw, p: null, q: null, side: 1 }, selected: on ? null : s.selected, preview: on ? null : s.preview }));
    },

    tapPoint(label) {
      const s = get();
      if (!s.drawing) set({ drawing: true, selected: null, preview: null });
      const d = get().draw;
      if (!d.p || (d.p && d.q)) set({ draw: { ...d, p: label, q: null, side: 1 }, selected: null, preview: null });
      else if (label === d.p) set({ draw: { ...d, p: null } });
      else {
        set({ draw: { ...d, q: label } });
        void previewCustom();
      }
    },

    setDraw(patch) {
      set((s) => ({ draw: { ...s.draw, ...patch } }));
      const d = get().draw;
      if (d.p && d.q) void previewCustom();
    },

    setView3d(on) {
      set({ view3d: on });
    },

    async myModel() {
      const r = await call("session.steps", {});
      return r.model;
    },

    async exportFiles(kinds) {
      const needsPictures = kinds.some((k) => k === "png_zip" || k === "cp_png");
      set({ busy: needsPictures ? "Loading the picture maker (about 4 MB)…" : "Exporting…", message: null });
      try {
        if (needsPictures) await ensurePython("export");
        set({ busy: "Exporting…" });
        const r = await call("export", { kinds });
        for (const f of r.files) saveFile(f.name, f.bytes as BlobPart, f.mime);
        if (r.errors.length) set({ message: { kind: "error", text: r.errors.map((e) => e.message).join("; ") } });
      } catch (e) {
        set({ message: { kind: "error", text: (e as Error).message } });
      } finally {
        set({ busy: null });
      }
    },
  };

  async function previewCustom() {
    const { draw: d, rev } = get();
    if (!d.p || !d.q) return;
    // "Switch side" for P-onto-Q folds Q onto P instead
    const [p, q] = d.kind === "onto" && d.side === -1 ? [d.q, d.p] : [d.p, d.q];
    const params: CustomParams = { p, q, kind: d.kind, op: d.op, side: d.side, layers: d.layers, preview: true };
    set({ selected: { kind: "custom" }, previewLoading: true, preview: null, message: null });
    try {
      const r = await call("session.custom", params);
      const s = get();
      if (r.rev === rev && JSON.stringify(s.draw) === JSON.stringify(d)) set({ preview: r });
    } catch (e) {
      set({ message: { kind: "error", text: (e as Error).message } });
    } finally {
      set({ previewLoading: false });
    }
  }
});


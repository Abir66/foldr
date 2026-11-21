import { useEffect, useRef, useState } from "react";
import { resetDiagramView, useDiagramReset, usePanZoom } from "./usePanZoom";
import { getRun } from "../lib/examples";
import type { Route } from "../lib/router";
import { useUi, type CanvasView, type Speed } from "../lib/store";
import { creasePattern, DASH, flatView, frame, PAPER } from "../lib/diagrams";
import type { StateJson } from "../lib/timeline";
import { Thumb } from "../ui/Thumb";
import { FoldOverlay } from "../features/fold/FoldOverlay";
import { useFold } from "../features/fold/foldStore";
import { useAi } from "../features/ai/aiStore";
import { LivePill } from "../features/ai/LivePill";
import { ExportMenu } from "../features/export/ExportMenu";
import { getViewer, poseAt, replayStep, resetCamera, scrubTo, stepNext, stepPrev, Viewer } from "../viewer/Viewer";
import type { Source } from "./source";

/** The paper on the canvas (3D, flat, front + back, crease pattern), the view toolbar and the transport. */
export function Stage({ route, source }: { route: Route; source: Source }) {
  const step = useUi((s) => s.step);
  const view = useUi((s) => s.view);
  const live = useAi((s) => route.mode === "ai" && !!s.source && !s.run.finished);
  const { timeline, n } = source;
  const own = route.mode === "fold" && route.own;

  useEffect(() => {
    if (useFold.getState().view3d !== (view !== "flat")) useFold.getState().setView3d(view !== "flat");
  }, [view]);

  const label =
    source.kind === "own" ? `Your paper, step ${step} of ${n}` : source.kind === "ai" ? "The AI's paper" : `${source.label}, step ${step} of ${n}`;
  // the transport covers the bottom of the stage; the camera centres the paper above it
  const [transportH, setTransportH] = useState(0);
  // while a timeline loads, a picture of the finished model (an empty AI canvas shows nothing)
  const thumb = source.example?.name ?? getRun(route.run)?.example;
  const state = timeline?.states[Math.min(step, timeline.states.length - 1)];
  const sceneRef = useRef<HTMLDivElement>(null);
  usePanZoom(
    sceneRef,
    {
      zoom: (x, y, f) => getViewer()?.zoomFlat(x, y, f),
      pan: (dx, dy) => getViewer()?.panFlat(dx, dy),
      reset: resetCamera,
    },
    // the fold overlay owns drags on the paper while folding; panning starts on the bare canvas
    { enabled: view === "flat" && !!timeline, accept: (e) => e.target instanceof HTMLCanvasElement, capture: false },
  );
  return (
    <section className="stage" aria-label="The paper">
      <div className={`stage-scene${view === "flat" && timeline ? " pan-zoom" : ""}`} role="img" aria-label={label} ref={sceneRef}>
        {!timeline && thumb && <Thumb name={thumb} size={320} className="stage-paper" />}
        <Viewer timeline={source.play} params={route.params} topDown={view === "flat"} insetBottom={timeline ? transportH : 0} />
        {own && <FoldOverlay />}
        {state && (view === "sides" || view === "cp") && (
          <Diagram view={view} bottom={transportH}>
            {view === "sides" ? <Sides state={state} coloredSide={source.coloredSide} /> : <Crease state={state} />}
          </Diagram>
        )}
      </div>
      <Toolbar source={source} own={own} />
      {route.mode === "ai" && timeline && <LivePill n={n} />}
      {timeline && <Transport n={n} onHeight={setTransportH} live={live} />}
    </section>
  );
}

/** Sides / Creases overlay: wheel or pinch to zoom, drag to pan, double-click or double-tap to reset. */
function Diagram({ view, bottom, children }: { view: CanvasView; bottom: number; children: React.ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  const [t, setT] = useState({ k: 1, x: 0, y: 0 });
  const reset = () => setT({ k: 1, x: 0, y: 0 });
  useEffect(reset, [view]);
  useDiagramReset(reset);
  usePanZoom(ref, {
    zoom: (cx, cy, f) =>
      setT((t) => {
        const r = ref.current!.getBoundingClientRect();
        const k = Math.min(20, Math.max(0.3, t.k * f));
        const g = k / t.k,
          px = cx - r.left,
          py = cy - r.top;
        return { k, x: px - (px - t.x) * g, y: py - (py - t.y) * g };
      }),
    pan: (dx, dy) => setT((t) => ({ ...t, x: t.x + dx, y: t.y + dy })),
    reset,
  });
  return (
    <div className="diagram pan-zoom" style={{ bottom }} ref={ref}>
      <div className="diagram-pan" style={{ transform: `translate(${t.x}px, ${t.y}px) scale(${t.k})` }}>
        {children}
      </div>
      {view === "cp" && <CreaseLegend />}
    </div>
  );
}

function Sides({ state, coloredSide }: { state: StateJson; coloredSide: "up" | "down" }) {
  return (
    <div className="diagram-sides">
      {([false, true] as const).map((back) => {
        const v = flatView(state, coloredSide, back);
        const f = frame(v.box, 320);
        return (
          <figure key={String(back)} className="diagram-fig">
            <svg viewBox={f.viewBox} aria-label={back ? "Back of the paper" : "Front of the paper"}>
              {v.faces.map((x, i) => (
                <g key={i}>
                  <path d={x.d} fill={x.colored ? PAPER.front : PAPER.back} stroke={PAPER.edge} strokeWidth={f.unit * 1.4} strokeLinejoin="round" />
                  {x.lines.map(([a, b], j) => (
                    <line key={j} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke={PAPER.crease} strokeWidth={f.unit} strokeLinecap="round" />
                  ))}
                </g>
              ))}
            </svg>
            <figcaption>{back ? "Back" : "Front"}</figcaption>
          </figure>
        );
      })}
    </div>
  );
}

function Crease({ state }: { state: StateJson }) {
  const cp = creasePattern(state);
  const f = frame(cp.box, 480);
  const u = f.unit;
  return (
    <figure className="diagram-fig diagram-cp">
      <svg viewBox={f.viewBox} aria-label="Crease pattern: the unfolded sheet with every crease so far">
        <path d={cp.outline} fill={PAPER.back} stroke={PAPER.edge} strokeWidth={u * 2} />
        {cp.creases.map((c, i) => (
          <line
            key={i}
            x1={c.a[0]}
            y1={c.a[1]}
            x2={c.b[0]}
            y2={c.b[1]}
            stroke={c.kind === "mountain" ? PAPER.mountain : c.kind === "valley" ? PAPER.valley : PAPER.crease}
            strokeWidth={c.kind === "flat" ? u : u * 2}
            strokeDasharray={DASH[c.kind].map((d) => d * u).join(" ") || undefined}
            strokeLinecap="round"
          />
        ))}
      </svg>
    </figure>
  );
}

function CreaseLegend() {
  return (
    <div className="cp-legend">
      <span className="cp-key cp-valley">Valley</span>
      <span className="cp-key cp-mountain">Mountain</span>
      <span className="cp-key cp-flat">Unfolded</span>
    </div>
  );
}

const VIEWS: { v: CanvasView; label: string; title: string; icon: JSX.Element }[] = [
  { v: "3d", label: "3D", title: "3D: drag to turn", icon: <path d="M10 2.8 16.5 6.5v7L10 17.2 3.5 13.5v-7zM3.5 6.5 10 10l6.5-3.5M10 10v7.2" /> },
  { v: "flat", label: "Flat", title: "Flat: looking straight down", icon: <path d="M3.5 3.5h13v13h-13z" /> },
  { v: "sides", label: "Sides", title: "Front and back side by side", icon: <path d="M2.5 5h6.5v10H2.5zM11 5h6.5v10H11z" /> },
  { v: "cp", label: "Creases", title: "Crease pattern: the sheet unfolded", icon: <path d="M3.5 3.5h13v13h-13zM3.5 3.5l13 13M16.5 3.5 10 10" /> },
];

function Toolbar({ source, own }: { source: Source; own: boolean }) {
  const view = useUi((s) => s.view);
  const setView = useUi((s) => s.setView);
  const folding = useFold((s) => s.phase === "folding");
  const busy = useFold((s) => !!s.busy);
  const apply = useFold((s) => s.apply);
  const camera = view === "3d" || view === "flat";
  return (
    <div className="toolbar-float" role="toolbar" aria-label="View">
      <div className="tool-seg" role="radiogroup" aria-label="View">
        {VIEWS.map((x) => (
          <button key={x.v} role="radio" aria-checked={view === x.v} className="tool-btn" title={x.title} onClick={() => setView(x.v)}>
            <svg viewBox="0 0 20 20" aria-hidden="true">{x.icon}</svg>
            <span className="tool-label">{x.label}</span>
          </button>
        ))}
      </div>
      <div className="tool-group">
        {source.timeline && (
          <button className="tool-btn" onClick={camera ? resetCamera : resetDiagramView} title="Back to the starting angle and zoom" aria-label="Reset view">
            <svg viewBox="0 0 20 20" aria-hidden="true">
              <path d="M3.5 10a6.5 6.5 0 1 0 2-4.7" />
              <path d="M3.2 3.5v3.2h3.2" />
            </svg>
          </button>
        )}
        {own && folding && (
          <>
            <button className="tool-btn" disabled={busy} onClick={() => apply({ op: "flip" })} title="Turn the paper over (adds a step)" aria-label="Turn over">
              <svg viewBox="0 0 20 20" aria-hidden="true">
                <path d="M4 7h10.5l-3-3M16 13H5.5l3 3" />
              </svg>
              <span className="tool-label">Turn over</span>
            </button>
            <button className="tool-btn" disabled={busy} onClick={() => apply({ op: "rotate", degrees: 45 })} title="Rotate the paper 45° (adds a step)" aria-label="Rotate 45°">
              <svg viewBox="0 0 20 20" aria-hidden="true">
                <path d="M15.5 10a5.5 5.5 0 1 1-1.6-3.9" />
                <path d="M14.5 2.8v3.6h-3.6" />
              </svg>
              <span className="tool-label">45°</span>
            </button>
          </>
        )}
        <ExportMenu source={source} />
      </div>
    </div>
  );
}

const SPEEDS: Speed[] = [0.5, 1, 2];

function Transport({ n, onHeight, live = false }: { n: number; onHeight: (px: number) => void; live?: boolean }) {
  const { step, pos, foldK, playing, speed, setStep, setPlaying, setSpeed } = useUi();
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const stage = el.parentElement!.getBoundingClientRect();
      onHeight(Math.round(stage.bottom - el.getBoundingClientRect().top + 8));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [onHeight]);

  const togglePlay = () => {
    // an AI run still going waits at the end for new folds instead of starting over
    if (!playing && step >= n && !live) setStep(0);
    setPlaying(!playing);
  };
  // Name the fold the same way the step list does: resting after fold k shows "Step k, done",
  // not "Step k+1, 0%" (the viewer parks on the next fold's start).
  let k = Math.min(Math.max(foldK, 0), Math.max(n - 1, 0));
  let t = Math.min(1, Math.max(0, pos - k));
  if (t === 0 && k > 0 && k === step) {
    k -= 1;
    t = 1;
  }
  const pct = Math.round(t * 100);

  return (
    <div className="transport" ref={ref} role="group" aria-label="Playback">
      <div className="t-row">
        <button className="t-btn" onClick={() => { setPlaying(false); setStep(0); }} disabled={pos <= 0} aria-label="Back to the start" title="Back to the start">
          <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 4.5v11M15 4.5 9 10l6 5.5" /></svg>
        </button>
        <button className="t-btn" onClick={stepPrev} disabled={pos <= 0} aria-label="Previous step">
          <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M12.5 4.5 7 10l5.5 5.5" /></svg>
        </button>
        <button className="t-btn t-play" onClick={togglePlay} disabled={n === 0} aria-label={playing ? "Pause" : "Play"}>
          {playing ? (
            <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M7 4.5v11M13 4.5v11" /></svg>
          ) : (
            <svg viewBox="0 0 20 20" aria-hidden="true"><path className="fill" d="M7 4.2v11.6L16 10z" /></svg>
          )}
        </button>
        <button className="t-btn" onClick={() => stepNext(n)} disabled={step >= n} aria-label="Next step">
          <svg viewBox="0 0 20 20" aria-hidden="true"><path d="M7.5 4.5 13 10l-5.5 5.5" /></svg>
        </button>
        <label className="scrub">
          <span className="visually-hidden">Whole model</span>
          <input
            type="range"
            min={0}
            max={n}
            step={0.001}
            value={pos}
            disabled={n === 0}
            aria-valuetext={`Step ${step} of ${n}`}
            onChange={(e) => scrubTo(Number(e.target.value))}
            onKeyDown={(e) => {
              if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
              e.preventDefault();
              if (e.key === "ArrowRight") stepNext(n);
              else stepPrev();
            }}
            style={{ ["--fill" as string]: `${n ? (pos / n) * 100 : 0}%` }}
          />
        </label>
        <span className="t-count" aria-live="polite">
          {step}<span className="t-of">/{n}</span>
        </span>
        <div className="t-speed" role="radiogroup" aria-label="Speed">
          {SPEEDS.map((s) => (
            <button key={s} role="radio" aria-checked={speed === s} className="t-speed-opt" onClick={() => setSpeed(s)}>
              {s === 0.5 ? "½×" : `${s}×`}
            </button>
          ))}
        </div>
      </div>
      {n > 0 && (
        <div className="t-row t-fold">
          <span className="t-fold-label">Step {k + 1}</span>
          <label className="scrub scrub-fold">
            <span className="visually-hidden">Progress of step {k + 1}</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.001}
              value={t}
              aria-valuetext={t >= 1 ? "done" : `${pct}%`}
              onChange={(e) => poseAt(k, Number(e.target.value))}
              style={{ ["--fill" as string]: `${t * 100}%` }}
            />
          </label>
          <span className="t-pct">{t >= 1 ? "done" : `${pct}%`}</span>
          <button className="chip chip-small" onClick={() => replayStep(k + 1)} aria-label={`Replay step ${k + 1}`}>
            <svg viewBox="0 0 20 20" aria-hidden="true">
              <path d="M4 10a6 6 0 1 0 1.8-4.3" />
              <path d="M4.2 3.6v3h3" />
            </svg>
            <span className="hide-tiny">Replay</span>
          </button>
        </div>
      )}
    </div>
  );
}

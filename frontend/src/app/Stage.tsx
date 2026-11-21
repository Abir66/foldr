import { useEffect, useMemo, useRef, useState } from "react";
import { getExample, getRun } from "../lib/examples";
import type { Route } from "../lib/router";
import { useUi, type Speed } from "../lib/store";
import { blankTimeline, useExampleTimeline } from "../lib/timeline";
import { Thumb } from "../ui/Thumb";
import { FoldOverlay } from "../features/fold/FoldOverlay";
import { useFold } from "../features/fold/foldStore";
import { useAi } from "../features/ai/aiStore";
import { LivePill } from "../features/ai/LivePill";
import { poseAt, replayStep, resetCamera, scrubTo, stepNext, stepPrev, Viewer } from "../viewer/Viewer";

const useNarrow = () => {
  const q = "(max-width: 959px)";
  const [narrow, setNarrow] = useState(() => window.matchMedia?.(q).matches ?? false);
  useEffect(() => {
    const mq = window.matchMedia?.(q);
    const on = () => setNarrow(mq.matches);
    mq?.addEventListener?.("change", on);
    return () => mq?.removeEventListener?.("change", on);
  }, []);
  return narrow;
};

/** The 3D paper, camera tools, and (Watch) the playback transport. */
export function Stage({ route }: { route: Route }) {
  const example = getExample(route.example)!;
  const step = useUi((s) => s.step);
  const foldSide = useUi((s) => s.foldSide);
  const narrow = useNarrow();
  let shape = example.name;
  let label = "";
  if (route.mode === "fold") {
    shape = foldSide === "blue" ? "blank_blue" : "blank";
    label = `A blank square of paper, ${foldSide} side up`;
  } else if (route.mode === "ai") {
    shape = getRun(route.run)?.example ?? "hat_behind";
    label = "The AI's paper";
  }
  const aiBase = useAi((s) => (route.mode === "ai" ? s.run.base ?? null : null));
  const aiN = useAi((s) => s.run.steps.length);
  const aiLive = useAi((s) => !!s.source && !s.run.finished);
  const exampleTl = useExampleTimeline(route.mode === "watch" ? shape : null);
  const n = aiBase ? aiN : (exampleTl?.steps.length ?? example.steps.length);
  if (route.mode === "watch") label = `${example.label}, step ${step} of ${n}`;
  const blank = useMemo(() => blankTimeline(foldSide === "blue" ? "up" : "down"), [foldSide]);
  const foldTl = useFold((s) => s.timeline);
  const view3d = useFold((s) => s.view3d);
  const timeline = route.mode === "fold" ? (foldTl ?? blank) : (aiBase ?? (route.mode === "ai" ? null : exampleTl));
  const topDown = route.mode === "fold" && !view3d;
  // the transport covers the bottom of the stage; the camera centres the paper above it
  const [transportH, setTransportH] = useState(0);
  return (
    <section className="stage" aria-label="The paper">
      <div className="stage-scene" role="img" aria-label={label}>
        {!timeline && <Thumb name={shape} size={320} className="stage-paper" />}
        <Viewer
          timeline={timeline}
          params={route.params}
          topDown={topDown}
          insetBottom={route.mode === "watch" || aiBase ? transportH : 0}
        />
        {route.mode === "fold" && <FoldOverlay />}
      </div>
      {!topDown && timeline && <CameraTools narrow={narrow} />}
      {route.mode === "ai" && aiBase && <LivePill n={n} />}
      {(route.mode === "watch" || aiBase) && <Transport n={n} onHeight={setTransportH} live={aiLive} />}
    </section>
  );
}

function CameraTools({ narrow }: { narrow: boolean }) {
  return (
    <div className="stage-tools">
      <p className="stage-hint">
        {narrow ? "Drag to turn · two fingers to move or zoom" : "Drag to turn · right-drag or Shift-drag to move · scroll to zoom"}
      </p>
      <button className="chip" onClick={resetCamera} title="Back to the starting angle and zoom">
        <svg viewBox="0 0 20 20" aria-hidden="true">
          <path d="M3.5 10a6.5 6.5 0 1 0 2-4.7" />
          <path d="M3.2 3.5v3.2h3.2" />
          <circle className="fill" cx="10" cy="10" r="1.6" />
        </svg>
        Reset view
      </button>
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
        <button className="t-btn t-play" onClick={togglePlay} aria-label={playing ? "Pause" : "Play"}>
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
          <span className="t-fold-label">
            Step {k + 1}
          </span>
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

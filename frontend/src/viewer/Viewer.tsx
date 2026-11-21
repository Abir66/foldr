// React wrapper: owns one PaperViewer and keeps it in sync with the playback store
// (step / playing / speed). The viewer drives playback; the store mirrors where it is.
import { useEffect, useRef } from "react";
import { useUi } from "../lib/store";
import type { Timeline } from "../lib/timeline";
import { PaperViewer, type ViewerInfo } from "./PaperViewer";

declare global {
  interface Window {
    /** the mounted viewer, for scripted checks and screenshots */
    __viewer?: PaperViewer & { ready?: boolean };
  }
}

let active: PaperViewer | null = null;

/** The mounted viewer once it has a timeline (null before). */
export function getViewer(): PaperViewer | null {
  return active && active.timeline.states.length ? active : null;
}

/** Next step: animates the fold when the viewer is up, else just moves the store. */
export function stepNext(n: number) {
  const v = getViewer();
  if (v) v.next();
  else useUi.getState().setStep(Math.min(useUi.getState().step + 1, n));
}

export function stepPrev() {
  const v = getViewer();
  if (v) v.prev();
  else useUi.getState().setStep(Math.max(useUi.getState().step - 1, 0));
}

/** Scrub to a continuous position (k + t). */
export function scrubTo(pos: number) {
  const ui = useUi.getState();
  ui.setPlaying(false);
  const v = getViewer();
  if (v) v.scrub(pos);
  else ui.setStep(Math.round(pos));
}

/** Show the current fold (step k, 0-based) frozen at progress t. */
export function poseAt(k: number, t: number) {
  useUi.getState().setPlaying(false);
  getViewer()?.pose(k, t);
}

/** Play step k (1-based) again: from the paper before it to the paper after it. */
export function replayStep(k: number) {
  const v = getViewer();
  if (v) v.replay(k - 1);
  else {
    useUi.getState().setPlaying(false);
    useUi.getState().setStep(k);
  }
}

/** Camera back to the default angle, centred on the paper and following it again. */
export function resetCamera() {
  getViewer()?.resetView();
}

function pair(params: URLSearchParams | undefined, key: string): [number, number] | null {
  const raw = params?.get(key);
  if (!raw) return null;
  const [a, b] = raw.split(",").map(Number);
  return Number.isFinite(a) && Number.isFinite(b) ? [a, b] : null;
}

interface Props {
  timeline: Timeline | null;
  /** ?pose=k,t freezes step k at progress t; ?orbit=a,b sets the view angles (degrees) */
  params?: URLSearchParams;
  /** orthographic camera straight down, no orbit */
  topDown?: boolean;
  /** px at the bottom covered by the transport */
  insetBottom?: number;
  className?: string;
}

export function Viewer({ timeline, params, topDown = false, insetBottom = 0, className }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const ref = useRef<PaperViewer | null>(null);
  // false while the next timeline loads: store changes then belong to it, not to the old one
  const live = useRef(false);

  useEffect(() => {
    const mq = window.matchMedia?.("(prefers-reduced-motion: reduce)");
    let syncing = false;
    const onChange = (info: ViewerInfo) => {
      const s = useUi.getState();
      const patch: Partial<{ step: number; pos: number; foldK: number; playing: boolean }> = {};
      if (s.step !== info.cur) patch.step = info.cur;
      if (s.pos !== info.pos) patch.pos = info.pos;
      if (s.foldK !== info.k) patch.foldK = info.k;
      if (s.playing !== info.playing) patch.playing = info.playing;
      if (!Object.keys(patch).length) return;
      syncing = true;
      useUi.setState(patch);
      syncing = false;
    };
    const v = new PaperViewer(host.current!, { reducedMotion: mq?.matches ?? false, insetBottom, onChange });
    ref.current = v;
    active = v;
    window.__viewer = v;

    const unsub = useUi.subscribe((s, prev) => {
      if (syncing || !live.current) return;
      if (s.speed !== prev.speed) v.setSpeed(s.speed);
      if (s.step !== prev.step) v.goto(s.step, { animate: true });
      if (s.playing !== prev.playing) {
        if (s.playing) v.play({ speed: s.speed });
        else v.pause();
      }
    });
    const onMotion = () => v.setReducedMotion(mq.matches);
    mq?.addEventListener?.("change", onMotion);
    return () => {
      unsub();
      mq?.removeEventListener?.("change", onMotion);
      v.dispose();
      if (active === v) active = null;
      if (window.__viewer === v) window.__viewer = undefined;
      ref.current = null;
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    ref.current?.setInsetBottom(insetBottom);
  }, [insetBottom]);

  useEffect(() => {
    ref.current?.setView({ topDown });
  }, [topDown]);

  useEffect(() => {
    const v = ref.current;
    live.current = !!timeline;
    if (!v || !timeline) return;
    const ui = useUi.getState();
    v.setSpeed(ui.speed);
    v.setTimeline(timeline, { step: ui.step });
    const pose = pair(params, "pose");
    if (pose) v.pose(pose[0], pose[1]);
    const orbit = pair(params, "orbit");
    if (orbit) v.orbit(orbit[0], orbit[1]);
    if (useUi.getState().playing) v.play({ speed: ui.speed });
    (v as PaperViewer & { ready?: boolean }).ready = true;
  }, [timeline]); // eslint-disable-line react-hooks/exhaustive-deps

  return <div ref={host} className={className ?? "stage-canvas"} />;
}

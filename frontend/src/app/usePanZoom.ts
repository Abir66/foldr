import { useEffect, useRef } from "react";

export interface PanZoomHandlers {
  /** zoom by factor f around a screen point (client coords) */
  zoom(clientX: number, clientY: number, f: number): void;
  /** move by a screen-pixel offset */
  pan(dx: number, dy: number): void;
  reset(): void;
}

const TAP_MS = 300;
const TAP_PX = 24;

/**
 * Wheel / pinch zoom, drag pan and double-click or double-tap reset on an element (mouse, pen and touch).
 * accept: which pointer-downs may start a pan; wheel zoom always applies.
 */
export function usePanZoom(
  ref: React.RefObject<HTMLElement>,
  handlers: PanZoomHandlers,
  opts: { enabled?: boolean; accept?: (e: PointerEvent) => boolean; capture?: boolean } = {},
) {
  const h = useRef(handlers);
  h.current = handlers;
  const accept = useRef(opts.accept);
  accept.current = opts.accept;
  const enabled = opts.enabled ?? true;
  const capture = opts.capture ?? true;

  useEffect(() => {
    const el = ref.current;
    if (!el || !enabled) return;
    const pts = new Map<number, { x: number; y: number }>();
    let moved = 0;
    let lastTap = { t: 0, x: 0, y: 0 };

    const wheel = (e: WheelEvent) => {
      e.preventDefault();
      // trackpad pinch arrives as ctrl+wheel with small deltas
      const k = e.ctrlKey ? 0.01 : 0.0015;
      const dy = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
      h.current.zoom(e.clientX, e.clientY, Math.exp(-dy * k));
    };
    const down = (e: PointerEvent) => {
      if (e.pointerType === "mouse" && e.button !== 0) return;
      if (pts.size === 0 && accept.current && !accept.current(e)) return;
      pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (pts.size === 1) moved = 0;
      if (capture) el.setPointerCapture?.(e.pointerId);
    };
    const move = (e: PointerEvent) => {
      const p = pts.get(e.pointerId);
      if (!p) return;
      if (pts.size === 1) {
        h.current.pan(e.clientX - p.x, e.clientY - p.y);
        moved += Math.abs(e.clientX - p.x) + Math.abs(e.clientY - p.y);
      } else {
        const [a, b] = [...pts.entries()].map(([id, q]) => (id === e.pointerId ? { x: e.clientX, y: e.clientY } : q));
        const [a0, b0] = [...pts.values()];
        const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 },
          mid0 = { x: (a0.x + b0.x) / 2, y: (a0.y + b0.y) / 2 };
        const d = Math.hypot(a.x - b.x, a.y - b.y),
          d0 = Math.hypot(a0.x - b0.x, a0.y - b0.y);
        h.current.pan(mid.x - mid0.x, mid.y - mid0.y);
        if (d0 > 1) h.current.zoom(mid.x, mid.y, d / d0);
        moved += TAP_PX;
      }
      p.x = e.clientX;
      p.y = e.clientY;
    };
    const up = (e: PointerEvent) => {
      if (!pts.delete(e.pointerId)) return;
      if (pts.size > 0 || moved > 6) return;
      const now = performance.now();
      if (now - lastTap.t < TAP_MS && Math.hypot(e.clientX - lastTap.x, e.clientY - lastTap.y) < TAP_PX) {
        h.current.reset();
        lastTap = { t: 0, x: 0, y: 0 };
      } else lastTap = { t: now, x: e.clientX, y: e.clientY };
    };
    const cancel = (e: PointerEvent) => void pts.delete(e.pointerId);
    el.addEventListener("wheel", wheel, { passive: false });
    el.addEventListener("pointerdown", down);
    el.addEventListener("pointermove", move);
    el.addEventListener("pointerup", up);
    el.addEventListener("pointercancel", cancel);
    if (!capture) el.addEventListener("pointerleave", cancel);
    return () => {
      el.removeEventListener("wheel", wheel);
      el.removeEventListener("pointerdown", down);
      el.removeEventListener("pointermove", move);
      el.removeEventListener("pointerup", up);
      el.removeEventListener("pointercancel", cancel);
      el.removeEventListener("pointerleave", cancel);
    };
  }, [ref, enabled, capture]);
}

const resetters = new Set<() => void>();

/** Reset every mounted diagram's zoom and pan (the toolbar's reset-view button). */
export function resetDiagramView() {
  resetters.forEach((f) => f());
}

export function useDiagramReset(f: () => void) {
  const fn = useRef(f);
  fn.current = f;
  useEffect(() => {
    const g = () => fn.current();
    resetters.add(g);
    return () => void resetters.delete(g);
  }, []);
}

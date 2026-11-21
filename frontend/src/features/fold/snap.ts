// Snapping a screen position to the paper (labeled point, edge, or anywhere on a face) and the
// live "fold P onto Q" line. Pure geometry so it can run every frame without the worker.
import type { StateJson, Vec2 } from "../../lib/timeline";

/** A fold point: a label from `points`, or folded engine coords. */
export type PointRef = string | Vec2;

export type SnapKind = "point" | "edge" | "free";

export interface Snap {
  kind: SnapKind;
  ref: PointRef;
  /** engine coords */
  xy: Vec2;
}

export interface SnapInput {
  labels: { label: string; xy: Vec2 }[];
  polys: Vec2[][];
  toScreen(p: Vec2): Vec2;
  fromScreen(s: Vec2): Vec2 | null;
  pointPx: number;
  edgePx: number;
}

export const round4 = (v: number) => Math.round(v * 1e4) / 1e4 + 0;
const r4 = (p: Vec2): Vec2 => [round4(p[0]), round4(p[1])];

export function samePoint(a: PointRef | null, b: PointRef | null): boolean {
  if (a === null || b === null) return false;
  if (typeof a === "string" || typeof b === "string") return a === b;
  return Math.abs(a[0] - b[0]) < 1e-6 && Math.abs(a[1] - b[1]) < 1e-6;
}

export function refName(r: PointRef): string {
  return typeof r === "string" ? r : `(${r[0].toFixed(2)}, ${r[1].toFixed(2)})`;
}

/** Face outlines of a state in folded engine coords. */
export function facePolys(state: StateJson | undefined): Vec2[][] {
  if (!state) return [];
  return state.faces.map((f) =>
    f.orig.map((o) => [f.T[0][0] * o[0] + f.T[0][1] * o[1] + f.T[0][2], f.T[1][0] * o[0] + f.T[1][1] * o[1] + f.T[1][2]] as Vec2),
  );
}

export function insidePoly(p: Vec2, poly: Vec2[]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i],
      [xj, yj] = poly[j];
    if (yi > p[1] !== yj > p[1] && p[0] < ((xj - xi) * (p[1] - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/** Labeled point within pointPx, else nearest face edge within edgePx, else a point inside a face, else null. */
export function snapAt(s: Vec2, inp: SnapInput): Snap | null {
  let best: Snap | null = null,
    bd = inp.pointPx;
  for (const l of inp.labels) {
    const [x, y] = inp.toScreen(l.xy);
    const d = Math.hypot(x - s[0], y - s[1]);
    if (d < bd) {
      bd = d;
      best = { kind: "point", ref: l.label, xy: l.xy };
    }
  }
  if (best) return best;
  bd = inp.edgePx;
  for (const poly of inp.polys) {
    for (let i = 0; i < poly.length; i++) {
      const A = poly[i],
        B = poly[(i + 1) % poly.length];
      const a = inp.toScreen(A),
        b = inp.toScreen(B);
      const dx = b[0] - a[0],
        dy = b[1] - a[1];
      const len2 = dx * dx + dy * dy;
      if (len2 < 1e-9) continue;
      const t = Math.max(0, Math.min(1, ((s[0] - a[0]) * dx + (s[1] - a[1]) * dy) / len2));
      const d = Math.hypot(a[0] + t * dx - s[0], a[1] + t * dy - s[1]);
      if (d < bd) {
        bd = d;
        const xy = r4([A[0] + t * (B[0] - A[0]), A[1] + t * (B[1] - A[1])]);
        best = { kind: "edge", ref: xy, xy };
      }
    }
  }
  if (best) return best;
  const e = inp.fromScreen(s);
  if (e && inp.polys.some((poly) => insidePoly(e, poly))) {
    const xy = r4(e);
    return { kind: "free", ref: xy, xy };
  }
  return null;
}

/** Perpendicular bisector of p -> q (the crease that lays p onto q), clipped to the paper's outline. */
export function bisectorSegment(p: Vec2, q: Vec2, polys: Vec2[][]): [Vec2, Vec2] | null {
  const dx = q[0] - p[0],
    dy = q[1] - p[1];
  if (Math.hypot(dx, dy) < 1e-9) return null;
  const m: Vec2 = [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2];
  const d: Vec2 = [-dy, dx];
  let lo = Infinity,
    hi = -Infinity;
  for (const poly of polys) {
    for (let i = 0; i < poly.length; i++) {
      const A = poly[i],
        B = poly[(i + 1) % poly.length];
      const ex = B[0] - A[0],
        ey = B[1] - A[1];
      const den = d[0] * ey - d[1] * ex;
      if (Math.abs(den) < 1e-12) continue;
      const wx = A[0] - m[0],
        wy = A[1] - m[1];
      const t = (wx * ey - wy * ex) / den;
      const u = (wx * d[1] - wy * d[0]) / den;
      if (u < -1e-9 || u > 1 + 1e-9) continue;
      lo = Math.min(lo, t);
      hi = Math.max(hi, t);
    }
  }
  if (!(hi - lo > 1e-9)) return null;
  return [
    [m[0] + lo * d[0], m[1] + lo * d[1]],
    [m[0] + hi * d[0], m[1] + hi * d[1]],
  ];
}

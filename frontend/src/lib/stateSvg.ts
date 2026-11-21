// Flat SVG picture of a paper state (as the fold menu returns it): every face mapped by its T,
// painted bottom layer first, coloured by which side of the sheet faces the viewer.
import type { Affine, StateJson, Vec2 } from "./timeline";

/** 2x2 map from engine coords to screen-like coords (y down). Default: x right, y up. */
export type Lin2 = [[number, number], [number, number]];
export const ENGINE_TO_SCREEN: Lin2 = [
  [1, 0],
  [0, -1],
];

export interface SvgFace {
  d: string;
  colored: boolean;
}

export interface StateSvg {
  viewBox: string;
  faces: SvgFace[];
  /** stroke width that looks like ~1px at the picture's size */
  unit: number;
}

const apply = (T: Affine, p: Vec2): Vec2 => [T[0][0] * p[0] + T[0][1] * p[1] + T[0][2], T[1][0] * p[0] + T[1][1] * p[1] + T[1][2]];
const lin = (A: Lin2, p: Vec2): Vec2 => [A[0][0] * p[0] + A[0][1] * p[1], A[1][0] * p[0] + A[1][1] * p[1]];
const r = (v: number) => Math.round(v * 1e4) / 1e4;

export function stateSvg(state: StateJson, coloredSide: "up" | "down", A: Lin2 = ENGINE_TO_SCREEN, px = 64): StateSvg {
  const faces = [...state.faces].sort((a, b) => a.layer - b.layer);
  let lo: Vec2 = [Infinity, Infinity];
  let hi: Vec2 = [-Infinity, -Infinity];
  const out: SvgFace[] = [];
  for (const f of faces) {
    const pts = f.orig.map((p) => lin(A, apply(f.T, p)));
    for (const [x, y] of pts) {
      lo = [Math.min(lo[0], x), Math.min(lo[1], y)];
      hi = [Math.max(hi[0], x), Math.max(hi[1], y)];
    }
    out.push({ d: "M" + pts.map(([x, y]) => `${r(x)} ${r(y)}`).join("L") + "Z", colored: f.front_up === (coloredSide === "up") });
  }
  if (!isFinite(lo[0])) return { viewBox: "0 0 1 1", faces: [], unit: 1 / px };
  const span = Math.max(hi[0] - lo[0], hi[1] - lo[1], 1e-6);
  const pad = span * 0.06;
  const unit = (span + 2 * pad) / px;
  const cx = (lo[0] + hi[0]) / 2,
    cy = (lo[1] + hi[1]) / 2,
    h = span / 2 + pad;
  return { viewBox: `${r(cx - h)} ${r(cy - h)} ${r(2 * h)} ${r(2 * h)}`, faces: out, unit };
}

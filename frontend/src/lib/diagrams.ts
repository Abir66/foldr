// Flat diagrams of one paper state, built from timeline data: the front and back views and the
// unfolded crease pattern. Output is plain data (paths + segments) so React and SVG export share it.
import { ENGINE_TO_SCREEN, type Lin2 } from "./stateSvg";
import type { Affine, StateJson, Vec2 } from "./timeline";

export const PAPER = { front: "#e07a3f", back: "#fbf6ec", edge: "#3a2a1a", crease: "#5c3d24", valley: "#2a211a", mountain: "#b3261e" };

const apply = (T: Affine, p: Vec2): Vec2 => [T[0][0] * p[0] + T[0][1] * p[1] + T[0][2], T[1][0] * p[0] + T[1][1] * p[1] + T[1][2]];
const lin = (A: Lin2, p: Vec2): Vec2 => [A[0][0] * p[0] + A[0][1] * p[1], A[1][0] * p[0] + A[1][1] * p[1]];
const r = (v: number) => Math.round(v * 1e4) / 1e4;
const path = (pts: Vec2[]) => "M" + pts.map(([x, y]) => `${r(x)} ${r(y)}`).join("L") + "Z";

export interface Box {
  lo: Vec2;
  hi: Vec2;
}

export interface FlatView {
  /** in paint order (farthest layer first); each face carries the creases visible on it */
  faces: { d: string; colored: boolean; lines: [Vec2, Vec2][] }[];
  box: Box;
}

function grow(b: Box, [x, y]: Vec2) {
  b.lo = [Math.min(b.lo[0], x), Math.min(b.lo[1], y)];
  b.hi = [Math.max(b.hi[0], x), Math.max(b.hi[1], y)];
}

/**
 * The folded paper seen from above (back = false) or from below (back = true, mirrored left-right,
 * layers painted in reverse). Faces come bottom-up in timeline order, like the exported step images,
 * so with opaque fills each layer hides the edges and creases under it.
 * colored = the coloured side of the sheet faces the viewer.
 */
export function flatView(state: StateJson, coloredSide: "up" | "down", back = false, A: Lin2 = ENGINE_TO_SCREEN): FlatView {
  const M: Lin2 = back ? [[-A[0][0], -A[0][1]], A[1]] : A;
  const faces = back ? [...state.faces].reverse() : state.faces;
  const box: Box = { lo: [Infinity, Infinity], hi: [-Infinity, -Infinity] };
  const out: FlatView["faces"] = [];
  for (const f of faces) {
    const pts = f.orig.map((p) => lin(M, apply(f.T, p)));
    pts.forEach((p) => grow(box, p));
    const up = f.front_up === (coloredSide === "up");
    const lines = (f.lines ?? []).map(([a, b]) => [lin(M, apply(f.T, a)), lin(M, apply(f.T, b))] as [Vec2, Vec2]);
    out.push({ d: path(pts), colored: back ? !up : up, lines });
  }
  return { faces: out, box };
}

export type CreaseKind = "valley" | "mountain" | "flat";

export interface CreasePattern {
  outline: string;
  creases: { a: Vec2; b: Vec2; kind: CreaseKind }[];
  box: Box;
}

const EPS = 1e-6;

/** Overlap of two segments when they lie on one line; null otherwise. */
function overlap(p: Vec2, q: Vec2, s: Vec2, t: Vec2): [Vec2, Vec2] | null {
  const dx = q[0] - p[0],
    dy = q[1] - p[1];
  const len2 = dx * dx + dy * dy;
  if (len2 < EPS * EPS) return null;
  const cross = (v: Vec2) => (dx * (v[1] - p[1]) - dy * (v[0] - p[0])) / Math.sqrt(len2);
  if (Math.abs(cross(s)) > 1e-5 || Math.abs(cross(t)) > 1e-5) return null;
  const u = (v: Vec2) => (dx * (v[0] - p[0]) + dy * (v[1] - p[1])) / len2;
  const lo = Math.max(0, Math.min(u(s), u(t))),
    hi = Math.min(1, Math.max(u(s), u(t)));
  if (hi - lo < 1e-5) return null;
  return [
    [p[0] + dx * lo, p[1] + dy * lo],
    [p[0] + dx * hi, p[1] + dy * hi],
  ];
}

/**
 * The unfolded sheet, seen from the side that faced up at the start, with every crease made so far.
 * A shared edge between two faces is folded when the faces lie with opposite sides up: valley when the
 * face still front-up is the lower one (front sides meet), mountain otherwise. Face lines are creases
 * folded and undone.
 */
export function creasePattern(state: StateJson, A: Lin2 = ENGINE_TO_SCREEN): CreasePattern {
  const box: Box = { lo: [Infinity, Infinity], hi: [-Infinity, -Infinity] };
  const edges: { a: Vec2; b: Vec2; f: number }[] = [];
  state.faces.forEach((f, i) => {
    f.orig.forEach((p, j) => {
      grow(box, p);
      edges.push({ a: p, b: f.orig[(j + 1) % f.orig.length], f: i });
    });
  });
  const creases: CreasePattern["creases"] = [];
  for (let i = 0; i < edges.length; i++) {
    for (let j = i + 1; j < edges.length; j++) {
      const e = edges[i],
        g = edges[j];
      if (e.f === g.f) continue;
      const seg = overlap(e.a, e.b, g.a, g.b);
      if (!seg) continue;
      const fa = state.faces[e.f],
        fb = state.faces[g.f];
      let kind: CreaseKind = "flat";
      if (fa.front_up !== fb.front_up) {
        const [up, down] = fa.front_up ? [fa, fb] : [fb, fa];
        kind = up.layer < down.layer ? "valley" : "mountain";
      }
      creases.push({ a: seg[0], b: seg[1], kind });
    }
  }
  const seen = new Set<string>();
  for (const f of state.faces)
    for (const [a, b] of f.lines ?? []) {
      const key = [a, b].map((p) => p.map((v) => v.toFixed(4)).join(",")).sort().join("|");
      if (seen.has(key) || creases.some((c) => overlap(c.a, c.b, a, b))) continue;
      seen.add(key);
      creases.push({ a, b, kind: "flat" });
    }
  const out: CreasePattern = { outline: "", creases: [], box: { lo: [Infinity, Infinity], hi: [-Infinity, -Infinity] } };
  const corners: Vec2[] = [box.lo, [box.hi[0], box.lo[1]], box.hi, [box.lo[0], box.hi[1]]].map((p) => lin(A, p as Vec2));
  corners.forEach((p) => grow(out.box, p));
  out.outline = isFinite(box.lo[0]) ? path(corners) : "";
  out.creases = creases.map((c) => ({ a: lin(A, c.a), b: lin(A, c.b), kind: c.kind }));
  return out;
}

/** Square viewBox around a box with a margin; unit = one pixel at `px` wide. */
export function frame(b: Box, px: number, pad = 0.06): { viewBox: string; unit: number } {
  if (!isFinite(b.lo[0])) return { viewBox: "0 0 1 1", unit: 1 / px };
  const span = Math.max(b.hi[0] - b.lo[0], b.hi[1] - b.lo[1], 1e-6);
  const h = span / 2 + span * pad;
  const cx = (b.lo[0] + b.hi[0]) / 2,
    cy = (b.lo[1] + b.hi[1]) / 2;
  return { viewBox: `${r(cx - h)} ${r(cy - h)} ${r(2 * h)} ${r(2 * h)}`, unit: (2 * h) / px };
}

const seg = (a: Vec2, b: Vec2) => `x1="${r(a[0])}" y1="${r(a[1])}" x2="${r(b[0])}" y2="${r(b[1])}"`;

/** Dash pattern (in px) per crease kind: valley dashed, mountain dash-dot. */
export const DASH: Record<CreaseKind, number[]> = { valley: [8, 5], mountain: [10, 4, 2, 4], flat: [] };

function flatSvgBody(v: FlatView, unit: number): string {
  return v.faces
    .map(
      (x) =>
        `<path d="${x.d}" fill="${x.colored ? PAPER.front : PAPER.back}" stroke="${PAPER.edge}" stroke-width="${r(unit * 1.4)}" stroke-linejoin="round"/>` +
        x.lines.map(([a, b]) => `<line ${seg(a, b)} stroke="${PAPER.crease}" stroke-width="${r(unit)}" stroke-linecap="round"/>`).join(""),
    )
    .join("");
}

/** Front and back side by side, as a standalone SVG document. */
export function sidesSvgText(state: StateJson, coloredSide: "up" | "down", A?: Lin2, px = 600): string {
  const parts = [false, true].map((back, i) => {
    const v = flatView(state, coloredSide, back, A);
    const fr = frame(v.box, px);
    const label = back ? "Back" : "Front";
    return `<svg x="${i * px}" y="0" width="${px}" height="${px}" viewBox="${fr.viewBox}">${flatSvgBody(v, fr.unit)}</svg><text x="${i * px + px / 2}" y="${px + 28}" text-anchor="middle" font-family="sans-serif" font-size="20" fill="#3a2a1a">${label}</text>`;
  });
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${2 * px}" height="${px + 44}" viewBox="0 0 ${2 * px} ${px + 44}"><rect width="100%" height="100%" fill="#ffffff"/>${parts.join("")}</svg>`;
}

/** The crease pattern as a standalone SVG document. */
export function cpSvgText(state: StateJson, A?: Lin2, px = 800): string {
  const cp = creasePattern(state, A);
  const fr = frame(cp.box, px);
  const u = fr.unit;
  const lines = cp.creases
    .map((c) => {
      const color = c.kind === "mountain" ? PAPER.mountain : c.kind === "valley" ? PAPER.valley : PAPER.crease;
      const dash = DASH[c.kind].length ? ` stroke-dasharray="${DASH[c.kind].map((d) => r(d * u)).join(" ")}"` : "";
      const w = c.kind === "flat" ? u : u * 2;
      return `<line ${seg(c.a, c.b)} stroke="${color}" stroke-width="${r(w)}"${dash} stroke-linecap="round"/>`;
    })
    .join("");
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${px}" height="${px}" viewBox="${fr.viewBox}"><rect x="-1e3" y="-1e3" width="2e3" height="2e3" fill="#ffffff"/><path d="${cp.outline}" fill="${PAPER.back}" stroke="${PAPER.edge}" stroke-width="${r(u * 2)}"/>${lines}</svg>`;
}

/** Rasterise an SVG document to a PNG blob. */
export function svgToPng(svg: string): Promise<Blob> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(new Blob([svg], { type: "image/svg+xml" }));
    img.onload = () => {
      const c = document.createElement("canvas");
      c.width = img.naturalWidth || 800;
      c.height = img.naturalHeight || 800;
      c.getContext("2d")!.drawImage(img, 0, 0);
      URL.revokeObjectURL(url);
      c.toBlob((b) => (b ? resolve(b) : reject(new Error("could not make the picture"))), "image/png");
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("could not draw the picture"));
    };
    img.src = url;
  });
}

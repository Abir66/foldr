import { useEffect, useState } from "react";
import { exampleFileUrl, getExample } from "./examples";

// Timeline JSON: what origami's steps_export / TimelineBuilder writes and the 3D viewer plays.
// Points are 2D [x, y]; T is a 2x3 affine map (orig -> folded engine coords).

export type Vec2 = [number, number];
export type Affine = number[][]; // [[a, b, tx], [c, d, ty]]

/** Engine coords -> world coords. A flip mirrors M and negates s, so the paper never moves. */
export interface WorldMap {
  M: Affine;
  s: number;
  zoff: number;
}

export interface FaceJson {
  id: number;
  orig: Vec2[];
  T: Affine;
  /** compact stacking height in layer units */
  layer: number;
  front_up: boolean;
  lines: [Vec2, Vec2][];
}

/** A crease where the sheet wraps around between two stacked faces. */
export interface BendJson {
  a: number;
  b: number;
  seg: [Vec2, Vec2];
}

export interface StateJson {
  faces: FaceJson[];
  bends?: BendJson[];
  bbox: [Vec2, Vec2];
  world: WorldMap;
}

export interface RootMotion {
  type: "hinge" | "rotate" | "translate";
  p?: Vec2;
  d?: Vec2;
  angle?: number;
  degrees?: number;
  center?: Vec2;
  t?: Vec2;
}

export interface PieceJson {
  orig: Vec2[];
  lines?: [Vec2, Vec2][];
  Tb: Affine;
  parent: number;
  root_motion?: RootMotion | null;
  /** root piece height (layer units) at t = 0 / 1 */
  h: [number, number];
  /** hinge to the parent: [point, direction] in the parent's orig frame */
  hinge: [Vec2, Vec2];
  theta: [number, number];
  delta: [number, number];
  win: [number, number];
  /** solved hinge angles sampled uniformly in the path parameter */
  keys?: number[];
  /** scripted frames: [R 3x3 row-major, t (3), w (3)] per sample */
  fkeys?: number[][];
}

export interface KinBendJson {
  f: number;
  o: number;
  seg: [Vec2, Vec2];
  n: Vec2;
}

export interface AxisJson {
  seg: [Vec2, Vec2];
  sign: number;
  phase: number;
}

export interface StepJson {
  index: number;
  op: string;
  description: string;
  axis?: [Vec2, Vec2] | null;
  arrow?: { from: Vec2; to: Vec2 } | null;
  degrees?: number | null;
  motions?: unknown[] | null;
  phases?: number;
  axes?: AxisJson[];
  pieces: PieceJson[];
  kin_order: number[];
  kin_bends?: KinBendJson[];
  retime?: number[];
  path_len?: number;
  side_before: number;
  side_after: number;
}

export interface Timeline {
  name: string;
  title?: string;
  size?: number;
  colored_side?: "up" | "down";
  original_faces?: { id: number; orig: Vec2[] }[];
  /** states[k] = the paper after k steps; states.length = steps.length + 1 */
  states: StateJson[];
  steps: StepJson[];
}

const ID: Affine = [
  [1, 0, 0],
  [0, 1, 0],
];

/** A flat unfolded square, the same shape TimelineBuilder starts from. */
export function blankTimeline(coloredSide: "up" | "down" = "up", size = 1): Timeline {
  const orig: Vec2[] = [
    [0, 0],
    [size, 0],
    [size, size],
    [0, size],
  ];
  return {
    name: "blank",
    title: "Blank square",
    size,
    colored_side: coloredSide,
    original_faces: [{ id: 0, orig }],
    states: [
      {
        bends: [],
        faces: [{ id: 0, orig, T: ID, layer: 0, front_up: true, lines: [] }],
        bbox: [
          [0, 0],
          [size, size],
        ],
        world: { M: ID, s: 1, zoff: 0 },
      },
    ],
    steps: [],
  };
}

const cache = new Map<string, Promise<Timeline>>();

/** Register a timeline built in the page (own fold, uploaded file) under an example name. */
export function putTimeline(name: string, tl: Timeline): void {
  cache.set(name, Promise.resolve(tl));
}

/** The precomputed timeline of an example (cached). */
export function loadExampleTimeline(name: string): Promise<Timeline> {
  const example = getExample(name);
  if (!example) return Promise.reject(new Error(`unknown example ${name}`));
  let p = cache.get(name);
  if (!p) {
    p = fetch(exampleFileUrl(example, "timeline")).then((r) => {
      if (!r.ok) throw new Error(`timeline ${name}: HTTP ${r.status}`);
      return r.json() as Promise<Timeline>;
    });
    p.catch(() => cache.delete(name));
    cache.set(name, p);
  }
  return p;
}

/** The timeline of an example; null while it loads (or if it is missing). */
export function useExampleTimeline(name: string | null): Timeline | null {
  const [got, setGot] = useState<{ name: string; tl: Timeline } | null>(null);
  useEffect(() => {
    if (!name) return;
    let alive = true;
    loadExampleTimeline(name).then(
      (tl) => alive && setGot({ name, tl }),
      (err) => console.warn(err),
    );
    return () => {
      alive = false;
    };
  }, [name]);
  return got && got.name === name ? got.tl : null;
}

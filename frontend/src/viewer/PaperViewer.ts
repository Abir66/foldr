// 3D paper viewer: plays a timeline (states + steps from origami's steps_export) with three.js.
// Framework-free; React mounts it through <Viewer>. The animation math is a line-by-line port of
// the original viewer_template.html (hinge-tree kinematics, solved keys + retime, scripted
// spread/tuck frames, flip = camera swing, bend strips, fold-line and arrow hints).
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { Line2 } from "three/examples/jsm/lines/Line2.js";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";
import { LineGeometry } from "three/examples/jsm/lines/LineGeometry.js";
import type { Affine, KinBendJson, PieceJson, StateJson, StepJson, Timeline, Vec2, WorldMap } from "../lib/timeline";

type V3 = [number, number, number];

export interface ViewerColors {
  front: number;
  back: number;
  edge: number;
  crease: number;
  valley: number;
  mountain: number;
  unfold: number;
  flip: number;
  label: string;
}

export const DEFAULT_COLORS: ViewerColors = {
  front: 0xe07a3f,
  back: 0xfbf6ec,
  edge: 0x3a2a1a,
  crease: 0x5c3d24,
  valley: 0x2a211a,
  mountain: 0xb3261e,
  unfold: 0x6b6b6b,
  flip: 0x444444,
  label: "#1E2422",
};

/** Where playback is. pos = k + t runs continuously from 0 (flat) to n (finished). */
export interface ViewerInfo {
  /** number of finished steps */
  cur: number;
  n: number;
  /** step being shown (0-based) and its progress */
  k: number;
  t: number;
  pos: number;
  busy: boolean;
  playing: boolean;
}

export interface ViewerOptions {
  colors?: Partial<ViewerColors>;
  /** height between stacked layers, in paper sizes */
  layerGap?: number;
  /** px at the bottom covered by other UI; the model is centred in the area above it */
  insetBottom?: number;
  /** jump between steps instead of animating (prefers-reduced-motion) */
  reducedMotion?: boolean;
  onChange?: (info: ViewerInfo) => void;
}

export interface PlayOptions {
  speed?: number;
  /** keep waiting at the end and animate steps appended later (live runs) */
  follow?: boolean;
}

const PRE = 280; // ms showing the fold line before moving
const HINT_RENDER = 20;
const BK = 10;

const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const ap = (T: Affine, p: Vec2): Vec2 => [
  T[0][0] * p[0] + T[0][1] * p[1] + T[0][2],
  T[1][0] * p[0] + T[1][1] * p[1] + T[1][2],
];
const nPhases = (step: StepJson) => (step.motions ? Math.max(1, step.phases || 1) : 1);
const winEase = (t: number, w: [number, number]) =>
  w[1] - w[0] < 1e-12 ? (t >= w[1] ? 1 : 0) : ease(Math.min(1, Math.max(0, (t - w[0]) / (w[1] - w[0]))));
// table lookup: tab sampled uniformly on [0, 1]
function lerpTab(tab: number[], x: number) {
  const S = tab.length - 1,
    v = Math.min(S, Math.max(0, x * S)),
    i = Math.min(S - 1, Math.floor(v));
  return tab[i] + (tab[i + 1] - tab[i]) * (v - i);
}
// path parameter u at linear progress t: eased, then (solved steps) retimed so the paper moves at an even pace
const pathU = (step: StepJson, t: number) => (step.retime ? lerpTab(step.retime, ease(t)) : ease(t));
function hingeAngle(pc: PieceJson, t: number, u: number) {
  if (pc.keys) return lerpTab(pc.keys, u); // solved angles (loops kept closed), sampled uniformly in u
  const w = pc.win,
    f = w[0] <= 0 && w[1] >= 1 ? u : winEase(t, w);
  return pc.theta[0] + (pc.theta[1] - pc.theta[0]) * f;
}
const hintOpacity = (step: StepJson, t: number) => Math.max(0, 1 - 3 * ease(Math.min(1, t * nPhases(step)))); // fades with the first phase

const _ax = new THREE.Vector3(),
  _m = new THREE.Matrix4(),
  _p = new THREE.Vector3();
function frameE3(T: Affine) {
  const det = T[0][0] * T[1][1] - T[0][1] * T[1][0] > 0 ? 1 : -1;
  return new THREE.Matrix4().set(T[0][0], T[0][1], 0, T[0][2], T[1][0], T[1][1], 0, T[1][2], 0, 0, det, 0, 0, 0, 0, 1);
}
// rotation by th about the line (p, d) in z = 0
function hingeMat(p: Vec2, d: Vec2, th: number, out = new THREE.Matrix4()) {
  _ax.set(d[0], d[1], 0).normalize();
  out.makeRotationAxis(_ax, th);
  _p.set(p[0], p[1], 0).applyMatrix4(out);
  out.elements[12] = p[0] - _p.x;
  out.elements[13] = p[1] - _p.y;
  out.elements[14] = -_p.z;
  return out;
}
const zMat = (z: number) => new THREE.Matrix4().makeTranslation(0, 0, z);

// ----- flat diagram arrow geometry (polylines in world coords) -----
function resample(pts: V3[], step: number): V3[] {
  const out: V3[] = [pts[0]];
  let acc = 0;
  for (let i = 1; i < pts.length; i++) {
    const p = pts[i - 1],
      q = pts[i],
      L = Math.hypot(q[0] - p[0], q[1] - p[1]);
    let t = step - acc;
    while (t <= L) {
      const f = t / L;
      out.push([p[0] + (q[0] - p[0]) * f, p[1] + (q[1] - p[1]) * f, p[2] + (q[2] - p[2]) * f]);
      t += step;
    }
    acc = (acc + L) % step;
  }
  const last = pts[pts.length - 1],
    o = out[out.length - 1];
  if (Math.hypot(last[0] - o[0], last[1] - o[1]) > step * 0.3) out.push(last);
  return out;
}
function pathLen(pts: number[][]) {
  let L = 0;
  for (let i = 1; i < pts.length; i++) L += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
  return L;
}
// part of a fine polyline between arc lengths s0..s1
function subPath(pts: V3[], s0: number, s1: number): V3[] {
  const out: V3[] = [];
  let acc = 0;
  const at = (p: V3, q: V3, f: number): V3 => [p[0] + (q[0] - p[0]) * f, p[1] + (q[1] - p[1]) * f, p[2] + (q[2] - p[2]) * f];
  for (let i = 1; i < pts.length; i++) {
    const p = pts[i - 1],
      q = pts[i],
      L = Math.hypot(q[0] - p[0], q[1] - p[1]);
    if (acc + L >= s0 && acc <= s1 && L > 0) {
      const f0 = Math.max(0, (s0 - acc) / L),
        f1 = Math.min(1, (s1 - acc) / L);
      if (!out.length) out.push(at(p, q, f0));
      out.push(at(p, q, f1));
    }
    acc += L;
  }
  return out;
}
function ribbonTris(pts: V3[], w: number): V3[] {
  const tris: V3[] = [],
    off: Vec2[] = [];
  for (let i = 0; i < pts.length; i++) {
    const p = pts[Math.max(0, i - 1)],
      q = pts[Math.min(pts.length - 1, i + 1)];
    let dx = q[0] - p[0],
      dy = q[1] - p[1];
    const d = Math.hypot(dx, dy) || 1;
    dx /= d;
    dy /= d;
    off.push([(-dy * w) / 2, (dx * w) / 2]);
  }
  for (let i = 0; i + 1 < pts.length; i++) {
    const a = pts[i],
      b = pts[i + 1],
      oa = off[i],
      ob = off[i + 1];
    const A: V3 = [a[0] + oa[0], a[1] + oa[1], a[2]],
      B: V3 = [a[0] - oa[0], a[1] - oa[1], a[2]];
    const C: V3 = [b[0] + ob[0], b[1] + ob[1], b[2]],
      D: V3 = [b[0] - ob[0], b[1] - ob[1], b[2]];
    tris.push(A, B, C, B, D, C);
  }
  return tris;
}
function headTris(tip: V3, dir: Vec2, hl: number, hw: number, hollow: boolean): V3[] {
  const d = Math.hypot(dir[0], dir[1]) || 1,
    ux = dir[0] / d,
    uy = dir[1] / d,
    z = tip[2];
  const P: V3[] = [
    [tip[0], tip[1], z],
    [tip[0] - ux * hl - uy * hw, tip[1] - uy * hl + ux * hw, z],
    [tip[0] - ux * hl + uy * hw, tip[1] - uy * hl - ux * hw, z],
  ];
  if (!hollow) return P;
  const cx = (P[0][0] + P[1][0] + P[2][0]) / 3,
    cy = (P[0][1] + P[1][1] + P[2][1]) / 3,
    k = 0.45;
  const I: V3[] = P.map((p) => [cx + (p[0] - cx) * k, cy + (p[1] - cy) * k, z]);
  const tris: V3[] = [];
  for (let i = 0; i < 3; i++) {
    const j = (i + 1) % 3;
    tris.push(P[i], P[j], I[i], P[j], I[j], I[i]);
  }
  return tris;
}

interface Piece {
  update(map: (p: Vec2) => V3): void;
}
interface KinBend {
  update(G: THREE.Matrix4[], W: WorldMap): void;
}
interface FlipCtx {
  k: number;
  base: THREE.Vector3;
  orbit: boolean;
}

export class PaperViewer {
  private host: HTMLElement;
  private opts: ViewerOptions;
  private colors: ViewerColors;
  private tl: Timeline = { name: "", states: [], steps: [] };
  private size = 1;
  private gap = 0.004;
  private DZ = 0.004;
  private reduced: boolean;

  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private persp: THREE.PerspectiveCamera;
  private ortho: THREE.OrthographicCamera;
  private camera: THREE.Camera;
  private controls: OrbitControls;
  private headlight = new THREE.DirectionalLight(0xffffff, 1.5);
  private paper = new THREE.Group();
  private hints = new THREE.Group();
  private lineMats: LineMaterial[] = [];
  private frontMat: THREE.MeshLambertMaterial;
  private backMat: THREE.MeshLambertMaterial;
  private edgeMat: THREE.LineBasicMaterial;
  private creaseMat: THREE.LineBasicMaterial;

  private center = new THREE.Vector3();
  private span = 1;
  private hintOp = 1;
  private hintFade = 1;
  private hintSide = 1;
  private hintPlaneZ = 0;
  private hintSpan = 1; // labels scale with the state they belong to

  // cur = number of finished steps. Step sk is shown at progress st in [0, 1].
  // Invariant: st < 1 -> cur = sk (step sk not finished; st > 0 = partly folded, held there);
  //            st = 1 -> cur = sk + 1 (step sk done).
  private cur = 0;
  private sk = 0;
  private st = 0;
  private busy = false;
  private anim: { k: number } | null = null;
  private posed: string | null = null; // what is built for a partial pose: "kin:k" / "flip:k"
  private hintK = -1; // step whose hints are built
  private flipCtx: FlipCtx | null = null; // camera swing of the flip step being posed
  private animPieces: { pc: PieceJson; obj: Piece }[] | null = null;
  private animBends: KinBend[] = [];

  private playing = false;
  private follow = false;
  private speed = 1;
  private beat = 0; // reduced-motion playback timer
  private topDown = false;
  private raf = 0;
  private ro: ResizeObserver;
  private width = 1;
  private height = 1;
  private disposed = false;
  private insetBottom = 0;
  // camera follows the shown state; true once the user pans or zooms (until resetView)
  private manual = false;
  private dirGoal: THREE.Vector3 | null = null; // view direction resetView eases to
  private lastFrame = 0;
  private touches = new Set<number>();
  private orthoScale = 1; // Flat view zoom (> 1 = closer)
  private orthoShift = new THREE.Vector3(); // Flat view pan, world units

  constructor(host: HTMLElement, opts: ViewerOptions = {}) {
    this.host = host;
    this.opts = opts;
    this.colors = { ...DEFAULT_COLORS, ...opts.colors };
    this.gap = opts.layerGap ?? 0.004;
    this.reduced = !!opts.reducedMotion;
    this.insetBottom = opts.insetBottom ?? 0;

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x000000, 0);
    const el = this.renderer.domElement;
    el.style.display = "block";
    el.style.touchAction = "none";
    host.appendChild(el);

    this.persp = new THREE.PerspectiveCamera(40, 1, 0.01, 100);
    this.persp.up.set(0, 1, 0);
    this.ortho = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 100);
    this.camera = this.persp;
    this.scene.add(this.persp, this.ortho);
    this.headlight.position.set(0.3, 0.5, 1);
    this.persp.add(this.headlight);
    this.scene.add(new THREE.AmbientLight(0xffffff, 1.6));
    const fill = new THREE.DirectionalLight(0xffffff, 0.5);
    fill.position.set(0, 0, 1);
    this.scene.add(fill);

    this.controls = new OrbitControls(this.persp, el);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    // right-drag, shift/ctrl-drag or two fingers move the model; one finger / left-drag turns it
    this.controls.enablePan = true;
    this.controls.screenSpacePanning = true;
    this.controls.touches = { ONE: THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_PAN };
    const takeOver = () => {
      this.manual = true;
      this.dirGoal = null;
    };
    el.addEventListener("wheel", takeOver, { passive: true });
    el.addEventListener("pointerdown", (e) => {
      if (e.pointerType === "touch") {
        this.touches.add(e.pointerId);
        if (this.touches.size >= 2) takeOver();
      } else if (e.button === 2 || e.button === 1 || e.shiftKey || e.ctrlKey || e.metaKey) takeOver();
      else this.dirGoal = null;
    });
    const lift = (e: PointerEvent) => this.touches.delete(e.pointerId);
    el.addEventListener("pointerup", lift);
    el.addEventListener("pointercancel", lift);

    const poly = { polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 };
    this.frontMat = new THREE.MeshLambertMaterial({ color: 0, side: THREE.FrontSide, ...poly });
    this.backMat = new THREE.MeshLambertMaterial({ color: 0, side: THREE.BackSide, ...poly });
    this.edgeMat = new THREE.LineBasicMaterial({ color: this.colors.edge });
    this.creaseMat = new THREE.LineBasicMaterial({ color: this.colors.crease }); // opaque: no transparent sorting on paper
    this.scene.add(this.paper);
    this.scene.add(this.hints);

    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(host);
    this.resize();
    const loop = (now: number) => {
      if (this.disposed) return;
      const dt = Math.min(0.1, Math.max(0, (now - (this.lastFrame || now)) / 1000));
      this.lastFrame = now;
      this.trackFrame(dt);
      if (this.controls.enabled) this.controls.update();
      this.fitClip();
      this.updateHintFade();
      this.renderer.render(this.scene, this.camera);
      this.raf = requestAnimationFrame(loop);
    };
    this.raf = requestAnimationFrame(loop);
  }

  // ---------------- public API ----------------

  /** Load a timeline and rest at state `step` (default 0). */
  setTimeline(tl: Timeline, opts: { step?: number } = {}) {
    this.stopAnim();
    this.stopPlaying(false);
    // own copies of the step/state lists, so appendStep/truncate never touch the caller's object
    this.tl = { ...tl, states: [...tl.states], steps: [...tl.steps] };
    this.size = tl.size || 1;
    this.DZ = this.gap * this.size;
    const c = this.colors;
    const [front, back] = tl.colored_side === "down" ? [c.back, c.front] : [c.front, c.back];
    this.frontMat.color.setHex(front);
    this.backMat.color.setHex(back);
    this.frame();
    const N = this.n;
    const k0 = Math.min(N, Math.max(0, opts.step ?? 0));
    this.flipCtx = null;
    if (k0 === N && N) {
      this.sk = N - 1;
      this.st = 1;
    } else {
      this.sk = k0;
      this.st = 0;
    }
    this.resetView(k0, { smooth: false });
    this.rest(k0);
  }

  get timeline(): Timeline {
    return this.tl;
  }

  get n() {
    return this.tl.steps.length;
  }

  get info(): ViewerInfo {
    return {
      cur: this.cur,
      n: this.n,
      k: this.sk,
      t: this.st,
      pos: this.n ? this.sk + this.st : 0,
      busy: this.busy,
      playing: this.playing,
    };
  }

  /** Add one committed step; `state` is the paper after it (with its world map). */
  appendStep(step: StepJson, state: StateJson) {
    const wasEnd = this.cur === this.n;
    this.tl.steps.push(step);
    this.tl.states.push(state);
    this.frame();
    if (wasEnd && !this.busy && !(this.st > 0 && this.st < 1)) {
      // state n now has a next step: rebuild its hints
      this.hintK = -1;
      this.sk = this.cur;
      this.st = 0;
      this.rest(this.cur);
    } else this.emit();
    this.pump();
  }

  /** Drop steps after the first `nSteps` (undo). */
  truncate(nSteps: number) {
    const keep = Math.max(0, Math.min(this.n, nSteps));
    if (keep === this.n) return;
    const showing = this.sk + this.st;
    this.tl.steps.length = keep;
    this.tl.states.length = keep + 1;
    if (this.sk >= keep || this.cur > keep) {
      this.stopAnim();
      this.jumpTo(Math.min(this.cur, keep));
    } else if (this.cur === keep && showing <= keep) {
      this.hintK = -1;
      if (!this.busy && !(this.st > 0 && this.st < 1)) this.rest(this.cur);
    }
    this.frame();
    this.emit();
  }

  /** Go to state k. animate: fold/unfold smoothly when k is the next or previous state. */
  goto(k: number, opts: { animate?: boolean } = {}) {
    const N = this.n;
    k = Math.min(N, Math.max(0, Math.round(k)));
    const partial = this.sk === this.cur && this.st > 0 && this.st < 1;
    if (k === this.cur && !partial && !this.busy) return;
    this.stopAnim();
    const anim = opts.animate && !this.reduced;
    if (anim && k === this.cur + 1) this.animateTo(this.cur, 1);
    else if (anim && k === this.cur && partial) this.animateTo(this.cur, 0);
    else if (anim && k === this.cur - 1 && !partial) this.animateTo(this.cur - 1, 0);
    else this.jumpTo(k);
  }

  next() {
    if (this.cur < this.n) this.goto(this.cur + 1, { animate: true });
  }

  prev() {
    if (this.sk === this.cur && this.st > 0 && this.st < 1) this.goto(this.cur, { animate: true });
    else if (this.cur > 0) this.goto(this.cur - 1, { animate: true });
  }

  /** Continuous position: pos = k + t, 0 = flat sheet, n = finished. */
  scrub(pos: number) {
    const N = this.n;
    if (!N) return;
    this.stopPlaying(false);
    this.stopAnim();
    pos = Math.min(N, Math.max(0, pos));
    let k = Math.floor(pos),
      t = pos - k;
    if (k >= N) {
      k = N - 1;
      t = 1;
    }
    if (k !== this.sk) this.jumpTo(k);
    this.show(k, t);
  }

  /** Freeze step k (0-based) at progress t. */
  pose(k: number, t: number) {
    if (!this.n) return;
    this.stopPlaying(false);
    this.stopAnim();
    k = Math.min(this.n - 1, Math.max(0, Math.floor(k)));
    if (k !== this.sk) this.jumpTo(k);
    this.show(k, t);
  }

  play(opts: PlayOptions = {}) {
    if (opts.speed) this.speed = opts.speed;
    this.follow = !!opts.follow;
    if (this.playing) return;
    if (this.cur >= this.n && this.n && !this.follow) this.jumpTo(0);
    this.playing = true;
    this.emit();
    this.pump();
  }

  /** Stop where the paper is (a fold in progress stays part-folded). */
  pause() {
    this.stopPlaying(true);
  }

  setSpeed(speed: number) {
    this.speed = speed;
  }

  setInsetBottom(px: number) {
    this.insetBottom = px;
    this.resize();
  }

  setReducedMotion(reduced: boolean) {
    this.reduced = reduced;
  }

  setLayerGap(gap: number) {
    this.gap = gap;
    this.DZ = gap * this.size;
    if (this.busy) return; // during a fold the next frame picks up DZ
    this.hintK = -1;
    if (this.st > 0 && this.st < 1) {
      this.posed = null;
      this.show(this.sk, this.st);
    } else this.rest(this.cur);
  }

  /** Default camera: working side of the paper, framing the shown state; the camera follows it again. */
  resetView(k = this.cur, opts: { smooth?: boolean } = {}) {
    this.orthoScale = 1;
    this.orthoShift.set(0, 0, 0);
    if (this.topDown) this.placeOrtho();
    const st = this.tl.states[Math.min(k, this.tl.states.length - 1)];
    const sd = st ? st.world.s : 1;
    const dir = new THREE.Vector3(-sd * 0.35, -0.8, sd * 1.8).normalize();
    this.manual = false;
    if (opts.smooth ?? true) {
      // a flip in progress owns the camera; trackFrame eases everything else
      this.dirGoal = this.flipCtx ? null : dir;
      return;
    }
    this.dirGoal = null;
    const g = this.frameGoal(Math.min(k, Math.max(0, this.tl.states.length - 1)));
    const c = g?.c ?? this.center;
    this.controls.target.copy(c);
    this.persp.position.copy(c).addScaledVector(dir, this.fitDist(g?.span ?? this.span));
    this.persp.lookAt(c);
    this.controls.update();
    if (this.topDown) this.placeOrtho();
  }

  /** Play step k (0-based) again from its start. */
  replay(k = this.sk) {
    if (!this.n) return;
    k = Math.min(this.n - 1, Math.max(0, Math.floor(k)));
    this.stopPlaying(true);
    this.stopAnim();
    this.jumpTo(k);
    if (this.reduced) this.jumpTo(k + 1);
    else this.animateTo(k, 1);
  }

  /** Spherical view angles in degrees around the model centre (theta around, phi from straight above). */
  orbit(theta: number, phi: number) {
    const cam = this.persp;
    const r = cam.position.distanceTo(this.controls.target),
      th = (theta * Math.PI) / 180,
      ph = (phi * Math.PI) / 180;
    const c = this.controls.target;
    this.dirGoal = null;
    cam.position.set(c.x + r * Math.sin(ph) * Math.sin(th), c.y - r * Math.sin(ph) * Math.cos(th), r * Math.cos(ph));
    this.controls.update();
  }

  /** topDown: orthographic camera straight above the working side, no orbit (so taps map to the paper). */
  setView(v: { topDown: boolean }) {
    this.topDown = v.topDown;
    this.camera = v.topDown ? this.ortho : this.persp;
    this.camera.add(this.headlight);
    if (v.topDown) this.placeOrtho();
    this.syncControls();
  }

  /** Engine coords of the paper plane under a screen point, or null if the ray misses it. */
  pickPoint(clientX: number, clientY: number): Vec2 | null {
    const r = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
    const ray = new THREE.Raycaster();
    ray.setFromCamera(ndc, this.camera);
    const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), -this.hintPlaneZ);
    const hit = ray.ray.intersectPlane(plane, new THREE.Vector3());
    if (!hit) return null;
    const W = this.tl.states[this.cur]?.world;
    if (!W) return null;
    const M = W.M,
      det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
    const x = hit.x - M[0][2],
      y = hit.y - M[1][2];
    return [(M[1][1] * x - M[0][1] * y) / det, (-M[1][0] * x + M[0][0] * y) / det];
  }

  /** Screen position (px from the canvas's top-left) of an engine point of the shown state, on the paper plane. */
  toScreen(xy: Vec2): Vec2 | null {
    const W = this.tl.states[this.cur]?.world;
    if (!W) return null;
    const M = W.M;
    const v = new THREE.Vector3(
      M[0][0] * xy[0] + M[0][1] * xy[1] + M[0][2],
      M[1][0] * xy[0] + M[1][1] * xy[1] + M[1][2],
      this.hintPlaneZ,
    ).project(this.camera);
    return [((v.x + 1) / 2) * this.width, ((1 - v.y) / 2) * this.height];
  }

  /** Unit-scale 2x2 map engine -> screen (y down) for the shown state: how the stage orients the paper. */
  screenLinear(): [[number, number], [number, number]] | null {
    const o = this.toScreen([0, 0]),
      ex = this.toScreen([1, 0]),
      ey = this.toScreen([0, 1]);
    if (!o || !ex || !ey) return null;
    const s = Math.hypot(ex[0] - o[0], ex[1] - o[1]) || 1;
    return [
      [(ex[0] - o[0]) / s, (ey[0] - o[0]) / s],
      [(ex[1] - o[1]) / s, (ey[1] - o[1]) / s],
    ];
  }

  /** PNG of the current view; background fills the transparent canvas (e.g. the mat colour). */
  screenshot(background?: string): Promise<Blob> {
    this.renderer.render(this.scene, this.camera);
    const src = this.renderer.domElement;
    let out: HTMLCanvasElement = src;
    if (background) {
      out = document.createElement("canvas");
      out.width = src.width;
      out.height = src.height;
      const g = out.getContext("2d")!;
      g.fillStyle = background;
      g.fillRect(0, 0, out.width, out.height);
      g.drawImage(src, 0, 0);
    }
    return new Promise((res, rej) => out.toBlob((b) => (b ? res(b) : rej(new Error("screenshot failed"))), "image/png"));
  }

  showHints(v: boolean) {
    this.hints.visible = v;
  }

  camSide() {
    return Math.sign(this.camera.position.z - this.controls.target.z);
  }

  /** Engine-space position of every piece's first vertex at step k, progress t (scripted checks). */
  kinPoints(k: number, t: number) {
    const s = this.tl.steps[k],
      G = this.kinFrames(s, t);
    return s.pieces.map((pc, i) => new THREE.Vector3(pc.orig[0][0], pc.orig[0][1], 0).applyMatrix4(G[i]).toArray());
  }

  dispose() {
    this.disposed = true;
    this.stopAnim();
    this.stopPlaying(false);
    cancelAnimationFrame(this.raf);
    this.ro.disconnect();
    this.controls.dispose();
    this.clearGroup(this.paper);
    this.clearHints();
    for (const m of [this.frontMat, this.backMat, this.edgeMat, this.creaseMat]) m.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }

  // ---------------- playback ----------------

  private emit() {
    this.opts.onChange?.(this.info);
  }

  private stopPlaying(notify: boolean) {
    if (this.beat) window.clearTimeout(this.beat);
    this.beat = 0;
    if (!this.playing) return;
    this.playing = false;
    this.follow = false;
    this.stopAnim();
    if (notify) this.emit();
  }

  // starts the next fold while playing (or following a live run)
  private pump() {
    if (!this.playing || this.busy || this.beat) return;
    if (this.cur >= this.n) {
      if (!this.follow) {
        this.playing = false;
        this.emit();
      }
      return;
    }
    if (this.reduced) {
      this.beat = window.setTimeout(() => {
        this.beat = 0;
        if (!this.playing) return;
        this.jumpTo(this.cur + 1);
        this.pump();
      }, 600 / this.speed);
      return;
    }
    this.animateTo(this.cur, 1);
  }

  private stopAnim() {
    if (this.anim) {
      this.anim = null;
      this.busy = false;
    }
  }

  // rest at state k without animation; turns the camera to state k's working side if needed
  private jumpTo(k: number) {
    const N = this.n;
    k = Math.min(N, Math.max(0, k));
    if (this.flipCtx && this.flipCtx.orbit) this.flipCamera(this.flipCtx.base, k > this.flipCtx.k ? 1 : 0);
    this.flipCtx = null;
    const side = this.tl.states[k].world.s;
    if (this.flipNeeded(side)) {
      const o = this.persp.position.clone().sub(this.controls.target);
      this.persp.position.set(this.controls.target.x - o.x, this.controls.target.y + o.y, this.controls.target.z - o.z);
      this.persp.lookAt(this.controls.target);
      this.controls.update();
      if (this.topDown) this.placeOrtho();
    }
    if (k === N && N) {
      this.sk = N - 1;
      this.st = 1;
    } else {
      this.sk = k;
      this.st = 0;
    }
    this.rest(k);
  }

  // state k at rest: layers, bends, creases + hints for the next step
  private rest(k: number) {
    this.cur = k;
    this.posed = null;
    if (this.flipCtx) this.flipCtx = null;
    this.syncControls();
    this.showState(k);
    this.setHints(k);
    this.hintK = k;
    this.emit();
  }

  // camera swing for flip step k; t0 = the progress the camera currently shows (0 or 1)
  private flipContext(k: number, t0: number) {
    if (this.flipCtx && this.flipCtx.k === k) return this.flipCtx;
    const s = this.tl.steps[k],
      o = this.persp.position.clone().sub(this.controls.target);
    const orbit = this.flipNeeded(t0 >= 1 ? s.side_before : s.side_after);
    const base = orbit && t0 >= 1 ? new THREE.Vector3(-o.x, o.y, -o.z) : o; // turned back by 180 deg
    this.flipCtx = { k, base, orbit };
    this.syncControls();
    return this.flipCtx;
  }

  // Show step k at progress t (pure function of k, t). t = 0 / 1 rest at state k / k + 1.
  private show(k: number, t: number) {
    const step = this.tl.steps[k];
    t = Math.min(1, Math.max(0, t));
    const tFrom = this.cur > k ? 1 : 0; // what the screen showed for step k so far
    if (step.op === "flip") {
      const fc = this.flipContext(k, this.sk === k ? this.st : tFrom);
      if (fc.orbit) this.flipCamera(fc.base, ease(t));
    }
    this.sk = k;
    this.st = t;
    if (t >= 1) return this.rest(k + 1);
    if (t <= 0) return this.rest(k);
    this.cur = k;
    if (step.op === "flip") {
      if (this.posed !== "flip:" + k) {
        this.showState(k);
        this.posed = "flip:" + k;
      }
    } else {
      if (this.posed !== "kin:" + k) {
        this.buildStepPieces(step);
        this.posed = "kin:" + k;
      }
      this.poseStep(step, t);
    }
    if (this.hintK !== k) {
      this.setHints(k);
      this.hintK = k;
    }
    this.setHintOpacity(hintOpacity(step, t));
    this.emit();
  }

  // fold time: grows with how far the paper travels; multi-crease (solved / macro) steps get longer
  private stepDuration(step: StepJson) {
    if (step.op === "flip")
      return this.flipNeeded(this.cur > step.index - 1 ? step.side_before : step.side_after) ? 1000 : 300;
    const multi = !!(step.motions || step.retime);
    const d = 850 * Math.sqrt(Math.max(0, step.path_len ?? this.size) / this.size) * (multi ? 1.6 : 1);
    return Math.min(3000, Math.max(900, d));
  }

  // animate step k from its current progress to `target` (0 or 1)
  private animateTo(k: number, target: number) {
    if (this.busy) return;
    const step = this.tl.steps[k];
    const from = this.sk === k ? this.st : target > 0.5 ? 0 : 1;
    if (this.sk !== k || this.st !== from) this.show(k, from);
    const dur = (this.stepDuration(step) * Math.abs(target - from)) / this.speed;
    const pre = target > from && from === 0 && step.op !== "flip" ? PRE / this.speed : 0; // show the fold line before moving
    const me = (this.anim = { k });
    this.busy = true;
    this.emit();
    const t0 = performance.now();
    const frame = (now: number) => {
      if (this.anim !== me) return; // cancelled
      const f = dur > 0 ? Math.min(1, Math.max(0, (now - t0 - pre) / dur)) : 1;
      if (f <= 0 && from <= 0) {
        requestAnimationFrame(frame); // still showing the fold line
        return;
      }
      if (f >= 1) {
        this.anim = null;
        this.busy = false;
        this.show(k, target);
        this.pump();
        return;
      }
      this.show(k, from + (target - from) * f);
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }

  // ---------------- camera ----------------

  private resize() {
    const w = Math.max(1, this.host.clientWidth),
      h = Math.max(1, this.host.clientHeight);
    this.width = w;
    this.height = h;
    const inset = Math.min(this.insetBottom, h * 0.4);
    this.persp.aspect = w / (h + inset);
    // render the lower part of a taller view, so the view centre sits in the middle of the uncovered area
    this.persp.setViewOffset(w, h + inset, 0, inset, w, h);
    this.persp.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    for (const m of this.lineMats) m.resolution.set(w, h);
    if (this.topDown) this.placeOrtho();
  }

  // frame all states (in world coords)
  private frame() {
    let lo = [Infinity, Infinity],
      hi = [-Infinity, -Infinity];
    for (const s of this.tl.states) {
      for (const cx of [s.bbox[0][0], s.bbox[1][0]])
        for (const cy of [s.bbox[0][1], s.bbox[1][1]]) {
          const q = this.wpt(s.world, cx, cy, 0);
          lo = [Math.min(lo[0], q[0]), Math.min(lo[1], q[1])];
          hi = [Math.max(hi[0], q[0]), Math.max(hi[1], q[1])];
        }
    }
    if (!isFinite(lo[0])) {
      lo = [0, 0];
      hi = [this.size, this.size];
    }
    this.center.set((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, 0);
    this.span = Math.max(hi[0] - lo[0], hi[1] - lo[1], this.size);
  }

  // world-space box of state k (flat, z ignored)
  private stateBox(k: number) {
    const s = this.tl.states[k];
    let lo = [Infinity, Infinity],
      hi = [-Infinity, -Infinity];
    if (s)
      for (const cx of [s.bbox[0][0], s.bbox[1][0]])
        for (const cy of [s.bbox[0][1], s.bbox[1][1]]) {
          const q = this.wpt(s.world, cx, cy, 0);
          lo = [Math.min(lo[0], q[0]), Math.min(lo[1], q[1])];
          hi = [Math.max(hi[0], q[0]), Math.max(hi[1], q[1])];
        }
    return { lo, hi };
  }

  // what the camera should frame: the resting state, or both ends of the fold in progress
  private frameGoal(k?: number) {
    if (!this.tl.states.length) return null;
    const a = this.stateBox(k ?? (this.st > 0 && this.st < 1 ? this.sk : this.cur));
    if (k === undefined && this.st > 0 && this.st < 1) {
      const b = this.stateBox(this.sk + 1);
      a.lo = [Math.min(a.lo[0], b.lo[0]), Math.min(a.lo[1], b.lo[1])];
      a.hi = [Math.max(a.hi[0], b.hi[0]), Math.max(a.hi[1], b.hi[1])];
    }
    if (!isFinite(a.lo[0])) return null;
    return {
      c: new THREE.Vector3((a.lo[0] + a.hi[0]) / 2, (a.lo[1] + a.hi[1]) / 2, 0),
      span: Math.max(a.hi[0] - a.lo[0], a.hi[1] - a.lo[1], 0.3 * this.size),
    };
  }

  // camera distance at which a model of this span fills ~3/4 of the uncovered view
  private fitDist(span: number) {
    const inset = Math.min(this.insetBottom, this.height * 0.4);
    const tanHalf =
      (Math.tan((this.persp.fov * Math.PI) / 360) * Math.min(this.width, this.height - inset)) / (this.height + inset);
    return (span * 0.5 * 1.35) / Math.max(tanHalf, 1e-3);
  }

  // eases target, distance (and after resetView, direction) toward the framing of what is shown
  private trackFrame(dt: number) {
    if (this.topDown || this.manual || !dt) return;
    const g = this.frameGoal();
    if (!g) return;
    const a = 1 - Math.exp(-dt / 0.28);
    const tg = this.controls.target,
      cam = this.persp;
    const d = g.c.sub(tg).multiplyScalar(a);
    tg.add(d);
    cam.position.add(d);
    if (this.flipCtx) return; // the flip swing sets the camera from its own offset
    const off = cam.position.clone().sub(tg);
    const len = off.length();
    off.normalize();
    if (this.dirGoal) {
      off.lerp(this.dirGoal, a).normalize();
      if (off.angleTo(this.dirGoal) < 2e-3) this.dirGoal = null;
    }
    cam.position.copy(tg).addScaledVector(off, len + (this.fitDist(g.span) - len) * a);
    cam.lookAt(tg);
  }

  /** Flat view: zoom by factor f keeping the paper under the screen point still. */
  zoomFlat(clientX: number, clientY: number, f: number) {
    if (!this.topDown) return;
    const ndc = this.ndc(clientX, clientY);
    const before = ndc.clone().unproject(this.ortho);
    this.orthoScale = Math.min(20, Math.max(0.3, this.orthoScale * f));
    this.placeOrtho();
    this.orthoShift.add(before.sub(ndc.unproject(this.ortho))).setZ(0);
    this.placeOrtho();
  }

  /** Flat view: move the paper by a screen-pixel offset. */
  panFlat(dx: number, dy: number) {
    if (!this.topDown) return;
    const r = this.renderer.domElement.getBoundingClientRect();
    const o = new THREE.Vector3(0, 0, 0).unproject(this.ortho);
    const d = new THREE.Vector3((2 * dx) / r.width, (-2 * dy) / r.height, 0).unproject(this.ortho);
    this.orthoShift.sub(d.sub(o)).setZ(0);
    this.placeOrtho();
  }

  private ndc(clientX: number, clientY: number) {
    const r = this.renderer.domElement.getBoundingClientRect();
    return new THREE.Vector3(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1, 0);
  }

  private placeOrtho() {
    const side = this.tl.states[this.cur]?.world.s ?? 1;
    const half = (this.span * 0.62) / this.orthoScale,
      aspect = this.width / this.height;
    const o = this.ortho;
    o.left = -half * Math.max(1, aspect);
    o.right = half * Math.max(1, aspect);
    o.top = half / Math.min(1, aspect);
    o.bottom = -half / Math.min(1, aspect);
    o.near = 0.01;
    o.far = 10 * this.span;
    const c = this.center.clone().add(this.orthoShift);
    o.position.set(c.x, c.y, side * 4 * this.span);
    o.up.set(0, side, 0);
    o.lookAt(c);
    o.updateMatrixWorld();
    o.updateProjectionMatrix();
  }

  private syncControls() {
    this.controls.enabled = !this.flipCtx && !this.topDown;
  }

  private flipNeeded(side: number) {
    return (this.persp.position.z - this.controls.target.z) * side < 0;
  }

  // flip = camera move: the paper stays put, the camera swings 180 deg around the vertical axis
  private flipCamera(base: THREE.Vector3, e: number) {
    const ph = Math.PI * e,
      c = Math.cos(ph),
      s = Math.sin(ph),
      tg = this.controls.target;
    this.persp.position.set(tg.x + base.x * c + base.z * s, tg.y + base.y, tg.z - base.x * s + base.z * c);
    this.persp.lookAt(tg);
    this.controls.update();
    if (this.topDown && e >= 1) this.placeOrtho();
  }

  // depth precision: keep near / far tight around the paper (thin layer gaps must not z-fight)
  private fitClip() {
    const cam = this.persp;
    const dist = cam.position.distanceTo(this.controls.target);
    const near = Math.max(dist * 0.02, dist - 2 * this.span),
      far = dist + 3 * this.span;
    if (Math.abs(near - cam.near) > 1e-3 * near || Math.abs(far - cam.far) > 1e-3 * far) {
      cam.near = near;
      cam.far = far;
      cam.updateProjectionMatrix();
    }
  }

  // ---------------- paper geometry ----------------

  // engine coords -> world coords
  private wpt(W: WorldMap, x: number, y: number, z: number): V3 {
    const M = W.M;
    return [M[0][0] * x + M[0][1] * y + M[0][2], M[1][0] * x + M[1][1] * y + M[1][2], W.s * z + W.zoff * this.DZ];
  }

  private clearGroup(g: THREE.Group) {
    for (const c of [...g.children]) {
      g.remove(c);
      (c as THREE.Mesh).geometry?.dispose();
    }
  }

  // One flat convex piece: front mesh + back mesh + outline + creases, sharing a position buffer.
  private makePiece(orig: Vec2[], lines: [Vec2, Vec2][]): Piece {
    const n = orig.length;
    const pos = new THREE.BufferAttribute(new Float32Array(n * 3), 3);
    const nrm = new THREE.BufferAttribute(new Float32Array(n * 3), 3);
    const idx: number[] = [];
    for (let i = 1; i < n - 1; i++) idx.push(0, i, i + 1); // orig polygons are CCW -> front normal
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", pos);
    geom.setAttribute("normal", nrm);
    geom.setIndex(idx);
    const front = new THREE.Mesh(geom, this.frontMat);
    const back = new THREE.Mesh(geom, this.backMat);
    const outline = new THREE.LineLoop(new THREE.BufferGeometry().setAttribute("position", pos), this.edgeMat);
    const cpos = new THREE.BufferAttribute(new Float32Array(Math.max(lines.length, 1) * 6), 3);
    const creases = new THREE.LineSegments(new THREE.BufferGeometry().setAttribute("position", cpos), this.creaseMat);
    creases.geometry.setDrawRange(0, lines.length * 2);
    for (const o of [front, back, outline, creases]) {
      o.frustumCulled = false;
      this.paper.add(o);
    }
    return {
      update(map) {
        for (let i = 0; i < n; i++) pos.setXYZ(i, ...map(orig[i]));
        lines.forEach((l, j) => {
          cpos.setXYZ(2 * j, ...map(l[0]));
          cpos.setXYZ(2 * j + 1, ...map(l[1]));
        });
        // Newell normal (robust for near-collinear vertices)
        let nx = 0,
          ny = 0,
          nz = 0;
        for (let i = 0; i < n; i++) {
          const j = (i + 1) % n;
          const a = [pos.getX(i), pos.getY(i), pos.getZ(i)],
            b = [pos.getX(j), pos.getY(j), pos.getZ(j)];
          nx += (a[1] - b[1]) * (a[2] + b[2]);
          ny += (a[2] - b[2]) * (a[0] + b[0]);
          nz += (a[0] - b[0]) * (a[1] + b[1]);
        }
        const L = Math.hypot(nx, ny, nz) || 1;
        for (let i = 0; i < n; i++) nrm.setXYZ(i, nx / L, ny / L, nz / L);
        pos.needsUpdate = true;
        nrm.needsUpdate = true;
        cpos.needsUpdate = true;
      },
    };
  }

  // Folded connection: the sheet wraps around a crease between two stacked layers -> half-cylinder strip.
  // The outside shows the colour of the upper face's top side; raw (cut) edges are never bent.
  private addBend(st: StateJson, W: WorldMap, b: { a: number; b: number; seg: [Vec2, Vec2] }) {
    const DZ = this.DZ;
    const fa = st.faces[b.a],
      fb = st.faces[b.b];
    const wp = (f: typeof fa, p: Vec2) => {
      const q = ap(f.T, p);
      return this.wpt(W, q[0], q[1], f.layer * DZ);
    };
    const A = wp(fa, b.seg[0]),
      B = wp(fa, b.seg[1]),
      A2 = wp(fb, b.seg[0]),
      B2 = wp(fb, b.seg[1]);
    void B2;
    const lowF = A[2] <= A2[2] ? fa : fb,
      upF = lowF === fa ? fb : fa;
    const lowA = lowF === fa ? A : A2,
      upA = lowF === fa ? A2 : A;
    const r = Math.abs(upA[2] - lowA[2]) / 2,
      zc = (upA[2] + lowA[2]) / 2;
    if (r < 1e-9) return;
    const l = [B[0] - A[0], B[1] - A[1]],
      ll = Math.hypot(l[0], l[1]) || 1;
    l[0] /= ll;
    l[1] /= ll;
    const mid = [(A[0] + B[0]) / 2, (A[1] + B[1]) / 2];
    const c = [0, 0];
    for (const p of lowF.orig) {
      const q = wp(lowF, p);
      c[0] += q[0] / lowF.orig.length;
      c[1] += q[1] / lowF.orig.length;
    }
    let n = [c[0] - mid[0], c[1] - mid[1]];
    const dl = n[0] * l[0] + n[1] * l[1];
    n = [n[0] - dl * l[0], n[1] - dl * l[1]];
    const nl = Math.hypot(n[0], n[1]) || 1;
    const u = [-n[0] / nl, -n[1] / nl]; // outward (away from the faces)
    // does the upper face show its front side upward?
    let nz = 0;
    const w = upF.orig.map((p) => wp(upF, p));
    for (let i = 0; i < w.length; i++) {
      const a = w[i],
        d = w[(i + 1) % w.length];
      nz += (a[0] - d[0]) * (a[1] + d[1]);
    }
    const frontOut = nz > 0;
    const K = 10,
      pos: number[] = [],
      idx: number[] = [];
    for (let i = 0; i <= K; i++) {
      const th = (Math.PI * i) / K,
        du = r * Math.sin(th),
        z = zc - r * Math.cos(th);
      pos.push(A[0] + u[0] * du, A[1] + u[1] * du, z, B[0] + u[0] * du, B[1] + u[1] * du, z);
    }
    // winding so the triangle front faces outward at the bulge (th = 90deg): outward = u in xy
    const t = [0, 0, 1],
      e = [l[0], l[1], 0],
      nrm = [t[1] * e[2] - t[2] * e[1], t[2] * e[0] - t[0] * e[2], 0];
    const flipW = nrm[0] * u[0] + nrm[1] * u[1] > 0 !== frontOut;
    for (let i = 0; i < K; i++) {
      const a0 = 2 * i,
        b0 = 2 * i + 1,
        a1 = 2 * i + 2,
        b1 = 2 * i + 3;
      if (flipW) idx.push(a0, b0, a1, b0, b1, a1);
      else idx.push(a0, a1, b0, b0, a1, b1);
    }
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    geom.setIndex(idx);
    geom.computeVertexNormals();
    for (const m of [this.frontMat, this.backMat]) {
      const mesh = new THREE.Mesh(geom, m);
      mesh.frustumCulled = false;
      this.paper.add(mesh);
    }
  }

  private showState(k: number, withBends = true) {
    this.clearGroup(this.paper);
    const st = this.tl.states[k],
      W = st.world;
    for (const f of st.faces) {
      const piece = this.makePiece(f.orig, f.lines);
      const z = f.layer * this.DZ;
      piece.update((p) => {
        const q = ap(f.T, p);
        return this.wpt(W, q[0], q[1], z);
      });
    }
    if (withBends) for (const b of st.bends || []) this.addBend(st, W, b);
  }

  // ---------------- hints: fold line + tutorial arrows ----------------

  private hintMat(color: number, opacity = 1) {
    const m = new THREE.MeshBasicMaterial({ color, transparent: true, opacity, depthTest: false, depthWrite: false });
    m.userData.base = opacity;
    return m;
  }

  private addFlat(tris: V3[], color: number, opacity = 1) {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(tris.flat(), 3));
    const m = this.hintMat(color, opacity);
    m.side = THREE.DoubleSide;
    const mesh = new THREE.Mesh(g, m);
    mesh.renderOrder = HINT_RENDER;
    mesh.frustumCulled = false;
    this.hints.add(mesh);
  }

  // pts: world polyline at constant z. Solid ribbon (valley / plain) or dashed ribbon + outlined head (mountain).
  private addArrow(
    pts: V3[],
    color: number,
    { width, dashed = false, hollow = false, both = false, opacity = 1 }: { width: number; dashed?: boolean; hollow?: boolean; both?: boolean; opacity?: number },
  ) {
    const SIZE = this.size;
    const fine = resample(pts, width * 0.25),
      L = pathLen(fine);
    const hl = Math.min(0.05 * SIZE, L * 0.4),
      hw = hl * 0.5; // constant head, independent of arrow length
    const s0 = both ? hl * 0.9 : 0,
      s1 = L - hl * 0.9;
    const tris: V3[] = [];
    if (!dashed) tris.push(...ribbonTris(subPath(fine, s0, s1), width));
    else {
      const dash = width * 2.2,
        gap = width * 1.3;
      for (let s = s0; s < s1 - 1e-9; s += dash + gap) {
        const seg = subPath(fine, s, Math.min(s + dash, s1));
        if (seg.length > 1) tris.push(...ribbonTris(seg, width));
      }
    }
    const dirAt = (i: number, j: number): Vec2 => [fine[i][0] - fine[j][0], fine[i][1] - fine[j][1]];
    const n = fine.length;
    const heads = headTris(fine[n - 1], dirAt(n - 1, Math.max(0, n - 3)), hl, hw, hollow);
    const heads2 = both ? headTris(fine[0], dirAt(0, Math.min(n - 1, 2)), hl, hw, hollow) : [];
    this.addFlat([...tris, ...heads, ...heads2], color, opacity);
  }

  private addDot(p: V3, color: number, r: number) {
    const tris: V3[] = [],
      K = 16;
    for (let i = 0; i < K; i++) {
      const a = (2 * Math.PI * i) / K,
        b = (2 * Math.PI * (i + 1)) / K;
      tris.push([p[0], p[1], p[2]], [p[0] + r * Math.cos(a), p[1] + r * Math.sin(a), p[2]], [p[0] + r * Math.cos(b), p[1] + r * Math.sin(b), p[2]]);
    }
    this.addFlat(tris, color);
  }

  private addLabel(text: string, p: V3, color: string, scale = 0.07) {
    const c = document.createElement("canvas");
    c.width = 256;
    c.height = 80;
    const x = c.getContext("2d")!;
    x.fillStyle = "rgba(251,251,248,0.92)";
    x.beginPath();
    x.roundRect(4, 6, 248, 68, 18);
    x.fill();
    x.fillStyle = color;
    x.font = '700 36px "Zen Kaku Gothic New", system-ui, sans-serif';
    x.textAlign = "center";
    x.textBaseline = "middle";
    x.fillText(text, 128, 42);
    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false, depthWrite: false });
    mat.userData.base = 1;
    const sp = new THREE.Sprite(mat);
    sp.scale.set(scale * this.hintSpan * 3.2, scale * this.hintSpan, 1);
    sp.position.set(...p);
    sp.renderOrder = HINT_RENDER + 2;
    this.hints.add(sp);
  }

  private addFoldLine(a: V3, b: V3, color: number, dashSize: number, gapSize: number) {
    const geom = new LineGeometry();
    geom.setPositions([...a, ...b]);
    const mat = new LineMaterial({ color, linewidth: 6, dashed: true, dashSize, gapSize, transparent: true, depthTest: false, depthWrite: false });
    mat.resolution.set(this.width, this.height);
    mat.userData.base = 1;
    mat.userData.bothSides = true; // fold line stays visible from either side of the paper
    this.lineMats.push(mat);
    const line = new Line2(geom, mat);
    line.computeLineDistances();
    line.renderOrder = HINT_RENDER;
    line.frustumCulled = false;
    this.hints.add(line);
  }

  private clearHints() {
    for (const c of [...this.hints.children]) {
      this.hints.remove(c);
      const o = c as THREE.Mesh;
      o.geometry?.dispose();
      const mat = o.material as THREE.MeshBasicMaterial | undefined;
      if (mat) {
        mat.map?.dispose();
        mat.dispose();
      }
    }
    this.lineMats.length = 0;
  }

  private setHintOpacity(o?: number) {
    if (o !== undefined) this.hintOp = o;
    this.hints.traverse((c) => {
      const mat = (c as THREE.Mesh).material as THREE.Material | undefined;
      if (!mat) return;
      const f = this.hintOp * (mat.userData.bothSides ? 1 : this.hintFade);
      mat.opacity = (mat.userData.base ?? 1) * f;
      c.visible = f > 0.01;
    });
  }

  // hints belong to the working side: fade them out when the camera looks at the paper from behind
  private updateHintFade() {
    const cam = this.camera.position,
      tg = this.topDown ? this.center : this.controls.target;
    const dx = cam.x - tg.x,
      dy = cam.y - tg.y,
      dz = cam.z - this.hintPlaneZ;
    const d = (this.hintSide * dz) / (Math.hypot(dx, dy, dz) || 1);
    const f = Math.min(1, Math.max(0, (d - 0.03) / 0.15));
    if (Math.abs(f - this.hintFade) > 1e-4) {
      this.hintFade = f;
      this.setHintOpacity();
    }
  }

  // Hints for the *next* step k (shown while it is pending): fold line + arrow, turn-over or rotate indicator.
  private setHints(k: number, opacity = 1) {
    this.clearHints();
    const step = this.tl.steps[k];
    if (!step) return;
    const SIZE = this.size,
      DZ = this.DZ,
      COLORS = this.colors;
    const ARROW_W = 0.0135 * SIZE; // constant thin ribbon width
    const st = this.tl.states[k],
      W = st.world;
    this.hintSide = W.s;
    this.hintPlaneZ = W.zoff * DZ;
    const color = (COLORS as unknown as Record<string, number>)[step.op] ?? COLORS.unfold;
    const zTop = (Math.max(0, ...st.faces.map((f) => f.layer)) + 2) * DZ;
    const E = (x: number, y: number, z: number) => this.wpt(W, x, y, z); // engine -> world
    if (step.motions) {
      // multi-part technique step: fold lines of the first phase only, no arrows
      for (const ax of step.axes || []) {
        if (ax.phase !== 0) continue;
        const [a, b] = ax.seg,
          mountain = ax.sign < 0;
        const [ds, gs] = mountain ? [0.05 * SIZE, 0.03 * SIZE] : [0.06 * SIZE, 0.035 * SIZE];
        this.addFoldLine(E(a[0], a[1], zTop), E(b[0], b[1], zTop), mountain ? COLORS.mountain : COLORS.valley, ds, gs);
      }
    } else if (step.axis && step.op !== "flip") {
      const [a, b] = step.axis;
      const [ds, gs] = step.op === "mountain" ? [0.05 * SIZE, 0.03 * SIZE] : [0.06 * SIZE, 0.035 * SIZE];
      this.addFoldLine(E(a[0], a[1], zTop), E(b[0], b[1], zTop), color, ds, gs);
    }
    const lo_ = st.bbox[0],
      hi_ = st.bbox[1],
      cx = (lo_[0] + hi_[0]) / 2,
      cy = (lo_[1] + hi_[1]) / 2;
    const sp = Math.max(hi_[0] - lo_[0], hi_[1] - lo_[1]);
    this.hintSpan = Math.max(sp, 0.5 * SIZE);
    if ((step.op === "valley" || step.op === "mountain") && step.arrow && step.axis) {
      const [px, py] = step.arrow.from;
      const valley = step.op === "valley",
        wd = ARROW_W;
      // quadratic arc like the PNG diagrams (matplotlib arc3, rad = 0.35), lying flat on the paper
      const [tx, ty] = step.arrow.to,
        mx = (px + tx) / 2,
        my = (py + ty) / 2,
        rad = 0.35;
      const kx = mx + rad * (ty - py),
        ky = my - rad * (tx - px);
      const pts: V3[] = [],
        xy: Vec2[] = [];
      for (let i = 0; i <= 40; i++) {
        const t = i / 40,
          a0 = (1 - t) * (1 - t),
          a1 = 2 * t * (1 - t),
          a2 = t * t;
        xy.push([a0 * px + a1 * kx + a2 * tx, a0 * py + a1 * ky + a2 * ty]);
      }
      // low arch: ends just above the paper, middle raised ~15% of arrow length (capped); ribbon stays flat (normal ~ up)
      const aL = pathLen(xy),
        peak = Math.min(0.15 * aL, 0.06 * SIZE);
      xy.forEach(([x, y], i) => pts.push(E(x, y, zTop + peak * Math.sin((Math.PI * i) / 40))));
      this.addArrow(pts, color, { width: wd, dashed: !valley, hollow: !valley });
      this.addDot(E(px, py, zTop), color, wd * 0.9);
    } else if (step.op === "flip") {
      // turn-over: double-headed arc under the paper (like the PNG diagrams), on the camera's side
      const sd = W.s,
        c = E(cx, cy, 0),
        y0 = c[1] - 0.58 * sp,
        half = 0.4 * sp,
        hgt = 0.2 * sp;
      const pts: V3[] = [],
        z = c[2] + sd * zTop;
      for (let i = 0; i <= 30; i++) {
        const th = (Math.PI * i) / 30;
        pts.push([c[0] - half * Math.cos(th), y0 - hgt * Math.sin(th), z]);
      }
      this.addArrow(pts, COLORS.flip, { width: ARROW_W, both: true });
      this.addLabel("turn over", [c[0], y0 - hgt - 0.1 * sp, z], COLORS.label);
    } else if (step.op === "rotate") {
      const r = 0.62 * sp,
        deg = step.degrees ?? 0,
        a0 = 60;
      const pts: V3[] = [],
        n = 32;
      for (let i = 0; i <= n; i++) {
        const a = ((a0 + (deg * i) / n) * Math.PI) / 180;
        pts.push(E(cx + r * Math.cos(a), cy + r * Math.sin(a), zTop));
      }
      this.addArrow(pts, COLORS.flip, { width: ARROW_W });
      const am = ((a0 + deg / 2) * Math.PI) / 180;
      this.addLabel(`rotate ${deg}°`, E(cx + (r + 0.12 * sp) * Math.cos(am), cy + (r + 0.12 * sp) * Math.sin(am), zTop), COLORS.label);
    }
    this.hintOp = opacity;
    this.setHintOpacity();
  }

  // ---------------- connected (kinematic) animation, see kinematics.py ----------------
  // Every piece hangs off its parent by a hinge along their shared edge; its frame is the parent's frame
  // turned about that edge by the current fold angle, so the paper never comes apart at a tree hinge.
  // Frames map orig (x, y, 0) -> engine 3D; z offsets are in layer units * DZ.

  // scripted frames (inside reverse / tuck): key = [R 3x3 row-major, t, w], frame = [R | t + DZ w]
  private keyFrame(fk: number[][], u: number) {
    const DZ = this.DZ;
    const S = fk.length - 1,
      x = Math.min(S, Math.max(0, u * S)),
      i = Math.min(S - 1, Math.floor(x)),
      f = x - i;
    const a = fk[i],
      b = fk[i + 1],
      k = a.map((v, j) => v + (b[j] - v) * f);
    return new THREE.Matrix4().set(
      k[0], k[1], k[2], k[9] + DZ * k[12],
      k[3], k[4], k[5], k[10] + DZ * k[13],
      k[6], k[7], k[8], k[11] + DZ * k[14],
      0, 0, 0, 1,
    );
  }

  // frames of all pieces at linear progress t
  private kinFrames(step: StepJson, t: number) {
    const DZ = this.DZ;
    const G: THREE.Matrix4[] = new Array(step.pieces.length),
      e = pathU(step, t);
    if (step.pieces.length && step.pieces[0].fkeys) return step.pieces.map((pc) => this.keyFrame(pc.fkeys!, e));
    for (const i of step.kin_order) {
      const pc = step.pieces[i];
      if (pc.parent < 0) {
        const M = frameE3(pc.Tb),
          m = pc.root_motion;
        if (m && m.type === "hinge") M.premultiply(hingeMat(m.p!, m.d!, m.angle! * e));
        else if (m && m.type === "rotate") {
          const a = ((m.degrees! * Math.PI) / 180) * e,
            [cx, cy] = m.center!;
          M.premultiply(
            new THREE.Matrix4()
              .makeTranslation(cx, cy, 0)
              .multiply(new THREE.Matrix4().makeRotationZ(a))
              .multiply(new THREE.Matrix4().makeTranslation(-cx, -cy, 0)),
          );
        } else if (m && m.type === "translate") M.premultiply(new THREE.Matrix4().makeTranslation(m.t![0] * e, m.t![1] * e, 0));
        G[i] = M.multiply(zMat((pc.h[0] + (pc.h[1] - pc.h[0]) * e) * DZ));
      } else {
        const th = hingeAngle(pc, t, e);
        // layer offset follows the hinge's own progress (or the whole step if the angle doesn't change)
        const span = pc.theta[1] - pc.theta[0];
        const f = Math.abs(span) > 1e-9 && !pc.keys ? Math.min(1, Math.max(0, (th - pc.theta[0]) / span)) : e;
        const dl = pc.delta[0] + (pc.delta[1] - pc.delta[0]) * f;
        G[i] = G[pc.parent].clone().multiply(hingeMat(pc.hinge[0], pc.hinge[1], th, _m)).multiply(zMat(dl * DZ));
      }
    }
    return G;
  }

  // bend strip that follows its hinge while the two pieces lie folded flat on each other
  private makeKinBend(b: KinBendJson): KinBend {
    const pos = new THREE.BufferAttribute(new Float32Array((BK + 1) * 6), 3);
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", pos);
    const [p, q] = b.seg,
      n = b.n;
    const cr = (q[0] - p[0]) * n[1] - (q[1] - p[1]) * n[0]; // winding: front of paper = +z of the frame piece
    const idx: number[] = [];
    for (let i = 0; i < BK; i++) {
      const a0 = 2 * i,
        b0 = 2 * i + 1,
        a1 = 2 * i + 2,
        b1 = 2 * i + 3;
      if (cr > 0) idx.push(a0, b0, a1, b0, b1, a1);
      else idx.push(a0, a1, b0, b0, a1, b1);
    }
    geom.setIndex(idx);
    const meshes = [this.frontMat, this.backMat].map((m) => {
      const o = new THREE.Mesh(geom, m);
      o.frustumCulled = false;
      this.paper.add(o);
      return o;
    });
    const v = new THREE.Vector3(),
      inv = new THREE.Matrix4();
    return {
      update: (G, W) => {
        const Gf = G[b.f],
          Go = G[b.o];
        // visible only while o lies folded flat onto f (normals opposite)
        const ef = Gf.elements,
          eo = Go.elements;
        const dot = ef[8] * eo[8] + ef[9] * eo[9] + ef[10] * eo[10];
        // o's surface height in f's frame
        v.set(p[0], p[1], 0).applyMatrix4(Go).applyMatrix4(inv.copy(Gf).invert());
        const z1 = v.z,
          r = Math.abs(z1) / 2;
        const show = dot < -0.995 && r > 1e-9;
        for (const o of meshes) o.visible = show;
        if (!show) return;
        const sg = Math.sign(z1),
          zc = z1 / 2;
        for (let i = 0; i <= BK; i++) {
          const ph = (Math.PI * i) / BK,
            du = r * Math.sin(ph),
            z = zc - sg * r * Math.cos(ph);
          for (const [k, s] of [
            [0, p],
            [1, q],
          ] as [number, Vec2][]) {
            v.set(s[0] + n[0] * du, s[1] + n[1] * du, z).applyMatrix4(Gf);
            pos.setXYZ(2 * i + k, ...this.wpt(W, v.x, v.y, v.z));
          }
        }
        pos.needsUpdate = true;
        geom.computeVertexNormals();
      },
    };
  }

  private buildStepPieces(step: StepJson) {
    this.clearGroup(this.paper);
    this.animPieces = step.pieces.map((pc) => ({ pc, obj: this.makePiece(pc.orig, pc.lines || []) }));
    this.animBends = (step.kin_bends || []).map((b) => this.makeKinBend(b));
  }

  // t: linear progress 0..1 over the whole step (eased inside)
  private poseStep(step: StepJson, t: number) {
    const W = this.tl.states[step.index - 1].world;
    const G = this.kinFrames(step, t),
      v = new THREE.Vector3();
    this.animPieces!.forEach(({ obj }, i) =>
      obj.update((p) => {
        v.set(p[0], p[1], 0).applyMatrix4(G[i]);
        return this.wpt(W, v.x, v.y, v.z);
      }),
    );
    for (const b of this.animBends) b.update(G, W);
  }
}

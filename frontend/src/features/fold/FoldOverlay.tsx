// Drawn over the top-down paper in Fold mode: tappable point stickers, the chosen points,
// and the preview (fold line + a ghost of the result). Engine coords map to the screen via the viewer.
// Taps snap to a labeled point, then a paper edge, then anywhere on the paper; dragging from a point
// shows the live "fold P onto Q" crease and previews it on release.
import { useEffect, useReducer, useRef, useState } from "react";
import type { Vec2 } from "../../lib/timeline";
import { getViewer } from "../../viewer/Viewer";
import { useFold } from "./foldStore";
import { bisectorSegment, facePolys, samePoint, snapAt, type Snap } from "./snap";

const SNAP_PX = 28;
const EDGE_PX = 14;
const TOUCH_SNAP_PX = 40;
const TOUCH_EDGE_PX = 26;
const DRAG_PX = 10;

interface Drag {
  start: Snap;
  at: Vec2;
  cur: Snap | null;
  moved: boolean;
}

export function FoldOverlay() {
  const { points, drawing, draw, preview, selected, phase, view3d, coloredSide, busy } = useFold();
  const tapPoint = useFold((s) => s.tapPoint);
  const dragOnto = useFold((s) => s.dragOnto);
  const timeline = useFold((s) => s.timeline);
  const [hover, setHover] = useState<Snap | null>(null);
  const [drag, setDrag] = useState<Drag | null>(null);
  const [, redraw] = useReducer((x: number) => x + 1, 0);
  const svgRef = useRef<SVGSVGElement>(null);
  const hidden = useRef(true);

  // follow the camera (resize, flips, the panel sliding) and hide while the paper is moving
  useEffect(() => {
    let key = "";
    let raf = 0;
    const tick = () => {
      const v = getViewer();
      const a = v?.toScreen([0, 0]),
        b = v?.toScreen([1, 1]);
      const moving = !v || v.info.busy || (v.info.t > 0 && v.info.t < 1);
      const k = `${moving}|${a?.map((x) => x.toFixed(1))}|${b?.map((x) => x.toFixed(1))}`;
      if (k !== key) {
        key = k;
        hidden.current = moving;
        redraw();
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  const v = getViewer();
  if (phase !== "folding" || view3d || !v || hidden.current) return null;
  const S = (p: Vec2) => v.toScreen(p) ?? ([0, 0] as Vec2);

  const polys = facePolys(timeline?.states[timeline.states.length - 1]);
  const local = (e: React.PointerEvent<SVGSVGElement>): Vec2 => {
    const r = svgRef.current!.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  };
  const snap = (e: React.PointerEvent<SVGSVGElement>, at: Vec2) => {
    const r = svgRef.current!.getBoundingClientRect();
    const touch = e.pointerType === "touch";
    return snapAt(at, {
      labels: visible,
      polys,
      toScreen: S,
      fromScreen: (s) => v.pickPoint(s[0] + r.left, s[1] + r.top),
      pointPx: touch ? TOUCH_SNAP_PX : SNAP_PX,
      edgePx: touch ? TOUCH_EDGE_PX : EDGE_PX,
    });
  };

  const onPointerDown = (e: React.PointerEvent<SVGSVGElement>) => {
    if (busy || !e.isPrimary) return;
    const at = local(e);
    const start = snap(e, at);
    if (!start) return;
    e.preventDefault();
    e.currentTarget.setPointerCapture(e.pointerId);
    setHover(null);
    setDrag({ start, at, cur: null, moved: false });
  };
  const onPointerMove = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!e.isPrimary) return;
    const at = local(e);
    if (drag) {
      const moved = drag.moved || Math.hypot(at[0] - drag.at[0], at[1] - drag.at[1]) > DRAG_PX;
      setDrag({ ...drag, moved, cur: moved ? snap(e, at) : null });
    } else if (e.pointerType !== "touch" && !busy) {
      setHover(snap(e, at));
    }
  };
  const onPointerUp = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!drag || !e.isPrimary) return;
    setDrag(null);
    if (!drag.moved) tapPoint(drag.start.ref);
    else if (drag.cur && !samePoint(drag.cur.ref, drag.start.ref)) dragOnto(drag.start.ref, drag.cur.ref);
  };

  const dragging = drag?.moved ? drag : null;
  const liveLine = dragging?.cur ? bisectorSegment(dragging.start.xy, dragging.cur.xy, polys) : null;
  const ghost = !dragging && preview?.ok && preview.state;
  const line = dragging ? liveLine : preview?.line;
  const selOp = preview?.step?.op ?? (selected?.kind === "menu" ? selected.option.step.op : draw.op);
  const showLetters = drawing;
  const kami = coloredSide === "up";
  // one sticker per spot (stacked points share it), chosen points drawn last so they stay on top
  const seen = new Set<string>();
  const visible = points
    .filter((p) => {
      if (seen.has(p.label)) return false;
      p.stacked.forEach((l) => seen.add(l));
      return true;
    })
    .map((p) => (typeof draw.p === "string" && p.stacked.includes(draw.p) ? { ...p, label: draw.p } : typeof draw.q === "string" && p.stacked.includes(draw.q) ? { ...p, label: draw.q } : p))
    .sort((a, b) => Number(a.label === draw.p || a.label === draw.q) - Number(b.label === draw.p || b.label === draw.q));

  return (
    <svg
      ref={svgRef}
      className="fold-overlay"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={() => setDrag(null)}
      onPointerLeave={() => setHover(null)}
      aria-hidden="true"
    >
      {ghost &&
        [...ghost.faces]
          .sort((a, b) => a.layer - b.layer)
          .map((f) => {
            const pts = f.orig.map((o) => S([f.T[0][0] * o[0] + f.T[0][1] * o[1] + f.T[0][2], f.T[1][0] * o[0] + f.T[1][1] * o[1] + f.T[1][2]]));
            const colored = f.front_up === kami;
            return <polygon key={f.id} className={colored ? "ghost ghost-kami" : "ghost"} points={pts.map((p) => p.join(",")).join(" ")} />;
          })}
      {line && (() => {
        const [a, b] = [S(line[0]), S(line[1])];
        return <line className={!dragging && selOp === "mountain" ? "fold-line is-mountain" : "fold-line"} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} />;
      })()}
      {dragging?.cur && (() => {
        const [a, b] = [S(dragging.start.xy), S(dragging.cur.xy)];
        return <line className="drag-arrow" x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} />;
      })()}
      {preview?.move && (() => {
        const [x, y] = S(preview.move);
        return <circle className="move-dot" cx={x} cy={y} r={5} />;
      })()}
      {visible.map((p) => {
        const [x, y] = S(p.xy);
        const chosen = p.label === draw.p ? "P" : p.label === draw.q ? "Q" : null;
        return (
          <g key={p.label} className={`pt-sticker${chosen ? " is-chosen" : ""}${showLetters ? "" : " is-small"}`} transform={`translate(${x},${y})`}>
            <circle r={chosen ? 13 : showLetters ? 11 : 8.5} />
            <text dy="0.35em">{chosen ?? p.label}</text>
          </g>
        );
      })}
      {(dragging ? [dragging.start.ref, dragging.cur?.ref ?? null] : [draw.p, draw.q]).map((r, i) =>
        r !== null && typeof r !== "string" ? <Chosen key={i} at={S(r)} letter={i ? "Q" : "P"} /> : null,
      )}
      {dragging?.cur && <SnapMarker snap={dragging.cur} at={S(dragging.cur.xy)} />}
      {!drag && hover && <SnapMarker snap={hover} at={S(hover.xy)} />}
    </svg>
  );
}

function Chosen({ at, letter }: { at: Vec2; letter: string }) {
  return (
    <g className="pt-sticker is-chosen is-free" transform={`translate(${at[0]},${at[1]})`}>
      <circle r={11} />
      <text dy="0.35em">{letter}</text>
    </g>
  );
}

function SnapMarker({ snap, at }: { snap: Snap; at: Vec2 }) {
  const t = `translate(${at[0]},${at[1]})`;
  if (snap.kind === "point") return <circle className="snap-marker is-point" transform={t} r={16} />;
  if (snap.kind === "edge") return <rect className="snap-marker is-edge" transform={`${t} rotate(45)`} x={-6} y={-6} width={12} height={12} />;
  return <path className="snap-marker is-free" transform={t} d="M-7 0H7M0 -7V7" />;
}


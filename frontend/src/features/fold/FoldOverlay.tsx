// Drawn over the top-down paper in Fold mode: tappable point stickers, the chosen points,
// and the preview (fold line + a ghost of the result). Engine coords map to the screen via the viewer.
import { useEffect, useReducer, useRef } from "react";
import type { Vec2 } from "../../lib/timeline";
import { getViewer } from "../../viewer/Viewer";
import { useFold } from "./foldStore";

const SNAP_PX = 28;

export function FoldOverlay() {
  const { points, drawing, draw, preview, selected, phase, view3d, coloredSide, busy } = useFold();
  const tapPoint = useFold((s) => s.tapPoint);
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

  const onPointerDown = (e: React.PointerEvent<SVGSVGElement>) => {
    if (busy) return;
    const r = svgRef.current!.getBoundingClientRect();
    const x = e.clientX - r.left,
      y = e.clientY - r.top;
    let best: string | null = null,
      bd = SNAP_PX;
    for (const p of visible) {
      const [px, py] = S(p.xy);
      const d = Math.hypot(px - x, py - y);
      if (d < bd) {
        bd = d;
        best = p.label;
      }
    }
    if (best) {
      e.preventDefault();
      tapPoint(best);
    }
  };

  const ghost = preview?.ok && preview.state;
  const line = preview?.line;
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
    .map((p) => (draw.p && p.stacked.includes(draw.p) ? { ...p, label: draw.p } : draw.q && p.stacked.includes(draw.q) ? { ...p, label: draw.q } : p))
    .sort((a, b) => Number(a.label === draw.p || a.label === draw.q) - Number(b.label === draw.p || b.label === draw.q));

  return (
    <svg ref={svgRef} className="fold-overlay" onPointerDown={onPointerDown} aria-hidden="true">
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
        return <line className={selOp === "mountain" ? "fold-line is-mountain" : "fold-line"} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} />;
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
    </svg>
  );
}

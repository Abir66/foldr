import { memo } from "react";
import { stateSvg, type Lin2 } from "../../lib/stateSvg";
import type { StateJson } from "../../lib/timeline";

/** Small flat picture of a paper state (menu option result). */
export const StateThumb = memo(function StateThumb({
  state,
  coloredSide,
  lin,
  size = 56,
}: {
  state: StateJson;
  coloredSide: "up" | "down";
  lin?: Lin2 | null;
  size?: number;
}) {
  const s = stateSvg(state, coloredSide, lin ?? undefined, size);
  return (
    <svg className="state-thumb" width={size} height={size} viewBox={s.viewBox} aria-hidden="true">
      {s.faces.map((f, i) => (
        <path key={i} d={f.d} className={f.colored ? "face face-kami" : "face"} strokeWidth={s.unit} />
      ))}
    </svg>
  );
});

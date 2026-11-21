import { useRef, type ReactNode } from "react";
import { useUi, type SheetSnap } from "../lib/store";

const ORDER: SheetSnap[] = ["peek", "half", "full"];

/** Right-hand sheet on desktop; a bottom sheet with a handle (peek / half / full) on phones. */
export function Panel({ label, children }: { label: string; children: ReactNode }) {
  const sheet = useUi((s) => s.sheet);
  const setSheet = useUi((s) => s.setSheet);
  const drag = useRef<{ y: number; moved: boolean } | null>(null);
  const suppressClick = useRef(false);

  const step = (dir: 1 | -1) => {
    const i = ORDER.indexOf(sheet) + dir;
    setSheet(ORDER[Math.min(Math.max(i, 0), ORDER.length - 1)]);
  };

  return (
    <aside className="panel" data-snap={sheet} aria-label={label}>
      <button
        className="sheet-handle"
        aria-label={`Panel size: ${sheet}. Tap to change.`}
        onPointerDown={(e) => {
          drag.current = { y: e.clientY, moved: false };
          (e.target as HTMLElement).setPointerCapture(e.pointerId);
        }}
        onPointerMove={(e) => {
          if (drag.current && Math.abs(e.clientY - drag.current.y) > 8) drag.current.moved = true;
        }}
        onPointerUp={(e) => {
          const d = drag.current;
          drag.current = null;
          if (d?.moved) {
            suppressClick.current = true;
            step(e.clientY < d.y ? 1 : -1);
          }
        }}
        onClick={() => {
          if (suppressClick.current) {
            suppressClick.current = false;
            return;
          }
          setSheet(ORDER[(ORDER.indexOf(sheet) + 1) % ORDER.length]);
        }}
      >
        <span className="sheet-grip" />
      </button>
      <div className="panel-body">{children}</div>
    </aside>
  );
}

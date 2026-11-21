import { useUi } from "../../lib/store";
import { getViewer } from "../../viewer/Viewer";
import { follow, useAi } from "./aiStore";

/** "Jump to live" when the user has scrubbed back during a run that's still going. */
export function LivePill({ n }: { n: number }) {
  const step = useUi((s) => s.step);
  const playing = useUi((s) => s.playing);
  const going = useAi((s) => !!s.source && !s.run.finished);
  const phase = useAi((s) => s.run.status?.phase);
  if (!going) return null;
  if (playing || step >= n) {
    if (!phase || !playing) return null;
    return (
      <p className="ai-pill ai-pill-status" aria-live="polite">
        <span className="ai-dot" /> {phase === "thinking" ? "AI is thinking" : phase === "planning" ? "Making a plan" : phase === "menu" ? "Building choices" : "Folding"}
      </p>
    );
  }
  return (
    <button
      className="ai-pill"
      onClick={() => {
        getViewer()?.goto(n);
        follow();
      }}
    >
      Jump to live
    </button>
  );
}

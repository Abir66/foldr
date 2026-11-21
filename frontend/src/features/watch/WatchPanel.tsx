import { useEffect, useRef } from "react";
import type { Example } from "../../lib/examples";
import { useUi } from "../../lib/store";
import { useExampleTimeline } from "../../lib/timeline";
import { OpBadge, StepNum } from "../../ui/Notation";
import { replayStep } from "../../viewer/Viewer";
import { ExportMenu } from "./ExportMenu";

export function WatchPanel({ example }: { example: Example }) {
  const timeline = useExampleTimeline(example.name);
  const steps = timeline?.steps ?? example.steps;
  const n = steps.length;
  const step = useUi((s) => s.step);
  const listRef = useRef<HTMLOListElement>(null);

  // Keep the current row in view as playback moves (inside the list only, not the page).
  useEffect(() => {
    const scroller = listRef.current;
    const row = scroller?.querySelector<HTMLElement>('[aria-current="step"]');
    if (!row || !scroller) return;
    const r = row.getBoundingClientRect();
    const s = scroller.getBoundingClientRect();
    if (r.top < s.top || r.bottom > s.bottom) scroller.scrollTop += r.top - s.top - s.height / 3;
  }, [step]);

  const cur = step > 0 ? steps[step - 1] : null;

  return (
    <div className="watch">
      <div className="watch-top">
        <div className="watch-head">
          <h1 className="model-title">{example.label}</h1>
          <ExportMenu example={example} />
        </div>
        <p className="model-meta">{n} steps from one square</p>

        <div className="now" aria-live="polite">
          {cur ? (
            <>
              <StepNum n={step} current />
              <p className="now-text">{cur.description}</p>
            </>
          ) : (
            <p className="now-text now-start">Start with a flat square. Press play to fold.</p>
          )}
        </div>
      </div>

      <ol className="steps" ref={listRef} aria-label="All steps">
        {steps.map((s, i) => {
          const k = i + 1;
          const current = k === step;
          return (
            <li key={k}>
              <button
                className="step-row"
                aria-current={current ? "step" : undefined}
                title="Play this step"
                onClick={() => replayStep(k)}
              >
                <StepNum n={k} current={current} />
                <OpBadge op={s.op} />
                <span className="step-text">{s.description}</span>
              </button>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

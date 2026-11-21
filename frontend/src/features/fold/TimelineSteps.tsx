import { useEffect, useRef } from "react";
import type { Source } from "../../app/source";
import { useUi } from "../../lib/store";
import { OpBadge, StepNum } from "../../ui/Notation";
import { replayStep } from "../../viewer/Viewer";

/** Clickable list of a timeline's steps; the current one stays in view. */
export function StepList({ steps, label = "All steps" }: { steps: { op: string; description: string }[]; label?: string }) {
  const step = useUi((s) => s.step);
  const listRef = useRef<HTMLOListElement>(null);

  useEffect(() => {
    const scroller = listRef.current;
    const row = scroller?.querySelector<HTMLElement>('[aria-current="step"]');
    if (!row || !scroller) return;
    const r = row.getBoundingClientRect();
    const s = scroller.getBoundingClientRect();
    if (r.top < s.top || r.bottom > s.bottom) scroller.scrollTop += r.top - s.top - s.height / 3;
  }, [step, steps.length]);

  return (
    <ol className="steps" ref={listRef} aria-label={label}>
      {steps.map((s, i) => {
        const k = i + 1;
        const current = k === step;
        return (
          <li key={k}>
            <button className="step-row" aria-current={current ? "step" : undefined} title="Play this step" onClick={() => replayStep(k)}>
              <StepNum n={k} current={current} />
              <OpBadge op={s.op} />
              <span className="step-text">{s.description}</span>
            </button>
          </li>
        );
      })}
    </ol>
  );
}

/** The step being shown, in words. */
export function NowStep({ source }: { source: Source }) {
  const step = useUi((s) => s.step);
  const cur = step > 0 ? source.steps[step - 1] : null;
  return (
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
  );
}

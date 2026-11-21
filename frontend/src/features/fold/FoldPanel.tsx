import { useEffect, useState } from "react";
import type { Source } from "../../app/source";
import { hrefFor } from "../../lib/router";
import { useUi } from "../../lib/store";
import type { MenuOption } from "../../worker/protocol";
import { ensurePython, usePython } from "../../worker/rpc";
import { foldFromHere } from "./foldFromHere";
import { SECTION_TITLES, useFold, type Section } from "./foldStore";
import { StateThumb } from "./StateThumb";
import { NowStep, StepList } from "./TimelineSteps";
import "./fold.css";

/** Fold mode side panel: the steps of what is on the canvas, or your own folding tools. */
export function FoldPanel({ source }: { source: Source }) {
  const phase = useFold((s) => s.phase);

  // entering your own folding starts the download while the user picks a side
  useEffect(() => {
    if (source.kind === "own") ensurePython("core").catch(() => {});
  }, [source.kind]);

  if (source.kind !== "own") return <Watching source={source} />;
  return (
    <div className="fold">
      <h1 className="model-title">Fold it yourself</h1>
      <p className="fold-sample">
        <button className="text-link" onClick={() => useUi.getState().setLibraryOpen(true)}>
          Load a sample
        </button>
      </p>
      {phase === "folding" ? <Folding source={source} /> : <StartSheet />}
    </div>
  );
}

function FromHereButton({ source, primary = false }: { source: Source; primary?: boolean }) {
  const step = useUi((s) => s.step);
  const busy = useFold((s) => !!s.busy || s.phase === "loading");
  const [working, setWorking] = useState(false);
  return (
    <button
      className={`button ${primary ? "button-primary" : "button-quiet"}`}
      disabled={busy || working}
      title={`Start folding from the paper at step ${step}`}
      onClick={async () => {
        setWorking(true);
        try {
          await foldFromHere(source, step);
        } finally {
          setWorking(false);
        }
      }}
    >
      Fold yourself from step {step}
    </button>
  );
}

function Watching({ source }: { source: Source }) {
  return (
    <div className="watch">
      <div className="watch-top">
        <h1 className="model-title">{source.label}</h1>
        <p className="model-meta">{source.n} steps from one square</p>
        <div className="watch-actions">
          <FromHereButton source={source} primary />
          <a className="button button-quiet" href={hrefFor("fold")}>
            Start from scratch
          </a>
        </div>
        <NowStep source={source} />
      </div>
      <StepList steps={source.steps} />
    </div>
  );
}

function LoadBar() {
  const { core, progress, error } = usePython();
  if (core === "error") return <p className="fold-msg is-error" role="alert">{error ?? "Python could not load."} This needs a newer browser.</p>;
  if (core !== "loading") return null;
  const frac = progress ? Math.min(0.97, progress.loaded / progress.total) : 0.02;
  return (
    <div className="loadbar" role="status" aria-live="polite">
      <p className="loadbar-label">
        Getting the paper ready… <span className="loadbar-detail">{progress?.label ?? "Starting"}</span>
      </p>
      <div className="loadbar-track" aria-hidden="true">
        <div className="loadbar-fill" style={{ width: `${frac * 100}%` }} />
      </div>
      <p className="hint">
        The fold checker runs in your browser. The first visit downloads about 7 MB
        {progress ? ` (${(progress.loaded / 1e6).toFixed(1)} of ~${(progress.total / 1e6).toFixed(0)} MB unpacked)` : ""}.
      </p>
    </div>
  );
}

function StartSheet() {
  const side = useUi((s) => s.foldSide);
  const setSide = useUi((s) => s.setFoldSide);
  const { phase, message, start, busy } = useFold();
  const core = usePython((s) => s.core);
  const starting = phase === "loading";
  return (
    <>
      <p className="model-meta">Tap a suggested fold, or tap two points on the paper to draw your own.</p>
      <fieldset className="choice">
        <legend>Start with</legend>
        <label className="choice-opt">
          <input type="radio" name="side" checked={side === "white"} onChange={() => setSide("white")} />
          <span>White side up</span>
        </label>
        <label className="choice-opt">
          <input type="radio" name="side" checked={side === "colored"} onChange={() => setSide("colored")} />
          <span>Coloured side up</span>
        </label>
      </fieldset>
      <button className="button button-primary fold-start" disabled={starting || core === "error"} onClick={() => start(side === "colored" ? "up" : "down")}>
        {starting ? "Getting ready…" : "Start folding"}
      </button>
      {starting && busy && <p className="fold-msg" role="status">{busy}</p>}
      <LoadBar />
      {message && <p className="fold-msg is-error" role="alert">{message.text}</p>}
    </>
  );
}

function Folding({ source }: { source: Source }) {
  const { steps, busy, message, selected, drawing } = useFold();
  const step = useUi((s) => s.step);
  return (
    <>
      <Toolbar />
      {step < steps.length && (
        <div className="fold-behind" role="status">
          <p className="hint">
            You are looking at step {step}. New folds go after step {steps.length}.
          </p>
          <FromHereButton source={source} />
        </div>
      )}
      {busy && <p className="fold-msg" role="status">{busy}</p>}
      {message && (
        <p className={`fold-msg ${message.kind === "error" ? "is-error" : ""}`} role="alert">
          {message.text}
        </p>
      )}
      {(selected || drawing) && <ActionBar />}
      {steps.length > 0 && (
        <section className="mysteps">
          <h2 className="section-title">Your steps</h2>
          <StepList steps={source.steps} label="Your steps" />
        </section>
      )}
      <Suggestions />
    </>
  );
}

function Toolbar() {
  const { steps, busy, undo, startOver, setDrawing, drawing } = useFold();
  const disabled = !!busy;
  return (
    <div className="toolbar fold-toolbar" role="toolbar" aria-label="Folding tools">
      <button className={`button ${drawing ? "button-primary" : "button-quiet"}`} aria-pressed={drawing} disabled={disabled} onClick={() => setDrawing(!drawing)}>
        Draw a fold
      </button>
      <button className="button button-quiet" disabled={disabled || !steps.length} onClick={undo}>
        Undo
      </button>
      <button
        className="button button-quiet"
        disabled={disabled}
        onClick={() => {
          if (!steps.length || window.confirm("Start over with a new sheet?")) startOver();
        }}
      >
        Start over
      </button>
    </div>
  );
}

function ActionBar() {
  const { selected, preview, previewLoading, busy, drawing, draw, setDraw, applySelected, select, setDrawing } = useFold();
  const ok = preview?.ok;
  const title =
    selected?.kind === "menu"
      ? selected.option.description
      : draw.p && draw.q
        ? draw.kind === "through"
          ? "Line through P and Q"
          : "Fold P onto Q"
        : draw.p
          ? "Now tap point Q"
          : "Tap point P anywhere on the paper, or drag it onto Q";
  return (
    <section className="actionbar" aria-label="Chosen fold">
      <p className="actionbar-title">{title}</p>
      {drawing && draw.p && draw.q && (
        <div className="draw-opts">
          <Seg
            label="Fold"
            value={draw.kind}
            options={[
              ["through", "Line through P and Q"],
              ["onto", "P onto Q"],
            ]}
            onChange={(kind) => setDraw({ kind: kind as "through" | "onto", side: 1 })}
          />
          <Seg
            label="Kind"
            value={draw.op}
            options={[
              ["valley", "Valley"],
              ["mountain", "Mountain"],
            ]}
            onChange={(op) => setDraw({ op: op as "valley" | "mountain" })}
          />
          <Seg
            label="Layers"
            value={draw.layers}
            options={[
              ["all", "All"],
              ["top", "Top layer"],
              ["flap", "Top flap"],
            ]}
            onChange={(layers) => setDraw({ layers: layers as "all" | "top" | "flap" })}
          />
        </div>
      )}
      {previewLoading && <p className="hint">Checking the fold…</p>}
      {preview && !ok && <p className="fold-msg is-error">{preview.text.replace(/^REJECTED \[[A-Z_]+\]:\s*/, "")}</p>}
      <div className="actionbar-buttons">
        <button className="button button-primary" disabled={!ok || !!busy || previewLoading} onClick={applySelected}>
          Fold
        </button>
        {drawing && draw.p && draw.q && (
          <button className="button button-quiet" onClick={() => setDraw({ side: draw.side === 1 ? -1 : 1 })}>
            Switch side
          </button>
        )}
        <button className="button button-quiet" onClick={() => (drawing ? setDrawing(false) : select(null))}>
          Cancel
        </button>
      </div>
    </section>
  );
}

function Seg({ label, value, options, onChange }: { label: string; value: string; options: [string, string][]; onChange: (v: string) => void }) {
  return (
    <div className="seg" role="radiogroup" aria-label={label}>
      {options.map(([v, text]) => (
        <button key={v} role="radio" aria-checked={value === v} className="seg-opt" onClick={() => onChange(v)}>
          {text}
        </button>
      ))}
    </div>
  );
}

function Suggestions() {
  const { sections, menuLoading } = useFold();
  return (
    <section className="suggest" aria-label="Suggested folds" aria-busy={menuLoading}>
      <h2 className="section-title">Suggested folds</h2>
      {menuLoading && !sections.length && <p className="hint">Finding folds…</p>}
      {sections.map((s) => (
        <SectionList key={s.name} sec={s} />
      ))}
      {menuLoading && sections.length > 0 && <p className="hint">Finding more…</p>}
      {!menuLoading && !sections.length && <p className="hint">No suggestions here. Try drawing a fold.</p>}
    </section>
  );
}

function SectionList({ sec }: { sec: Section }) {
  const { selected, select, showMore, coloredSide, lin, busy, applySelected } = useFold();
  const cur = selected?.kind === "menu" ? selected.option : null;
  return (
    <div className="suggest-sec">
      <h3 className="suggest-title">{SECTION_TITLES[sec.name] ?? sec.name}</h3>
      <ul className="suggest-list">
        {sec.options.map((o: MenuOption) => {
          const on = cur?.id === o.id && cur.section === o.section;
          return (
            <li key={`${o.section}-${o.id}`}>
              <button
                className="suggest-row"
                aria-pressed={on}
                disabled={!!busy}
                onClick={() => (on ? applySelected() : select(o))}
                title={on ? "Tap again to fold" : "Preview this fold"}
              >
                {o.result_state ? <StateThumb state={o.result_state} coloredSide={coloredSide} lin={lin} /> : <span className="state-thumb" />}
                <span className="suggest-text">{o.description}</span>
              </button>
            </li>
          );
        })}
      </ul>
      {!sec.exhausted && (
        <button className="text-link suggest-more" disabled={sec.loading || !!busy} onClick={() => showMore(sec.name)}>
          {sec.loading ? "Looking…" : sec.truncated ? "Show these folds" : "Show more"}
        </button>
      )}
      {sec.truncated && !sec.loading && <p className="hint">Skipped to save time.</p>}
    </div>
  );
}

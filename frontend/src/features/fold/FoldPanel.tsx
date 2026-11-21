import { useEffect, useRef, useState } from "react";
import { useUi } from "../../lib/store";
import { showInWatch } from "../../lib/openLocal";
import { OpBadge, StepNum } from "../../ui/Notation";
import type { ModelJson, MenuOption } from "../../worker/protocol";
import { ensurePython, usePython } from "../../worker/rpc";
import { SECTION_TITLES, useFold, type Section } from "./foldStore";
import { StateThumb } from "./StateThumb";
import "./fold.css";

export function FoldPanel() {
  const phase = useFold((s) => s.phase);

  // entering Fold starts the download while the user picks a side
  useEffect(() => {
    ensurePython("core").catch(() => {});
  }, []);

  return (
    <div className="fold">
      <h1 className="model-title">Fold it yourself</h1>
      {phase === "folding" ? <Folding /> : <StartSheet />}
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
  const { phase, message, start } = useFold();
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
          <input type="radio" name="side" checked={side === "blue"} onChange={() => setSide("blue")} />
          <span>Blue side up</span>
        </label>
      </fieldset>
      <button className="button button-primary fold-start" disabled={starting || core === "error"} onClick={() => start(side === "blue" ? "up" : "down")}>
        {starting ? "Getting ready…" : "Start folding"}
      </button>
      <LoadBar />
      {message && <p className="fold-msg is-error" role="alert">{message.text}</p>}
    </>
  );
}

function Folding() {
  const { steps, busy, message, selected, drawing } = useFold();
  return (
    <>
      <Toolbar />
      {busy && <p className="fold-msg" role="status">{busy}</p>}
      {message && (
        <p className={`fold-msg ${message.kind === "error" ? "is-error" : ""}`} role="alert">
          {message.text}
        </p>
      )}
      {(selected || drawing) && <ActionBar />}
      {steps.length > 0 && <MySteps />}
      <Suggestions />
    </>
  );
}

function Toolbar() {
  const { steps, busy, view3d, apply, undo, startOver, setView3d, setDrawing, drawing } = useFold();
  const disabled = !!busy;
  return (
    <div className="toolbar fold-toolbar" role="toolbar" aria-label="Folding tools">
      <button className="button button-quiet" disabled={disabled || !steps.length} onClick={undo}>
        Undo
      </button>
      <button className="button button-quiet" disabled={disabled} onClick={() => apply({ op: "flip" })}>
        Turn over
      </button>
      <button className="button button-quiet" disabled={disabled} onClick={() => apply({ op: "rotate", degrees: 45 })}>
        Rotate 45°
      </button>
      <button className={`button ${drawing ? "button-primary" : "button-quiet"}`} aria-pressed={drawing} disabled={disabled} onClick={() => setDrawing(!drawing)}>
        Draw a fold
      </button>
      <button className="button button-quiet" aria-pressed={view3d} onClick={() => setView3d(!view3d)}>
        {view3d ? "Flat view" : "3D view"}
      </button>
      <ExportFolds />
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

function ExportFolds() {
  const [open, setOpen] = useState(false);
  const { steps, busy, exportFiles, myModel, timeline } = useFold();
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const close = (e: PointerEvent) => {
      if (!ref.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", close);
    return () => document.removeEventListener("pointerdown", close);
  }, [open]);
  const none = !steps.length;
  const run = (f: () => unknown) => () => {
    setOpen(false);
    void f();
  };
  return (
    <div className="export fold-export" ref={ref}>
      <button className="button button-quiet" aria-expanded={open} aria-haspopup="menu" disabled={none || !!busy} onClick={() => setOpen(!open)}>
        Export
      </button>
      {open && (
        <div className="export-menu" role="menu">
          <button role="menuitem" className="export-item" onClick={run(async () => {
            const model = (await myModel()) as ModelJson;
            if (timeline) showInWatch(model, timeline, "My fold");
          })}>
            Watch my folds
          </button>
          <button role="menuitem" className="export-item" onClick={run(() => exportFiles(["model_json"]))}>
            Steps (.json)
          </button>
          <button role="menuitem" className="export-item" onClick={run(() => exportFiles(["fold"]))}>
            Crease pattern (.fold)
          </button>
          <button role="menuitem" className="export-item" onClick={run(() => exportFiles(["png_zip"]))}>
            Step pictures (.zip)
            <span className="export-soon">loads 4 MB more</span>
          </button>
        </div>
      )}
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
          : "Tap point P on the paper";
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

function MySteps() {
  const steps = useFold((s) => s.steps);
  return (
    <section className="mysteps">
      <h2 className="section-title">Your steps</h2>
      <ol className="steps fold-steps">
        {steps.map((s, i) => (
          <li key={i} className="step-row is-static">
            <StepNum n={i + 1} current={i === steps.length - 1} />
            <OpBadge op={s.op} />
            <span className="step-text">{s.text}</span>
          </li>
        ))}
      </ol>
    </section>
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

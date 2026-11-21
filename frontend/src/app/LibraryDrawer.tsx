import { useEffect, useRef, useState } from "react";
import { EXAMPLES, RUNS } from "../lib/examples";
import { hrefFor, type Route } from "../lib/router";
import { useUi } from "../lib/store";
import { Thumb } from "../ui/Thumb";
import { RunMark } from "../features/ai/RunMark";
import { openFile } from "../lib/openLocal";

export function LibraryDrawer({ route }: { route: Route }) {
  const open = useUi((s) => s.libraryOpen);
  const setOpen = useUi((s) => s.setLibraryOpen);
  const setAboutOpen = useUi((s) => s.setAboutOpen);
  const ref = useRef<HTMLDialogElement>(null);
  const [fileNote, setFileNote] = useState<string | null>(null);

  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
  }, [open]);

  const close = () => setOpen(false);
  const onFile = async (f: File | undefined) => {
    if (!f) return;
    setFileNote(`Opening “${f.name}”… (the first time this downloads about 7 MB)`);
    try {
      const r = await openFile(f);
      setFileNote(r.note ?? null);
      if (r.ok && !r.note) close();
    } catch (e) {
      setFileNote(`Could not open “${f.name}”: ${(e as Error).message}`);
    }
  };

  return (
    <dialog
      ref={ref}
      className="drawer"
      aria-labelledby="library-title"
      onClose={close}
      onClick={(e) => {
        if (e.target === ref.current) close();
      }}
    >
      <div
        className="drawer-inner"
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          onFile(e.dataTransfer.files[0]);
        }}
      >
        <div className="drawer-head">
          <h2 id="library-title" className="drawer-title">Library</h2>
          <button className="icon-button" onClick={close} aria-label="Close the menu">
            <svg width="20" height="20" viewBox="0 0 20 20" aria-hidden="true">
              <path d="M5 5l10 10M15 5 5 15" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
            </svg>
          </button>
        </div>

        <h3 className="drawer-section">Examples</h3>
        <ul className="lib-list">
          {EXAMPLES.map((e) => {
            const current = route.mode === "watch" && route.example === e.name;
            return (
              <li key={e.name}>
                <a className="lib-row" href={hrefFor("watch", e.name)} onClick={close} aria-current={current ? "true" : undefined}>
                  <Thumb name={e.name} size={48} className="lib-thumb" />
                  <span className="lib-text">
                    <span className="lib-name">{e.label}</span>
                    <span className="lib-meta">{e.steps.length} steps</span>
                  </span>
                </a>
              </li>
            );
          })}
        </ul>

        <h3 className="drawer-section">AI runs</h3>
        <ul className="lib-list">
          {RUNS.map((r) => (
            <li key={r.slug}>
              <a className="lib-row" href={hrefFor("ai", r.slug)} onClick={close} aria-current={route.run === r.slug ? "true" : undefined}>
                <Thumb name={r.example} size={48} className="lib-thumb" />
                <span className="lib-text">
                  <span className="lib-name">{r.title}</span>
                  <span className="lib-meta">
                    <RunMark passed={r.passed} /> {r.model}
                    {r.costUsd != null && <>, ${r.costUsd.toFixed(2)}</>}
                  </span>
                </span>
              </a>
            </li>
          ))}
        </ul>

        <div className="drawer-file">
          <label className="button button-quiet">
            Open a file…
            <input type="file" accept=".json,.jsonl,application/json" className="visually-hidden" onChange={(e) => onFile(e.target.files?.[0])} />
          </label>
          <p className="hint">A model, an AI solution or a run log. You can also drop it here.</p>
          {fileNote && <p className="hint" role="status">{fileNote}</p>}
        </div>

        <h3 className="drawer-section">About</h3>
        <p className="drawer-about-text">
          Foldr folds paper on a computer, checked like real paper. Watch classic models, fold your own, or watch an AI try.
        </p>
        <button
          className="text-link"
          onClick={() => {
            close();
            setAboutOpen(true);
          }}
        >
          About Foldr and keyboard keys
        </button>
      </div>
    </dialog>
  );
}

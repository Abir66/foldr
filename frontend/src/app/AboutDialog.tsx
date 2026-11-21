import { useEffect, useRef } from "react";
import { useUi } from "../lib/store";

export function AboutDialog() {
  const open = useUi((s) => s.aboutOpen);
  const setOpen = useUi((s) => s.setAboutOpen);
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
  }, [open]);

  return (
    <dialog
      ref={ref}
      className="modal"
      aria-labelledby="about-title"
      onClose={() => setOpen(false)}
      onClick={(e) => {
        if (e.target === ref.current) setOpen(false);
      }}
    >
      <h2 id="about-title" className="modal-title">About Foldr</h2>
      <p>
        Foldr folds paper on a computer. Every fold is checked like real paper: it can't tear, stretch or pass through
        itself. You can watch the classic models, fold your own, or give an AI a goal and watch it try.
      </p>
      <p>It grew out of a research project that asks a simple question: can an AI fold origami if it can only see the paper and make one fold at a time?</p>
      <h3 className="modal-sub">Keys</h3>
      <dl className="keys">
        <dt><kbd>←</kbd> <kbd>→</kbd></dt>
        <dd>Previous or next step</dd>
        <dt><kbd>Space</kbd></dt>
        <dd>Play or pause</dd>
      </dl>
      <form method="dialog">
        <button className="button button-primary">Close</button>
      </form>
    </dialog>
  );
}

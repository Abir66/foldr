import { useEffect, useRef, useState } from "react";
import { exampleFileUrl, type Example } from "../../lib/examples";
import { getViewer } from "../../viewer/Viewer";
import type { ExportKind } from "../../worker/protocol";
import { call, ensurePython, saveFile } from "../../worker/rpc";

function save(name: string, href: string) {
  const a = Object.assign(document.createElement("a"), { href, download: name });
  a.click();
}

const FILES = [
  { label: "Step pictures (.zip)", kind: "stepsZip", ext: "steps.zip", py: "png_zip" },
  { label: "Crease pattern (.png)", kind: "creasePattern", ext: "crease_pattern.png", py: "cp_png" },
  { label: "Fold file (.fold)", kind: "fold", ext: "fold", py: "fold" },
  { label: "Steps (.json)", kind: "model", ext: "json", py: "model_json" },
  { label: "Timeline (.json)", kind: "timeline", ext: "timeline.json", py: "timeline_json" },
] as const;

export function ExportMenu({ example }: { example: Example }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const away = (e: Event) => {
      if (!ref.current?.contains(e.target as Node)) setOpen(false);
    };
    const esc = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("pointerdown", away);
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("pointerdown", away);
      document.removeEventListener("keydown", esc);
    };
  }, [open]);

  const [busy, setBusy] = useState(false);
  const local = example.local;

  const exportLocal = async (kind: ExportKind) => {
    setOpen(false);
    setBusy(true);
    try {
      await ensurePython(kind === "png_zip" || kind === "cp_png" ? "export" : "core");
      const r = await call("export", { model: local!.model, kinds: [kind] });
      for (const f of r.files) saveFile(f.name, f.bytes as BlobPart, f.mime);
      if (r.errors.length) window.alert(r.errors.map((e) => e.message).join("; "));
    } catch (e) {
      window.alert((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const savePicture = async () => {
    setOpen(false);
    const blob = await getViewer()?.screenshot();
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    save(`${example.name}.png`, url);
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  return (
    <div className="export" ref={ref}>
      <button className="button button-quiet" aria-expanded={open} onClick={() => setOpen(!open)}>
        Export
      </button>
      {open && (
        <div className="export-menu">
          {FILES.map((f) =>
            local ? (
              <button key={f.kind} className="export-item" disabled={busy} onClick={() => exportLocal(f.py)}>
                {f.label}
              </button>
            ) : (
              <a
              key={f.kind}
              className="export-item"
              href={exampleFileUrl(example, f.kind)}
              download={`${example.name}.${f.ext}`}
              onClick={() => setOpen(false)}
            >
                {f.label}
              </a>
            ),
          )}
          <button className="export-item" onClick={savePicture}>
            Picture of this view (.png)
          </button>
        </div>
      )}
    </div>
  );
}

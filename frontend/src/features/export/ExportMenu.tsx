import { useEffect, useRef, useState } from "react";
import type { Source } from "../../app/source";
import { cpSvgText, sidesSvgText, svgToPng } from "../../lib/diagrams";
import { exampleFileUrl, type Example } from "../../lib/examples";
import { useUi } from "../../lib/store";
import { getViewer } from "../../viewer/Viewer";
import type { ExportKind } from "../../worker/protocol";
import { call, ensurePython, saveFile } from "../../worker/rpc";

type Item =
  | { label: string; py: ExportKind; prebuilt: keyof Example["files"]; ext: string; heavy?: boolean }
  | { label: string; client: "sides" | "cp" | "view" | "model" | "timeline"; prebuilt?: keyof Example["files"]; ext: string };

const ITEMS: Item[] = [
  { label: "Step pictures (.zip)", py: "png_zip", prebuilt: "stepsZip", ext: "steps.zip", heavy: true },
  { label: "Front and back of this step (.png)", client: "sides", ext: "front_back.png" },
  { label: "Crease pattern of this step (.png)", client: "cp", ext: "crease_pattern.png" },
  { label: "Fold file (.fold)", py: "fold", prebuilt: "fold", ext: "fold" },
  { label: "Steps (.json)", client: "model", prebuilt: "model", ext: "json" },
  { label: "Timeline (.json)", client: "timeline", prebuilt: "timeline", ext: "timeline.json" },
  { label: "Picture of this view (.png)", client: "view", ext: "png" },
];

const json = (v: unknown) => new TextEncoder().encode(JSON.stringify(v, null, 2));

/** One Export menu for every source: examples, opened files, your own folding and AI runs. */
export function ExportMenu({ source }: { source: Source }) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const away = (e: Event) => {
      if (!ref.current?.contains(e.target as Node)) setOpen(false);
    };
    const esc = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("pointerdown", away);
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("pointerdown", away);
      document.removeEventListener("keydown", esc);
    };
  }, [open]);

  const tl = source.timeline;
  const prebuilt = source.example && !source.example.local ? source.example : null;
  const state = () => tl?.states[Math.min(useUi.getState().step, tl.states.length - 1)];
  const stepTag = () => `step${useUi.getState().step}`;

  const run = async (it: Item) => {
    setOpen(false);
    setBusy("Exporting…");
    try {
      const base = source.name;
      if ("py" in it) {
        if (it.heavy) setBusy("Loading the picture maker (about 4 MB)…");
        await ensurePython(it.heavy ? "export" : "core");
        setBusy("Exporting…");
        const model = source.kind === "own" ? undefined : await source.model();
        if (source.kind !== "own" && !model) throw new Error("This run has no steps to export.");
        const r = await call("export", { model: model ?? undefined, kinds: [it.py] });
        for (const f of r.files) saveFile(f.name, f.bytes as BlobPart, f.mime);
        if (r.errors.length) throw new Error(r.errors.map((e) => e.message).join("; "));
        return;
      }
      const s = state();
      if (it.client === "sides" && s) saveFile(`${base}_${stepTag()}_front_back.png`, await svgToPng(sidesSvgText(s, source.coloredSide)), "image/png");
      else if (it.client === "cp" && s) saveFile(`${base}_${stepTag()}_crease_pattern.png`, await svgToPng(cpSvgText(s)), "image/png");
      else if (it.client === "timeline" && tl) saveFile(`${base}.timeline.json`, json(tl), "application/json");
      else if (it.client === "model") {
        const m = await source.model();
        if (!m) throw new Error("There are no steps to save yet.");
        saveFile(`${base}.json`, json(m), "application/json");
      } else if (it.client === "view") {
        const blob = await getViewer()?.screenshot();
        if (blob) saveFile(`${base}.png`, blob, "image/png");
      }
    } catch (e) {
      window.alert((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const view = useUi((s) => s.view);
  return (
    <div className="export" ref={ref}>
      <button
        className="tool-btn tool-export"
        aria-expanded={open}
        aria-haspopup="menu"
        disabled={!!busy}
        onClick={() => setOpen(!open)}
        title={busy ?? "Export"}
      >
        <svg viewBox="0 0 20 20" aria-hidden="true">
          <path d="M10 3v9M6.5 8.5 10 12l3.5-3.5M4 14v2.5h12V14" />
        </svg>
        <span className="tool-label">{busy ? "Exporting…" : "Export"}</span>
      </button>
      {open && (
        <div className="export-menu" role="menu">
          {ITEMS.map((it) => {
            const key = it.label;
            if (prebuilt && it.prebuilt)
              return (
                <a
                  key={key}
                  role="menuitem"
                  className="export-item"
                  href={exampleFileUrl(prebuilt, it.prebuilt)}
                  download={`${prebuilt.name}.${it.ext}`}
                  onClick={() => setOpen(false)}
                >
                  {it.label}
                </a>
              );
            const needsSteps = "py" in it || ("client" in it && it.client === "model");
            const off = !tl || (needsSteps && source.n === 0) || ("client" in it && it.client === "view" && view !== "3d" && view !== "flat");
            return (
              <button key={key} role="menuitem" className="export-item" disabled={off} onClick={() => run(it)}>
                {it.label}
                {"heavy" in it && it.heavy && <span className="export-soon">loads 4 MB more</span>}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

import { useEffect } from "react";
import { stepParam, useRoute } from "../lib/router";
import { useUi } from "../lib/store";
import { Header } from "./Header";
import { Stage } from "./Stage";
import { Panel } from "./Panel";
import { LibraryDrawer } from "./LibraryDrawer";
import { AboutDialog } from "./AboutDialog";
import { useSource } from "./source";
import { FoldPanel } from "../features/fold/FoldPanel";
import { AiPanel } from "../features/ai/AiPanel";
import { stepNext, stepPrev } from "../viewer/Viewer";

export function App() {
  const route = useRoute();
  const source = useSource(route);
  const n = source.n;
  const setStep = useUi((s) => s.setStep);
  const setPlaying = useUi((s) => s.setPlaying);
  const setLibraryOpen = useUi((s) => s.setLibraryOpen);

  // ?library opens the library drawer (handy for links and screenshots).
  useEffect(() => {
    if (route.params.has("library")) setLibraryOpen(true);
    const v = route.params.get("view");
    if (v === "3d" || v === "flat" || v === "sides" || v === "cp") useUi.getState().setView(v);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // On a new timeline, show the finished model (or ?step=k).
  useEffect(() => {
    setPlaying(false);
    setStep(stepParam(route.params, n) ?? n);
  }, [route.example, route.mode, route.own]); // eslint-disable-line react-hooks/exhaustive-deps

  // Keyboard: arrows step, space plays.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t.closest("input, textarea, select, dialog[open]")) return;
      const { step, playing } = useUi.getState();
      if (e.key === "ArrowRight") stepNext(n);
      else if (e.key === "ArrowLeft") stepPrev();
      else if (e.key === " " && !t.closest("button, a") && n > 0) {
        e.preventDefault();
        if (!playing && step >= n) setStep(0);
        setPlaying(!playing);
      } else return;
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [n, setStep, setPlaying]);

  return (
    <div className="app" data-mode={route.mode}>
      <Header route={route} />
      <main className="workspace">
        {/* a fresh viewer per tab (and for an empty AI canvas): it would keep showing its last timeline */}
        <Stage key={route.mode === "ai" && !source.play ? "ai-empty" : route.mode} route={route} source={source} />
        <Panel label={route.mode === "fold" ? (route.own ? "Folding tools" : "Steps") : "AI"}>
          {route.mode === "fold" && <FoldPanel source={source} />}
          {route.mode === "ai" && <AiPanel runSlug={route.run} />}
        </Panel>
      </main>
      <LibraryDrawer route={route} />
      <AboutDialog />
    </div>
  );
}

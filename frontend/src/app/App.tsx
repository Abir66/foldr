import { useEffect } from "react";
import { getExample } from "../lib/examples";
import { stepParam, useRoute } from "../lib/router";
import { useUi } from "../lib/store";
import { useExampleTimeline } from "../lib/timeline";
import { Header } from "./Header";
import { Stage } from "./Stage";
import { Panel } from "./Panel";
import { LibraryDrawer } from "./LibraryDrawer";
import { AboutDialog } from "./AboutDialog";
import { WatchPanel } from "../features/watch/WatchPanel";
import { FoldPanel } from "../features/fold/FoldPanel";
import { AiPanel } from "../features/ai/AiPanel";
import { stepNext, stepPrev } from "../viewer/Viewer";

export function App() {
  const route = useRoute();
  const example = getExample(route.example)!;
  const n = useExampleTimeline(example.name)?.steps.length ?? example.steps.length;
  const setStep = useUi((s) => s.setStep);
  const setPlaying = useUi((s) => s.setPlaying);
  const setLibraryOpen = useUi((s) => s.setLibraryOpen);

  // ?library opens the library drawer (handy for links and screenshots).
  useEffect(() => {
    if (route.params.has("library")) setLibraryOpen(true);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // On a new example, show the finished model (or ?step=k).
  useEffect(() => {
    setPlaying(false);
    setStep(stepParam(route.params, n) ?? n);
  }, [route.example, route.mode]); // eslint-disable-line react-hooks/exhaustive-deps

  // Keyboard: arrows step, space plays (Watch only).
  useEffect(() => {
    if (route.mode !== "watch") return;
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t.closest("input, textarea, select, dialog[open]")) return;
      const { step, playing } = useUi.getState();
      if (e.key === "ArrowRight") stepNext(n);
      else if (e.key === "ArrowLeft") stepPrev();
      else if (e.key === " " && !t.closest("button, a")) {
        e.preventDefault();
        if (!playing && step >= n) setStep(0);
        setPlaying(!playing);
      } else return;
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [route.mode, n, setStep, setPlaying]);

  return (
    <div className="app" data-mode={route.mode}>
      <Header mode={route.mode} />
      <main className="workspace">
        <Stage route={route} />
        <Panel label={route.mode === "watch" ? "Steps" : route.mode === "fold" ? "Folding tools" : "AI"}>
          {route.mode === "watch" && <WatchPanel example={example} />}
          {route.mode === "fold" && <FoldPanel />}
          {route.mode === "ai" && <AiPanel runSlug={route.run} />}
        </Panel>
      </main>
      <LibraryDrawer route={route} />
      <AboutDialog />
    </div>
  );
}

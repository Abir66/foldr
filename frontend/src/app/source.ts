// The timeline on the canvas, whatever it came from: an example or opened file, your own folding,
// or an AI run. Export, "Fold from here" and the transport all work from this.
import { useMemo } from "react";
import { exampleFileUrl, getExample, getRun, type Example } from "../lib/examples";
import type { Route } from "../lib/router";
import { useUi } from "../lib/store";
import { blankTimeline, useExampleTimeline, type Timeline } from "../lib/timeline";
import { useFold } from "../features/fold/foldStore";
import { useAi } from "../features/ai/aiStore";
import type { ModelJson } from "../worker/protocol";

export interface Source {
  kind: "example" | "own" | "ai";
  /** base of exported file names */
  name: string;
  label: string;
  /** every state and step so far */
  timeline: Timeline | null;
  /** what the 3D viewer is given; it grows in place (own folding) or via appendStep (AI runs) */
  play: Timeline | null;
  n: number;
  coloredSide: "up" | "down";
  steps: { op: string; description: string }[];
  /** prebuilt example (its files can be downloaded as they are) */
  example?: Example;
  /** the steps as a model JSON; null when there is nothing to fold */
  model(): Promise<ModelJson | null>;
}

export function useSource(route: Route): Source {
  const example = getExample(route.example)!;
  const exampleTl = useExampleTimeline(route.mode === "fold" && !route.own ? example.name : null);
  const foldSide = useUi((s) => s.foldSide);
  const foldTl = useFold((s) => s.timeline);
  const mySteps = useFold((s) => s.steps);
  const foldColored = useFold((s) => s.coloredSide);
  const run = useAi((s) => s.run);
  const blank = useMemo(() => blankTimeline(foldSide === "colored" ? "up" : "down"), [foldSide]);
  const aiTl = useMemo<Timeline | null>(
    () =>
      run.base
        ? { ...run.base, steps: run.steps.map((s) => s.step), states: [run.base.states[0], ...run.steps.map((s) => s.state)] }
        : null,
    [run.base, run.steps],
  );

  if (route.mode === "ai") {
    const meta = getRun(route.run);
    const name = meta?.slug ?? "ai-run";
    return {
      kind: "ai",
      name,
      label: meta?.title ?? "AI run",
      timeline: aiTl,
      play: run.base ?? null,
      n: run.steps.length,
      coloredSide: run.base?.colored_side ?? "up",
      steps: run.steps.map((s) => ({ op: s.step.op, description: s.step.description })),
      async model() {
        if (!run.base || !run.steps.every((s) => s.src)) return null;
        return {
          name,
          title: meta?.title,
          paper: { size: run.base.size ?? 1, colored_side: run.base.colored_side ?? "up" },
          steps: run.steps.map((s) => ({ ...s.src!, description: s.src!.description ?? s.step.description })),
        };
      },
    };
  }
  if (route.own) {
    const tl = foldTl ?? blank;
    return {
      kind: "own",
      name: "my_fold",
      label: "My fold",
      timeline: tl,
      play: tl,
      n: mySteps.length,
      coloredSide: foldTl ? foldColored : tl.colored_side ?? "down",
      steps: mySteps.map((s) => ({ op: s.op, description: s.text })),
      async model() {
        return foldTl && mySteps.length ? ((await useFold.getState().myModel()) as ModelJson) : null;
      },
    };
  }
  return {
    kind: "example",
    name: example.name,
    label: example.label,
    timeline: exampleTl,
    play: exampleTl,
    n: exampleTl?.steps.length ?? example.steps.length,
    coloredSide: exampleTl?.colored_side ?? "up",
    steps: example.steps,
    example,
    async model() {
      if (example.local) return example.local.model as ModelJson;
      const r = await fetch(exampleFileUrl(example, "model"));
      if (!r.ok) throw new Error(`model ${example.name}: HTTP ${r.status}`);
      return (await r.json()) as ModelJson;
    },
  };
}

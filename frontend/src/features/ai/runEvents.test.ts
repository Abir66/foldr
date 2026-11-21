import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { emptyRun, parseJsonl, reduce, turnForStep } from "./runEvents";

const log = (slug: string) => parseJsonl(readFileSync(`public/examples/run_${slug}.events.jsonl`, "utf8"));

describe("run reducer", () => {
  it("replays a saved run with undos to its final committed steps", () => {
    const events = log("haiku-cup-command");
    const v = events.reduce(reduce, emptyRun());
    const fin = events.at(-1)!;
    expect(v.finished?.type).toBe("run_finished");
    expect(v.steps.length).toBe(fin.solution.steps.length);
    expect(v.turns.length).toBe(fin.totals.turns);
    expect(v.spent).toBeCloseTo(fin.totals.cost_usd, 6);
    expect(v.base?.states.length).toBe(1);
    // the step now shown maps to the latest turn that made it
    expect(turnForStep(v.turns, v.steps.length)).toBe(11);
  });

  it("ignores events it already has (SSE reconnect)", () => {
    const events = log("haiku-cup-command").slice(0, 10);
    const v = events.reduce(reduce, emptyRun());
    expect(events.reduce(reduce, v)).toBe(v);
  });
});

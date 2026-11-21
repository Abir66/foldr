import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { creasePattern, cpSvgText, flatView, sidesSvgText } from "./diagrams";
import { blankTimeline, type Timeline } from "./timeline";

const crane = JSON.parse(readFileSync(new URL("../../public/examples/crane.timeline.json", import.meta.url), "utf8")) as Timeline;

describe("diagrams", () => {
  it("blank sheet has no creases", () => {
    const cp = creasePattern(blankTimeline().states[0]);
    expect(cp.creases).toHaveLength(0);
    expect(cp.outline).toMatch(/^M/);
  });
  it("first valley fold of the crane is one valley crease", () => {
    const cp = creasePattern(crane.states[2]);
    expect(cp.creases.map((c) => c.kind)).toEqual(["valley"]);
  });
  it("finished crane has valley and mountain creases", () => {
    const kinds = new Set(creasePattern(crane.states.at(-1)!).creases.map((c) => c.kind));
    expect(kinds.has("valley") && kinds.has("mountain")).toBe(true);
  });
  it("back view flips colours of a flat sheet", () => {
    const s = crane.states[0];
    expect(flatView(s, "down").faces[0].colored).toBe(!flatView(s, "down", true).faces[0].colored);
    expect(sidesSvgText(s, "down")).toContain("Back");
    expect(cpSvgText(crane.states[3])).toContain("stroke-dasharray");
  });
  it("paints faces bottom-up (back view top-down), each with its own creases", () => {
    const s = crane.states[8];
    const front = flatView(s, "up"),
      back = flatView(s, "up", true);
    expect(front.faces).toHaveLength(s.faces.length);
    expect(front.faces.map((f) => f.lines.length)).toEqual(s.faces.map((f) => (f.lines ?? []).length));
    expect(back.faces.map((f) => f.lines.length)).toEqual([...front.faces].reverse().map((f) => f.lines.length));
    const svg = sidesSvgText(s, "up");
    // a face's crease lines come right after its outline, before the next face covers them
    expect(svg.indexOf("<line")).toBeLessThan(svg.lastIndexOf("<path"));
  });
});

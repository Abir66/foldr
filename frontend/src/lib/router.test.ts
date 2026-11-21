import { describe, expect, it } from "vitest";
import { setExamples, type Example } from "./examples";
import { hrefFor, parseHash, stepParam } from "./router";

const fake = (name: string) => ({ name }) as Example;
setExamples(["crane", "cup", "hat"].map(fake));

describe("parseHash", () => {
  it("defaults to watching the crane", () => {
    const r = parseHash("");
    expect(r.mode).toBe("watch");
    expect(r.example).toBe("crane");
  });
  it("reads watch example and hash query", () => {
    const r = parseHash("#/watch/cup?step=3");
    expect(r.example).toBe("cup");
    expect(stepParam(r.params, 7)).toBe(3);
  });
  it("falls back for unknown example", () => {
    expect(parseHash("#/watch/nope").example).toBe("crane");
  });
  it("reads fold and ai routes", () => {
    expect(parseHash("#/fold").mode).toBe("fold");
    const ai = parseHash("#/ai/opus-hat-text");
    expect(ai.mode).toBe("ai");
    expect(ai.run).toBe("opus-hat-text");
  });
  it("merges the real query string and clamps step", () => {
    const r = parseHash("#/watch/crane", "?pose=99,0.5");
    expect(stepParam(r.params, 20)).toBe(20);
  });
  it("builds hrefs", () => {
    expect(hrefFor("watch", "hat")).toBe("#/watch/hat");
    expect(hrefFor("ai")).toBe("#/ai");
    expect(hrefFor("fold")).toBe("#/fold");
  });
});

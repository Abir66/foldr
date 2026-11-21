import { describe, expect, it } from "vitest";
import type { Vec2 } from "../../lib/timeline";
import { bisectorSegment, samePoint, snapAt, type SnapInput } from "./snap";

// unit square drawn at 100 px per unit, y down
const square: Vec2[][] = [
  [
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1],
  ],
];
const inp: SnapInput = {
  labels: [{ label: "A", xy: [0, 0] }],
  polys: square,
  toScreen: (p) => [p[0] * 100, 100 - p[1] * 100],
  fromScreen: (s) => [s[0] / 100, (100 - s[1]) / 100],
  pointPx: 20,
  edgePx: 10,
};

describe("snapAt", () => {
  it("prefers a labeled point", () => {
    expect(snapAt([5, 95], inp)).toMatchObject({ kind: "point", ref: "A" });
  });
  it("snaps to the nearest edge", () => {
    const s = snapAt([50, 96], inp)!;
    expect(s.kind).toBe("edge");
    expect(s.xy).toEqual([0.5, 0]);
  });
  it("gives a free point inside the paper, rounded", () => {
    const s = snapAt([33.333333, 50], inp)!;
    expect(s.kind).toBe("free");
    expect(s.xy).toEqual([0.3333, 0.5]);
  });
  it("ignores taps off the paper", () => {
    expect(snapAt([150, 50], inp)).toBeNull();
  });
});

describe("bisectorSegment", () => {
  it("clips the crease of corner onto opposite corner to the diagonal", () => {
    const seg = bisectorSegment([1, 1], [0, 0], square)!;
    const ends = seg.map((p) => p.map((v) => Math.round(v * 1e6) / 1e6)).sort((a, b) => a[0] - b[0]);
    expect(ends).toEqual([
      [0, 1],
      [1, 0],
    ]);
  });
  it("is null for the same point", () => {
    expect(bisectorSegment([0.5, 0.5], [0.5, 0.5], square)).toBeNull();
  });
});

it("samePoint compares labels and coords", () => {
  expect(samePoint("A", "A")).toBe(true);
  expect(samePoint([0.1, 0.2], [0.1, 0.2])).toBe(true);
  expect(samePoint("A", [0, 0])).toBe(false);
});

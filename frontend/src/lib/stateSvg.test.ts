import { describe, expect, it } from "vitest";
import { stateSvg } from "./stateSvg";
import { blankTimeline, type StateJson } from "./timeline";

describe("stateSvg", () => {
  it("draws a blank square coloured by the side facing up", () => {
    const s = blankTimeline("up").states[0];
    const up = stateSvg(s, "up");
    expect(up.faces).toHaveLength(1);
    expect(up.faces[0].colored).toBe(true);
    expect(up.faces[0].d).toBe("M0 0L1 0L1 -1L0 -1Z");
    expect(stateSvg(s, "down").faces[0].colored).toBe(false);
  });

  it("paints lower layers first and maps faces by T", () => {
    const state: StateJson = {
      faces: [
        { id: 1, orig: [[0, 0.5], [1, 0.5], [1, 1], [0, 1]], T: [[1, 0, 0], [0, -1, 1]], layer: 1, front_up: false, lines: [] },
        { id: 0, orig: [[0, 0], [1, 0], [1, 0.5], [0, 0.5]], T: [[1, 0, 0], [0, 1, 0]], layer: 0, front_up: true, lines: [] },
      ],
      bbox: [[0, 0], [1, 0.5]],
      world: { M: [[1, 0, 0], [0, 1, 0]], s: 1, zoff: 0 },
    };
    const svg = stateSvg(state, "up");
    expect(svg.faces.map((f) => f.colored)).toEqual([true, false]);
    // top face folded down onto the bottom half: y in [0, 0.5] -> svg y in [-0.5, 0]
    expect(svg.faces[1].d).toBe("M0 -0.5L1 -0.5L1 0L0 0Z");
  });
});

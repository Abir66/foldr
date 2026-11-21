# Fold step format (v0.1)

A model is one JSON file in `models/`. Run it with:

```
uv run origami fold models/cup.json --out out/cup/
```

```json
{
  "name": "cup",                  // used for the .fold file name
  "title": "Paper cup",
  "paper": {"size": 1.0},         // square sheet, (0,0) = bottom-left, front (colored) side up
                                  // optional "colored_side": "down" = start colored side down (only swaps colors)
  "steps": [ { "op": "...", "description": "human text shown in the PNG", ... } ]
}
```

Coordinates: x right, y up. "Folded space" = where the paper is now, as seen by the viewer.
"Original" (orig) = position on the flat, unfolded sheet (crease-pattern coords).

## Points
| Spec | Meaning |
|---|---|
| `"bl"`, `"br"`, `"tr"`, `"tl"` | corners of the original sheet, wherever they are now |
| `"bottom_mid"`, `"right_mid"`, `"top_mid"`, `"left_mid"`, `"center"` | edge midpoints / center of the original sheet |
| `[x, y]` | explicit folded-space point |
| `{"orig": [x, y]}` | a point of the original sheet, followed to where it is now |
| `{"mid": [P, Q]}` | midpoint of two points |
| `{"lerp": [P, Q], "t": 0.25}` | point 25% of the way from P to Q |
| `{"intersect": [L1, L2]}` | intersection of two lines. Each `L` is a line reference (see below), e.g. `[["bl","br"], ["tl","tr"]]` |

Named points resolve through the topmost face that contains them (all faces agree, the paper is continuous).

## Lines
A **line reference** `L` (an input line, e.g. an edge) is either a point pair `[P, Q]`
(directed P → Q; e.g. `["tl", "tr"]` = the edge from tl to tr, wherever those corners are NOW)
or any line spec dict below. All points are resolved on the current folded state.

| Spec | Huzita–Justin | Meaning |
|---|---|---|
| `{"through": [P, Q]}` | O1 | line through two points |
| `{"onto": [P, Q]}` | O2 | perpendicular bisector: the fold that puts P onto Q. P's side moves by default |
| `{"bisect": [L1, L2]}` | O3 | fold line L1 onto line L2 (angle bisector, see below) |
| `{"perpendicular": L, "at": P}` | O4 | line through P, perpendicular to L |
| `{"point_onto_line": P, "line": L, "pivot": R}` | O5 | fold P onto line L with the crease passing through R. P's side moves by default |
| `{"point_onto_line": P, "line": L, "perpendicular_to": L2}` | O7 | fold P onto L with the crease perpendicular to L2 (P travels parallel to L2) |
| `{"horizontal": P}` / `{"vertical": P}` | — | horizontal / vertical line through P |

**`bisect` — which bisector.** Two crossing lines have two bisectors. We take the one whose
reflection maps the direction P→Q of `L1 = [P, Q]` onto the direction R→S of `L2 = [R, S]`
(i.e. crease direction = unit(PQ) + unit(RS)). So `[["bl","br"], ["bl","tl"]]` is the diagonal
through bl, while `[["bl","br"], ["tl","bl"]]` is the other one. Parallel lines → the midline
(directions don't matter). Default `move`: whichever of P, Q is farther from the crease
(for a dict `L1`, its anchor point); give `move` explicitly if that is not what you want.

**`point_onto_line` with `pivot` — which solution.** There can be 0, 1 or 2 landing spots
(circle around the pivot through P, cut with L). No solution → error. Choice:
1. `"near": X` given → the landing spot closest to point X;
2. else if `L` is a point pair `[S, T]` → prefer spots inside segment ST (on the actual edge),
   ties broken by closeness to S;
3. else (dict `L`) → the spot closest to the line's anchor point.

Example (cup step 3, no magic numbers): the pivot is where the left corner's angle bisector meets
the right edge.
```json
{"point_onto_line": "br", "line": ["tl", "tr"],
 "pivot": {"intersect": [{"bisect": [["tl", "br"], ["tl", "tr"]]}, ["br", "tr"]]}}
```

## Ops
| Op | Fields | Effect |
|---|---|---|
| `valley` | `line`, `move` (point; optional for `onto`), `layers` | moving part is reflected and lands **on top** (toward viewer) |
| `mountain` | same | moving part is reflected and goes **underneath** |
| `flip` | `axis`: `"vertical"` (default, turn left-right) or `"horizontal"` | turn the paper over; layer order reverses |
| `rotate` | `degrees` (counter-clockwise) | rotate around the paper's bounding-box center |
| `unfold` | — | undo the most recent not-yet-undone fold; its creases stay. Flips/rotates done since are kept |

`move`: any point strictly on the side of the fold line that should move.

`layers` (default `"all"`):
- `"all"` — every layer on the moving side.
- `{"top": n}` — only the top n layers (locally: a piece moves if fewer than n moving-side pieces overlap it from above).
- `{"bottom": n}` — same, counted from below (natural with `mountain`).
- add `"flap": true` (e.g. `{"top": 2, "flap": true}`) — of the chosen layers keep only the
  connected flap under the `move` point (pieces joined by paper edges, incl. old creases).
  Use it when the fold line also crosses a *different* flap on the moving side that must stay
  put (e.g. the kabuto horns: one ear at a time). Error if `move` is not on a chosen layer.

- add `"local": true` (e.g. `{"top": 2, "local": true}`) — count the layers **only in the stack
  at the `move` point**: the faces of the moving side that cover `move`, top (or bottom) n of
  them. Those layers, plus all paper attached to them (see drag-along below), move — i.e. the
  whole flap(s) those layers belong to. Faces elsewhere on the moving side are never counted,
  so a different flap lying over another part of the same layer (e.g. the crane's neck
  crossing the wing) can't push the wanted layer out of the "top n". Put `move` on a spot
  where the flap you want is the top (bottom) n layers; if `move` sits on a crease, pieces of
  one layer that meet there count as one level. Recommended for folding a flap on a busy
  model (crane wings: `{"top": 2, "local": true}` with `move` on the wing).

Paper attached to a chosen layer by an edge that is **not** on the fold line is dragged along
(otherwise it would tear). So `{"top": 1}` on a folded edge still moves both layers — correct physically.

## Tucks (`tuck` on valley / mountain)
Where the moved flap lands in the layer order (default: valley = on top of everything it covers,
mountain = under everything). See `docs/design/layers.md`.
- `"tuck": "inside"` — the flap lands right against the layer(s) it is hinged to and goes *under*
  (valley) / *over* (mountain) everything else: "tuck the flap inside the pocket". "The layer it
  is hinged to" includes the paper continuing that layer flat past old (unfolded) creases. Kabuto last step:
  `{"op": "valley", "line": ..., "move": "bl", "tuck": "inside"}`.
- `"tuck": {"under": k}` (valley) — flap slides under the top k layers it would cover.
- `"tuck": {"over": k}` (mountain) — flap stays above the bottom k layers.

## Macros
| Op | Fields | Effect |
|---|---|---|
| `inside_reverse` | `line`, `move` (point on the tip), `layers` (default all), `split` (optional) | the tip of the flap under `move`, beyond `line`, is pushed inside between its own layers (each layer reflected across the line; front half goes behind the front half, back half in front of the back half). `split` = number of front layers if the default (half) is wrong |
| `squash` | `move` (point on the flap), `hinge` (line ref: edge the flap is attached by), `spine` (line ref: the flap's folded edge that gets opened), `layers` (default `{"top": 2}`) | lift the flap upright on its hinge, open it, flatten symmetrically: the spine lands on the hinge line |
| `petal` | `move` (bottom corner of the top layer), `line` (horizontal crease the corner is lifted along), `layers` (default `{"top": 2}`) | lift the corner up along `line`; the sides fold in along the kite creases from `move` to where `line` meets the flap's edges and end inside the side flaps, under the lifted petal (physically valid bird base: Maekawa 3/1 at the kite points) |

`outside_reverse`, `rabbit_ear`, `sink` are still reserved ("not implemented" error).

Each resulting state has `motions` (list of sub-motions: reflect axis, sign, phase, moving regions)
for animation; see `docs/design/layers.md`.

## Outputs (per model)
- `step_00.png` … `step_NN.png`: state after k steps, with the line + arrow for step k+1
  (valley = blue dashed, mountain = red dash-dot). Orange = front side, cream = back side.
- `crease_pattern.png`: creases on the unfolded sheet.
- `<name>.fold`: FOLD 1.1. Key frame = crease pattern (B/M/V). `file_frames[0]` = folded form
  (faces, folded vertex coords, `faceOrders`, plus `origami:vertices_orig_coords`).

import type { Source } from "../../app/source";
import { hrefFor } from "../../lib/router";
import { useFold } from "./foldStore";

/** Start folding from step k of whatever is on the canvas. Returns false when the user cancels. */
export async function foldFromHere(source: Source, k: number): Promise<boolean> {
  const fold = useFold.getState();
  const later = source.n - k;
  if (source.kind === "own") {
    if (later > 0 && !window.confirm(`Fold from step ${k}? Your ${later} later step${later > 1 ? "s" : ""} will be removed.`)) return false;
    await fold.undoTo(k);
    return true;
  }
  const asks: string[] = [];
  if (later > 0) asks.push(`The ${later} step${later > 1 ? "s" : ""} after step ${k} are left out.`);
  if (fold.phase === "folding" && fold.steps.length) asks.push("Your current folding will be replaced.");
  if (asks.length && !window.confirm(`Fold from step ${k}?\n${asks.join("\n")}`)) return false;
  let model;
  try {
    model = await source.model();
  } catch (e) {
    window.alert((e as Error).message);
    return false;
  }
  if (!model) {
    window.alert("These steps cannot be folded again here.");
    return false;
  }
  window.location.hash = hrefFor("fold");
  return fold.startFrom(model, k, source.timeline);
}

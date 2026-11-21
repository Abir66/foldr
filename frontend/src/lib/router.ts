// Tiny hash router. Routes: #/fold/<example> (watch and fold an example), #/fold (your own folding),
// #/ai, #/ai/<run-slug>. Old #/watch/<example> links open the example in Fold.
// Viewer URL params (?step=k, ?pose=k,t, ?orbit=a,b) may sit in the hash query or the real query.
import { useEffect, useState } from "react";
import { DEFAULT_EXAMPLE, getExample } from "./examples";

export type Mode = "fold" | "ai";

export interface Route {
  mode: Mode;
  example: string; // fold: which example is shown (when not own)
  /** fold: your own folding session instead of an example */
  own: boolean;
  run?: string; // ai: which recorded run
  params: URLSearchParams;
}

export function parseHash(hash: string, search = ""): Route {
  const raw = hash.replace(/^#/, "");
  const [path, query = ""] = raw.split("?");
  const parts = path.split("/").filter(Boolean);
  const params = new URLSearchParams(search);
  new URLSearchParams(query).forEach((v, k) => params.set(k, v));

  const head = parts[0];
  if (head === "ai") return { mode: "ai", example: DEFAULT_EXAMPLE, own: false, run: parts[1], params };
  if (head === "fold" && !parts[1]) return { mode: "fold", example: DEFAULT_EXAMPLE, own: true, params };
  const example = getExample(parts[1]) ? parts[1] : DEFAULT_EXAMPLE;
  return { mode: "fold", example, own: false, params };
}

/** #/fold/<example>, or #/fold (own folding) without an id; #/ai[/<run>]. */
export function hrefFor(mode: Mode, id?: string): string {
  if (mode === "ai") return id ? `#/ai/${id}` : "#/ai";
  return id ? `#/fold/${id}` : "#/fold";
}

/** Reads ?step=k (1-based, like the old viewer) and clamps it to [1, n]. */
export function stepParam(params: URLSearchParams, n: number): number | undefined {
  const pose = params.get("pose")?.split(",")[0];
  const raw = params.get("step") ?? pose;
  if (raw == null) return undefined;
  const k = Number.parseInt(raw, 10);
  if (Number.isNaN(k)) return undefined;
  return Math.min(Math.max(k, 1), n);
}

export function useRoute(): Route {
  const read = () => parseHash(window.location.hash, window.location.search);
  const [route, setRoute] = useState<Route>(read);
  useEffect(() => {
    const on = () => setRoute(read());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}

// Tiny hash router. Routes: #/watch/<example>, #/fold, #/ai, #/ai/<run-slug>.
// Viewer URL params (?step=k, ?pose=k,t, ?orbit=a,b) may sit in the hash query or the real query.
import { useEffect, useState } from "react";
import { DEFAULT_EXAMPLE, getExample } from "./examples";

export type Mode = "watch" | "fold" | "ai";

export interface Route {
  mode: Mode;
  example: string; // watch: which example
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
  if (head === "fold") return { mode: "fold", example: DEFAULT_EXAMPLE, params };
  if (head === "ai") return { mode: "ai", example: DEFAULT_EXAMPLE, run: parts[1], params };
  const example = getExample(parts[1]) ? parts[1] : DEFAULT_EXAMPLE;
  return { mode: "watch", example, params };
}

export function hrefFor(mode: Mode, id?: string): string {
  if (mode === "watch") return `#/watch/${id ?? DEFAULT_EXAMPLE}`;
  if (mode === "ai") return id ? `#/ai/${id}` : "#/ai";
  return "#/fold";
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

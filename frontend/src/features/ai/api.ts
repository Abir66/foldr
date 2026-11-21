// Backend client: /api/config, start/stop a run, and the run's SSE event stream.
import type { RunEvent } from "./runEvents";

export const API_URL: string = ((import.meta.env.VITE_API_URL as string | undefined) ?? "").replace(/\/$/, "");

export interface ModelInfo {
  id: string;
  label: string;
  provider: string;
  images: boolean;
  default?: boolean;
  owner_key: boolean;
  user_key: boolean;
  price?: { input?: number; output?: number } | null;
}

export interface AiConfig {
  ai: {
    enabled: boolean;
    allow_user_keys: boolean;
    max_turns: number;
    per_run_usd?: number | null;
    daily_left_usd?: number | null;
  };
  models: ModelInfo[];
  providers: { id: string; kind: string; user_key: boolean }[];
  examples: string[];
}

/** null = no backend configured, unreachable, or AI turned off there. */
export async function fetchConfig(signal?: AbortSignal): Promise<AiConfig | null> {
  if (!API_URL) return null;
  try {
    const r = await fetch(`${API_URL}/api/config`, { signal });
    if (!r.ok) return null;
    const c = (await r.json()) as AiConfig;
    return c.ai?.enabled && c.models?.length ? c : null;
  } catch {
    return null;
  }
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

const FRIENDLY: Record<number, string> = {
  402: "Today's AI budget is used up. Try again tomorrow, or use your own key.",
  403: "This server won't start that run",
  409: "Someone else's run is going right now. Wait for it to finish, then try again.",
  429: "Please wait a minute between runs.",
};

/** What to show the user for a failed start. */
export function errorText(e: unknown): string {
  if (e instanceof ApiError) {
    const f = FRIENDLY[e.status];
    if (e.status === 403) return `${f}: ${e.message}.`;
    return f ?? e.message;
  }
  return "Couldn't reach the AI server. Is it running?";
}

export async function startRun(body: unknown): Promise<string> {
  let r: Response;
  try {
    r = await fetch(`${API_URL}/api/runs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    throw new Error("network");
  }
  const data = (await r.json().catch(() => ({}))) as { run_id?: string; detail?: unknown };
  if (!r.ok || !data.run_id) {
    const d = typeof data.detail === "string" ? data.detail : `HTTP ${r.status}`;
    throw new ApiError(r.status, d);
  }
  return data.run_id;
}

export async function stopRun(id: string): Promise<void> {
  await fetch(`${API_URL}/api/runs/${id}/stop`, { method: "POST" }).catch(() => undefined);
}

/** Subscribes to a run. EventSource reconnects by itself and sends Last-Event-ID. Returns a closer. */
export function streamRun(id: string, onEvent: (e: RunEvent) => void, onLost: () => void): () => void {
  const es = new EventSource(`${API_URL}/api/runs/${id}/events`);
  let fails = 0;
  es.onmessage = (m) => {
    fails = 0;
    const e = JSON.parse(m.data) as RunEvent;
    onEvent(e);
    // the server ends the stream after run_finished; without close() the browser would reconnect forever
    if (e.type === "run_finished") es.close();
  };
  es.onerror = () => {
    if (++fails > 5 || es.readyState === EventSource.CLOSED) {
      es.close();
      onLost();
    }
  };
  return () => es.close();
}

/** File -> base64 (no data: prefix), for the reference picture. */
export function fileToB64(f: File): Promise<string> {
  return new Promise((res, rej) => {
    const rd = new FileReader();
    rd.onload = () => res(String(rd.result).replace(/^data:[^,]*,/, ""));
    rd.onerror = () => rej(rd.error);
    rd.readAsDataURL(f);
  });
}

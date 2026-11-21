// LLM inspector: per turn, what went in and what came out.
import { useEffect, useRef, useState, type ReactNode } from "react";
import { useUi } from "../../lib/store";
import {
  callLabel,
  kfmt,
  turnForStep,
  turnMark,
  usd,
  type MenuItem,
  type Part,
  type RunEvent,
  type RunView,
  type ToolCall,
  type Turn,
  type Usage,
} from "./runEvents";

type Zoom = (src: string, caption?: string) => void;

const CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳";
const num = (n: number) => (n >= 1 && n <= 20 ? CIRCLED[n - 1] : `${n}.`);
const MARK = { ok: "✓", bad: "✗", undo: "↺", none: "·" } as const;

/** "option 2: Fold the corner ... -> OK: valley ..." -> "Fold the corner ..." */
function plainResult(text: string | undefined): string {
  if (!text) return "";
  return text.replace(/^option \d+:\s*/, "").replace(/\s*->\s*(OK|rejected).*$/s, "").replace(/^OK:\s*/, "");
}

function turnSummary(t: Turn): string {
  const a = t.actions.find((x) => x.ok && x.text) ?? t.actions[0];
  if (a) return plainResult(a.text);
  if (t.errors[0]) return t.errors[0].text;
  return t.response ? "(no move)" : "waiting for the AI…";
}

function Thumbs({ parts, images, zoom }: { parts: { img: string; caption?: string }[]; images: RunView["images"]; zoom: Zoom }) {
  const shown = parts.filter((p) => images[p.img]);
  if (!shown.length) return null;
  return (
    <div className="ai-thumbs">
      {shown.map((p, i) => {
        const im = images[p.img];
        const cap = p.caption ?? im.caption;
        return (
          <button key={i} className="ai-thumb" onClick={() => zoom(im.data, cap)} title={cap ? `${cap} (tap to enlarge)` : "Tap to enlarge"}>
            <img src={im.data} alt={cap ?? "picture sent to the AI"} loading="lazy" />
            {cap && <span>{cap}</span>}
          </button>
        );
      })}
    </div>
  );
}

function Clip({ text, lines = 6 }: { text: string; lines?: number }) {
  const [all, setAll] = useState(false);
  const long = text.split("\n").length > lines || text.length > lines * 90;
  return (
    <div>
      <p className={`ai-prose${all || !long ? "" : " is-clipped"}`} style={{ ["--lines" as string]: lines }}>
        {text}
      </p>
      {long && (
        <button className="text-link ai-more" onClick={() => setAll(!all)}>
          {all ? "Show less" : "Show all"}
        </button>
      )}
    </div>
  );
}

function Sub({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="ai-sub">
      <h4>{title}</h4>
      {children}
    </section>
  );
}

// big objects (paper states, data URLs) would swamp the raw view
const rawReplacer = (k: string, v: unknown) =>
  k === "state" || k === "timeline_step" || k === "timeline"
    ? "…(paper geometry)…"
    : typeof v === "string" && v.startsWith("data:")
      ? `…(picture, ${Math.round(v.length / 1365)} kB)…`
      : v;

function Raw({ events }: { events: (RunEvent | undefined)[] }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="ai-raw">
      <button className="text-link" onClick={() => setOpen(!open)} aria-expanded={open}>
        {open ? "Hide raw JSON" : "Raw JSON"}
      </button>
      {open && <pre>{events.filter(Boolean).map((e) => JSON.stringify(e, rawReplacer, 2)).join("\n\n")}</pre>}
    </div>
  );
}

function TurnBlock({ t, run, current, zoom }: { t: Turn; run: RunView; current: boolean; zoom: Zoom }) {
  const ref = useRef<HTMLDetailsElement>(null);
  const [open, setOpen] = useState(false);
  const setStep = useUi((s) => s.setStep);
  const setPlaying = useUi((s) => s.setPlaying);
  useEffect(() => {
    if (current) ref.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [current]);
  const req = t.request;
  const res = t.response;
  const u = (res?.usage ?? {}) as Usage;
  const mark = turnMark(t);
  const parts = ((req?.messages ?? []) as { role: string; parts?: Part[] }[]).flatMap((m) => (Array.isArray(m.parts) ? m.parts : []));
  const menu = (req?.menu ?? []) as MenuItem[];
  const calls = (res?.tool_calls ?? []) as ToolCall[];
  const chosen = new Set(calls.map((c) => (typeof c.args === "object" && c.args ? Number(c.args.option) : NaN)));
  const imgs = parts.filter((p) => p.kind === "image" && p.img) as { img: string; caption?: string }[];
  return (
    <details ref={ref} className={`ai-turn is-${mark}`} aria-current={current ? "step" : undefined} onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary>
        <span className="ai-num">{num(t.turn)}</span>
        <span className="ai-chips-inline">{calls.map((c, i) => <code key={i}>{callLabel(c)}</code>)}</span>
        <span className="ai-sum-text">{turnSummary(t)}</span>
        <span className={`ai-mark is-${mark}`} aria-label={mark}>
          {MARK[mark]}
        </span>
        {res && (u.input || u.output || u.cost_usd) ? (
          <span className="ai-sum-meta">
            {kfmt(u.input ?? 0)} in / {kfmt(u.output ?? 0)} out{u.cost_usd != null ? ` · ${usd(u.cost_usd)}` : ""}
            {res.latency_s != null ? ` · ${Math.round(res.latency_s)}s` : ""}
          </span>
        ) : null}
      </summary>
      {open && (
        <div className="ai-turn-body">
          <Sub title="What the AI saw">
            {parts
              .filter((p, i) => p.kind !== "image" && !(p.kind === "goal" && parts.findIndex((q) => q.kind === "goal") !== i))
              .map((p, i) =>
                p.kind === "goal" ? (
                  <p key={i} className="ai-ref">
                    The goal (see Setup)
                  </p>
                ) : p.text ? (
                  <details key={i} className="ai-text">
                    <summary>
                      {p.kind === "turn" ? "Turn text" : p.kind} · {kfmt(p.chars ?? p.text.length)} characters
                    </summary>
                    <pre>{p.text}</pre>
                  </details>
                ) : null,
              )}
            {run.plans.length > 0 && <p className="ai-ref">The expert plan (see Setup)</p>}
            <Thumbs parts={imgs} images={run.images} zoom={zoom} />
            {menu.length > 0 && (
              <details className="ai-text">
                <summary>
                  Menu · {menu.length} choices{req?.menu_truncated ? " (more hidden)" : ""}
                </summary>
                <ol className="ai-menu">
                  {menu.map((m) => (
                    <li key={m.id} className={chosen.has(m.id) ? "is-chosen" : undefined}>
                      <b>{m.id}</b> {m.description}
                      {chosen.has(m.id) && <em> ← chosen</em>}
                    </li>
                  ))}
                </ol>
              </details>
            )}
            {req && (
              <p className="ai-size">
                {kfmt(req.prompt_chars ?? 0)} characters · {req.images ?? imgs.length} pictures
              </p>
            )}
          </Sub>
          {(res?.reasoning || res?.content) && (
            <Sub title="What it thought">
              {res.reasoning && <Clip text={res.reasoning} />}
              {res.content && <Clip text={res.content} lines={4} />}
            </Sub>
          )}
          {calls.length > 0 && (
            <Sub title="What it did">
              {calls.map((c, i) => {
                const args = typeof c.args === "object" && c.args ? { ...c.args } : {};
                const plan = typeof args.plan === "string" ? args.plan : null;
                delete args.plan;
                return (
                  <div key={i} className="ai-call">
                    <code className="ai-chip">{callLabel(c)}</code>
                    {Object.keys(args).length > 0 && <pre className="ai-args">{JSON.stringify(args, null, 1)}</pre>}
                    {plan && <blockquote>{plan}</blockquote>}
                  </div>
                );
              })}
            </Sub>
          )}
          {(t.actions.length > 0 || t.errors.length > 0) && (
            <Sub title="What happened">
              <ul className="ai-results">
                {t.actions.map((a, i) => (
                  <li key={i} className={a.ok ? "is-ok" : "is-bad"}>
                    {a.ok ? "Accepted: " : "Rejected: "}
                    {a.text}
                  </li>
                ))}
                {t.errors.map((e, i) => (
                  <li key={`e${i}`} className="is-bad">
                    {e.kind === "format_error" ? "Bad reply: " : "Error: "}
                    {e.text}
                  </li>
                ))}
              </ul>
              {t.made != null && (
                <button
                  className="chip chip-small"
                  onClick={() => {
                    setPlaying(false);
                    setStep(t.made!);
                  }}
                >
                  Show on paper
                </button>
              )}
            </Sub>
          )}
          <Raw events={[req, res, ...t.actions, ...t.errors]} />
        </div>
      )}
    </details>
  );
}

function Setup({ run, zoom }: { run: RunView; zoom: Zoom }) {
  const s = run.started;
  if (!s) return null;
  const goal = s.goal ?? {};
  const goalImgs = ((s.goal_parts ?? []) as Part[]).filter((p) => p.img) as { img: string; caption?: string }[];
  return (
    <details className="ai-turn ai-setup">
      <summary>
        <span className="ai-num">★</span>
        <span className="ai-sum-text">
          Setup: {goal.kind === "text" ? "goal in words" : goal.kind === "image" ? "goal as a picture" : `fold the ${goal.target_name}`}
          {run.plans.length ? " · with a plan" : ""}
        </span>
      </summary>
      <div className="ai-turn-body">
        <Sub title="Goal">
          {goal.text && <p className="ai-prose">{goal.text}</p>}
          <Thumbs parts={goalImgs} images={run.images} zoom={zoom} />
          {goal.kind === "example" && !goalImgs.length && <p className="ai-prose">Target model: {goal.target_name}</p>}
        </Sub>
        {run.plans.map((p, i) => (
          <Sub key={i} title={p.source === "model" ? `Expert plan from ${p.model}` : "Expert plan"}>
            <Clip text={p.text ?? ""} />
          </Sub>
        ))}
        <Sub title="Instructions">
          <details className="ai-text">
            <summary>System prompt · {kfmt((s.system_prompt ?? "").length)} characters</summary>
            <pre>{s.system_prompt}</pre>
          </details>
          {Array.isArray(s.tools) && (
            <p className="ai-size">Tools: {(s.tools as { name: string }[]).map((t) => t.name).join(", ")}</p>
          )}
        </Sub>
        <Raw events={[{ ...s, goal_parts: undefined } as RunEvent]} />
      </div>
    </details>
  );
}

export function Inspector({ run }: { run: RunView }) {
  const step = useUi((s) => s.step);
  const [onlyBad, setOnlyBad] = useState(false);
  const [zoomed, setZoomed] = useState<{ src: string; caption?: string } | null>(null);
  const dlg = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    if (zoomed) dlg.current?.showModal?.();
  }, [zoomed]);
  const cur = turnForStep(run.turns, step);
  const turns = onlyBad ? run.turns.filter((t) => turnMark(t) !== "ok") : run.turns;
  const zoom: Zoom = (src, caption) => setZoomed({ src, caption });
  return (
    <div className="ai-inspector">
      <div className="ai-inspector-head">
        <h2 className="section-title">Turn by turn</h2>
        <label className="ai-check">
          <input type="checkbox" checked={onlyBad} onChange={(e) => setOnlyBad(e.target.checked)} /> Only problems
        </label>
      </div>
      <Setup run={run} zoom={zoom} />
      {turns.map((t) => (
        <TurnBlock key={t.turn} t={t} run={run} current={t.turn === cur} zoom={zoom} />
      ))}
      {zoomed && (
        <dialog
          ref={dlg}
          className="ai-lightbox"
          onClose={() => setZoomed(null)}
          onClick={() => dlg.current?.close()}
          aria-label={zoomed.caption ?? "Picture"}
        >
          <img src={zoomed.src} alt={zoomed.caption ?? ""} />
          {zoomed.caption && <p>{zoomed.caption}</p>}
        </dialog>
      )}
    </div>
  );
}

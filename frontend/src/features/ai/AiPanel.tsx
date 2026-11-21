import { useEffect, useState } from "react";
import { getRun, RUNS, type RunMeta } from "../../lib/examples";
import { hrefFor } from "../../lib/router";
import { Thumb } from "../../ui/Thumb";
import { fetchConfig, type AiConfig } from "./api";
import { clearRun, follow, loadReplay, skipToEnd, stopLive, syncViewer, useAi } from "./aiStore";
import { Inspector } from "./Inspector";
import { LiveForm } from "./LiveForm";
import { RunMark } from "./RunMark";
import { kfmt, usd, type RunView } from "./runEvents";
import "./ai.css";

const cost = (r: RunMeta) => (r.costUsd != null ? `, $${r.costUsd.toFixed(2)}` : "");

const COMMANDS = [
  "git clone https://github.com/Abir66/foldr && cd foldr",
  "cd backend && cp config.example.toml foldr.toml",
  "# put your key in backend/.env, e.g. OPENROUTER_API_KEY=sk-...",
  "uv run python -m foldr_backend",
  "cd ../frontend && npm ci",
  "VITE_API_URL=http://localhost:8000 npm run dev",
].join("\n");

let configPromise: Promise<AiConfig | null> | null = null;

function useConfig(): AiConfig | null | undefined {
  const [cfg, setCfg] = useState<AiConfig | null | undefined>(undefined);
  useEffect(() => {
    configPromise ??= fetchConfig(AbortSignal.timeout?.(8000));
    let alive = true;
    configPromise.then((c) => alive && setCfg(c));
    return () => {
      alive = false;
    };
  }, []);
  return cfg;
}

const STOP: Record<string, string> = {
  submitted: "The AI said it was done",
  stopped: "Stopped",
  turns: "Ran out of turns",
  turn_limit: "Ran out of turns",
  budget: "Hit the money limit",
  format_errors: "Too many bad replies",
  error: "Something went wrong",
};

function Counter({ run, max }: { run: RunView; max?: number }) {
  return (
    <p className="ai-counter" aria-live="polite">
      Turn {run.turns.length}
      {max ? ` of ${max}` : ""} · {usd(run.spent)} · {kfmt(run.tokensIn)} in / {kfmt(run.tokensOut)} out
    </p>
  );
}

function Result({ run }: { run: RunView }) {
  const f = run.finished;
  if (!f) return null;
  const passed = f.score?.passed;
  return (
    <div className={`ai-result ${passed ? "is-pass" : "is-fail"}`}>
      {f.score ? <RunMark passed={!!passed} /> : null}
      <span>
        {STOP[f.stop_reason] ?? f.stop_reason}
        {f.totals ? ` · ${f.totals.turns} turns · ${usd(f.totals.cost_usd ?? 0)} · ${Math.round(f.totals.seconds ?? 0)}s` : ""}
      </span>
      {f.error && <span className="ai-error-text">{String(f.error)}</span>}
    </div>
  );
}

function RunList() {
  return (
    <>
      <h2 className="section-title">Recorded runs</h2>
      <ul className="run-list">
        {RUNS.map((r) => (
          <li key={r.slug}>
            <a className="run-row" href={hrefFor("ai", r.slug)}>
              <Thumb name={r.example} size={44} />
              <span className="run-text">
                <span className="run-title">{r.title}</span>
                <span className="run-meta">
                  <RunMark passed={r.passed} /> {r.model}
                  {cost(r)}
                </span>
              </span>
            </a>
          </li>
        ))}
      </ul>
    </>
  );
}

export function AiPanel({ runSlug }: { runSlug?: string }) {
  const meta = getRun(runSlug);
  const cfg = useConfig();
  const { source, run, loading, error, lost, stopping } = useAi();

  useEffect(() => {
    if (meta) loadReplay(meta);
    else if (useAi.getState().source?.kind === "replay") clearRun();
  }, [meta]);

  // the new run's blank sheet reached the stage in this commit (sibling effects run first): add steps, follow
  useEffect(() => {
    if (!run.base) return;
    syncViewer();
    follow();
  }, [run.base]);

  if (meta) {
    return (
      <div className="ai">
        <a className="text-link back" href={hrefFor("ai")}>
          All runs
        </a>
        <h1 className="model-title">{meta.title}</h1>
        <p className="model-meta">
          <RunMark passed={meta.passed} /> {meta.model}, {meta.mode === "menu" ? "picking from suggestions" : "describing each fold"}
          {meta.goal === "text" ? ", goal given in words only" : ""}
          {cost(meta)}
        </p>
        {meta.blurb && <p className="lead">{meta.blurb}</p>}
        {loading && <p className="ai-note">Loading the run…</p>}
        {error && (
          <p className="ai-error" role="alert">
            {error}
          </p>
        )}
        {run.started && (
          <div className="ai-bar">
            <Counter run={run} max={run.started.limits?.max_turns} />
            {!run.finished && (
              <button className="button button-quiet" onClick={skipToEnd}>
                Skip to the end
              </button>
            )}
          </div>
        )}
        <Result run={run} />
        {run.started && <Inspector run={run} />}
      </div>
    );
  }

  if (source?.kind === "live" || loading || error) {
    return (
      <div className="ai">
        <h1 className="model-title">Live AI run</h1>
        {loading && <p className="ai-note">Starting…</p>}
        {error && (
          <p className="ai-error" role="alert">
            {error}
          </p>
        )}
        {lost && !run.finished && <p className="ai-error">Lost the connection to the AI server.</p>}
        {run.started && (
          <div className="ai-bar">
            <Counter run={run} max={run.started.limits?.max_turns} />
            {!run.finished && (
              <button className="button" onClick={stopLive} disabled={stopping}>
                {stopping ? "Stopping after this turn…" : "Stop"}
              </button>
            )}
          </div>
        )}
        <Result run={run} />
        {(run.finished || error || lost) && (
          <button className="button button-primary ai-again" onClick={clearRun}>
            {error ? "Back" : "Start another run"}
          </button>
        )}
        {run.started && <Inspector run={run} />}
      </div>
    );
  }

  return (
    <div className="ai">
      <h1 className="model-title">Watch an AI fold</h1>
      {cfg === undefined && <p className="ai-note">Checking whether the AI is on…</p>}
      {cfg && <LiveForm cfg={cfg} />}
      {cfg === null && (
        <>
          <p className="lead">
            <strong>AI is off in this demo</strong>, because I am too broke to afford API calls :') Watch a recorded run below, or run Foldr on your own
            computer with your own key:
          </p>
          <pre className="commands" tabIndex={0} aria-label="Commands to run Foldr on your computer">
            <code>{COMMANDS}</code>
          </pre>
        </>
      )}
      <RunList />
    </div>
  );
}

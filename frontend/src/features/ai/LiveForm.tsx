import { useState } from "react";
import { fileToB64, type AiConfig } from "./api";
import { startLive } from "./aiStore";

type GoalKind = "example" | "text" | "image";

/** Start a live run: goal, how the AI acts, plan, model, optional own key, turn cap. */
export function LiveForm({ cfg }: { cfg: AiConfig }) {
  const examples = cfg.examples;
  const defModel = cfg.models.find((m) => m.default) ?? cfg.models[0];
  const keyProviders = cfg.providers.filter((p) => p.user_key);
  const [goalKind, setGoalKind] = useState<GoalKind>("example");
  const [example, setExample] = useState(examples.includes("cup") ? "cup" : (examples[0] ?? ""));
  const [text, setText] = useState("");
  const [checkAgainst, setCheckAgainst] = useState("");
  const [image, setImage] = useState<File | null>(null);
  const [actions, setActions] = useState<"menu" | "command">("menu");
  const [planOn, setPlanOn] = useState(false);
  const [model, setModel] = useState(defModel?.id ?? "");
  const [planner, setPlanner] = useState(defModel?.id ?? "");
  const [maxTurns, setMaxTurns] = useState(Math.min(20, cfg.ai.max_turns));
  const [keyProvider, setKeyProvider] = useState(keyProviders[0]?.id ?? "");
  const [key, setKey] = useState("");
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);

  const usingKey = cfg.ai.allow_user_keys && key.trim() !== "";
  const usable = cfg.models.filter((m) => m.owner_key || (usingKey && m.user_key));
  const chosen = cfg.models.find((m) => m.id === model);

  const start = async () => {
    setProblem(null);
    let goal: Record<string, unknown>;
    if (goalKind === "example") goal = { kind: "example", example };
    else if (goalKind === "text") {
      if (!text.trim()) return setProblem("Describe what the AI should fold.");
      goal = { kind: "text", text: text.trim(), check_against: checkAgainst || null };
    } else {
      if (!image) return setProblem("Pick a picture first.");
      if (image.size > 2_000_000) return setProblem("That picture is too big (max 2 MB).");
      goal = { kind: "image", image_b64: await fileToB64(image), text: text.trim() || null, check_against: checkAgainst || null };
    }
    if (!usable.some((m) => m.id === model)) return setProblem("That model needs your own key.");
    setBusy(true);
    await startLive({
      goal,
      actions,
      plan: planOn ? { kind: "model", model: planner } : { kind: "none" },
      model,
      images: chosen?.images ?? true,
      max_turns: maxTurns,
      user_key: usingKey ? { provider: keyProvider, key: key.trim() } : null,
    });
    setBusy(false);
  };

  return (
    <form
      className="ai-form"
      onSubmit={(e) => {
        e.preventDefault();
        start();
      }}
    >
      <p className="lead">Give the AI a goal and watch it fold, one move at a time.</p>
      <fieldset>
        <legend>Goal</legend>
        <div className="ai-seg" role="radiogroup">
          {(["example", "text", "image"] as GoalKind[]).map((k) => (
            <label key={k} className={goalKind === k ? "is-on" : undefined}>
              <input type="radio" name="goal" checked={goalKind === k} onChange={() => setGoalKind(k)} />
              {k === "example" ? "A model" : k === "text" ? "In words" : "A picture"}
            </label>
          ))}
        </div>
        {goalKind === "example" && (
          <select value={example} onChange={(e) => setExample(e.target.value)} aria-label="Target model">
            {examples.map((x) => (
              <option key={x}>{x}</option>
            ))}
          </select>
        )}
        {goalKind === "image" && (
          <input type="file" accept="image/png,image/jpeg,image/webp" onChange={(e) => setImage(e.target.files?.[0] ?? null)} aria-label="Reference picture" />
        )}
        {goalKind !== "example" && (
          <>
            <textarea
              rows={3}
              value={text}
              placeholder={goalKind === "text" ? "e.g. a samurai helmet you could wear" : "Optional hint about the picture"}
              onChange={(e) => setText(e.target.value)}
              aria-label="Goal in words"
            />
            <label className="ai-field">
              Score it against
              <select value={checkAgainst} onChange={(e) => setCheckAgainst(e.target.value)}>
                <option value="">nothing (no score)</option>
                {examples.map((x) => (
                  <option key={x}>{x}</option>
                ))}
              </select>
            </label>
          </>
        )}
      </fieldset>

      <fieldset>
        <legend>How the AI folds</legend>
        <div className="ai-seg" role="radiogroup">
          <label className={actions === "menu" ? "is-on" : undefined}>
            <input type="radio" name="actions" checked={actions === "menu"} onChange={() => setActions("menu")} />
            Picks from suggestions
          </label>
          <label className={actions === "command" ? "is-on" : undefined}>
            <input type="radio" name="actions" checked={actions === "command"} onChange={() => setActions("command")} />
            Describes each fold
          </label>
        </div>
      </fieldset>

      <fieldset>
        <legend>Model</legend>
        <select value={model} onChange={(e) => setModel(e.target.value)} aria-label="Model">
          {cfg.models.map((m) => (
            <option key={m.id} value={m.id} disabled={!m.owner_key && !usingKey}>
              {m.label}
              {!m.owner_key ? " (needs your key)" : ""}
            </option>
          ))}
        </select>
        <label className="ai-check">
          <input type="checkbox" checked={planOn} onChange={(e) => setPlanOn(e.target.checked)} /> First ask a model for an expert plan
        </label>
        {planOn && (
          <select value={planner} onChange={(e) => setPlanner(e.target.value)} aria-label="Planner model">
            {usable.map((m) => (
              <option key={m.id} value={m.id}>
                Plan by {m.label}
              </option>
            ))}
          </select>
        )}
        <label className="ai-field">
          Max turns
          <input
            type="number"
            min={1}
            max={cfg.ai.max_turns}
            value={maxTurns}
            onChange={(e) => setMaxTurns(Math.max(1, Math.min(cfg.ai.max_turns, Number(e.target.value) || 1)))}
          />
        </label>
      </fieldset>

      {cfg.ai.allow_user_keys && keyProviders.length > 0 && (
        <fieldset>
          <legend>Your own key (optional)</legend>
          <div className="ai-key">
            {keyProviders.length > 1 && (
              <select value={keyProvider} onChange={(e) => setKeyProvider(e.target.value)} aria-label="Key provider">
                {keyProviders.map((p) => (
                  <option key={p.id}>{p.id}</option>
                ))}
              </select>
            )}
            <input type="password" autoComplete="off" value={key} placeholder={`${keyProvider} key`} onChange={(e) => setKey(e.target.value)} aria-label="Your API key" />
          </div>
          <p className="ai-note">Used for this run only, never stored.</p>
        </fieldset>
      )}

      {problem && (
        <p className="ai-error" role="alert">
          {problem}
        </p>
      )}
      <button className="button button-primary" type="submit" disabled={busy || !model}>
        {busy ? "Starting…" : "Start"}
      </button>
      {cfg.ai.per_run_usd != null && !usingKey && <p className="ai-note">Each run stops at {`$${cfg.ai.per_run_usd.toFixed(2)}`}.</p>}
    </form>
  );
}

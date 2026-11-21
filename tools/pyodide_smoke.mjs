// Pyodide smoke test: load the origami wheel in Pyodide (Node) and drive web/bridge.py.
//
//   cd tools && npm install && node pyodide_smoke.mjs [--wheel path.whl] [--crane] [--export]
//
// Without --wheel it builds one with `uv build --wheel` (origami/dist/). --crane adds a crane-size
// menu timing (slow), --export loads matplotlib and checks the PNG exports.
// Exit code 0 = all checks passed.

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadPyodide } from "pyodide";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const PKG = path.join(ROOT, "origami");
const MODELS = path.join(ROOT, "examples", "models");
const args = process.argv.slice(2);
const flag = (f) => args.includes(f);
const opt = (f) => (args.includes(f) ? args[args.indexOf(f) + 1] : null);

const LIMITS = { cupMenuS: 10, deadlineMs: 2500 };

function buildWheel() {
  const out = path.join(PKG, "dist");
  execFileSync("uv", ["build", "--wheel", "--out-dir", out], { cwd: PKG, stdio: "inherit" });
  const whl = fs.readdirSync(out).filter((f) => f.endsWith(".whl")).map((f) => path.join(out, f));
  whl.sort((a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs);
  return whl[0];
}

let failures = 0;
function check(cond, msg) {
  console.log(`${cond ? "ok  " : "FAIL"} ${msg}`);
  if (!cond) failures++;
}

const wheel = opt("--wheel") || buildWheel();
console.log(`wheel: ${path.relative(ROOT, wheel)} (${(fs.statSync(wheel).size / 1024).toFixed(0)} KB)`);

let t = Date.now();
const py = await loadPyodide();
await py.loadPackage(["numpy"]);
py.unpackArchive(new Uint8Array(fs.readFileSync(wheel)), "wheel");
console.log(`pyodide ${py.version} + numpy + wheel: ${Date.now() - t} ms`);

const bridge = py.pyimport("origami.web.bridge");

function call(method, params = {}, progress = null) {
  const t0 = Date.now();
  const out = JSON.parse(bridge.call(method, JSON.stringify(params), progress));
  const ms = Date.now() - t0;
  if (out.exception) throw new Error(`${method}: ${out.exception.type}: ${out.exception.message}`);
  return [out, ms];
}
const model = (name) => JSON.parse(fs.readFileSync(path.join(MODELS, `${name}.json`), "utf8"));
const stripDesc = ({ description, ...s }) => s;

// init
let [r, ms] = call("init", { stage: "core" });
check(r.versions.platform === "emscripten", `init core (${ms} ms): ${JSON.stringify(r.versions)}`);
check(py.runPython("__import__('origami.menu').menu._workers()") === 0, "menu uses no worker processes");

// timeline
const cup = model("cup");
[r, ms] = call("timeline", { model: cup });
check(r.error === null && r.timeline.steps.length === cup.steps.length, `timeline(cup): ${r.timeline?.steps.length} steps, ${ms} ms`);
[r] = call("timeline", { model: { ...cup, steps: [...cup.steps.slice(0, 2), { op: "nope" }] } });
check(r.failed_step === 3 && r.shown_steps === 2, `timeline with a bad step 3: partial timeline of ${r.shown_steps} steps`);

// session: new / apply / undo / custom / preview
[r] = call("session.new", { size: 1, colored_side: cup.paper?.colored_side || "up" });
check(r.rev === 0 && r.points.length === 9 && r.timeline.steps.length === 0, `session.new: ${r.points.length} points`);
const labels = Object.fromEntries(r.points.map((p) => [p.name, p.label]));
[r] = call("session.apply", { step: stripDesc(cup.steps[0]) });
check(r.ok && r.rev === 1 && r.timeline_step && r.state, `session.apply: ${r.text}`);
[r] = call("session.apply", { step: { op: "valley", line: { onto: ["tr", "tr"] } } });
check(!r.ok && r.rev === 1, `bad fold rejected: ${r.text}`);
[r] = call("session.undo", { n: 1 });
check(r.ok && r.timeline_len === 0 && r.rev === 2, `session.undo: ${r.text}`);
[r] = call("session.custom", { p: labels.tr, q: labels.bl, kind: "onto" });
check(r.ok && r.timeline_step, `session.custom onto: ${r.text}`);
[r] = call("session.preview", { step: stripDesc(cup.steps[1]) });
check(r.ok && r.state, `session.preview: ${r.text}`);
call("session.undo", { n: 1 });

// cup mid-fold menu (same state as the design-phase timing: first 4 steps)
for (const s of cup.steps.slice(0, 4)) {
  [r] = call("session.apply", { step: stripDesc(s) });
  if (!r.ok) throw new Error(`cup step failed: ${r.text}`);
}
const pushed = [];
[r, ms] = call("session.menu", {}, (s) => pushed.push(JSON.parse(s)));
check(r.options.length > 10 && ms < LIMITS.cupMenuS * 1000, `cup menu (full): ${r.options.length} options, ${ms} ms (< ${LIMITS.cupMenuS} s)`);
check(pushed.length > 0 && pushed.every((p) => p.rev === r.rev), `progress pushes: ${pushed.map((p) => `${p.section}:${p.options.length}`).join(" ")}`);
check(r.options.every((o) => o.result_state && o.step && o.section), "options carry step + result_state + section");
const full = r.options;
[r, ms] = call("session.menu", { deadline_ms: 1 });
check(r.skipped.length > 0 && Object.values(r.truncated).some(Boolean), `deadline 1 ms: ${ms} ms, skipped ${r.skipped.join(",")}`);
const prefix = r.options.filter((o) => o.section !== "unfold").map((o) => o.description);
check(JSON.stringify(prefix) === JSON.stringify(full.slice(0, prefix.length).map((o) => o.description)), "deadline menu is a prefix of the full menu");
const flapSec = r.skipped.find((s) => s !== "repeat") || "flap";
[r] = call("session.menu", { sections: [flapSec], more: { [flapSec]: 2 } });
check(r.options.every((o) => o.section === flapSec), `show more of '${flapSec}': ${r.options.length} options`);

// one menu option applied
[r] = call("session.apply", { step: full[0].step });
check(r.ok, `apply menu option 1: ${r.text}`);
[r] = call("session.steps");
check(r.model.steps.length === 5, `session.steps: ${r.model.steps.length} steps`);

// exports (core)
[r, ms] = call("export", { kinds: ["fold", "model_json", "timeline_json"] });
check(r.errors.length === 0 && r.files.length === 3, `export core: ${r.files.map((f) => f.name).join(", ")} (${ms} ms)`);
check(!py.runPython("'matplotlib' in __import__('sys').modules"), "core never imported matplotlib");

if (flag("--export")) {
  t = Date.now();
  await py.loadPackage(["matplotlib"]);
  [r] = call("init", { stage: "export" });
  console.log(`matplotlib stage: ${Date.now() - t} ms`);
  [r, ms] = call("export", { model: cup, kinds: ["png_zip", "cp_png"] });
  check(r.errors.length === 0 && r.files.length === 2, `export png: ${r.files.map((f) => `${f.name} ${f.b64.length}b64`).join(", ")} (${ms} ms)`);
}

if (flag("--crane")) {
  const crane = model("crane");
  call("session.new", { size: 1, colored_side: crane.paper?.colored_side || "up" });
  t = Date.now();
  for (const s of crane.steps.slice(0, 13)) call("session.apply", { step: stripDesc(s) });
  console.log(`crane 13 steps applied: ${Date.now() - t} ms`);
  const secs = [];
  [r, ms] = call("session.menu", { deadline_ms: LIMITS.deadlineMs }, (s) => secs.push(JSON.parse(s).section));
  check(r.options.length > 0, `crane menu deadline ${LIMITS.deadlineMs} ms: ${ms} ms, sections ${secs.join(",")}, skipped ${r.skipped.join(",")}`);
  [r, ms] = call("session.menu", {});
  console.log(`crane menu full: ${r.options.length} options, ${ms} ms`);
}

console.log(failures ? `\n${failures} check(s) FAILED` : "\nall checks passed");
process.exit(failures ? 1 : 0);

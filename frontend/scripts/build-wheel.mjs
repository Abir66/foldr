// Builds the origami wheel into public/py/ (served next to the app) and writes public/py/wheel.json,
// which the Pyodide worker reads to find the file. Needs `uv` on PATH.
import { execFileSync } from "node:child_process";
import { mkdirSync, readdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const pkg = path.resolve(here, "../../origami");
const out = path.resolve(here, "../public/py");

rmSync(out, { recursive: true, force: true });
mkdirSync(out, { recursive: true });
execFileSync("uv", ["build", "--wheel", "--out-dir", out], { cwd: pkg, stdio: "inherit" });
const file = readdirSync(out).find((f) => f.endsWith(".whl"));
if (!file) throw new Error("uv build produced no wheel");
const version = file.split("-")[1];
const size = statSync(path.join(out, file)).size;
writeFileSync(path.join(out, "wheel.json"), JSON.stringify({ file, version, size, built: new Date().toISOString() }) + "\n");
// uv drops a .gitignore into the out dir; keep only what the site serves
for (const f of readdirSync(out)) if (f !== file && f !== "wheel.json") rmSync(path.join(out, f), { force: true });
console.log(`wheel: public/py/${file} (${(size / 1024).toFixed(0)} KB)`);

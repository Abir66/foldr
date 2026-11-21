// `npm run dev` needs the generated example files; build them once if they are missing.
import { existsSync } from "node:fs";
import { spawnSync } from "node:child_process";

if (!existsSync(new URL("../public/examples/index.json", import.meta.url))) {
  console.log("building examples (first run)...");
  const r = spawnSync("npm", ["run", "examples"], { stdio: "inherit" });
  process.exit(r.status ?? 1);
}

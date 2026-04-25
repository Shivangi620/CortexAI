import { existsSync } from "node:fs";
import { spawnSync } from "node:child_process";
import path from "node:path";

const rootDir = process.cwd();
const candidates = [
  path.join(rootDir, "venv", "bin", "python"),
  path.join(rootDir, "venv", "Scripts", "python.exe"),
  "python",
];

const python = candidates.find((candidate) => candidate === "python" || existsSync(candidate));
const targets = process.argv.slice(2);

const result = spawnSync(python, ["-m", "ruff", "check", ...(targets.length ? targets : ["backend"])], {
  cwd: rootDir,
  stdio: "inherit",
});

if (result.error) {
  console.error(result.error.message);
  process.exit(1);
}

process.exit(result.status ?? 1);

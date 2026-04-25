import { build } from "esbuild";
import { copyFile, mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, "..");
const outDir = path.join(rootDir, "frontend", "static");
const logoSourcePath = path.join(rootDir, "frontend", "WhatsApp Image 2026-04-26 at 1.33.22 AM.jpeg");
const logoOutputPath = path.join(outDir, "assets", "project-logo.jpeg");

await mkdir(path.join(outDir, "assets"), { recursive: true });
await copyFile(logoSourcePath, logoOutputPath);

await build({
  entryPoints: [path.join(rootDir, "frontend", "react-src", "main.jsx")],
  bundle: true,
  format: "esm",
  minify: false,
  sourcemap: false,
  target: ["es2020"],
  outfile: path.join(outDir, "assets", "app.js"),
  loader: {
    ".js": "jsx",
    ".jsx": "jsx",
  },
});

await build({
  entryPoints: [path.join(rootDir, "frontend", "react-src", "styles.css")],
  bundle: true,
  minify: false,
  outfile: path.join(outDir, "assets", "app.css"),
});

const html = `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Inferyx Neural Studio</title>
    <link rel="stylesheet" href="/static/assets/app.css" />
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/static/assets/app.js"></script>
  </body>
</html>
`;

await writeFile(path.join(outDir, "index.html"), html, "utf8");

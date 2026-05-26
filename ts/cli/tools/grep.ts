import * as fs from "node:fs/promises";
import * as path from "node:path";
import type { Tool } from "@core/types";
import {
  EXCLUDED,
  MAX_FILE_SIZE,
  resolveInWorkspace,
  workspaceRoot,
} from "./workspace";

const MAX_MATCHES = 200;
const MAX_FILES = 2000;

type Counter = { files: number };

// バイナリ判定の簡易ヒューリスティック。先頭チャンクに NUL があればバイナリ扱いで除外する。
function looksBinary(buf: Buffer): boolean {
  const len = Math.min(buf.length, 8000);
  for (let i = 0; i < len; i++) {
    if (buf[i] === 0) return true;
  }
  return false;
}

async function searchDir(
  dir: string,
  rootAbs: string,
  re: RegExp,
  out: string[],
  counter: Counter,
): Promise<void> {
  if (out.length >= MAX_MATCHES || counter.files >= MAX_FILES) return;
  const entries = await fs.readdir(dir, { withFileTypes: true });
  entries.sort((a, b) => a.name.localeCompare(b.name));
  for (const entry of entries) {
    if (out.length >= MAX_MATCHES || counter.files >= MAX_FILES) return;
    if (EXCLUDED.has(entry.name)) continue;
    if (entry.name.startsWith(".") && entry.name !== ".github") continue;
    const abs = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      await searchDir(abs, rootAbs, re, out, counter);
      continue;
    }
    if (!entry.isFile()) continue;

    const stat = await fs.stat(abs);
    if (stat.size > MAX_FILE_SIZE) continue;
    counter.files++;

    const buf = await fs.readFile(abs);
    if (looksBinary(buf)) continue;

    const rel = path.relative(rootAbs, abs) || entry.name;
    const lines = buf.toString("utf-8").split("\n");
    for (const [idx, line] of lines.entries()) {
      if (out.length >= MAX_MATCHES) return;
      // RegExp に g フラグを付けないため lastIndex を持ち越さず、行ごとに独立判定できる。
      if (re.test(line)) {
        out.push(`${rel}:${idx + 1}:${line}`);
      }
    }
  }
}

export const grep: Tool = {
  name: "grep",
  description:
    "Search file contents under WORKSPACE_ROOT with a JavaScript regular expression. Returns matches as 'path:line:text'. Skips node_modules, .git, dist, binary files and files larger than 1 MB. Up to 200 matches.",
  parameters: {
    type: "object",
    properties: {
      pattern: {
        type: "string",
        description: "JavaScript regular expression to match per line",
      },
      path: {
        type: "string",
        description:
          "Directory or file to search, relative to WORKSPACE_ROOT (default '.')",
      },
      ignoreCase: {
        type: "boolean",
        description: "Case-insensitive match. Default false.",
      },
    },
    required: ["pattern"],
  },
  needsApproval: false,
  async execute(args) {
    const pattern = args.pattern;
    if (typeof pattern !== "string") {
      throw new Error("grep: 'pattern' must be a string");
    }
    const searchPath = typeof args.path === "string" ? args.path : ".";

    let re: RegExp;
    try {
      re = new RegExp(pattern, args.ignoreCase === true ? "i" : "");
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new Error(`grep: invalid regular expression: ${msg}`);
    }

    const root = workspaceRoot();
    const target = resolveInWorkspace(searchPath);

    let stat: Awaited<ReturnType<typeof fs.stat>>;
    try {
      stat = await fs.stat(target);
    } catch (err) {
      const e = err as NodeJS.ErrnoException;
      if (e.code === "ENOENT") {
        throw new Error(`path not found: ${searchPath}`);
      }
      throw err;
    }

    const matches: string[] = [];
    const counter: Counter = { files: 0 };
    if (stat.isDirectory()) {
      await searchDir(target, root, re, matches, counter);
    } else if (stat.isFile()) {
      if (stat.size <= MAX_FILE_SIZE) {
        const buf = await fs.readFile(target);
        if (!looksBinary(buf)) {
          const rel = path.relative(root, target) || searchPath;
          for (const [idx, line] of buf.toString("utf-8").split("\n").entries()) {
            if (matches.length >= MAX_MATCHES) break;
            if (re.test(line)) matches.push(`${rel}:${idx + 1}:${line}`);
          }
        }
      }
    }

    if (matches.length === 0) {
      return "(no matches)";
    }
    if (matches.length >= MAX_MATCHES) {
      matches.push(`... (truncated at ${MAX_MATCHES} matches)`);
    }
    return matches.join("\n");
  },
};

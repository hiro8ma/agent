import * as fs from "node:fs/promises";
import * as path from "node:path";
import type { Tool } from "@core/types";

const EXCLUDED = new Set([
  "node_modules",
  ".git",
  "dist",
  "build",
  ".next",
  ".turbo",
  ".cache",
  "vendor",
  ".venv",
  "__pycache__",
]);

const MAX_ENTRIES = 500;

function workspaceRoot(): string {
  return path.resolve(process.env.WORKSPACE_ROOT ?? process.cwd());
}

async function walk(
  dir: string,
  rootAbs: string,
  recursive: boolean,
  out: string[],
): Promise<void> {
  if (out.length >= MAX_ENTRIES) return;
  const entries = await fs.readdir(dir, { withFileTypes: true });
  entries.sort((a, b) => a.name.localeCompare(b.name));
  for (const entry of entries) {
    if (out.length >= MAX_ENTRIES) return;
    if (EXCLUDED.has(entry.name)) continue;
    if (entry.name.startsWith(".") && entry.name !== ".github") continue;
    const abs = path.join(dir, entry.name);
    const rel = path.relative(rootAbs, abs) || entry.name;
    if (entry.isDirectory()) {
      out.push(`${rel}/`);
      if (recursive) {
        await walk(abs, rootAbs, recursive, out);
      }
    } else if (entry.isFile()) {
      out.push(rel);
    }
  }
}

export const listFiles: Tool = {
  name: "listFiles",
  description:
    "List files and directories under WORKSPACE_ROOT. Directories end with '/'. Skips node_modules, .git, dist and similar. Up to 500 entries.",
  parameters: {
    type: "object",
    properties: {
      path: {
        type: "string",
        description: "Directory path relative to WORKSPACE_ROOT (use '.' for root)",
      },
      recursive: {
        type: "boolean",
        description: "If true, walk subdirectories. Default false.",
      },
    },
    required: ["path"],
  },
  needsApproval: false,
  async execute(args) {
    const dirPath = args.path;
    if (typeof dirPath !== "string") {
      throw new Error("listFiles: 'path' must be a string");
    }
    const recursive = args.recursive === true;

    const root = workspaceRoot();
    const target = path.resolve(root, dirPath);
    const allowedPrefix = root + path.sep;
    if (target !== root && !target.startsWith(allowedPrefix)) {
      throw new Error(`access denied: ${dirPath} is outside workspace`);
    }

    let stat: Awaited<ReturnType<typeof fs.stat>>;
    try {
      stat = await fs.stat(target);
    } catch (err) {
      const e = err as NodeJS.ErrnoException;
      if (e.code === "ENOENT") {
        throw new Error(`directory not found: ${dirPath}`);
      }
      throw err;
    }
    if (!stat.isDirectory()) {
      throw new Error(`not a directory: ${dirPath}`);
    }

    const entries: string[] = [];
    await walk(target, root, recursive, entries);
    if (entries.length === 0) {
      return "(empty directory)";
    }
    if (entries.length >= MAX_ENTRIES) {
      entries.push(`... (truncated at ${MAX_ENTRIES} entries)`);
    }
    return entries.join("\n");
  },
};

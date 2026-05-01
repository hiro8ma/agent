import * as fs from "node:fs/promises";
import * as path from "node:path";
import { z } from "zod";
import type { Tool } from "@core/types";

const MAX_FILE_SIZE = 1024 * 1024;

const parameters = z.object({
  path: z
    .string()
    .describe("Path to the file relative to WORKSPACE_ROOT (e.g. README.md)"),
});

function workspaceRoot(): string {
  return path.resolve(process.env.WORKSPACE_ROOT ?? process.cwd());
}

export const readFile: Tool<typeof parameters> = {
  name: "readFile",
  description:
    "Read a file from WORKSPACE_ROOT as utf-8 text. Rejects access outside the workspace and files larger than 1 MB.",
  parameters,
  needsApproval: false,
  async execute(args) {
    const root = workspaceRoot();
    const target = path.resolve(root, args.path);
    const allowedPrefix = root + path.sep;
    if (target !== root && !target.startsWith(allowedPrefix)) {
      throw new Error(`access denied: ${args.path} is outside workspace`);
    }

    let realTarget: string;
    try {
      realTarget = await fs.realpath(target);
    } catch (err) {
      const e = err as NodeJS.ErrnoException;
      if (e.code === "ENOENT") {
        throw new Error(`file not found: ${args.path}`);
      }
      throw err;
    }
    if (realTarget !== root && !realTarget.startsWith(allowedPrefix)) {
      throw new Error(
        `access denied: ${args.path} resolves outside workspace via symlink`,
      );
    }

    const stat = await fs.stat(realTarget);
    if (!stat.isFile()) {
      throw new Error(`not a regular file: ${args.path}`);
    }
    if (stat.size > MAX_FILE_SIZE) {
      throw new Error(
        `file too large: ${args.path} (${Math.round(stat.size / 1024)} KB > 1024 KB)`,
      );
    }
    return await fs.readFile(realTarget, "utf-8");
  },
};

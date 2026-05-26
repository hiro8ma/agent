import * as fs from "node:fs/promises";
import type { Tool } from "@core/types";
import {
  MAX_FILE_SIZE,
  realpathInWorkspace,
  resolveInWorkspace,
} from "./workspace";

export const readFile: Tool = {
  name: "readFile",
  description:
    "Read a file from WORKSPACE_ROOT as utf-8 text. Rejects access outside the workspace and files larger than 1 MB.",
  parameters: {
    type: "object",
    properties: {
      path: {
        type: "string",
        description: "Path to the file relative to WORKSPACE_ROOT (e.g. README.md)",
      },
    },
    required: ["path"],
  },
  needsApproval: false,
  async execute(args) {
    const filePath = args.path;
    if (typeof filePath !== "string") {
      throw new Error("readFile: 'path' must be a string");
    }

    const target = resolveInWorkspace(filePath);

    let realTarget: string;
    try {
      realTarget = await realpathInWorkspace(target, filePath);
    } catch (err) {
      const e = err as NodeJS.ErrnoException;
      if (e.code === "ENOENT") {
        throw new Error(`file not found: ${filePath}`);
      }
      throw err;
    }

    const stat = await fs.stat(realTarget);
    if (!stat.isFile()) {
      throw new Error(`not a regular file: ${filePath}`);
    }
    if (stat.size > MAX_FILE_SIZE) {
      throw new Error(
        `file too large: ${filePath} (${Math.round(stat.size / 1024)} KB > 1024 KB)`,
      );
    }
    return await fs.readFile(realTarget, "utf-8");
  },
};

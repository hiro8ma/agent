import * as fs from "node:fs/promises";
import * as path from "node:path";
import type { Tool } from "@core/types";
import {
  MAX_FILE_SIZE,
  realpathInWorkspace,
  resolveInWorkspace,
} from "./workspace";

// needsApproval: true。破壊的操作なので、HITL 承認（phase3）が入るまでは
// 呼び出し側が承認フローを差し込む前提で扱う。
export const writeFile: Tool = {
  name: "writeFile",
  description:
    "Create or overwrite a file under WORKSPACE_ROOT with utf-8 text. Creates parent directories as needed. Rejects paths outside the workspace and content larger than 1 MB. Destructive: requires approval.",
  parameters: {
    type: "object",
    properties: {
      path: {
        type: "string",
        description: "Path to the file relative to WORKSPACE_ROOT",
      },
      content: {
        type: "string",
        description: "Full file contents to write (overwrites any existing file)",
      },
    },
    required: ["path", "content"],
  },
  needsApproval: true,
  async execute(args) {
    const filePath = args.path;
    const content = args.content;
    if (typeof filePath !== "string") {
      throw new Error("writeFile: 'path' must be a string");
    }
    if (typeof content !== "string") {
      throw new Error("writeFile: 'content' must be a string");
    }

    const byteLength = Buffer.byteLength(content, "utf-8");
    if (byteLength > MAX_FILE_SIZE) {
      throw new Error(
        `content too large: ${Math.round(byteLength / 1024)} KB > 1024 KB`,
      );
    }

    const target = resolveInWorkspace(filePath);

    // 親ディレクトリが既存なら、symlink 経由で workspace 外を指していないか確認する。
    // ファイル自体は未作成のことがあるため realpath はディレクトリに対してかける。
    const parent = path.dirname(target);
    try {
      await fs.stat(parent);
      await realpathInWorkspace(parent, path.dirname(filePath));
    } catch (err) {
      const e = err as NodeJS.ErrnoException;
      if (e.code !== "ENOENT") throw err;
      // 親が無ければこの後 mkdir で作る。resolveInWorkspace で境界確認済み。
    }

    await fs.mkdir(parent, { recursive: true });
    await fs.writeFile(target, content, "utf-8");
    return `wrote ${byteLength} bytes to ${filePath}`;
  },
};

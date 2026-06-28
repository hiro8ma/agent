import * as fs from "node:fs/promises";
import type { Tool } from "@core/types";
import {
  MAX_FILE_SIZE,
  realpathInWorkspace,
  resolveInWorkspace,
} from "./workspace";

// needsApproval: true。既存ファイルを書き換える破壊的操作。
// 全文置換の writeFile と違い、oldText を一意に特定できる範囲に限って差分置換する。
// 一意性を保証することで「無関係な箇所を巻き込む」事故を防ぐのがこの tool の肝。
export const editFile: Tool = {
  name: "editFile",
  description:
    "Edit a file under WORKSPACE_ROOT by replacing oldText with newText. oldText must match exactly once in the file; zero or multiple matches are rejected so the edit stays unambiguous. Cheaper in tokens than rewriting the whole file. Destructive: requires approval.",
  parameters: {
    type: "object",
    properties: {
      path: {
        type: "string",
        description: "Path to the file relative to WORKSPACE_ROOT",
      },
      oldText: {
        type: "string",
        description:
          "Exact text to replace. Must identify a single, unique location in the file",
      },
      newText: {
        type: "string",
        description: "Replacement text",
      },
    },
    required: ["path", "oldText", "newText"],
  },
  needsApproval: true,
  async execute(args) {
    const filePath = args.path;
    const oldText = args.oldText;
    const newText = args.newText;
    if (typeof filePath !== "string") {
      throw new Error("editFile: 'path' must be a string");
    }
    if (typeof oldText !== "string") {
      throw new Error("editFile: 'oldText' must be a string");
    }
    if (typeof newText !== "string") {
      throw new Error("editFile: 'newText' must be a string");
    }
    if (oldText.length === 0) {
      throw new Error("editFile: 'oldText' must not be empty");
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

    const content = await fs.readFile(realTarget, "utf-8");

    // あいまい性チェック: oldText が一意に一致するときだけ編集する。
    const matches = content.split(oldText).length - 1;
    if (matches === 0) {
      const preview =
        oldText.length > 50 ? `${oldText.slice(0, 50)}...` : oldText;
      throw new Error(`no match for oldText: ${preview}`);
    }
    if (matches > 1) {
      throw new Error(
        `oldText matched ${matches} times; provide a larger unique snippet`,
      );
    }

    // 置換は関数形式で行う。文字列を渡すと newText 内の $& 等が置換パターンとして
    // 解釈されてしまうため、リテラルとして扱うために replacer 関数を使う。
    const next = content.replace(oldText, () => newText);

    const byteLength = Buffer.byteLength(next, "utf-8");
    if (byteLength > MAX_FILE_SIZE) {
      throw new Error(
        `result too large: ${Math.round(byteLength / 1024)} KB > 1024 KB`,
      );
    }

    await fs.writeFile(realTarget, next, "utf-8");
    return `edited ${filePath} (1 replacement, ${byteLength} bytes)`;
  },
};

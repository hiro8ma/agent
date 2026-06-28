import { spawn } from "node:child_process";
import type { Tool } from "@core/types";
import { resolveInWorkspace } from "./workspace";

// execCommand は最も危険な tool。ファイル操作と違い system 全体に影響しうるため、
// 脆弱性ごとに個別防御を重ねる:
//   - allowlist        : 許可したバイナリ以外は実行しない
//   - shell:false      : shell を介さず argv 起動。メタ文字によるインジェクションを構造的に防ぐ
//                        (引数の ; | & 等は文字列として渡るだけなので無害。shell 経由の
//                         コマンド文字列なら危険文字検出が要るが、argv 方式はそれ自体が不要)
//   - cwd 制限         : 実行ディレクトリを WORKSPACE_ROOT 配下に閉じ込める
//   - timeout          : 終わらないコマンドを kill する
//   - output 上限      : context window を圧迫する巨大出力を切り詰める
//   - 最小 env         : 親の env(API キー等)を子に渡さない。機密漏洩を防ぐ
//   - 自然言語エラー   : LLM が次の行動を選べる形で返す
//   - needsApproval    : 破壊的なので承認前提

const DEFAULT_TIMEOUT_MS = 30_000;
const MAX_OUTPUT = 100 * 1024;

// 許可コマンド。env EXEC_ALLOWLIST(カンマ区切り)で上書きできる。
function allowlist(): Set<string> {
  const raw = process.env.EXEC_ALLOWLIST;
  if (raw) {
    return new Set(
      raw
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean),
    );
  }
  return new Set([
    "ls",
    "cat",
    "pwd",
    "echo",
    "head",
    "tail",
    "wc",
    "grep",
    "find",
    "git",
    "node",
    "bun",
    "npm",
    "tsc",
  ]);
}

export const execCommand: Tool = {
  name: "execCommand",
  description:
    "Run an allowlisted command inside WORKSPACE_ROOT without a shell. Pass the binary as 'command' and arguments as a string array; arguments are passed literally with no shell expansion. Rejects commands outside the allowlist, caps output at 100 KB, and times out after 30s. The child process gets a minimal environment so secrets are not exposed. Most dangerous tool: requires approval.",
  parameters: {
    type: "object",
    properties: {
      command: {
        type: "string",
        description: "Binary to run; must be allowlisted (e.g. 'git', 'node')",
      },
      args: {
        type: "array",
        items: { type: "string" },
        description: "Arguments passed to the binary (no shell expansion)",
      },
      cwd: {
        type: "string",
        description:
          "Working directory relative to WORKSPACE_ROOT (default: workspace root)",
      },
    },
    required: ["command"],
  },
  needsApproval: true,
  async execute(args) {
    const command = args.command;
    const argv = args.args ?? [];
    const cwdArg = args.cwd ?? ".";
    if (typeof command !== "string") {
      throw new Error("execCommand: 'command' must be a string");
    }
    if (!Array.isArray(argv) || argv.some((a) => typeof a !== "string")) {
      throw new Error("execCommand: 'args' must be a string array");
    }
    if (typeof cwdArg !== "string") {
      throw new Error("execCommand: 'cwd' must be a string");
    }

    if (!allowlist().has(command)) {
      throw new Error(`command not allowed: ${command}`);
    }
    const argvStrings = argv as string[];

    const cwd = resolveInWorkspace(cwdArg);

    // 機密漏洩対策: 親 env(ANTHROPIC_API_KEY 等)を子に渡さない。最小限のみ。
    const env: NodeJS.ProcessEnv = {
      PATH: process.env.PATH ?? "",
      HOME: process.env.HOME ?? "",
    };

    return await new Promise<string>((resolve, reject) => {
      const child = spawn(command, argvStrings, {
        cwd,
        env,
        shell: false,
        timeout: DEFAULT_TIMEOUT_MS,
      });

      let out = "";
      let truncated = false;
      const onData = (chunk: Buffer) => {
        if (truncated) return;
        out += chunk.toString("utf-8");
        if (out.length > MAX_OUTPUT) {
          out = out.slice(0, MAX_OUTPUT);
          truncated = true;
        }
      };
      child.stdout.on("data", onData);
      child.stderr.on("data", onData);

      child.on("error", (err) => {
        const e = err as NodeJS.ErrnoException;
        if (e.code === "ENOENT") {
          reject(new Error(`command not found: ${command}`));
        } else {
          reject(err);
        }
      });

      child.on("close", (code, signal) => {
        if (signal === "SIGTERM") {
          reject(
            new Error(
              `command timed out after ${DEFAULT_TIMEOUT_MS / 1000}s: ${command}`,
            ),
          );
          return;
        }
        const body = (out + (truncated ? "\n…(output truncated at 100 KB)" : ""))
          .trimEnd();
        if (code === 0) {
          resolve(body || "(no output)");
        } else {
          resolve(`exit code ${code}:\n${body || "(no output)"}`);
        }
      });
    });
  },
};

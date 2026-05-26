import * as fs from "node:fs/promises";
import * as path from "node:path";

// 全ツール共通のサンドボックス境界。tool は WORKSPACE_ROOT の外を読み書きできない。
// セキュリティ上の境界なので実装は 1 箇所に集約し、各 tool でコピーしない。

export const MAX_FILE_SIZE = 1024 * 1024;

// listFiles / grep が走査時にスキップするディレクトリ。
export const EXCLUDED = new Set([
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

export function workspaceRoot(): string {
  return path.resolve(process.env.WORKSPACE_ROOT ?? process.cwd());
}

// 相対パスを絶対パスへ解決し、workspace 外なら拒否する。
// 文字列レベルのチェックのみ。symlink 経由の脱出は realpathInWorkspace で別途防ぐ。
export function resolveInWorkspace(relPath: string): string {
  const root = workspaceRoot();
  const target = path.resolve(root, relPath);
  const allowedPrefix = root + path.sep;
  if (target !== root && !target.startsWith(allowedPrefix)) {
    throw new Error(`access denied: ${relPath} is outside workspace`);
  }
  return target;
}

// 既存パスを realpath で解決し直し、symlink 経由で workspace 外を指していないか再確認する。
// root 側も realpath で正規化してから比較する（macOS の /var → /private/var のように
// workspace 自体が symlink 配下にあっても誤検知しないため）。
export async function realpathInWorkspace(
  target: string,
  relPath: string,
): Promise<string> {
  const realRoot = await fs.realpath(workspaceRoot());
  const allowedPrefix = realRoot + path.sep;
  const real = await fs.realpath(target);
  if (real !== realRoot && !real.startsWith(allowedPrefix)) {
    throw new Error(
      `access denied: ${relPath} resolves outside workspace via symlink`,
    );
  }
  return real;
}

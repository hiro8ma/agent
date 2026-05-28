// メトリクスレコーダー
//
// 設計方針
//   - JSONL 1 行 1 record で append-only。後段の分析ツール（jq / DuckDB / BigQuery）に流しやすい
//   - 書き込み先は LOG_DIR（無指定なら ./.metrics）。既存 workspace.ts と同じく
//     環境変数で外から差し替えられる形にし、テストで一時ディレクトリへ向けられるようにする
//   - 失敗時は throw する。observability の配信失敗を握り潰さない方針

import * as fs from "node:fs/promises";
import * as path from "node:path";
import type {
  AIProxyMetric,
  MetricsRecord,
  ProductMetric,
  SystemMetric,
} from "./types";

export type RecordInput = {
  product?: ProductMetric | undefined;
  system?: SystemMetric | undefined;
  aiProxy?: AIProxyMetric | undefined;
};

// LOG_DIR 未設定時のデフォルト出力ディレクトリ
const DEFAULT_LOG_DIR = "./.metrics";
const DEFAULT_FILE = "runs.jsonl";

export class MetricsRecorder {
  private readonly filePath: string;

  constructor(filePath?: string) {
    if (filePath) {
      this.filePath = path.resolve(filePath);
    } else {
      const dir = path.resolve(process.env.LOG_DIR ?? DEFAULT_LOG_DIR);
      this.filePath = path.join(dir, DEFAULT_FILE);
    }
  }

  // 出力先のフルパスを返す。テストと運用時のログ確認用
  get path(): string {
    return this.filePath;
  }

  async recordRun(traceId: string, input: RecordInput): Promise<void> {
    const record: MetricsRecord = {
      traceId,
      timestamp: new Date().toISOString(),
      ...(input.product !== undefined ? { product: input.product } : {}),
      ...(input.system !== undefined ? { system: input.system } : {}),
      ...(input.aiProxy !== undefined ? { aiProxy: input.aiProxy } : {}),
    };

    await fs.mkdir(path.dirname(this.filePath), { recursive: true });
    await fs.appendFile(this.filePath, JSON.stringify(record) + "\n", "utf8");
  }
}

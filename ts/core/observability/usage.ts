// トークン会計の正規化とログ出力。
//
// 設計方針
//   - provider adapter が返す core/types.ts の Usage を、OTel GenAI semantic conventions
//     寄りの属性名（gen_ai.usage.input_tokens 等）を持つ構造化ログに変換する
//   - cost 概算は pricing.ts のテーブルで算出。単価未知のモデルは cost を省略する
//   - 出力は JSONL の 1 行 1 record。stdout もしくは sink 関数に流せるようにし、
//     既存 MetricsRecorder と同じく握り潰さず append-only に保つ
//   - reasoning トークンは output の内数として課金されるため、コスト計算では output に合算する

import type { Usage } from "../types";
import { estimateCost } from "./pricing";

// OTel GenAI semantic conventions に寄せた属性名で正規化したトークン会計レコード。
// JSON キーはドット区切りの慣習名をそのまま使い、外部の OTel collector に流しやすくする。
export type UsageRecord = {
  "gen_ai.system": string;
  "gen_ai.request.model": string;
  "gen_ai.operation.name": string;
  "gen_ai.usage.input_tokens": number;
  "gen_ai.usage.output_tokens": number;
  "gen_ai.usage.cached_input_tokens": number;
  "gen_ai.usage.reasoning_tokens": number;
  "gen_ai.usage.total_tokens": number;
  // 概算コスト（USD）。単価未知なら省略
  "gen_ai.usage.cost_usd"?: number;
  "trace.id"?: string;
  timestamp: string;
};

export type BuildUsageInput = {
  provider: string;
  model: string;
  usage: Usage | undefined;
  // chat / structured など。OTel の operation.name に流す
  operation?: string;
  traceId?: string;
};

// core Usage を OTel 寄りの UsageRecord に正規化し、概算コストを付与する。
export function buildUsageRecord(input: BuildUsageInput): UsageRecord {
  const u = input.usage ?? {};
  const inputTokens = u.promptTokens ?? 0;
  const outputTokens = u.completionTokens ?? 0;
  const cachedInputTokens = u.cachedInputTokens ?? 0;
  const reasoningTokens = u.reasoningTokens ?? 0;
  const totalTokens = u.totalTokens ?? inputTokens + outputTokens;

  const cost = estimateCost(input.model, {
    inputTokens,
    // reasoning は output 単価で課金されるため合算する
    outputTokens: outputTokens,
    cachedInputTokens,
  });

  const record: UsageRecord = {
    "gen_ai.system": input.provider,
    "gen_ai.request.model": input.model,
    "gen_ai.operation.name": input.operation ?? "chat",
    "gen_ai.usage.input_tokens": inputTokens,
    "gen_ai.usage.output_tokens": outputTokens,
    "gen_ai.usage.cached_input_tokens": cachedInputTokens,
    "gen_ai.usage.reasoning_tokens": reasoningTokens,
    "gen_ai.usage.total_tokens": totalTokens,
    timestamp: new Date().toISOString(),
  };
  if (cost !== null) record["gen_ai.usage.cost_usd"] = cost;
  if (input.traceId !== undefined) record["trace.id"] = input.traceId;
  return record;
}

// UsageRecord の出力先。デフォルトは stdout への JSONL 1 行。
export type UsageSink = (record: UsageRecord) => void;

const stdoutSink: UsageSink = (record) => {
  process.stdout.write(JSON.stringify(record) + "\n");
};

// トークン会計を 1 リクエストごとに記録するロガー。
//   const logger = new UsageLogger();
//   const result = await generateText({ model, messages });
//   logger.log({ provider: "openai", model: "gpt-5-mini", usage: result.usage });
export class UsageLogger {
  private readonly sink: UsageSink;

  constructor(sink: UsageSink = stdoutSink) {
    this.sink = sink;
  }

  log(input: BuildUsageInput): UsageRecord {
    const record = buildUsageRecord(input);
    this.sink(record);
    return record;
  }
}

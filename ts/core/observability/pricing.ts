// モデル別の単価テーブル（USD / 100 万トークン）。
//
// 設計方針
//   - input / output / cachedInput の 3 系統を持つ。cachedInput は prompt キャッシュヒット分の割引単価
//   - reasoning トークンは output と同単価で課金されるため completion 側に合算する
//   - 公開単価は改定されるため、ここを単一の更新点にする。未知モデルは null コストにフォールバック
//   - 値は「100 万トークンあたり USD」。estimateCost で 1 トークンあたりに割って計算する

export type ModelPricing = {
  // USD per 1M tokens
  input: number;
  output: number;
  // prompt キャッシュヒット分の割引単価（未指定なら input と同額扱い）
  cachedInput?: number;
};

const PER_MILLION = 1_000_000;

// 代表モデルの最小テーブル。プレフィックス一致で版差を吸収する。
const PRICING_TABLE: Record<string, ModelPricing> = {
  // OpenAI
  "gpt-5-mini": { input: 0.25, output: 2.0, cachedInput: 0.025 },
  "gpt-5": { input: 1.25, output: 10.0, cachedInput: 0.125 },
  // Anthropic
  "claude-haiku-4-5": { input: 1.0, output: 5.0, cachedInput: 0.1 },
  "claude-sonnet-4": { input: 3.0, output: 15.0, cachedInput: 0.3 },
  "claude-opus-4": { input: 15.0, output: 75.0, cachedInput: 1.5 },
  // Google
  "gemini-2.5-flash": { input: 0.3, output: 2.5, cachedInput: 0.075 },
  "gemini-2.5-pro": { input: 1.25, output: 10.0, cachedInput: 0.3125 },
};

// model ID から単価を引く。完全一致 → プレフィックス最長一致の順で解決する。
export function lookupPricing(modelId: string): ModelPricing | undefined {
  const direct = PRICING_TABLE[modelId];
  if (direct) return direct;
  let best: { key: string; pricing: ModelPricing } | undefined;
  for (const [key, pricing] of Object.entries(PRICING_TABLE)) {
    if (modelId.startsWith(key)) {
      if (!best || key.length > best.key.length) best = { key, pricing };
    }
  }
  return best?.pricing;
}

export type TokenCounts = {
  inputTokens: number;
  outputTokens: number;
  cachedInputTokens: number;
};

// トークン数と単価から概算コスト（USD）を返す。単価未知なら null。
//   - cached 分は input から差し引き、割引単価で別計算する
//   - reasoning は呼び出し側で outputTokens に含めて渡す
export function estimateCost(
  modelId: string,
  counts: TokenCounts,
): number | null {
  const pricing = lookupPricing(modelId);
  if (!pricing) return null;

  const cached = Math.min(counts.cachedInputTokens, counts.inputTokens);
  const uncachedInput = counts.inputTokens - cached;
  const cachedPrice = pricing.cachedInput ?? pricing.input;

  const cost =
    (uncachedInput * pricing.input) / PER_MILLION +
    (cached * cachedPrice) / PER_MILLION +
    (counts.outputTokens * pricing.output) / PER_MILLION;

  return cost;
}

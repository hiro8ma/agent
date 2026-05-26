// 論理名（model tier）→ 具体モデル ID のマッピング層。
//
// 実運用ではモデル ID をコードに散らさず、用途を表す論理名（fast / default / smart）に
// マッピングする。モデルの世代交代やプロバイダー切り替え時に、この 1 ファイルだけ直せば済む。
//   - fast    動作確認・大量処理向けの軽量モデル
//   - default 通常利用の標準モデル
//   - smart   複雑な推論・横断的なタスク向けの高性能モデル

export const PROVIDERS = ["anthropic", "openai", "google"] as const;
export type ProviderName = (typeof PROVIDERS)[number];

export const MODEL_TIERS = ["fast", "default", "smart"] as const;
export type ModelTier = (typeof MODEL_TIERS)[number];

// プロバイダーごとの tier → モデル ID テーブル。playground/README.md の表と整合させる。
const MODEL_TABLE: Record<ProviderName, Record<ModelTier, string>> = {
  anthropic: {
    fast: "claude-haiku-4-5-20251001",
    default: "claude-sonnet-4-6",
    smart: "claude-opus-4-7",
  },
  openai: {
    fast: "gpt-5-mini",
    default: "gpt-5",
    smart: "gpt-5",
  },
  google: {
    fast: "gemini-2.5-flash",
    default: "gemini-2.5-flash",
    smart: "gemini-2.5-pro",
  },
};

export function isProviderName(value: string): value is ProviderName {
  return (PROVIDERS as readonly string[]).includes(value);
}

export function isModelTier(value: string): value is ModelTier {
  return (MODEL_TIERS as readonly string[]).includes(value);
}

export type ResolveModelOptions = {
  // 生のモデル ID 明示。指定があれば tier より優先する（後方互換）
  modelId?: string | undefined;
  // 論理名。未指定なら "default"
  tier?: ModelTier | undefined;
};

// 解決の優先順位
//   1. modelId（生 ID 明示） — 後方互換のため最優先
//   2. tier（fast / default / smart）
//   3. tier 未指定なら "default"
export function resolveModel(
  provider: ProviderName,
  options: ResolveModelOptions = {},
): string {
  const trimmedModelId = options.modelId?.trim();
  if (trimmedModelId) {
    return trimmedModelId;
  }
  const tier = options.tier ?? "default";
  return MODEL_TABLE[provider][tier];
}

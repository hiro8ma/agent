import type { LanguageModel } from "../types";
import { createAnthropic } from "./anthropic";
import { createOpenAI } from "./openai";
import { createGoogle } from "./google";
import { isModelTier, resolveModel, type ProviderName } from "./models";

// 環境変数から Provider + Model を 1 発で得るユーティリティ。
//   LLM_PROVIDER=anthropic | openai | google
//   LLM_MODEL=claude-sonnet-4-6 | gpt-5 | gemini-2.5-pro   生 ID 明示（最優先・後方互換）
//   LLM_MODEL_TIER=fast | default | smart                  論理名（LLM_MODEL 未指定時に使用）
//
// 解決順は LLM_MODEL（生 ID）> LLM_MODEL_TIER > tier=default。
// より細かく制御したい場合は createAnthropic() / createOpenAI() / createGoogle() を直接呼ぶ。
export function selectProvider(): LanguageModel {
  const provider = (process.env.LLM_PROVIDER ?? "anthropic").toLowerCase();
  const modelId = process.env.LLM_MODEL;
  const tierEnv = process.env.LLM_MODEL_TIER?.toLowerCase();
  const tier = tierEnv && isModelTier(tierEnv) ? tierEnv : undefined;

  const model = (name: ProviderName): string =>
    resolveModel(name, { modelId, tier });

  switch (provider) {
    case "anthropic":
      return createAnthropic()(model("anthropic"));
    case "openai":
      return createOpenAI()(model("openai"));
    case "google":
      return createGoogle()(model("google"));
    default:
      throw new Error(
        `Unsupported LLM_PROVIDER: ${provider}. Supported providers: anthropic, openai, google`,
      );
  }
}

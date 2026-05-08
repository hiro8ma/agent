import type { LanguageModel } from "../types";
import { createAnthropic } from "./anthropic";

// 環境変数から Provider + Model を 1 発で得るユーティリティ。
//   LLM_PROVIDER=anthropic
//   LLM_MODEL=claude-haiku-4-5-20251001
//
// より細かく制御したい場合は createAnthropic() / createOpenAI() / createGoogle() を直接呼ぶ。
export function selectProvider(): LanguageModel {
  const provider = (process.env.LLM_PROVIDER ?? "anthropic").toLowerCase();
  const model = process.env.LLM_MODEL ?? "claude-haiku-4-5-20251001";

  switch (provider) {
    case "anthropic":
      return createAnthropic()(model);
    default:
      throw new Error(
        `Unsupported LLM_PROVIDER: ${provider}. Supported providers: anthropic`,
      );
  }
}

import type { LanguageModel } from "../types";
import { createAnthropic } from "./anthropic";
import { createOpenAI } from "./openai";

// 環境変数から Provider + Model を 1 発で得るユーティリティ。
//   LLM_PROVIDER=anthropic | openai
//   LLM_MODEL=claude-haiku-4-5-20251001 | gpt-5-mini
//
// より細かく制御したい場合は createAnthropic() / createOpenAI() / createGoogle() を直接呼ぶ。
export function selectProvider(): LanguageModel {
  const provider = (process.env.LLM_PROVIDER ?? "anthropic").toLowerCase();

  switch (provider) {
    case "anthropic": {
      const model = process.env.LLM_MODEL ?? "claude-haiku-4-5-20251001";
      return createAnthropic()(model);
    }
    case "openai": {
      const model = process.env.LLM_MODEL ?? "gpt-5-mini";
      return createOpenAI()(model);
    }
    default:
      throw new Error(
        `Unsupported LLM_PROVIDER: ${provider}. Supported providers: anthropic, openai`,
      );
  }
}

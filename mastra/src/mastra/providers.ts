import { openai } from "@ai-sdk/openai";
import { google } from "@ai-sdk/google";
import type { MastraModelConfig } from "@mastra/core/llm";

// 環境変数から Provider + Model を 1 発で得るユーティリティ。
//   LLM_PROVIDER=openai | google   （未指定なら openai）
//   LLM_MODEL=gpt-4o | gemini-2.5-flash | ...   （未指定なら provider 既定）
//
// Anthropic / Vertex に差し替える場合:
//   import { anthropic } from "@ai-sdk/anthropic";          // case "anthropic": return anthropic(modelId ?? "claude-sonnet-4-5");
//   import { vertex } from "@ai-sdk/google-vertex";         // case "vertex": return vertex(modelId ?? "gemini-2.5-flash");
const DEFAULT_MODEL: Record<string, string> = {
  openai: "gpt-4o-mini",
  google: "gemini-2.5-flash",
};

export function selectModel(): MastraModelConfig {
  const provider = (process.env.LLM_PROVIDER ?? "openai").toLowerCase();
  const modelId = process.env.LLM_MODEL;

  switch (provider) {
    case "openai":
      return openai(modelId ?? DEFAULT_MODEL.openai!);
    case "google":
      return google(modelId ?? DEFAULT_MODEL.google!);
    default:
      throw new Error(
        `Unsupported LLM_PROVIDER: ${provider}. Supported providers: openai, google`,
      );
  }
}

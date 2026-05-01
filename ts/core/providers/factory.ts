import type { LanguageModel } from "../types";
import { AnthropicLanguageModel } from "./anthropic";

export function selectProvider(): LanguageModel {
  const provider = (process.env.LLM_PROVIDER ?? "anthropic").toLowerCase();
  const model = process.env.LLM_MODEL ?? "claude-sonnet-4-5-20250929";

  switch (provider) {
    case "anthropic":
      return new AnthropicLanguageModel({ model });
    default:
      throw new Error(
        `Unsupported LLM_PROVIDER: ${provider}. Supported providers: anthropic`,
      );
  }
}

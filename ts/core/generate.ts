import type { GenerateParams, GenerateTextResult, LanguageModel } from "./types";

// 統一された生成 API。
//   const anthropic = createAnthropic();
//   const haiku = anthropic("claude-haiku-4-5-20251001");
//   const result = await generateText({ model: haiku, messages });
//
// model に依存しない top-level 関数として提供することで、Vercel AI SDK 互換の
// 呼び出し感を保ちつつ、内部抽象化（LanguageModel SPI）はリポ内に閉じる。
export async function generateText(
  params: { model: LanguageModel } & GenerateParams,
): Promise<GenerateTextResult> {
  const { model, ...rest } = params;
  return model.doGenerate(rest);
}

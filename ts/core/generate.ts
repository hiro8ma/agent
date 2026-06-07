import { zodToJsonSchema } from "zod-to-json-schema";
import type { ZodType } from "zod";
import type {
  GenerateParams,
  GenerateTextResult,
  LanguageModel,
  Message,
  Usage,
} from "./types";
import { StructuredOutputError } from "./types";

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

// 構造化出力の結果。検証済みのオブジェクトに加え、トークン会計のため usage も返す。
export type StructuredResult<T> = {
  object: T;
  finishReason: GenerateTextResult["finishReason"];
  usage?: Usage;
};

export type GenerateStructuredParams<T> = {
  model: LanguageModel;
  messages: Message[];
  schema: ZodType<T>;
  schemaName?: string;
  schemaDescription?: string;
  temperature?: number;
  maxTokens?: number;
  signal?: AbortSignal;
};

// 型安全な構造化出力 API。
//   const schema = z.object({ title: z.string(), score: z.number() });
//   const { object } = await generateStructured({ model, messages, schema });
//   object.title  // string として型付け
//
// zod スキーマを zod-to-json-schema で JSON Schema に変換して provider に渡し、
// 返ってきた生 JSON を schema.safeParse() で検証する。検証・パース・refusal の
// 失敗は StructuredOutputError として型付きで throw する。
export async function generateStructured<T>(
  params: GenerateStructuredParams<T>,
): Promise<StructuredResult<T>> {
  const { model, schema, schemaName, ...rest } = params;
  const name = schemaName ?? "response";

  const jsonSchema = zodToJsonSchema(schema, {
    // structured outputs は $ref/definitions 非対応の provider があるためインライン展開する
    $refStrategy: "none",
    target: "openApi3",
  }) as Record<string, unknown>;

  const raw = await model.doGenerateStructured({
    messages: rest.messages,
    jsonSchema,
    schemaName: name,
    ...(rest.schemaDescription !== undefined
      ? { schemaDescription: rest.schemaDescription }
      : {}),
    ...(rest.temperature !== undefined ? { temperature: rest.temperature } : {}),
    ...(rest.maxTokens !== undefined ? { maxTokens: rest.maxTokens } : {}),
    ...(rest.signal !== undefined ? { signal: rest.signal } : {}),
  });

  if (raw.refusal) {
    throw new StructuredOutputError(
      "refusal",
      `model refused structured output: ${raw.refusal}`,
      raw.refusal,
    );
  }

  const result = schema.safeParse(raw.data);
  if (!result.success) {
    throw new StructuredOutputError(
      "validation",
      `structured output failed schema validation: ${result.error.message}`,
      raw.data,
    );
  }

  return {
    object: result.data,
    finishReason: raw.finishReason,
    ...(raw.usage !== undefined ? { usage: raw.usage } : {}),
  };
}

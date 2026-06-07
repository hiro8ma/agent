import { describe, expect, test } from "bun:test";
import { z } from "zod";
import { generateStructured } from "@core/generate";
import { StructuredOutputError } from "@core/types";
import type {
  GenerateParams,
  GenerateTextResult,
  LanguageModel,
  StructuredParams,
  StructuredRawResult,
} from "@core/types";

// API を叩かないフェイク model。doGenerateStructured に渡された JSON Schema と
// 返す生データをテスト側で制御し、zod 検証経路を検証する。
class FakeModel implements LanguageModel {
  public lastSchema?: Record<string, unknown>;
  constructor(private readonly raw: StructuredRawResult) {}

  async doGenerate(_params: GenerateParams): Promise<GenerateTextResult> {
    return { text: "", finishReason: "stop" };
  }

  async doGenerateStructured(
    params: StructuredParams,
  ): Promise<StructuredRawResult> {
    this.lastSchema = params.jsonSchema;
    return this.raw;
  }
}

describe("generateStructured", () => {
  const schema = z.object({
    title: z.string(),
    score: z.number().int(),
  });

  test("zod スキーマを JSON Schema に変換して provider に渡す", async () => {
    const model = new FakeModel({
      data: { title: "ok", score: 3 },
      finishReason: "stop",
    });
    await generateStructured({
      model,
      messages: [{ role: "user", content: "hi" }],
      schema,
    });
    const js = model.lastSchema as {
      type?: string;
      properties?: Record<string, unknown>;
      required?: string[];
    };
    expect(js.type).toBe("object");
    expect(js.properties).toHaveProperty("title");
    expect(js.properties).toHaveProperty("score");
    expect(js.required).toContain("title");
    expect(js.required).toContain("score");
  });

  test("生データが schema に一致すれば型付きオブジェクトを返す", async () => {
    const model = new FakeModel({
      data: { title: "hello", score: 5 },
      finishReason: "stop",
      usage: { promptTokens: 10, completionTokens: 4 },
    });
    const { object, usage } = await generateStructured({
      model,
      messages: [{ role: "user", content: "x" }],
      schema,
    });
    expect(object.title).toBe("hello");
    expect(object.score).toBe(5);
    expect(usage?.promptTokens).toBe(10);
  });

  test("schema に一致しない生データは validation エラーを throw する", async () => {
    const model = new FakeModel({
      data: { title: "hello", score: "not-a-number" },
      finishReason: "stop",
    });
    const promise = generateStructured({
      model,
      messages: [{ role: "user", content: "x" }],
      schema,
    });
    await expect(promise).rejects.toBeInstanceOf(StructuredOutputError);
    await expect(promise).rejects.toMatchObject({ reason: "validation" });
  });

  test("refusal が返ると refusal エラーを throw する", async () => {
    const model = new FakeModel({
      data: undefined,
      refusal: "cannot comply",
      finishReason: "content_filter",
    });
    const promise = generateStructured({
      model,
      messages: [{ role: "user", content: "x" }],
      schema,
    });
    await expect(promise).rejects.toMatchObject({ reason: "refusal" });
  });
});

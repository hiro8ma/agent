import { GoogleGenAI } from "@google/genai";
import type {
  GenerateParams,
  GenerateTextResult,
  LanguageModel,
  Message,
  StructuredParams,
  StructuredRawResult,
  ToolCall,
  Usage,
} from "../types";
import { LLMApiError, StructuredOutputError } from "../types";

export type GoogleConfig = {
  apiKey?: string;
};

type GoogleRole = "user" | "model";

type GooglePart =
  | { text: string }
  | { functionCall: { name: string; args: Record<string, unknown> } }
  | {
      functionResponse: {
        name: string;
        response: Record<string, unknown>;
      };
    };

type GoogleContent = {
  role: GoogleRole;
  parts: GooglePart[];
};

function mapFinishReason(
  reason: string | null | undefined,
): GenerateTextResult["finishReason"] {
  switch (reason) {
    case "STOP":
      return "stop";
    case "MAX_TOKENS":
      return "length";
    case "SAFETY":
    case "BLOCKLIST":
    case "PROHIBITED_CONTENT":
    case "SPII":
      return "content_filter";
    case "MALFORMED_FUNCTION_CALL":
      return "error";
    default:
      return "stop";
  }
}

function mapMessages(messages: Message[]): {
  systemInstruction?: { parts: { text: string }[] };
  contents: GoogleContent[];
} {
  const systemTexts: string[] = [];
  const contents: GoogleContent[] = [];

  for (const msg of messages) {
    if (msg.role === "system") {
      systemTexts.push(msg.content);
    } else if (msg.role === "user") {
      contents.push({
        role: "user",
        parts: [{ text: msg.content }],
      });
    } else if (msg.role === "assistant") {
      const parts: GooglePart[] = [];
      if (msg.content) parts.push({ text: msg.content });
      for (const call of msg.toolCalls ?? []) {
        parts.push({
          functionCall: { name: call.name, args: call.args },
        });
      }
      if (parts.length === 0) parts.push({ text: "" });
      contents.push({ role: "model", parts });
    } else if (msg.role === "tool") {
      // Gemini は tool 結果を user role の parts.functionResponse として表現する
      let response: Record<string, unknown>;
      try {
        const parsed = JSON.parse(msg.content);
        response =
          typeof parsed === "object" && parsed !== null
            ? (parsed as Record<string, unknown>)
            : { result: msg.content };
      } catch {
        response = { result: msg.content };
      }
      contents.push({
        role: "user",
        parts: [
          {
            functionResponse: { name: msg.name, response },
          },
        ],
      });
    }
  }

  return {
    ...(systemTexts.length > 0
      ? { systemInstruction: { parts: [{ text: systemTexts.join("\n") }] } }
      : {}),
    contents,
  };
}

type GoogleUsageMeta = {
  promptTokenCount?: number;
  candidatesTokenCount?: number;
  totalTokenCount?: number;
  cachedContentTokenCount?: number;
  thoughtsTokenCount?: number;
};

// Gemini usageMetadata を共通 Usage に正規化する。
// promptTokenCount は cachedContentTokenCount を内包するため promptTokens にそのまま入れ、
// cachedInputTokens に内訳を入れてコスト計算側で割引する。thoughts は reasoning に対応。
function mapUsage(meta: GoogleUsageMeta | undefined): Usage {
  return {
    promptTokens: meta?.promptTokenCount,
    completionTokens: meta?.candidatesTokenCount,
    totalTokens: meta?.totalTokenCount,
    cachedInputTokens: meta?.cachedContentTokenCount,
    reasoningTokens: meta?.thoughtsTokenCount,
  };
}

function toLLMApiError(error: unknown): unknown {
  if (error instanceof LLMApiError) return error;
  const e = error as { status?: number; message?: string; code?: string };
  return new LLMApiError(
    e.status ?? 500,
    "google",
    e.code,
    e.message ?? String(error),
    error,
  );
}

class GoogleLanguageModel implements LanguageModel {
  constructor(
    private readonly client: GoogleGenAI,
    private readonly model: string,
  ) {}

  async doGenerate(params: GenerateParams): Promise<GenerateTextResult> {
    const { systemInstruction, contents } = mapMessages(params.messages);

    const tools =
      params.tools && params.tools.length > 0
        ? [
            {
              functionDeclarations: params.tools.map((tool) => ({
                name: tool.name,
                description: tool.description,
                parameters: tool.parameters,
              })),
            },
          ]
        : undefined;

    const generationConfig: Record<string, unknown> = {};
    if (params.temperature !== undefined) {
      generationConfig.temperature = params.temperature;
    }
    if (params.maxTokens !== undefined) {
      generationConfig.maxOutputTokens = params.maxTokens;
    }

    try {
      const response = await this.client.models.generateContent({
        model: this.model,
        contents,
        config: {
          ...(systemInstruction ? { systemInstruction } : {}),
          ...(tools ? { tools } : {}),
          ...(Object.keys(generationConfig).length > 0
            ? generationConfig
            : {}),
          ...(params.signal ? { abortSignal: params.signal } : {}),
        },
      });

      const candidate = response.candidates?.[0];
      if (!candidate) {
        throw new LLMApiError(
          500,
          "google",
          "no_candidate",
          "Google response had no candidates",
          response,
        );
      }

      let text = "";
      const toolCalls: ToolCall[] = [];
      const parts = (candidate.content?.parts ?? []) as GooglePart[];
      for (const part of parts) {
        if ("text" in part && typeof part.text === "string") {
          text += part.text;
        } else if ("functionCall" in part) {
          toolCalls.push({
            // Gemini は tool_call_id を発行しないため、name + 連番を ID に流用
            toolCallId: `${part.functionCall.name}-${toolCalls.length}`,
            name: part.functionCall.name,
            args: part.functionCall.args ?? {},
          });
        }
      }

      const result: GenerateTextResult = {
        text,
        finishReason: mapFinishReason(candidate.finishReason),
        usage: mapUsage(response.usageMetadata),
      };
      if (toolCalls.length > 0) result.toolCalls = toolCalls;
      return result;
    } catch (error) {
      throw toLLMApiError(error);
    }
  }

  // Gemini は responseMimeType + responseSchema で JSON 出力を強制できる。
  // 返ってきた JSON 文字列を parse して生データを返す。zod 検証は上位で行う。
  async doGenerateStructured(
    params: StructuredParams,
  ): Promise<StructuredRawResult> {
    const { systemInstruction, contents } = mapMessages(params.messages);

    const generationConfig: Record<string, unknown> = {
      responseMimeType: "application/json",
      responseJsonSchema: params.jsonSchema,
    };
    if (params.temperature !== undefined) {
      generationConfig.temperature = params.temperature;
    }
    if (params.maxTokens !== undefined) {
      generationConfig.maxOutputTokens = params.maxTokens;
    }

    try {
      const response = await this.client.models.generateContent({
        model: this.model,
        contents,
        config: {
          ...(systemInstruction ? { systemInstruction } : {}),
          ...generationConfig,
          ...(params.signal ? { abortSignal: params.signal } : {}),
        },
      });

      const candidate = response.candidates?.[0];
      if (!candidate) {
        throw new LLMApiError(
          500,
          "google",
          "no_candidate",
          "Google response had no candidates",
          response,
        );
      }

      let text = "";
      const parts = (candidate.content?.parts ?? []) as GooglePart[];
      for (const part of parts) {
        if ("text" in part && typeof part.text === "string") text += part.text;
      }

      let data: unknown;
      try {
        data = JSON.parse(text);
      } catch {
        throw new StructuredOutputError(
          "parse",
          `Google structured output was not valid JSON: ${text.slice(0, 200)}`,
          text,
        );
      }

      return {
        data,
        finishReason: mapFinishReason(candidate.finishReason),
        usage: mapUsage(response.usageMetadata),
      };
    } catch (error) {
      if (error instanceof StructuredOutputError) throw error;
      throw toLLMApiError(error);
    }
  }
}

// Factory パターン。createGoogle() は config を保持したクロージャを返し、
// model 名を渡すと LanguageModel が得られる。
//
//   const google = createGoogle();
//   const flash = google("gemini-2.5-flash");
//   const pro = google("gemini-2.5-pro");
export function createGoogle(config: GoogleConfig = {}) {
  const apiKey = config.apiKey ?? process.env.GOOGLE_API_KEY;
  if (!apiKey) {
    throw new Error("GOOGLE_API_KEY is not set");
  }
  const client = new GoogleGenAI({ apiKey });
  return (modelId: string): LanguageModel =>
    new GoogleLanguageModel(client, modelId);
}

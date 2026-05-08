import { GoogleGenAI } from "@google/genai";
import type {
  GenerateParams,
  GenerateTextResult,
  LanguageModel,
  Message,
  ToolCall,
  Usage,
} from "../types";
import { LLMApiError } from "../types";

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
    systemInstruction:
      systemTexts.length > 0
        ? { parts: [{ text: systemTexts.join("\n") }] }
        : undefined,
    contents,
  };
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

      const usageMeta = response.usageMetadata;
      const usage: Usage = {
        promptTokens: usageMeta?.promptTokenCount,
        completionTokens: usageMeta?.candidatesTokenCount,
        totalTokens: usageMeta?.totalTokenCount,
      };

      const result: GenerateTextResult = {
        text,
        finishReason: mapFinishReason(candidate.finishReason),
        usage,
      };
      if (toolCalls.length > 0) result.toolCalls = toolCalls;
      return result;
    } catch (error) {
      if (error instanceof LLMApiError) throw error;
      const e = error as { status?: number; message?: string; code?: string };
      throw new LLMApiError(
        e.status ?? 500,
        "google",
        e.code,
        e.message ?? String(error),
        error,
      );
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

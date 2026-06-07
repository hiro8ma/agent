import Anthropic from "@anthropic-ai/sdk";
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

export type AnthropicConfig = {
  apiKey?: string;
  baseURL?: string;
  maxRetries?: number;
};

type NonSystemMessage = Exclude<Message, { role: "system" }>;

function mapFinishReason(
  stopReason: string | null | undefined,
): GenerateTextResult["finishReason"] {
  switch (stopReason) {
    case "end_turn":
    case "stop_sequence":
      return "stop";
    case "tool_use":
      return "tool_calls";
    case "max_tokens":
      return "length";
    default:
      return "stop";
  }
}

function mapMessages(messages: NonSystemMessage[]): Anthropic.MessageParam[] {
  return messages.map((msg): Anthropic.MessageParam => {
    if (msg.role === "assistant") {
      const blocks: Anthropic.ContentBlockParam[] = [];
      if (msg.content) {
        blocks.push({ type: "text", text: msg.content });
      }
      if (msg.toolCalls) {
        for (const call of msg.toolCalls) {
          blocks.push({
            type: "tool_use",
            id: call.toolCallId,
            name: call.name,
            input: call.args,
          });
        }
      }
      return { role: "assistant", content: blocks };
    }
    if (msg.role === "tool") {
      return {
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: msg.toolCallId,
            content: msg.content,
          },
        ],
      };
    }
    return { role: "user", content: msg.content };
  });
}

// Anthropic usage を共通 Usage に正規化する。
// Anthropic は cache_read_input_tokens を input_tokens と別建てで返すため、
// promptTokens は両者の合算（実際にモデルが読んだ prompt 量）として扱い、
// cachedInputTokens にキャッシュヒット分を入れてコスト計算側で割引する。
function mapUsage(usage: Anthropic.Usage | undefined): Usage {
  if (!usage) return {};
  const cacheRead = usage.cache_read_input_tokens ?? 0;
  const cacheCreate = usage.cache_creation_input_tokens ?? 0;
  const promptTokens =
    usage.input_tokens + cacheRead + cacheCreate;
  return {
    promptTokens,
    completionTokens: usage.output_tokens,
    totalTokens: promptTokens + usage.output_tokens,
    cachedInputTokens: cacheRead,
  };
}

function toLLMApiError(error: unknown): unknown {
  if (error instanceof Anthropic.APIError) {
    return new LLMApiError(
      error.status ?? 500,
      "anthropic",
      undefined,
      error.message,
      error.error,
    );
  }
  return error;
}

class AnthropicLanguageModel implements LanguageModel {
  constructor(
    private readonly client: Anthropic,
    private readonly model: string,
  ) {}

  async doGenerate(params: GenerateParams): Promise<GenerateTextResult> {
    const systemMessages = params.messages.filter((m) => m.role === "system");
    const rest = params.messages.filter(
      (m): m is NonSystemMessage => m.role !== "system",
    );

    const system =
      systemMessages.length > 0
        ? systemMessages.map((m) => ({
            type: "text" as const,
            text: m.content,
          }))
        : undefined;

    const tools =
      params.tools && params.tools.length > 0
        ? params.tools.map((tool) => ({
            name: tool.name,
            description: tool.description,
            input_schema: tool.parameters as Anthropic.Tool.InputSchema,
          }))
        : undefined;

    try {
      const response = await this.client.messages.create(
        {
          model: this.model,
          max_tokens: params.maxTokens ?? 4096,
          ...(system ? { system } : {}),
          messages: mapMessages(rest),
          ...(params.temperature !== undefined
            ? { temperature: params.temperature }
            : {}),
          ...(tools ? { tools } : {}),
        },
        params.signal ? { signal: params.signal } : undefined,
      );

      let text = "";
      const toolCalls: ToolCall[] = [];
      for (const block of response.content) {
        if (block.type === "text") {
          text += block.text;
        } else if (block.type === "tool_use") {
          toolCalls.push({
            toolCallId: block.id,
            name: block.name,
            args: (block.input ?? {}) as Record<string, unknown>,
          });
        }
      }

      const result: GenerateTextResult = {
        text,
        finishReason: mapFinishReason(response.stop_reason),
        usage: mapUsage(response.usage),
      };
      if (toolCalls.length > 0) result.toolCalls = toolCalls;
      return result;
    } catch (error) {
      throw toLLMApiError(error);
    }
  }

  // Anthropic にネイティブの JSON Schema 強制は無いため、forced tool-use で構造化する。
  // 単一の "respond" tool を定義し tool_choice で強制、tool_use.input を生データとして返す。
  async doGenerateStructured(
    params: StructuredParams,
  ): Promise<StructuredRawResult> {
    const toolName = params.schemaName ?? "respond";
    const systemMessages = params.messages.filter((m) => m.role === "system");
    const rest = params.messages.filter(
      (m): m is NonSystemMessage => m.role !== "system",
    );
    const system =
      systemMessages.length > 0
        ? systemMessages.map((m) => ({ type: "text" as const, text: m.content }))
        : undefined;

    try {
      const response = await this.client.messages.create(
        {
          model: this.model,
          max_tokens: params.maxTokens ?? 4096,
          ...(system ? { system } : {}),
          messages: mapMessages(rest),
          ...(params.temperature !== undefined
            ? { temperature: params.temperature }
            : {}),
          tools: [
            {
              name: toolName,
              description:
                params.schemaDescription ??
                "Return the structured response using this schema.",
              input_schema: params.jsonSchema as Anthropic.Tool.InputSchema,
            },
          ],
          tool_choice: { type: "tool", name: toolName },
        },
        params.signal ? { signal: params.signal } : undefined,
      );

      for (const block of response.content) {
        if (block.type === "tool_use" && block.name === toolName) {
          return {
            data: block.input,
            finishReason: mapFinishReason(response.stop_reason),
            usage: mapUsage(response.usage),
          };
        }
      }

      throw new StructuredOutputError(
        "parse",
        "Anthropic structured output returned no tool_use block",
        response.content,
      );
    } catch (error) {
      if (error instanceof StructuredOutputError) throw error;
      throw toLLMApiError(error);
    }
  }
}

// Factory パターン。createAnthropic() は config を保持したクロージャを返し、
// model 名を渡すと LanguageModel が得られる。
//
//   const anthropic = createAnthropic();
//   const haiku = anthropic("claude-haiku-4-5-20251001");
//   const sonnet = anthropic("claude-sonnet-4-6");
export function createAnthropic(config: AnthropicConfig = {}) {
  const apiKey = config.apiKey ?? process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    throw new Error("ANTHROPIC_API_KEY is not set");
  }
  const client = new Anthropic({
    apiKey,
    ...(config.baseURL ? { baseURL: config.baseURL } : {}),
    maxRetries: config.maxRetries ?? 0,
  });
  return (modelId: string): LanguageModel =>
    new AnthropicLanguageModel(client, modelId);
}

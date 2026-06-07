import OpenAI from "openai";
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

export type OpenAIConfig = {
  apiKey?: string;
  baseURL?: string;
  organization?: string;
  maxRetries?: number;
};

function mapFinishReason(
  reason: string | null | undefined,
): GenerateTextResult["finishReason"] {
  switch (reason) {
    case "stop":
      return "stop";
    case "length":
      return "length";
    case "tool_calls":
    case "function_call":
      return "tool_calls";
    case "content_filter":
      return "content_filter";
    default:
      return "stop";
  }
}

function mapMessages(
  messages: Message[],
): OpenAI.Chat.Completions.ChatCompletionMessageParam[] {
  const out: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [];
  for (const msg of messages) {
    if (msg.role === "system") {
      out.push({ role: "system", content: msg.content });
    } else if (msg.role === "user") {
      out.push({ role: "user", content: msg.content });
    } else if (msg.role === "assistant") {
      const tool_calls = msg.toolCalls?.map((c) => ({
        id: c.toolCallId,
        type: "function" as const,
        function: {
          name: c.name,
          arguments: JSON.stringify(c.args),
        },
      }));
      out.push({
        role: "assistant",
        content: msg.content || null,
        ...(tool_calls && tool_calls.length > 0 ? { tool_calls } : {}),
      });
    } else if (msg.role === "tool") {
      out.push({
        role: "tool",
        tool_call_id: msg.toolCallId,
        content: msg.content,
      });
    }
  }
  return out;
}

// OpenAI usage を共通 Usage に正規化する。
// cached_tokens（prompt の内数）と reasoning_tokens（completion の内数）も拾う。
function mapUsage(
  usage: OpenAI.Completions.CompletionUsage | undefined,
): Usage {
  return {
    promptTokens: usage?.prompt_tokens,
    completionTokens: usage?.completion_tokens,
    totalTokens: usage?.total_tokens,
    cachedInputTokens: usage?.prompt_tokens_details?.cached_tokens,
    reasoningTokens: usage?.completion_tokens_details?.reasoning_tokens,
  };
}

function toLLMApiError(error: unknown): unknown {
  if (error instanceof OpenAI.APIError) {
    return new LLMApiError(
      error.status ?? 500,
      "openai",
      error.code ?? undefined,
      error.message,
      error,
    );
  }
  return error;
}

class OpenAILanguageModel implements LanguageModel {
  constructor(
    private readonly client: OpenAI,
    private readonly model: string,
  ) {}

  async doGenerate(params: GenerateParams): Promise<GenerateTextResult> {
    const tools =
      params.tools && params.tools.length > 0
        ? params.tools.map(
            (tool): OpenAI.Chat.Completions.ChatCompletionTool => ({
              type: "function",
              function: {
                name: tool.name,
                description: tool.description,
                parameters: tool.parameters,
              },
            }),
          )
        : undefined;

    try {
      const response = await this.client.chat.completions.create(
        {
          model: this.model,
          messages: mapMessages(params.messages),
          ...(params.temperature !== undefined
            ? { temperature: params.temperature }
            : {}),
          ...(params.maxTokens !== undefined
            ? { max_completion_tokens: params.maxTokens }
            : {}),
          ...(tools ? { tools } : {}),
        },
        params.signal ? { signal: params.signal } : undefined,
      );

      const choice = response.choices[0];
      if (!choice) {
        throw new LLMApiError(
          500,
          "openai",
          "no_choice",
          "OpenAI response had no choices",
          response,
        );
      }

      const text = choice.message.content ?? "";
      const toolCalls: ToolCall[] = [];
      for (const c of choice.message.tool_calls ?? []) {
        if (c.type !== "function") continue;
        let args: Record<string, unknown> = {};
        try {
          args = c.function.arguments
            ? (JSON.parse(c.function.arguments) as Record<string, unknown>)
            : {};
        } catch {
          // invalid JSON from model, surface as empty args
        }
        toolCalls.push({
          toolCallId: c.id,
          name: c.function.name,
          args,
        });
      }

      const result: GenerateTextResult = {
        text,
        finishReason: mapFinishReason(choice.finish_reason),
        usage: mapUsage(response.usage),
      };
      if (toolCalls.length > 0) result.toolCalls = toolCalls;
      return result;
    } catch (error) {
      throw toLLMApiError(error);
    }
  }

  // structured outputs（response_format: json_schema, strict）を使う。
  // strict 検証に通った JSON 文字列を parse して生データを返す。zod 検証は上位で行う。
  async doGenerateStructured(
    params: StructuredParams,
  ): Promise<StructuredRawResult> {
    try {
      const response = await this.client.chat.completions.create(
        {
          model: this.model,
          messages: mapMessages(params.messages),
          ...(params.temperature !== undefined
            ? { temperature: params.temperature }
            : {}),
          ...(params.maxTokens !== undefined
            ? { max_completion_tokens: params.maxTokens }
            : {}),
          response_format: {
            type: "json_schema",
            json_schema: {
              name: params.schemaName ?? "response",
              ...(params.schemaDescription !== undefined
                ? { description: params.schemaDescription }
                : {}),
              schema: params.jsonSchema,
              strict: true,
            },
          },
        },
        params.signal ? { signal: params.signal } : undefined,
      );

      const choice = response.choices[0];
      if (!choice) {
        throw new LLMApiError(
          500,
          "openai",
          "no_choice",
          "OpenAI response had no choices",
          response,
        );
      }

      const message = choice.message as typeof choice.message & {
        refusal?: string | null;
      };
      if (message.refusal) {
        return {
          data: undefined,
          refusal: message.refusal,
          finishReason: "content_filter",
          usage: mapUsage(response.usage),
        };
      }

      const content = message.content ?? "";
      let data: unknown;
      try {
        data = JSON.parse(content);
      } catch {
        throw new StructuredOutputError(
          "parse",
          `OpenAI structured output was not valid JSON: ${content.slice(0, 200)}`,
          content,
        );
      }

      return {
        data,
        finishReason: mapFinishReason(choice.finish_reason),
        usage: mapUsage(response.usage),
      };
    } catch (error) {
      if (error instanceof StructuredOutputError) throw error;
      throw toLLMApiError(error);
    }
  }
}

// Factory パターン。createOpenAI() は config を保持したクロージャを返し、
// model 名を渡すと LanguageModel が得られる。
//
//   const openai = createOpenAI();
//   const mini = openai("gpt-5-mini");
//   const full = openai("gpt-5");
export function createOpenAI(config: OpenAIConfig = {}) {
  const apiKey = config.apiKey ?? process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error("OPENAI_API_KEY is not set");
  }
  const client = new OpenAI({
    apiKey,
    ...(config.baseURL ? { baseURL: config.baseURL } : {}),
    ...(config.organization ? { organization: config.organization } : {}),
    maxRetries: config.maxRetries ?? 0,
  });
  return (modelId: string): LanguageModel =>
    new OpenAILanguageModel(client, modelId);
}

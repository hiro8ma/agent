import Anthropic from "@anthropic-ai/sdk";
import { zodToJsonSchema } from "zod-to-json-schema";
import type {
  GenerateInput,
  GenerateResult,
  LanguageModel,
  Message,
  ToolCall,
  FinishReason,
} from "../types";
import { LLMApiError } from "../types";

export type AnthropicConfig = {
  apiKey?: string;
  baseURL?: string;
  model: string;
  maxRetries?: number;
};

type NonSystemMessage = Exclude<Message, { role: "system" }>;

function mapFinishReason(
  stopReason: string | null | undefined,
): FinishReason {
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
            id: call.id,
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

export class AnthropicLanguageModel implements LanguageModel {
  private readonly client: Anthropic;
  private readonly model: string;

  constructor(config: AnthropicConfig) {
    const apiKey = config.apiKey ?? process.env.ANTHROPIC_API_KEY;
    if (!apiKey) {
      throw new Error("ANTHROPIC_API_KEY is not set");
    }
    this.client = new Anthropic({
      apiKey,
      ...(config.baseURL ? { baseURL: config.baseURL } : {}),
      maxRetries: config.maxRetries ?? 0,
    });
    this.model = config.model;
  }

  async doGenerate(input: GenerateInput): Promise<GenerateResult> {
    const systemMessages = input.messages.filter((m) => m.role === "system");
    const rest = input.messages.filter(
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
      input.tools && input.tools.length > 0
        ? input.tools.map((tool) => ({
            name: tool.name,
            description: tool.description,
            input_schema: zodToJsonSchema(tool.parameters, {
              target: "openApi3",
              $refStrategy: "none",
            }) as Anthropic.Tool.InputSchema,
          }))
        : undefined;

    try {
      const response = await this.client.messages.create(
        {
          model: this.model,
          max_tokens: input.maxTokens ?? 4096,
          ...(system ? { system } : {}),
          messages: mapMessages(rest),
          ...(input.temperature !== undefined
            ? { temperature: input.temperature }
            : {}),
          ...(tools ? { tools } : {}),
        },
        input.signal ? { signal: input.signal } : undefined,
      );

      let text = "";
      const toolCalls: ToolCall[] = [];
      for (const block of response.content) {
        if (block.type === "text") {
          text += block.text;
        } else if (block.type === "tool_use") {
          toolCalls.push({
            id: block.id,
            name: block.name,
            args: (block.input ?? {}) as Record<string, unknown>,
          });
        }
      }

      const result: GenerateResult = {
        finishReason: mapFinishReason(response.stop_reason),
      };
      if (text) result.text = text;
      if (toolCalls.length > 0) result.toolCalls = toolCalls;
      return result;
    } catch (error) {
      if (error instanceof Anthropic.APIError) {
        throw new LLMApiError({
          status: error.status ?? 500,
          provider: "anthropic",
          message: error.message,
          retryable:
            (error.status ?? 500) === 429 || (error.status ?? 500) >= 500,
          raw: error.error,
        });
      }
      throw error;
    }
  }
}

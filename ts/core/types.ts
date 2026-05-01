import type { ZodTypeAny, infer as zInfer } from "zod";

export type Role = "system" | "user" | "assistant" | "tool";

export type ToolCall = {
  id: string;
  name: string;
  args: Record<string, unknown>;
};

export type Message =
  | { role: "system"; content: string }
  | { role: "user"; content: string }
  | { role: "assistant"; content: string; toolCalls?: ToolCall[] }
  | { role: "tool"; content: string; toolCallId: string; name: string };

export type FinishReason = "stop" | "tool_calls" | "length" | "error";

export type GenerateResult = {
  text?: string;
  toolCalls?: ToolCall[];
  finishReason: FinishReason;
};

export type GenerateInput = {
  messages: Message[];
  tools?: AnyTool[];
  temperature?: number;
  maxTokens?: number;
  signal?: AbortSignal;
};

export interface LanguageModel {
  doGenerate(input: GenerateInput): Promise<GenerateResult>;
}

export type Tool<S extends ZodTypeAny = ZodTypeAny> = {
  name: string;
  description: string;
  parameters: S;
  execute: (args: zInfer<S>) => Promise<string>;
  needsApproval?: boolean;
};

// 各 Tool は固有の zod schema を持つので、配列に詰めると generic 不変で型が合わなくなる。
// Agent / Provider 側はスキーマの型情報を必要としないので、untyped な口を別に用意する。
export type AnyTool = Tool<ZodTypeAny>;

export class LLMApiError extends Error {
  public readonly status: number;
  public readonly code: string | undefined;
  public readonly retryable: boolean;
  public readonly provider: string;
  public readonly raw: unknown;

  constructor(params: {
    status: number;
    provider: string;
    message: string;
    code?: string;
    retryable?: boolean;
    raw?: unknown;
  }) {
    super(params.message);
    this.name = "LLMApiError";
    this.status = params.status;
    this.provider = params.provider;
    this.code = params.code;
    this.retryable =
      params.retryable ?? (params.status === 429 || params.status >= 500);
    this.raw = params.raw;
  }
}

// nano-code-core 互換の型定義。
//
// 設計方針
//   - 型は「会話の構造」だけを固定する。プロバイダー固有の表現はここに持ち込まない
//   - プロバイダー差（OpenAI / Anthropic / Google の messages 形式 / tool spec / finish reason）は
//     core/providers/* の adapter で吸収し、ここの型は安定維持する
//   - 型チェックは実行時（execute 側で必要に応じて）。コンパイル時の細かい args 型推論は持たない

// ツール定義の型
// parameters は JSON Schema 相当の生データ。zod 等の特定スキーマライブラリには依存しない
export type Tool = {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  execute: (args: Record<string, unknown>) => Promise<string>;
  needsApproval?: boolean;
};

// ツール呼び出しの型（LLM からの応答）
export type ToolCall = {
  toolCallId: string;
  name: string;
  args: Record<string, unknown>;
};

// ツール実行結果の型（会話履歴に追加する）
export type ToolResult = {
  toolCallId: string;
  result: string;
};

// 会話の最小単位
export type Message =
  | { role: "user" | "system"; content: string }
  | { role: "assistant"; content: string; toolCalls?: ToolCall[] }
  | { role: "tool"; toolCallId: string; name: string; content: string };

// トークン使用量のメタデータ（プロバイダーによって欠損する場合がある）
// 各フィールドは省略可かつ明示 undefined 可。プロバイダー adapter が値の有無を
// そのまま渡せるよう exactOptionalPropertyTypes 下でも undefined を許容する。
export type Usage = {
  promptTokens?: number | undefined;
  completionTokens?: number | undefined;
  totalTokens?: number | undefined;
};

// 統一された出力形式
export type GenerateTextResult = {
  text: string;
  finishReason:
    | "stop"
    | "length"
    | "content_filter"
    | "tool_calls"
    | "error";
  toolCalls?: ToolCall[];
  usage?: Usage;
};

// generateText に渡すパラメータ
export type GenerateParams = {
  messages: Message[];
  tools?: Tool[];
  temperature?: number;
  maxTokens?: number;
  signal?: AbortSignal;
};

// 言語モデルのインターフェース
export interface LanguageModel {
  doGenerate(params: GenerateParams): Promise<GenerateTextResult>;
}

// プロバイダー関数の型（factory パターン用）
//   const openai = createOpenAI();
//   const model: LanguageModel = openai("gpt-5-mini");
export type Provider = (modelId: string) => LanguageModel;

// LLM API エラーの統一型
export class LLMApiError extends Error {
  constructor(
    public status: number,
    public provider: string,
    public code?: string,
    message?: string,
    public raw?: unknown,
  ) {
    super(message || `LLM API Error: ${provider} returned ${status}`);
    this.name = "LLMApiError";
  }

  get retryable(): boolean {
    return this.status === 429 || this.status >= 500;
  }
}

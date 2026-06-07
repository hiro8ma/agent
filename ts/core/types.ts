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
//
// cachedInputTokens は promptTokens の内数（prompt のうちキャッシュヒットした分）。
// reasoningTokens は completionTokens の内数（推論モデルの思考トークン）。
// コスト計算側は cachedInputTokens を割引単価で再計算するため内数として扱う。
export type Usage = {
  promptTokens?: number | undefined;
  completionTokens?: number | undefined;
  totalTokens?: number | undefined;
  cachedInputTokens?: number | undefined;
  reasoningTokens?: number | undefined;
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

// 構造化出力に渡すパラメータ。
// schemaName は OpenAI structured outputs / Anthropic tool-use 等で要求される
// スキーマ名に流用する。省略時は各 adapter がデフォルト名を補う。
export type StructuredParams = {
  messages: Message[];
  jsonSchema: Record<string, unknown>;
  schemaName?: string;
  schemaDescription?: string;
  temperature?: number;
  maxTokens?: number;
  signal?: AbortSignal;
};

// 構造化出力の生データ。parse は呼び出し側（generateStructured）で zod により行う。
// data はスキーマ未検証の素の JSON 値。refusal はモデルが応答を拒否した場合に入る。
export type StructuredRawResult = {
  data: unknown;
  refusal?: string | undefined;
  finishReason: GenerateTextResult["finishReason"];
  usage?: Usage;
};

// 言語モデルのインターフェース
export interface LanguageModel {
  doGenerate(params: GenerateParams): Promise<GenerateTextResult>;
  // JSON Schema を渡して構造化出力の生データを得る。zod 検証は呼び出し側で行う。
  doGenerateStructured(params: StructuredParams): Promise<StructuredRawResult>;
}

// 構造化出力の失敗を表す型付きエラー。
//   - "refusal"     モデルが安全性等で応答を拒否した
//   - "parse"       JSON としてパースできなかった
//   - "validation"  JSON だが zod スキーマに一致しなかった
export class StructuredOutputError extends Error {
  constructor(
    public reason: "refusal" | "parse" | "validation",
    message: string,
    public raw?: unknown,
  ) {
    super(message);
    this.name = "StructuredOutputError";
  }
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

---
title: "ConnectRPC server-streaming × Genkit Go Flow — Vertex AI chunk をブラウザまで素通しする"
date: "2026-04-21"
tags: [connectrpc, genkit, streaming, vertex-ai, server-streaming]
---

# ConnectRPC server-streaming × Genkit Go Flow

LLM の部分生成チャンクをフロントエンドまでリアルタイムに流すための、Vertex AI → Genkit Flow → ConnectRPC → ブラウザの passthrough 設計パターン

## 全体像

```
ブラウザ（Connect Web）
  ↑ stream AskResponse（ConnectRPC server-streaming）
BFF / Agent Service（Go + Genkit）
  ↑ chunk（DefineStreamingFlow）
Genkit Flow
  ↑ chunk（generateContentStream）
Vertex AI（Gemini）
```

どの層も「バッファしないで順次転送する」ことを徹底する。中間でバッファすると体感レイテンシが Vertex AI の生成完了時間に戻ってしまう

## 各層の実装ポイント

### Vertex AI 呼び出し層

- 非ストリーミング版 `generateContent` ではなく `generateContentStream` を使う
- chunk 1 つに `{answer_delta, finish_reason?, tool_calls?, usage?}` を載せる

### Genkit Flow 層

- `DefineFlow` ではなく `DefineStreamingFlow` を使う
- Vertex AI から受けた chunk をそのまま Flow の出力チャネルに流す
- Flow 内でバッファして完了時に一括返却すると、せっかくの streaming が無効化される

### ConnectRPC 層

- proto 側で `returns (stream AskResponse);`（server-streaming RPC）
- Handler 実装で Genkit Flow の chunk を 1 つ受け取るたびに `stream.Send(resp)` する
- HTTP/1.1 で動く Connect プロトコルを選べば、ブラウザから直接叩ける（gRPC-Web プロキシ不要）

### ブラウザ層

- `@connectrpc/connect-web` の `AsyncIterable<AskResponse>` を `for await` で回す
- 受信ごとに UI に delta を追記（ChatGPT 風タイプライター表示）

## 設計上の落とし穴

### 1. 後から streaming に切り替えるのは proto breaking change

`rpc Ask(...) returns (AskResponse);` を後から `stream AskResponse` に変えるのは互換性のない変更。最初から streaming にしておくと後悔しない

マイルストーン 1 を「とりあえず非 streaming でリリース、次で streaming 化」と計画しがちだが、proto を 2 回切るコストのほうが大きい。最初の実装コストは streaming でも大差ない（Genkit の `DefineStreamingFlow` と Connect の streaming handler は boilerplate 量が同等）

### 2. Tool 実行ループとの整合

Tool 呼び出しは Flow 内で行い、Tool の結果は LLM の次の入力に詰め直してから次の chunk 生成を始める。Tool 実行中はクライアントに「tool_call.started」「tool_call.finished」を chunk として流すと UX が良い

chunk スキーマ例

```proto
message AskResponse {
  oneof payload {
    string answer_delta = 1;     // LLM の部分出力
    ToolCallEvent tool_event = 2; // Tool 実行の開始・完了
    FinishEvent finish = 3;       // 終了（finish_reason, token usage）
  }
}
```

### 3. キャンセル伝播

ブラウザがタブを閉じたら Context cancel → Genkit Flow cancel → Vertex AI stream cancel まで伝わるように、各層で `ctx` を正しく引き回す。Genkit の streaming flow は ctx cancel で iterator を閉じる

### 4. Prompt Caching との両立

Vertex AI の Prompt Caching は「プレフィックスが同一」の場合にのみヒットする。streaming でも非 streaming でも判定は同じ。キャッシュ対象は固定システムプロンプト + Tool 定義のみにして、会話履歴 sliding window や現行ユーザーメッセージはキャッシュ対象外として扱う

## 参考

- Genkit Go `DefineStreamingFlow`: 生成 chunk と最終結果を分けて扱う signature
- ConnectRPC server-streaming: HTTP/1.1 上で `Content-Type: application/connect+json` の chunked body として実装される
- Vertex AI `generateContentStream`: SSE 風の chunk を順次返す

import { Agent } from "@mastra/core/agent";
import { selectModel } from "../providers";

// instructions のみのシンプルなアシスタント。
// ツール / メモリ / ワークフローは後続トラックで追加する。
export const assistant = new Agent({
  id: "assistant",
  name: "assistant",
  instructions:
    "あなたは簡潔で正確なアシスタントです。質問には根拠を添えて日本語で答えます。わからないことは推測せず、わからないと伝えます。",
  model: selectModel(),
});

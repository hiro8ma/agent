import { Agent } from "@mastra/core/agent";
import { selectModel } from "../providers";
import {
  confluenceSearchPagesTool,
  confluenceGetPageTool,
} from "../tools/confluenceTool";

// instructions + Confluence ツールを持つアシスタント。
// Memory / Workflow は後続トラックで追加する。
export const assistant = new Agent({
  id: "assistant",
  name: "assistant",
  instructions:
    "あなたは簡潔で正確なアシスタントです。質問には根拠を添えて日本語で答えます。Confluence の情報が必要なときは confluence-search-pages で CQL 検索し、得た ID を confluence-get-page に渡して本文を取得します。わからないことは推測せず、わからないと伝えます。",
  model: selectModel(),
  tools: {
    confluenceSearchPagesTool,
    confluenceGetPageTool,
  },
});

// LLM-as-judge による応答品質スコアリング
//
// 設計方針
//   - 評価軸は「忠実性 / 関連性 / 簡潔性 + 総合」の 4 つを 1-5 で出させる
//   - provider は呼び出し側から LanguageModel を注入。未指定なら selectProvider() で
//     既存の env ベース解決にフォールバックし、judge 用に別 model を選べる余地を残す
//   - 返却は JSON 強制。zod で parse して型を保証し、判定崩れ（フォーマット崩れ）は throw
//   - judge 自体が幻覚を起こすため、rationale を必ず併記させて後で人間がレビュー可能にする

import { z } from "zod";
import { generateText } from "../generate";
import { selectProvider } from "../providers/factory";
import type { LanguageModel, Message } from "../types";
import type { JudgeScore } from "./types";

export type JudgeInput = {
  question: string;
  answer: string;
  context?: string | undefined;
  provider?: LanguageModel | undefined;
};

// judge プロンプト本文
// 評価軸の定義をプロンプト内に閉じ込め、呼び出し側で組み立て直さなくて済むようにする
export const JUDGE_PROMPT = `あなたは AI 応答の品質評価者です。以下の質問と回答を 3 軸で評価し、JSON のみを出力してください。

評価軸（各 1-5 の整数。5 が最良）
  - faithfulness  与えられた context に忠実か（context が無い場合は一般常識との整合で判断）
  - relevance     質問にどれだけ答えているか
  - conciseness   冗長でないか（重複・前置きの過多を減点）
  - overall       総合評価。上記の単純平均ではなく、ユーザー価値の観点で独立に判定する

出力は次の JSON スキーマに厳密に一致させてください。説明文・マークダウン・コードフェンスは禁止。

{
  "faithfulness": 1-5 の整数,
  "relevance": 1-5 の整数,
  "conciseness": 1-5 の整数,
  "overall": 1-5 の整数,
  "rationale": "1-2 文の判定理由（日本語）"
}`;

const scoreSchema = z.object({
  faithfulness: z.number().int().min(1).max(5),
  relevance: z.number().int().min(1).max(5),
  conciseness: z.number().int().min(1).max(5),
  overall: z.number().int().min(1).max(5),
  rationale: z.string().min(1),
});

export async function judgeResponse(input: JudgeInput): Promise<JudgeScore> {
  const model = input.provider ?? selectProvider();

  const userContent = [
    `# 質問`,
    input.question,
    ``,
    `# 回答`,
    input.answer,
    ...(input.context !== undefined
      ? [``, `# Context`, input.context]
      : []),
  ].join("\n");

  const messages: Message[] = [
    { role: "system", content: JUDGE_PROMPT },
    { role: "user", content: userContent },
  ];

  const result = await generateText({ model, messages, temperature: 0 });
  const parsed = parseJudgeJson(result.text);
  return scoreSchema.parse(parsed);
}

// LLM 出力から JSON 部を抽出する。コードフェンスで囲まれていた場合のフォールバックも行う
function parseJudgeJson(text: string): unknown {
  const trimmed = text.trim();
  try {
    return JSON.parse(trimmed);
  } catch {
    const match = trimmed.match(/\{[\s\S]*\}/);
    if (!match) {
      throw new Error(`judge response is not valid JSON: ${trimmed.slice(0, 200)}`);
    }
    return JSON.parse(match[0]);
  }
}

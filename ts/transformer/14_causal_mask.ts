// Step 14: Causal Mask（因果マスク）
//
// Decoder が「未来 token を見ない」ように attention の重みに細工する仕組み。
// これだけ追加すれば 13_encoder.ts が「Decoder-only モデル（GPT/Claude の本体構造）」になる。
//
// なぜ必要か
//   学習時、Decoder は「正解文を全部見ながら、各位置で次単語を予測」する練習をする。
//   例えば「私 は 猫 が 好き」を学習する時、位置 i での予測は位置 0..i-1 だけを見るべき。
//   未来（位置 i+1 以降）を見せたら「正解を見ながら正解を予測する」状態でカンニングになる。
//
//   そこで attention の重み行列で「未来位置を 0 にする」マスクをかける。
//
// マスクの形（4 token の場合）
//   位置 q から位置 k を見る重み行列
//
//            k=0   k=1   k=2   k=3
//   q=0   [   o  ,  ×  ,  ×  ,  ×   ]   ← 自分のみ
//   q=1   [   o  ,  o  ,  ×  ,  ×   ]   ← 自分と過去 1 つ
//   q=2   [   o  ,  o  ,  o  ,  ×   ]
//   q=3   [   o  ,  o  ,  o  ,  o   ]   ← 全部見れる
//
//   o = 計算したスコアをそのまま使う
//   × = -∞ にして softmax 後 0 になる
//
//   下三角が「見れる」、上三角が「見れない」 → 「下三角行列」
//
// 数式
//   masked_scores[i][j] = scaled_scores[i][j]   if j <= i  （過去 or 自分）
//                      = -∞                     if j > i   （未来）
//
//   exp(-∞) = 0 なので softmax 後の重みが 0 になる → 未来 token は完全に無視される
//
// 学習時 vs 推論時
//   学習時: 正解文を全部一度に入力して、causal mask で各位置の予測を「未来抜き」で並列計算
//          → 全位置の予測を一度に学習できる（Transformer が RNN より速い決定的理由）
//   推論時: これまでの生成済み単語列を入力、最後の位置の予測だけ取り出して次単語に
//          → 既に未来は無いから mask 不要だが、実装は同じものを使う

import type { Mask, Matrix } from "./types";
import { multiHeadAttention } from "./07_multi_head_attention";
import type { Head } from "./07_multi_head_attention";
import {
  scaledDotProductAttention,
  scaledDotProductAttentionTrace,
} from "./05_attention";

// causal mask を生成
//   形 (n × n)、mask[i][j] = (j > i) ※未来位置だけ true
//   true なら attention の softmax 前に -∞ にする
export function causalMask(n: number): Mask {
  const m: Mask = [];
  for (let i = 0; i < n; i++) {
    const row: boolean[] = [];
    for (let j = 0; j < n; j++) {
      row.push(j > i); // 未来は見ない
    }
    m.push(row);
  }
  return m;
}

// マスクの形を可視化（o / ×）
function visualizeMask(mask: Mask): string {
  return mask
    .map((row, i) => `  q=${i}  [ ${row.map((m) => (m ? "×" : "o")).join("  ")} ]`)
    .join("\n");
}

// 重み行列を整形
function fmtWeights(W: Matrix): string {
  return W.map(
    (row, i) =>
      `  q=${i}  [ ${row.map((v) => v.toFixed(4)).join(", ")} ]   合計 ${row.reduce((a, b) => a + b, 0).toFixed(4)}`,
  ).join("\n");
}

// === 動作確認 ===
// 直接実行: bun run transformer/14_causal_mask.ts
if (import.meta.main) {
  console.log("=== Causal Mask の動作確認 ===\n");

  // === Step A: causal mask の形を見る ===
  console.log("▼ causalMask(4) の形（× = -∞ にする, o = そのまま）");
  const mask4 = causalMask(4);
  console.log(visualizeMask(mask4));
  console.log("\n  → 下三角は o（見れる）、上三角は ×（見れない）");
  console.log("  → q=0 は自分のみ、q=3 は全 4 token 見える、というのが「自己回帰」の形\n");

  // === Step B: mask あり / なしで attention の重みを比較 ===
  console.log("▼ 同じ入力に mask あり / なしで attention を計算\n");

  // 4 token の self-attention 想定
  const x: Matrix = [
    [1, 0],
    [0, 1],
    [1, 1],
    [-1, 0],
  ];
  const V: Matrix = [
    [10, 0],
    [0, 10],
    [5, 5],
    [-10, 0],
  ];
  console.log(`入力 Q = K = ${JSON.stringify(x)}`);
  console.log(`V = ${JSON.stringify(V)}`);

  console.log("\n--- mask なし（Encoder の self-attention）---");
  const traceNoMask = scaledDotProductAttentionTrace(x, x, V);
  console.log("scaled scores:");
  traceNoMask.scaledScores.forEach((row, i) =>
    console.log(`  q=${i}  [${row.map((v) => v.toFixed(4)).join(", ")}]`),
  );
  console.log("softmax 重み（全位置を見る）:");
  console.log(fmtWeights(traceNoMask.weights));
  console.log("出力:");
  traceNoMask.output.forEach((row, i) =>
    console.log(`  q=${i}  [${row.map((v) => v.toFixed(4)).join(", ")}]`),
  );

  console.log("\n--- mask あり（Decoder の masked self-attention）---");
  const mask = causalMask(x.length);
  const traceMasked = scaledDotProductAttentionTrace(x, x, V, mask);
  console.log("masked scores（× の位置が -Infinity になる）:");
  traceMasked.maskedScores.forEach((row, i) =>
    console.log(
      `  q=${i}  [${row.map((v) => (v === -Infinity ? "  -Inf" : v.toFixed(4))).join(", ")}]`,
    ),
  );
  console.log("softmax 重み（未来は 0）:");
  console.log(fmtWeights(traceMasked.weights));
  console.log("出力（過去のみで作られた表現）:");
  traceMasked.output.forEach((row, i) =>
    console.log(`  q=${i}  [${row.map((v) => v.toFixed(4)).join(", ")}]`),
  );

  console.log("\n=== 観察ポイント ===");
  console.log("  - mask あり版では q=0 の重みが [1, 0, 0, 0] になる → 自分しか見ていない");
  console.log("  - q=1 の重みは [w0, w1, 0, 0] → 過去 1 つ + 自分");
  console.log("  - q=3 だけは mask なし版と同じ重み（全部見える）");
  console.log("  - 各行の合計は常に 1.0（softmax の性質、未来分が 0 でも残りで再正規化される）");

  // === Step C: Multi-Head Attention にも mask を渡せる ===
  console.log("\n\n▼ Multi-Head Attention（07）でも mask を渡せる\n");

  const xMHA: Matrix = [
    [1, 0, 1, 0],
    [0, 1, 1, 0],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
  ];
  const heads: Head[] = [
    {
      WQ: [[1, 0], [0, 1], [0, 0], [0, 0]],
      WK: [[1, 0], [0, 1], [0, 0], [0, 0]],
      WV: [[1, 0], [0, 1], [0, 0], [0, 0]],
    },
    {
      WQ: [[0, 0], [0, 0], [1, 0], [0, 1]],
      WK: [[0, 0], [0, 0], [1, 0], [0, 1]],
      WV: [[0, 0], [0, 0], [1, 0], [0, 1]],
    },
  ];
  const WO: Matrix = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ];

  const mhaMask = causalMask(xMHA.length);
  const mhaNoMask = multiHeadAttention(xMHA, xMHA, xMHA, heads, WO);
  const mhaMasked = multiHeadAttention(xMHA, xMHA, xMHA, heads, WO, mhaMask);

  console.log("入力 (4 × 4):");
  xMHA.forEach((row, i) => console.log(`  [${i}] ${JSON.stringify(row)}`));

  console.log("\nMHA mask なし（双方向）:");
  mhaNoMask.forEach((row, i) =>
    console.log(`  [${i}] ${JSON.stringify(row.map((v) => Number(v.toFixed(4))))}`),
  );

  console.log("\nMHA mask あり（causal、過去のみ）:");
  mhaMasked.forEach((row, i) =>
    console.log(`  [${i}] ${JSON.stringify(row.map((v) => Number(v.toFixed(4))))}`),
  );

  console.log("\n  → 行 0 の出力は mask 版だと token 0 自身の情報だけで作られる");
  console.log("  → 行 3 の出力はどちらも全 token を使うので同じ");

  // === Step D: 「自己回帰生成」の感覚を掴むためのデモ ===
  console.log("\n\n▼ 自己回帰生成の感覚\n");
  console.log("Decoder で 1 token ずつ生成する時、入力長は毎ステップ伸びる");
  console.log("  step 1: 入力 [<BOS>]                  → causalMask(1)");
  console.log("  step 2: 入力 [<BOS>, t1]              → causalMask(2)");
  console.log("  step 3: 入力 [<BOS>, t1, t2]          → causalMask(3)");
  console.log("  step n: 入力 [<BOS>, t1, ..., t_{n-1}] → causalMask(n)");
  console.log("");
  console.log("各ステップで「最後の行の出力」 = 「次に来る token の予測の元」になる");
  console.log("（後段で線形変換 + softmax → vocab 上の確率分布 → 最大確率を選ぶ、を Step 16 で）");

  console.log("\n=== なぜ「causal」mask と呼ぶか ===");
  console.log("  「原因 → 結果」の因果関係から取った名前");
  console.log("  - 過去の token（原因）が未来の token（結果）に影響する");
  console.log("  - 逆方向（未来が過去に影響）は物理的にありえない（causal violation）");
  console.log("  - だから Decoder では「未来からの影響を遮断」するマスクをかける");

  console.log("\n=== これだけで GPT/Claude の本体構造に到達 ===");
  console.log("  13_encoder.ts の transformerBlock に causal mask を渡せば");
  console.log("  → それが「Decoder-only モデル（GPT/Claude の本体）」");
  console.log("  違いは self-attention に mask を入れるかどうかだけ");
  console.log("  Cross-Attention は古典 Encoder-Decoder（翻訳タスク）用、現代 LLM では使わない");

  console.log("\n=== 次のステップ ===");
  console.log("  Step 15: Decoder-only Block（mask 付き transformerBlock）+ Decoder スタック");
  console.log("  Step 16: 出力層（線形変換 + softmax で次トークン予測）");
  console.log("  Step 17: 自己回帰生成ループ（greedy decoding）");
  console.log("  → ここまでで「動く mini-GPT」が完成");
}

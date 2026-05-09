// Step 12: Transformer Block（完全版）
//
// ここまでに作った全部品を組み合わせて 1 つの Transformer ブロックを作る。
// 実際の LLM はこの「ブロック」を 80-100 層スタックすることで作られる。
//
// 流れ
//   入力 x  (形 n × d_model)
//      ↓
//   Multi-Head Attention（Step 07）
//      ↓
//   Residual + LayerNorm（Step 10）
//      ↓ x_1
//   Feed-Forward Network（Step 11）
//      ↓
//   Residual + LayerNorm（Step 10）
//      ↓
//   出力 x_2  (形 n × d_model)
//
// 数式（Post-Norm 版、原典）
//   x_1 = LayerNorm(x + MultiHeadAttention(x, x, x))
//   x_2 = LayerNorm(x_1 + FFN(x_1))
//
// 役割
//   - Attention: token 間で「関連度に応じた情報のやり取り」
//   - FFN:        各 token 内で「非線形な変換」
//   - Residual:  「変化分を足す」設計で深い層でも学習可能
//   - LayerNorm: 数値スケールを安定化
//
// 実際の Transformer のフルパイプライン
//   token IDs
//      ↓ Embedding（Step 08）
//   embeddings
//      ↓ + Positional Encoding（Step 09）
//   位置情報入りの入力
//      ↓ Transformer Block × N 層（このファイル × N）
//      ↓
//   最終出力ベクトル
//      ↓ 出力層（線形変換 + softmax）
//   次トークンの確率分布

import type { Matrix } from "./types";
import { multiHeadAttention } from "./07_multi_head_attention";
import type { Head } from "./07_multi_head_attention";
import { embed } from "./08_embedding";
import type { EmbeddingTable } from "./08_embedding";
import {
  addMatrix,
  positionalEncoding,
} from "./09_positional_encoding";
import { residualLayerNorm } from "./10_layer_norm";
import { feedForward } from "./11_feed_forward";
import type { FFN } from "./11_feed_forward";

export type TransformerBlockConfig = {
  heads: Head[];
  WO: Matrix; // Multi-Head Attention の最終射影
  ffn: FFN;
};

// 1 つの Transformer ブロック
//   入力 x → MHA → Residual+LN → FFN → Residual+LN → 出力
export function transformerBlock(
  x: Matrix,
  config: TransformerBlockConfig,
): Matrix {
  // sublayer 1: Multi-Head Attention（self-attention なので Q = K = V = x）
  const attnOut = multiHeadAttention(x, x, x, config.heads, config.WO);
  const x1 = residualLayerNorm(x, attnOut);

  // sublayer 2: Feed-Forward
  const ffnOut = feedForward(x1, config.ffn);
  const x2 = residualLayerNorm(x1, ffnOut);

  return x2;
}

// === 動作確認 ===
// 直接実行: bun run transformer/12_transformer_block.ts
if (import.meta.main) {
  console.log("=== Transformer Block（フルパイプライン）の動作確認 ===\n");

  // === Step A: トークン化済み ID 列の準備 ===
  const vocab = ["<pad>", "猫", "は", "動物", "だ", "犬", "魚"];
  const tokenIds = [1, 2, 3, 4]; // 「猫 は 動物 だ」
  const sentence = tokenIds.map((id) => vocab[id]).join(" ");
  console.log(`▼ 入力文: "${sentence}"`);
  console.log(`  token IDs: ${JSON.stringify(tokenIds)}`);

  // === Step B: Embedding ===
  const dModel = 4;
  const E: EmbeddingTable = [
    [0, 0, 0, 0],
    [1.0, 0.0, 0.5, 0.0], // 猫
    [0.0, 0.0, 0.0, 1.0], // は
    [0.8, 0.0, 0.6, 0.0], // 動物
    [0.0, 0.0, 0.0, 0.9], // だ
    [0.9, 0.1, 0.5, 0.0], // 犬
    [0.5, -0.5, 0.3, 0.0], // 魚
  ];
  console.log("\n▼ Step B: Embedding（token ID → ベクトル）");
  const embeddings = embed(tokenIds, E);
  embeddings.forEach((row, i) =>
    console.log(`  ${vocab[tokenIds[i]!]} → ${JSON.stringify(row)}`),
  );

  // === Step C: Positional Encoding ===
  console.log("\n▼ Step C: Positional Encoding を加算");
  const PE = positionalEncoding(tokenIds.length, dModel);
  console.log("  PE:");
  PE.forEach((row, i) =>
    console.log(`    [${i}] ${JSON.stringify(row.map((v) => Number(v.toFixed(4))))}`),
  );

  const xInput = addMatrix(embeddings, PE);
  console.log("  embedding + PE:");
  xInput.forEach((row, i) =>
    console.log(`    [${i}] ${JSON.stringify(row.map((v) => Number(v.toFixed(4))))}`),
  );

  // === Step D: Transformer Block の構成 ===
  // Multi-Head Attention 用（07 と同じ 2 Head 構成）
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

  // FFN 用（11 と同じパラメータ）
  const ffn: FFN = {
    W1: [
      [1, 0, 0, 1, 0, 0, -1, 0],
      [0, 1, 0, 0, 1, 0, 0, -1],
      [0, 0, 1, 0, 0, 1, 0, 0],
      [1, 1, 1, 0, 0, 0, 0, 1],
    ],
    b1: [0, 0, 0, 0, 0, 0, 0, 0],
    W2: [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ],
    b2: [0, 0, 0, 0],
  };

  // === Step E: Transformer Block を通す ===
  console.log("\n▼ Step D-E: Transformer Block（MHA → Residual+LN → FFN → Residual+LN）");

  // 詳細表示用に手動で 1 ステップずつ
  const attnOut = multiHeadAttention(xInput, xInput, xInput, heads, WO);
  console.log("\n  Sublayer 1: Multi-Head Attention の出力");
  attnOut.forEach((row, i) =>
    console.log(`    [${i}] ${JSON.stringify(row.map((v) => Number(v.toFixed(4))))}`),
  );

  const x1 = residualLayerNorm(xInput, attnOut);
  console.log("\n  Residual + LayerNorm 後 (x_1):");
  x1.forEach((row, i) =>
    console.log(`    [${i}] ${JSON.stringify(row.map((v) => Number(v.toFixed(4))))}`),
  );

  const ffnOut = feedForward(x1, ffn);
  console.log("\n  Sublayer 2: FFN の出力");
  ffnOut.forEach((row, i) =>
    console.log(`    [${i}] ${JSON.stringify(row.map((v) => Number(v.toFixed(4))))}`),
  );

  const x2 = residualLayerNorm(x1, ffnOut);
  console.log("\n  Residual + LayerNorm 後 (x_2、ブロック最終出力):");
  x2.forEach((row, i) =>
    console.log(`    [${i}] ${JSON.stringify(row.map((v) => Number(v.toFixed(4))))}`),
  );

  // 関数版で確認
  const blockOutput = transformerBlock(xInput, { heads, WO, ffn });
  console.log("\n▼ transformerBlock 関数で一発実行（同じ結果のはず）");
  blockOutput.forEach((row, i) =>
    console.log(`  [${i}] ${JSON.stringify(row.map((v) => Number(v.toFixed(4))))}`),
  );

  console.log("\n=== 観察ポイント ===");
  console.log("  - 入力 x_input と出力 x_2 は同じ形 (n × d_model)");
  console.log("  - だから「次の Transformer ブロックの入力」にそのまま使える");
  console.log("  - 実際の LLM はこのブロックを 80-100 層スタックして使う");
  console.log("  - 各層で「Attention で関連性混合 → FFN で非線形変換」を繰り返し");
  console.log("    抽象度を上げていく");

  console.log("\n=== 全パイプラインの再確認 ===");
  console.log("  token IDs [1,2,3,4]");
  console.log("    ↓ Embedding（Step 08）");
  console.log("  意味ベクトル列");
  console.log("    ↓ + Positional Encoding（Step 09）");
  console.log("  「意味 + 位置」入りの入力");
  console.log("    ↓ Transformer Block（このファイル）× N 層");
  console.log("  最終ベクトル列");
  console.log("    ↓ 出力層（線形変換 + softmax）← 学習タスク次第（次トークン予測など）");
  console.log("  次トークンの確率分布");

  console.log("\n=== Transformer の核心、再確認 ===");
  console.log("  - Embedding（08）: token を方向ベクトルに");
  console.log("  - Position Encoding（09）: 位置も方向ベクトルに");
  console.log("  - Multi-Head Attention（07）: 関連度（内積）で情報を混ぜる");
  console.log("  - FFN（11）: 各 token を非線形に変換");
  console.log("  - Residual + LayerNorm（10）: 学習を安定化");
  console.log("  → 全部「ベクトル空間の角度ゲーム」+ 非線形変換 + 安定化");
}

// Step 5: Scaled Dot-Product Attention 完全版
//
// ここまでの Step を全部組み合わせると、Transformer の核となる Attention になる。
//
// 数式
//   Attention(Q, K, V) = softmax(Q K^T / √d_k) V
//
// 全体の流れ（Step 番号と対応）
//   Step 3: Q K^T               生スコア（n_q × n_k）
//   Step 4: / √d_k              スケーリング（極端化を防ぐ）
//   Step 1: softmax             重み化（行ごとに、合計 1 の確率分布に）
//   Step 5: × V                 重み × Value で加重平均（最終出力）
//
// 形のメモ
//   Q:           (n_q × d_k)
//   K:           (n_k × d_k)
//   V:           (n_k × d_v)
//   weights:     (n_q × n_k)    Q が各 K にどれだけ注目するか
//   output:      (n_q × d_v)    各 Q が引き出した Value の加重和
//
// 直感
//   - 「Query に近い Key を探し、その Value を強く取り込む」
//   - Step 3-4 で「どの Key に注目するか（weights）」を決め
//   - Step 5 で「実際に取り込む情報（V を加重平均）」を作る

import type { Matrix } from "./types";
import { matmul, transpose } from "./03_qk_similarity";
import { scale, softmaxRows } from "./04_scaled_dot_product";

export type AttentionTrace = {
  rawScores: Matrix; // Q K^T
  scaledScores: Matrix; // (Q K^T) / √d_k
  weights: Matrix; // softmax(scaled)
  output: Matrix; // weights V
};

// トレース付き版（学習用）。各段階の中間結果を返す。
export function scaledDotProductAttentionTrace(
  Q: Matrix,
  K: Matrix,
  V: Matrix,
): AttentionTrace {
  const dK = K[0]?.length ?? 0;

  // Step 1: 生スコア = Q K^T （Query × Keys の類似度）
  const rawScores = matmul(Q, transpose(K));

  // Step 2: √d_k で割る（極端化防止）
  const scaledScores = scale(rawScores, 1 / Math.sqrt(dK));

  // Step 3: softmax で行ごとに重み化
  const weights = softmaxRows(scaledScores);

  // Step 4: 重み × Value で加重平均
  const output = matmul(weights, V);

  return { rawScores, scaledScores, weights, output };
}

// 出力だけ返す版（プロダクション利用想定）
export function scaledDotProductAttention(Q: Matrix, K: Matrix, V: Matrix): Matrix {
  return scaledDotProductAttentionTrace(Q, K, V).output;
}

// === 動作確認 ===
// 直接実行: bun run transformer/05_attention.ts
if (import.meta.main) {
  console.log("=== Scaled Dot-Product Attention 完全版 ===\n");

  // 「猫っぽいクエリ」が、猫 / 犬 / 魚 という 3 つの Key に注目する例
  const Q: Matrix = [
    [1, 0], // Query: 「猫っぽい」
  ];
  const K: Matrix = [
    [1, 0], // Key 1: 「猫」     ← Q と同じ向き
    [0, 1], // Key 2: 「犬」     ← Q と直交
    [-1, 0], // Key 3: 「魚」    ← Q と反対
  ];
  const V: Matrix = [
    [10, 0, 0], // Value 1: 猫の情報
    [0, 10, 0], // Value 2: 犬の情報
    [0, 0, 10], // Value 3: 魚の情報
  ];

  const trace = scaledDotProductAttentionTrace(Q, K, V);

  console.log(`Q = ${JSON.stringify(Q)}   形 ${Q.length} × ${Q[0]!.length}`);
  console.log(`K = ${JSON.stringify(K)}   形 ${K.length} × ${K[0]!.length}`);
  console.log(`V = ${JSON.stringify(V)}   形 ${V.length} × ${V[0]!.length}`);

  console.log(`\nStep 1: 生スコア Q K^T`);
  console.log(`  ${JSON.stringify(trace.rawScores[0]!.map((v) => Number(v.toFixed(4))))}`);
  console.log("  → 同じ向き=1, 直交=0, 反対=-1");

  console.log(`\nStep 2: / √d_k （d_k = ${K[0]!.length}, √d_k = ${Math.sqrt(K[0]!.length).toFixed(4)}）`);
  console.log(
    `  ${JSON.stringify(trace.scaledScores[0]!.map((v) => Number(v.toFixed(4))))}`,
  );

  console.log(`\nStep 3: softmax で重み化`);
  console.log(`  ${JSON.stringify(trace.weights[0]!.map((v) => Number(v.toFixed(4))))}`);
  console.log(`  合計 = ${trace.weights[0]!.reduce((a, b) => a + b, 0).toFixed(4)} （必ず 1）`);
  console.log("  → Key 1（猫）に最大の重み、Key 2（犬）と Key 3（魚）にも一部");

  console.log(`\nStep 4: 重み × V`);
  console.log(`  ${JSON.stringify(trace.output[0]!.map((v) => Number(v.toFixed(4))))}`);
  console.log("  → Value 1（猫の情報）が大半、Value 2/3 が少し混じった結果");
  console.log("    これが「Query に近い Key の Value を強く取り込む」結果");

  console.log("\n--- 別の Query で試す ---");
  const Q2: Matrix = [
    [0.5, 0.5], // 「猫と犬の中間っぽい」
  ];
  const trace2 = scaledDotProductAttentionTrace(Q2, K, V);
  console.log(`Q = [0.5, 0.5]（猫と犬の中間）`);
  console.log(
    `weights = ${JSON.stringify(trace2.weights[0]!.map((v) => Number(v.toFixed(4))))}`,
  );
  console.log(
    `output  = ${JSON.stringify(trace2.output[0]!.map((v) => Number(v.toFixed(4))))}`,
  );
  console.log("  → 猫(K1)と犬(K2)に近い重み、出力も猫情報と犬情報が混ざる");

  console.log("\n--- 複数 Query を一気に処理 ---");
  const Q3: Matrix = [
    [1, 0], // 猫っぽい
    [0, 1], // 犬っぽい
    [0.5, 0.5], // 中間
  ];
  const result3 = scaledDotProductAttention(Q3, K, V);
  console.log(`Q（3 × 2）`);
  console.log(`output = ${JSON.stringify(result3.map((row) => row.map((v) => Number(v.toFixed(2)))))}`);
  console.log("  各行が 各 Query の Attention 結果");
}

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

// 計算過程を全部出すヘルパー
function fmt(v: number, digits = 4) {
  return v.toFixed(digits);
}
function fmtVec(v: number[], digits = 4) {
  return `[${v.map((x) => x.toFixed(digits)).join(", ")}]`;
}

// 1 つの Query について計算過程を全部出す
function attentionVerboseSingle(Q: Matrix, K: Matrix, V: Matrix, label = "") {
  console.log(`\n=== Attention 計算過程 ${label} ===`);
  console.log(`Q = ${JSON.stringify(Q)}   形 ${Q.length} × ${Q[0]!.length}`);
  console.log(`K = ${JSON.stringify(K)}   形 ${K.length} × ${K[0]!.length}`);
  console.log(`V = ${JSON.stringify(V)}   形 ${V.length} × ${V[0]!.length}`);

  const dK = K[0]!.length;
  const KT = transpose(K);

  console.log(`\n▼ Step 1: K を転置`);
  console.log(`  K^T = ${JSON.stringify(KT)}   形 ${KT.length} × ${KT[0]!.length}`);

  console.log(`\n▼ Step 2: 生スコア Q K^T を計算`);
  const rawScores = matmul(Q, KT);
  for (let i = 0; i < Q.length; i++) {
    for (let j = 0; j < KT[0]!.length; j++) {
      const aRow = Q[i]!;
      const bCol = KT.map((row) => row[j]!);
      const expr = aRow.map((a, k) => `${a}*${bCol[k]}`).join(" + ");
      console.log(`  (${i},${j}) = ${expr} = ${rawScores[i]![j]}`);
    }
  }
  console.log(`  生スコア = ${JSON.stringify(rawScores)}`);

  console.log(`\n▼ Step 3: √d_k で割る`);
  const sqrtDk = Math.sqrt(dK);
  console.log(`  √d_k = √${dK} = ${fmt(sqrtDk)}`);
  const scaledScores = scale(rawScores, 1 / sqrtDk);
  for (let i = 0; i < rawScores.length; i++) {
    for (let j = 0; j < rawScores[0]!.length; j++) {
      console.log(
        `  (${i},${j}): ${rawScores[i]![j]} / ${fmt(sqrtDk)} = ${fmt(scaledScores[i]![j]!)}`,
      );
    }
  }

  console.log(`\n▼ Step 4: 各行に softmax を適用（重み化）`);
  const weights: Matrix = [];
  for (let i = 0; i < scaledScores.length; i++) {
    console.log(`  行 ${i}（Query ${i} の重み）の計算:`);
    const row = scaledScores[i]!;
    const max = Math.max(...row);
    console.log(`    max = ${fmt(max)}`);
    const shifted = row.map((a) => a - max);
    console.log(`    a_i - max = ${fmtVec(shifted)}`);
    const exps = shifted.map((s) => Math.exp(s));
    console.log(`    exp(a_i - max) = ${fmtVec(exps)}`);
    const sum = exps.reduce((a, b) => a + b, 0);
    console.log(`    Σ exp = ${fmt(sum)}`);
    const w = exps.map((v) => v / sum);
    console.log(`    重み = ${fmtVec(w)}   合計 = ${fmt(w.reduce((a, b) => a + b, 0))}`);
    weights.push(w);
  }

  console.log(`\n▼ Step 5: 重み × V で加重平均`);
  const output = matmul(weights, V);
  for (let i = 0; i < weights.length; i++) {
    console.log(`  Query ${i} の出力:`);
    const w = weights[i]!;
    for (let j = 0; j < V[0]!.length; j++) {
      const terms = w.map((wk, k) => `${fmt(wk)}*${V[k]![j]}`).join(" + ");
      console.log(`    output[${i}][${j}] = ${terms} = ${fmt(output[i]![j]!)}`);
    }
  }

  console.log(`\n▼ 最終出力`);
  console.log(`  output = ${JSON.stringify(output.map((row) => row.map((v) => Number(fmt(v)))))}`);
  return output;
}

// === 動作確認 ===
// 直接実行: bun run transformer/05_attention.ts
if (import.meta.main) {
  console.log("=== Scaled Dot-Product Attention 完全版（計算過程を全部出す）===");

  // 「猫っぽいクエリ」が、猫 / 犬 / 魚 という 3 つの Key に注目する例
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

  // ケース 1: 猫っぽい Query
  attentionVerboseSingle([[1, 0]], K, V, "ケース 1: Q=[1,0] 猫っぽい");

  // ケース 2: 猫と犬の中間
  attentionVerboseSingle([[0.5, 0.5]], K, V, "ケース 2: Q=[0.5,0.5] 猫と犬の中間");

  // ケース 3: 複数 Query を一気に処理
  attentionVerboseSingle(
    [
      [1, 0],
      [0, 1],
      [0.5, 0.5],
    ],
    K,
    V,
    "ケース 3: 3 つの Query を同時に",
  );

  console.log("\n=== 観察ポイント ===");
  console.log("  - Q=[1,0] のとき重みは Key 1（猫）に集中、出力は猫の Value が支配");
  console.log("  - Q=[0.5,0.5] のとき重みは猫と犬に近い、出力も両者の混合");
  console.log("  - 複数 Query を渡すと各 Query 独立に注意分布が計算される");
  console.log("  - これが Transformer の「全部との関連度計算」の最小単位");
}

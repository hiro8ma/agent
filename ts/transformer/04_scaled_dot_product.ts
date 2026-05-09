// Step 4: スケール化された内積アテンション（Scaled Dot-Product Attention）の前半
//
// Step 3 で Q K^T で「生スコア」を得た。
// このまま softmax に入れると、d_k が大きいときに困った問題が起きる。
//
// 問題
//   ベクトルの次元 d_k が大きいと、内積の値が大きくなりやすい。
//   例: 各成分が標準正規分布に従うとき、内積の分散は d_k に比例する。
//   d_k = 512 のような実用サイズだと、生スコアが ±20 や ±50 まで膨らむことがある。
//
//   そんな大きな値を softmax に入れると：
//   - 1 つの値だけが 1 に近くなり、他は全部ほぼ 0
//     例: softmax([20, 5, 3]) ≈ [0.99999, 0.00001, ...]
//   - これは「one-hot に近い」状態
//   - 勾配がほぼ 0 になって学習が止まる（勾配消失）
//
// 解決
//   √d_k で割ってスケーリングする。
//   これでスコアの分散が d_k に依存しなくなり、softmax が極端化しない。
//
// 数式
//   scaled_scores = (Q K^T) / √d_k
//
// 直感
//   d_k = 4 なら 2 で割る、d_k = 16 なら 4 で割る、d_k = 64 なら 8 で割る。
//   次元が大きいほど割る数を大きくして、スコアの暴走を防ぐ。

import type { Matrix } from "./types";
import { matmul, transpose } from "./03_qk_similarity";
import { softmax } from "./01_softmax";

// 行列をスカラー倍
export function scale(A: Matrix, s: number): Matrix {
  return A.map((row) => row.map((v) => v * s));
}

// 行列の各行に softmax を適用
//   Attention の文脈では、各 Query について「複数 Key の重み」を softmax 化する。
//   なので「行ごと」に softmax を取る。
export function softmaxRows(A: Matrix): Matrix {
  return A.map((row) => softmax(row));
}

// 計算過程を全部出すヘルパー
function fmtVec(v: number[], digits = 4) {
  return `[${v.map((x) => x.toFixed(digits)).join(", ")}]`;
}

function softmaxRowVerbose(row: number[], indent = "    "): number[] {
  const max = Math.max(...row);
  console.log(`${indent}Step 1: max = ${max.toFixed(4)}`);
  const shifted = row.map((a) => a - max);
  console.log(`${indent}Step 2: a_i - max = ${fmtVec(shifted)}`);
  const exps = shifted.map((s) => Math.exp(s));
  console.log(`${indent}Step 3: exp(a_i - max) = ${fmtVec(exps)}`);
  const sum = exps.reduce((a, b) => a + b, 0);
  console.log(`${indent}Step 4: Σ exp = ${sum.toFixed(4)}`);
  const result = exps.map((v) => v / sum);
  console.log(`${indent}Step 5: 正規化 = ${fmtVec(result)}`);
  return result;
}

// === 動作確認 ===
// 直接実行: bun run transformer/04_scaled_dot_product.ts
if (import.meta.main) {
  console.log("=== Scaled Dot-Product（計算過程を全部出す版）===\n");

  // ケース 1: 小さい d_k（d_k = 2）
  console.log("=== ケース 1: d_k = 2（小さい次元）===");
  const Q1: Matrix = [[1, 0]];
  const K1: Matrix = [
    [1, 0],
    [0, 1],
    [-1, 0],
  ];
  const dK1 = K1[0]!.length;

  console.log(`Q = ${JSON.stringify(Q1)}`);
  console.log(`K = ${JSON.stringify(K1)}`);
  console.log(`d_k = ${dK1}（K の各行の長さ）`);

  console.log("\n▼ Step A: 生スコア Q K^T");
  const KT1 = transpose(K1);
  console.log(`  K^T = ${JSON.stringify(KT1)}`);
  const rawScores1 = matmul(Q1, KT1);
  for (let j = 0; j < KT1[0]!.length; j++) {
    const aRow = Q1[0]!;
    const bCol = KT1.map((row) => row[j]!);
    const expr = aRow.map((a, k) => `${a}*${bCol[k]}`).join(" + ");
    console.log(`  (0,${j}) = ${expr} = ${rawScores1[0]![j]}`);
  }
  console.log(`  生スコア = ${JSON.stringify(rawScores1[0])}`);

  console.log("\n▼ Step B: √d_k で割る（スケーリング）");
  const sqrtDk1 = Math.sqrt(dK1);
  console.log(`  √d_k = √${dK1} = ${sqrtDk1.toFixed(4)}`);
  const scaledScores1 = scale(rawScores1, 1 / sqrtDk1);
  for (let j = 0; j < rawScores1[0]!.length; j++) {
    console.log(
      `  ${rawScores1[0]![j]} / ${sqrtDk1.toFixed(4)} = ${scaledScores1[0]![j]!.toFixed(4)}`,
    );
  }
  console.log(`  スケール後 = ${fmtVec(scaledScores1[0]!)}`);

  console.log("\n▼ Step C: softmax で重み化");
  const weights1Row = softmaxRowVerbose(scaledScores1[0]!);
  console.log(`  重み = ${fmtVec(weights1Row)}`);
  console.log(`  合計 = ${weights1Row.reduce((a, b) => a + b, 0).toFixed(4)}`);

  // ケース 2: 大きいスコア（√d_k なしだと softmax が極端化）
  console.log("\n\n=== ケース 2: スコアが大きいとき（√d_k あり vs なし）===");
  const bigScores: Matrix = [[20, 5, 3]];
  console.log(`生スコア = ${JSON.stringify(bigScores[0])}`);

  console.log("\n▼ √d_k なし softmax の計算過程");
  const noScale = softmaxRowVerbose(bigScores[0]!);
  console.log(`  → 結果 = ${fmtVec(noScale, 6)}`);
  console.log(`  → 最大値の重み: ${noScale[0]!.toFixed(6)}（ほぼ one-hot、極端化）`);

  const dK2 = 64; // 実用的な d_k 例
  const sqrtDk2 = Math.sqrt(dK2);
  const scaledBig = scale(bigScores, 1 / sqrtDk2);

  console.log(`\n▼ √${dK2} = ${sqrtDk2} で割ってから softmax`);
  console.log(`  生スコア / √${dK2} = ${fmtVec(scaledBig[0]!)}`);
  const withScale = softmaxRowVerbose(scaledBig[0]!);
  console.log(`  → 結果 = ${fmtVec(withScale)}`);
  console.log(`  → 最大値の重み: ${withScale[0]!.toFixed(4)}（穏やか、他の Key にも注意が向く）`);

  console.log("\n=== 観察ポイント ===");
  console.log("  - √d_k なしだと最大値の重みがほぼ 1 に集中（極端化、勾配消失リスク）");
  console.log("  - √d_k で割るとスコアの分散が抑えられて softmax が穏やかになる");
  console.log("  - これが Transformer が「Scaled」と名乗る理由");
}

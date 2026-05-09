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

// === 動作確認 ===
// 直接実行: bun run transformer/04_scaled_dot_product.ts
if (import.meta.main) {
  console.log("=== Scaled Dot-Product（√d_k スケーリング）===\n");

  // ケース 1: 小さい d_k（d_k = 2）
  console.log("--- ケース 1: d_k = 2（小さい次元）---");
  const Q1: Matrix = [[1, 0]];
  const K1: Matrix = [
    [1, 0],
    [0, 1],
    [-1, 0],
  ];
  const dK1 = K1[0]!.length;

  const rawScores1 = matmul(Q1, transpose(K1));
  const scaledScores1 = scale(rawScores1, 1 / Math.sqrt(dK1));
  const weights1 = softmaxRows(scaledScores1);

  console.log(`生スコア Q K^T = ${JSON.stringify(rawScores1[0])}`);
  console.log(`√d_k = √${dK1} = ${Math.sqrt(dK1).toFixed(4)}`);
  console.log(
    `スケール後 / √d_k = ${JSON.stringify(scaledScores1[0]!.map((v) => Number(v.toFixed(4))))}`,
  );
  console.log(
    `softmax 適用後 = ${JSON.stringify(weights1[0]!.map((v) => Number(v.toFixed(4))))}`,
  );

  // ケース 2: 大きいスコア（√d_k なしだと softmax が極端化）
  console.log("\n--- ケース 2: スコアが大きいとき（√d_k あり vs なし）---");
  const bigScores: Matrix = [[20, 5, 3]];
  console.log(`生スコア = ${JSON.stringify(bigScores[0])}`);

  const noScale = softmaxRows(bigScores);
  console.log(
    `√d_k なし softmax = ${JSON.stringify(noScale[0]!.map((v) => Number(v.toFixed(6))))}`,
  );

  const dK2 = 64; // 実用的な d_k 例
  const scaledBig = scale(bigScores, 1 / Math.sqrt(dK2));
  const withScale = softmaxRows(scaledBig);
  console.log(
    `√${dK2} で割って softmax = ${JSON.stringify(withScale[0]!.map((v) => Number(v.toFixed(4))))}`,
  );

  console.log("\n観察ポイント");
  console.log("  - √d_k なしだと最大値が 0.9999 くらいまで集中（極端化、勾配消失）");
  console.log("  - √d_k で割ると重みが穏やか、他の Key にも注意が向く");
  console.log("  - これが Transformer が「scaled」と名乗る理由");
}

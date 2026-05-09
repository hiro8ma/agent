// Step 6: 線形変換（Linear Projection）
//
// Multi-Head Attention で初めて登場する操作。
//
// 数式
//   y = x W
//
//   x: 入力ベクトル / 行列     形 (n × d_model)
//   W: 重み行列（学習対象）    形 (d_model × d_k)
//   y: 出力                    形 (n × d_k)
//
// 役割
//   - 同じ入力 x を「別の意味空間」に射影する
//   - W が違えば、同じ x から違う y が得られる
//   - これが「Head ごとに視点を変える」仕組みの正体
//
// 直感
//   x には「単語の意味」が高次元ベクトルとして詰まっている。
//   W はその意味を「ある観点で抽出するフィルター」。
//   - W^Q（Query 用）: 「何を探したいか」を抽出
//   - W^K（Key 用）  : 「何を持っているか」を抽出
//   - W^V（Value 用）: 「実際に取り込む情報」を抽出
//
//   同じ単語でも、Query で見るか Key で見るかで使われ方が違う。
//   それを「線形変換で別の表現に変える」ことで実現している。
//
// なぜ線形変換が必要か
//   Multi-Head Attention では h 個の Head それぞれが
//   独自の W^Q_i, W^K_i, W^V_i を持つ。
//   各 Head は異なる W で射影するので、同じ入力から異なる注意分布を作れる。
//   これが「複数視点で同時に attention する」を実装する方法。

import type { Matrix } from "./types";
import { matmul } from "./03_qk_similarity";

// 線形変換: y = x W
// 単に行列積を取るだけ（既に matmul があるので薄いラッパ）。
// 名前を分けることで「これは射影」と意図を明示する。
export function linearProjection(x: Matrix, W: Matrix): Matrix {
  return matmul(x, W);
}

// === 動作確認 ===
// 直接実行: bun run transformer/06_linear_projection.ts
if (import.meta.main) {
  console.log("=== 線形変換（Linear Projection）の動作確認 ===\n");

  // 入力: 3 つのトークンが、それぞれ 4 次元のベクトルで表現されている
  // d_model = 4
  const x: Matrix = [
    [1, 0, 1, 0], // token 1: 「猫」（仮想的な埋め込み）
    [0, 1, 1, 0], // token 2: 「犬」
    [1, 1, 0, 0], // token 3: 「ペット」
  ];

  console.log("入力 x（3 トークン × d_model=4）:");
  console.log(`  ${JSON.stringify(x)}`);

  // ヘッド 0 用の W^Q: 最初の 2 次元を取り出すフィルター
  // これは学習で得られるべきものだが、デモ用に手作りで「見せたい挙動」を作る
  const W_Q_head0: Matrix = [
    [1, 0], // x[0] の 1 次元目を y[0] に
    [0, 1], // x[1] の 2 次元目を y[1] に
    [0, 0], // x[2] は使わない
    [0, 0], // x[3] は使わない
  ];

  console.log("\n▼ Head 0 の W^Q（4 × 2、最初の 2 次元を抽出）:");
  console.log(`  ${JSON.stringify(W_Q_head0)}`);

  console.log("\n▼ y = x W^Q（3 × 2）の計算");
  const y0 = linearProjection(x, W_Q_head0);
  for (let i = 0; i < x.length; i++) {
    for (let j = 0; j < W_Q_head0[0]!.length; j++) {
      const xRow = x[i]!;
      const wCol = W_Q_head0.map((row) => row[j]!);
      const expr = xRow.map((v, k) => `${v}*${wCol[k]}`).join(" + ");
      console.log(`  y[${i}][${j}] = ${expr} = ${y0[i]![j]}`);
    }
  }
  console.log(`  y = ${JSON.stringify(y0)}`);
  console.log("  → 各行が「token を Head 0 の視点で見た 2 次元表現」");

  // ヘッド 1 用の W^Q: 異なる視点（残りの 2 次元 + 線形混合）
  const W_Q_head1: Matrix = [
    [0, 0],
    [0, 0],
    [1, 0],
    [0, 1],
  ];

  console.log("\n▼ Head 1 の W^Q（後半 2 次元を抽出する別視点）");
  console.log(`  W = ${JSON.stringify(W_Q_head1)}`);
  const y1 = linearProjection(x, W_Q_head1);
  console.log(`  y = ${JSON.stringify(y1)}`);
  console.log("  → 同じ入力 x から、Head 1 の視点では別の 2 次元表現が得られる");

  console.log("\n=== 観察ポイント ===");
  console.log("  - 入力 x は 3 つとも同じ");
  console.log("  - W^Q_0 と W^Q_1 が違うので、出力も違う");
  console.log("  - これが「Head ごとに視点が違う」の正体");
  console.log("  - 実際の Transformer では W は学習で決まる（人間が手で書かない）");

  console.log("\n=== 重要な関係 ===");
  console.log("  Single Attention: Q = x をそのまま使う");
  console.log("  Multi-Head:       Q_i = x W^Q_i で射影してから使う");
  console.log("  W が学習可能なので、ネットワークが「どの視点で見るか」を自分で決められる");
}

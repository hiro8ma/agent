// Step 10: Residual Connection + Layer Normalization
//
// Transformer ブロック内で「Attention や FFN の前後」に必ず入る 2 つの操作。
// 学習を安定させる「縁の下の力持ち」。
//
// 数式（Residual: 残差接続）
//   y = x + sublayer(x)
//
//   sublayer は Attention や FFN
//   入力 x を「足し戻す」ことで「変化しすぎ」を防ぐ
//
// 数式（Layer Normalization）
//   各 token のベクトル（行）について:
//     mean = (1/d) Σ_k x[k]
//     var  = (1/d) Σ_k (x[k] - mean)^2
//     y[k] = (x[k] - mean) / √(var + ε)
//
//   ε は数値安定化用の小さな値（除算ゼロ防止）
//
// 役割
//   Residual:
//     - 「sublayer の出力 + 元の x」で、学習が壊れにくくなる
//     - 「sublayer は変化分を学習する」発想（変化が小さくても元の情報が保たれる）
//     - 100 層など深いネットワークで勾配が伝わるための必須テクニック
//
//   LayerNorm:
//     - 各 token のベクトルを「平均 0、分散 1」に揃える
//     - 数値が暴走しないので学習が安定する
//     - Attention や FFN の出力スケールが層ごとにバラバラになるのを抑える
//
// Transformer ブロック内での使われ方（後段の 12 で出てくる）
//   x_1 = LayerNorm(x + MultiHeadAttention(x))      ← Attention の前後
//   x_2 = LayerNorm(x_1 + FFN(x_1))                  ← FFN の前後
//
// Pre-Norm vs Post-Norm
//   オリジナル Transformer: LayerNorm を sublayer の後（Post-Norm）
//   最近の LLM: LayerNorm を sublayer の前に置くことも多い（Pre-Norm、学習が安定）
//   ここでは原典に従って Post-Norm 版を実装

import type { Matrix } from "./types";
import { addMatrix } from "./09_positional_encoding";

// 行ごとに LayerNorm を適用
//   各 token（各行）について平均 0、分散 1 になるよう正規化
export function layerNorm(X: Matrix, eps = 1e-5): Matrix {
  return X.map((row) => {
    const mean = row.reduce((a, b) => a + b, 0) / row.length;
    const variance =
      row.reduce((a, b) => a + (b - mean) ** 2, 0) / row.length;
    const std = Math.sqrt(variance + eps);
    return row.map((v) => (v - mean) / std);
  });
}

// Residual + LayerNorm を一括で（Transformer ブロック内で再利用）
export function residualLayerNorm(x: Matrix, sublayerOutput: Matrix): Matrix {
  return layerNorm(addMatrix(x, sublayerOutput));
}

// === 動作確認 ===
// 直接実行: bun run transformer/10_layer_norm.ts
if (import.meta.main) {
  console.log("=== Residual + LayerNorm の動作確認 ===\n");

  // 仮想的な入力
  const x: Matrix = [
    [1, 2, 3, 4],
    [10, 20, 30, 40],
    [-5, 0, 5, 10],
  ];
  console.log("入力 x:");
  x.forEach((row, i) => console.log(`  [${i}] ${JSON.stringify(row)}`));

  // 仮想的な sublayer 出力（Attention や FFN の出力を模擬）
  const sublayerOutput: Matrix = [
    [0.5, -0.5, 0.5, -0.5],
    [1, -1, 1, -1],
    [2, -2, 2, -2],
  ];
  console.log("\nsublayer 出力（Attention や FFN を模擬）:");
  sublayerOutput.forEach((row, i) => console.log(`  [${i}] ${JSON.stringify(row)}`));

  // Step 1: Residual
  const summed = addMatrix(x, sublayerOutput);
  console.log("\n▼ Step 1: Residual = x + sublayer(x)");
  summed.forEach((row, i) => console.log(`  [${i}] ${JSON.stringify(row)}`));

  // Step 2: LayerNorm の中身を verbose に
  console.log("\n▼ Step 2: LayerNorm（各行を平均 0 / 分散 1 に正規化）");
  for (let i = 0; i < summed.length; i++) {
    const row = summed[i]!;
    const mean = row.reduce((a, b) => a + b, 0) / row.length;
    const variance = row.reduce((a, b) => a + (b - mean) ** 2, 0) / row.length;
    const std = Math.sqrt(variance + 1e-5);
    const normed = row.map((v) => (v - mean) / std);
    console.log(`  行 [${i}]:`);
    console.log(`    入力      = ${JSON.stringify(row)}`);
    console.log(`    平均      = ${mean.toFixed(4)}`);
    console.log(`    分散      = ${variance.toFixed(4)}`);
    console.log(`    標準偏差  = ${std.toFixed(4)}`);
    console.log(
      `    正規化後  = ${JSON.stringify(normed.map((v) => Number(v.toFixed(4))))}`,
    );
    const checkMean = normed.reduce((a, b) => a + b, 0) / normed.length;
    const checkVar =
      normed.reduce((a, b) => a + (b - checkMean) ** 2, 0) / normed.length;
    console.log(
      `    確認: 平均 ≈ ${checkMean.toFixed(6)}, 分散 ≈ ${checkVar.toFixed(4)}`,
    );
  }

  console.log("\n▼ residualLayerNorm 関数で一発実行");
  const result = residualLayerNorm(x, sublayerOutput);
  result.forEach((row, i) =>
    console.log(`  [${i}] ${JSON.stringify(row.map((v) => Number(v.toFixed(4))))}`),
  );

  console.log("\n=== 観察ポイント ===");
  console.log("  - Residual で「元の x の情報」が保たれる（足し算なので消えない）");
  console.log("  - LayerNorm 後、各行の平均が ≈ 0、分散が ≈ 1 になっている");
  console.log("  - 元のスケール（[1,2,3,4] vs [10,20,30,40]）が消えて統一スケールに");
  console.log("  - これで sublayer の出力スケールに左右されず、次層が安定して学習できる");

  console.log("\n=== なぜ「正規化 = 学習安定化」なのか ===");
  console.log("  - 各層の出力スケールが層ごとにバラバラだと、勾配が爆発 / 消失する");
  console.log("  - LayerNorm で各層の入出力スケールを揃えると、勾配が綺麗に流れる");
  console.log("  - これが 100 層級の Transformer を学習可能にした立役者");
}

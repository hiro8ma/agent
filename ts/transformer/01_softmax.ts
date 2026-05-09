// Step 1: softmax
//
// 役割
//   生のスコア（バラバラの数値）を「確率分布っぽい重み」に変換する関数。
//
// 数式
//   softmax(a_i) = e^{a_i} / Σ_j e^{a_j}
//
// 直感
//   - 全部正になる（exp は必ず正）
//   - 合計が 1 になる（Σ で割っているから）
//   - 大きい値をさらに強調する（exp の効果）
//
// 例
//   入力: [1, 2, 10]
//   出力: [ほぼ 0, ほぼ 0, ほぼ 1]   ← 一番大きい値が圧倒的になる
//
// 数値安定化（重要）
//   素直に exp(a_i) を計算すると、a_i が大きいと overflow する。
//   例: e^1000 → Infinity
//   そこで「全要素から max を引いてから exp」する。
//   結果は数学的に等価（shift invariance、後述）。
//
//   softmax(a_i) = e^{a_i - max} / Σ_j e^{a_j - max}
//
// shift invariance（平行移動不変性）
//   softmax は「全要素から同じ定数 c を引いても結果が変わらない」性質を持つ。
//
//   softmax(a_i - c)
//     = e^{a_i - c} / Σ_j e^{a_j - c}
//     = (e^{a_i} / e^c) / (Σ_j e^{a_j} / e^c)
//     = e^{a_i} / Σ_j e^{a_j}      ← e^c が分子分母で打ち消される
//     = softmax(a_i)
//
//   なので「max を引く」操作は結果に影響を与えず、純粋に overflow 対策。
//   注意: shift invariance は「全体の平行移動」に対する不変性であって、
//   「差の絶対値」が変われば結果は変わる。
//   例: [1,2,3] と [-1,0,1] は同じ結果（差は両方 (2,1,0)）
//       [1,2,3] と [100,200,300] は違う結果（差が (2,1,0) vs (200,100,0)）

import type { Vector } from "./types";

export function softmax(input: Vector): Vector {
  // 1. 最大値を見つける（数値安定化のため）
  const max = Math.max(...input);

  // 2. 各要素から max を引いて exp を取る
  //    a_i - max は必ず ≤ 0 なので、exp は ≤ 1、overflow しない
  const expValues = input.map((a) => Math.exp(a - max));

  // 3. 合計で割って確率分布化
  const sum = expValues.reduce((acc, v) => acc + v, 0);
  return expValues.map((v) => v / sum);
}

// === 動作確認 ===
// 直接実行: bun run transformer/01_softmax.ts
if (import.meta.main) {
  console.log("=== softmax の動作確認 ===\n");

  const examples = [
    [1, 2, 3],
    [1, 2, 10],
    [-1, 0, 1],
    [100, 200, 300], // 数値安定化の効果（max 引きしないと overflow する範囲）
  ];

  for (const input of examples) {
    const output = softmax(input);
    const sum = output.reduce((a, b) => a + b, 0);
    console.log(`入力: [${input.join(", ")}]`);
    console.log(`出力: [${output.map((v) => v.toFixed(4)).join(", ")}]`);
    console.log(`合計: ${sum.toFixed(6)} （必ず 1 になる）`);
    console.log("");
  }

  console.log("観察ポイント");
  console.log("  - 入力 [1,2,10] のように差が大きいと、最大値の重みがほぼ 1 に集中");
  console.log("  - 入力 [1,2,3] のように差が小さいと、重みが分散する");
  console.log("  - 入力 [-1,0,1] は [1,2,3] と完全一致 → shift invariance（平行移動不変性）");
  console.log("    全要素から同じ値を引いても結果が変わらない。これが max 引き trick の根拠");
  console.log("  - 入力 [100,200,300] は差が極大（=200）なのでほぼ one-hot に集中");
  console.log("    → 大きい値でも overflow しない（max 引きで [-200,-100,0] に変換される）");
  console.log("    → ただし結果は [1,2,3] と異なる（差の大きさが違うため）");
}

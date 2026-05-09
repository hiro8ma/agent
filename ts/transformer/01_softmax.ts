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
//   結果は数学的に等価（分子分母で同じ定数を掛けても比は変わらないため）。
//
//   softmax(a_i) = e^{a_i - max} / Σ_j e^{a_j - max}

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
  console.log("  - 入力 [100,200,300] でも overflow せず、[1,2,3] と同じ結果になる");
  console.log("    （max 引きで a_i-max が同じになるため）");
}

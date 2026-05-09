// Step 11: Feed-Forward Network（FFN）
//
// Attention の後に必ず入る「2 層の MLP」。
// Attention が「token 間の情報のやり取り」を担うのに対し、
// FFN は「各 token 内での非線形な変換」を担う。
//
// 数式
//   FFN(x) = ReLU(x W_1 + b_1) W_2 + b_2
//
//   x:    入力           形 (n × d_model)
//   W_1:  最初の重み     形 (d_model × d_ff)、典型的に d_ff = 4 × d_model
//   b_1:  最初のバイアス 形 (d_ff,)
//   W_2:  2 つ目の重み   形 (d_ff × d_model)
//   b_2:  2 つ目のバイアス 形 (d_model,)
//
// 流れ
//   1. 線形変換で d_model → d_ff（次元を 4 倍に拡張）
//   2. ReLU で非線形性を入れる（負を 0 に）
//   3. 線形変換で d_ff → d_model（元の次元に戻す）
//
// 役割
//   - Attention だけだと「線形変換の組み合わせ」になり、表現力が足りない
//   - 非線形関数（ReLU）を挟むことで「複雑な関係」を表現できるようになる
//   - 「次元を一度広げて、絞り直す」ことで「中間表現」を作れる
//
// 「角度ゲーム」での位置づけ
//   - Attention は「向きの近さ」で情報を混ぜる
//   - FFN は各 token のベクトルを「歪める」ことで「意味の変換」を行う
//   - 例えば「猫」の方向 → 「ペット」の方向への変換などを学習する
//
// なぜ d_ff = 4 × d_model なのか
//   - 経験則。実験的に「広げて絞る」幅が 4 倍くらいが性能と計算量のバランスが良い
//   - 最近のモデル（LLaMA 等）は SwiGLU など改良版で d_ff = 8/3 × d_model の場合もあり

import type { Matrix } from "./types";
import { matmul } from "./03_qk_similarity";

// 行ベクトルにバイアスを足す（broadcast）
//   X: (n × d), b: (d,) → 各行に b を足す
function addBias(X: Matrix, b: number[]): Matrix {
  return X.map((row) => row.map((v, j) => v + b[j]!));
}

// ReLU: max(0, x) を要素ごとに適用
//   負の値を 0 に潰す、正の値はそのまま
//   これが「非線形性」の源
export function relu(X: Matrix): Matrix {
  return X.map((row) => row.map((v) => Math.max(0, v)));
}

// FFN のパラメータ
export type FFN = {
  W1: Matrix; // (d_model × d_ff)
  b1: number[]; // (d_ff,)
  W2: Matrix; // (d_ff × d_model)
  b2: number[]; // (d_model,)
};

// FFN(x) = ReLU(x W_1 + b_1) W_2 + b_2
export function feedForward(X: Matrix, ffn: FFN): Matrix {
  const hidden = relu(addBias(matmul(X, ffn.W1), ffn.b1));
  return addBias(matmul(hidden, ffn.W2), ffn.b2);
}

// === 動作確認 ===
// 直接実行: bun run transformer/11_feed_forward.ts
if (import.meta.main) {
  console.log("=== Feed-Forward Network の動作確認 ===\n");

  const dModel = 4;
  const dFf = 8; // 典型的に 4 × d_model

  // 入力（3 token、d_model = 4）
  const X: Matrix = [
    [1, 0, 1, 0],
    [0, 1, 1, 0],
    [-1, 1, 0, 1],
  ];
  console.log("入力 X:");
  X.forEach((row, i) => console.log(`  [${i}] ${JSON.stringify(row)}`));

  // FFN パラメータ（手書き、実際は学習で得られる）
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
  console.log(`\nFFN パラメータ（d_model=${dModel}, d_ff=${dFf}）:`);
  console.log(`  W_1 形 ${ffn.W1.length} × ${ffn.W1[0]!.length}`);
  console.log(`  b_1 形 ${ffn.b1.length}`);
  console.log(`  W_2 形 ${ffn.W2.length} × ${ffn.W2[0]!.length}`);
  console.log(`  b_2 形 ${ffn.b2.length}`);

  // Step 1: 線形変換で次元拡張
  console.log("\n▼ Step 1: 線形変換 X W_1（次元 4 → 8 に拡張）");
  const proj1 = matmul(X, ffn.W1);
  proj1.forEach((row, i) => console.log(`  [${i}] ${JSON.stringify(row)}`));

  // Step 2: バイアス加算
  console.log("\n▼ Step 2: バイアス b_1 を足す");
  const withBias1 = addBias(proj1, ffn.b1);
  withBias1.forEach((row, i) => console.log(`  [${i}] ${JSON.stringify(row)}`));

  // Step 3: ReLU
  console.log("\n▼ Step 3: ReLU で非線形性（負を 0 に）");
  const activated = relu(withBias1);
  activated.forEach((row, i) => console.log(`  [${i}] ${JSON.stringify(row)}`));

  console.log("  → 元の値で負だった成分が 0 になっている");

  // Step 4: 2 つ目の線形変換で次元を元に戻す
  console.log("\n▼ Step 4: 線形変換 W_2 で次元 8 → 4 に戻す");
  const proj2 = matmul(activated, ffn.W2);
  proj2.forEach((row, i) => console.log(`  [${i}] ${JSON.stringify(row)}`));

  // Step 5: バイアス加算
  console.log("\n▼ Step 5: バイアス b_2 を足す（最終出力）");
  const output = addBias(proj2, ffn.b2);
  output.forEach((row, i) => console.log(`  [${i}] ${JSON.stringify(row)}`));

  // 関数版で確認
  const result = feedForward(X, ffn);
  console.log("\n▼ feedForward 関数で一発実行（同じ結果のはず）");
  result.forEach((row, i) => console.log(`  [${i}] ${JSON.stringify(row)}`));

  console.log("\n=== 観察ポイント ===");
  console.log("  - 入力 X と出力は同じ形 (n × d_model)、サイズが保たれる");
  console.log("  - 中間で d_ff = 8 に拡張され、ReLU で「広い場所で非線形変換」が行われる");
  console.log("  - 最後にまた d_model に戻すので、Transformer ブロック内で再利用しやすい");

  console.log("\n=== なぜ ReLU（非線形性）が必要なのか ===");
  console.log("  - ReLU なしだと W_2(W_1 x) = (W_2 W_1) x で「ただの線形変換 1 回」と等価");
  console.log("  - 線形変換だけでは「角度を変える」「縮める」「回す」しかできない");
  console.log("  - ReLU で「方向を曲げる / 切り捨てる」ことで複雑な関係を表現できる");
  console.log("  - これがディープラーニング全般で活性化関数が必須な理由");
}

// Step 7: Multi-Head Attention
//
// Single Attention（05）の上に「複数視点で並列に attention する」を乗せたもの。
//
// 数式
//   各 Head：
//     O_i = Attention(Q W^Q_i, K W^K_i, V W^V_i)
//
//   全体：
//     O = concat(O_1, O_2, ..., O_h) W^O
//
// 流れ
//   Step 1: 入力 Q, K, V を各 Head の W で線形変換 → Head ごとの Q_i, K_i, V_i
//   Step 2: 各 Head で Scaled Dot-Product Attention を計算 → Head ごとの O_i
//   Step 3: 全 Head の出力を concat（横に結合）
//   Step 4: 最後に W^O で線形変換 → 統合された出力
//
// 形のメモ（n_q トークン, d_model = 4, h = 2 Head, d_k = d_v = d_model / h = 2）
//   入力 Q:    (n_q × d_model=4)
//   W^Q_i:     (d_model=4 × d_k=2)
//   Q_i:       (n_q × d_k=2)
//   各 Head 出力 O_i: (n_q × d_v=2)
//   concat 後: (n_q × h*d_v=4)
//   W^O:       (h*d_v=4 × d_model=4)
//   最終出力 O: (n_q × d_model=4)
//
// 直感
//   - 同じ入力でも、各 Head が「違うフィルター」をかけてから attention する
//   - 結果として、各 Head は異なる関係性（主語/述語、修飾、距離等）を捉える
//   - concat + W^O で全 Head の知識を 1 つの表現に統合する

import type { Mask, Matrix } from "./types";
import { matmul, transpose } from "./03_qk_similarity";
import { scale, softmaxRows } from "./04_scaled_dot_product";
import { scaledDotProductAttention } from "./05_attention";
import { linearProjection } from "./06_linear_projection";

// concat: 行列を「横方向」に結合
//   A = [[1, 2], [3, 4]]      (2 × 2)
//   B = [[5, 6], [7, 8]]      (2 × 2)
//   concat(A, B) = [[1, 2, 5, 6], [3, 4, 7, 8]]    (2 × 4)
export function concatColumns(matrices: Matrix[]): Matrix {
  if (matrices.length === 0) return [];
  const n = matrices[0]!.length;
  const result: Matrix = Array.from({ length: n }, () => []);
  for (let i = 0; i < n; i++) {
    for (const M of matrices) {
      result[i]!.push(...M[i]!);
    }
  }
  return result;
}

export type Head = {
  WQ: Matrix; // (d_model × d_k)
  WK: Matrix; // (d_model × d_k)
  WV: Matrix; // (d_model × d_v)
};

export function multiHeadAttention(
  Q: Matrix,
  K: Matrix,
  V: Matrix,
  heads: Head[],
  WO: Matrix,
  mask?: Mask,
): Matrix {
  // 各 Head で Q, K, V を射影してから attention
  // mask は全 Head で共通（同じ位置を見ない / 見るは Head によらず決まる）
  const headOutputs = heads.map((h) => {
    const Qi = linearProjection(Q, h.WQ);
    const Ki = linearProjection(K, h.WK);
    const Vi = linearProjection(V, h.WV);
    return scaledDotProductAttention(Qi, Ki, Vi, mask);
  });

  // concat（横結合）
  const concatenated = concatColumns(headOutputs);

  // W^O で最終射影
  return linearProjection(concatenated, WO);
}

// === 動作確認 ===
// 直接実行: bun run transformer/07_multi_head_attention.ts
if (import.meta.main) {
  console.log("=== Multi-Head Attention の動作確認（計算過程を全部出す）===\n");

  // 入力: 3 つのトークン、d_model = 4
  // self-attention 想定: Q = K = V = 同じ入力
  const x: Matrix = [
    [1, 0, 1, 0], // token 1: 「猫」
    [0, 1, 1, 0], // token 2: 「犬」
    [1, 1, 0, 0], // token 3: 「ペット」
  ];
  console.log(`入力 x（3 × 4） = ${JSON.stringify(x)}`);
  console.log("self-attention なので Q = K = V = x\n");

  // 2 Head 構成（d_model=4, h=2, d_k=d_v=2）
  // Head 0: 入力の前半 2 次元に注目
  const head0: Head = {
    WQ: [[1, 0], [0, 1], [0, 0], [0, 0]],
    WK: [[1, 0], [0, 1], [0, 0], [0, 0]],
    WV: [[1, 0], [0, 1], [0, 0], [0, 0]],
  };
  // Head 1: 入力の後半 2 次元に注目
  const head1: Head = {
    WQ: [[0, 0], [0, 0], [1, 0], [0, 1]],
    WK: [[0, 0], [0, 0], [1, 0], [0, 1]],
    WV: [[0, 0], [0, 0], [1, 0], [0, 1]],
  };
  const heads = [head0, head1];

  // 簡略化のため W^O は単位行列（恒等射影）
  // 実際の Transformer では学習対象
  const WO: Matrix = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ];

  for (let h = 0; h < heads.length; h++) {
    const head = heads[h]!;
    console.log(`▼ Head ${h}`);
    console.log(`  W^Q = ${JSON.stringify(head.WQ)}`);
    console.log(`  W^K = ${JSON.stringify(head.WK)}`);
    console.log(`  W^V = ${JSON.stringify(head.WV)}`);

    // Step 1: 線形変換
    const Qi = linearProjection(x, head.WQ);
    const Ki = linearProjection(x, head.WK);
    const Vi = linearProjection(x, head.WV);
    console.log(`\n  Step 1: 線形変換`);
    console.log(`    Q_${h} = x W^Q = ${JSON.stringify(Qi)}`);
    console.log(`    K_${h} = x W^K = ${JSON.stringify(Ki)}`);
    console.log(`    V_${h} = x W^V = ${JSON.stringify(Vi)}`);

    // Step 2: Scaled Dot-Product Attention
    const dK = Ki[0]!.length;
    const rawScores = matmul(Qi, transpose(Ki));
    console.log(`\n  Step 2: 生スコア Q_${h} K_${h}^T`);
    console.log(`    ${JSON.stringify(rawScores)}`);

    const scaledScores = scale(rawScores, 1 / Math.sqrt(dK));
    console.log(`\n  Step 3: / √${dK} = ${Math.sqrt(dK).toFixed(4)}`);
    console.log(
      `    ${JSON.stringify(scaledScores.map((row) => row.map((v) => Number(v.toFixed(4)))))}`,
    );

    const weights = softmaxRows(scaledScores);
    console.log(`\n  Step 4: softmax で重み化（注意マップ）`);
    console.log(
      `    ${JSON.stringify(weights.map((row) => row.map((v) => Number(v.toFixed(4)))))}`,
    );
    console.log(`    各行 i は「token i が他の各 token にどれだけ注目するか」`);

    const out = matmul(weights, Vi);
    console.log(`\n  Step 5: 重み × V_${h} → Head ${h} の出力`);
    console.log(
      `    O_${h} = ${JSON.stringify(out.map((row) => row.map((v) => Number(v.toFixed(4)))))}`,
    );

    console.log("");
  }

  // Multi-Head 全体
  const headOutputs = heads.map((h) => {
    const Qi = linearProjection(x, h.WQ);
    const Ki = linearProjection(x, h.WK);
    const Vi = linearProjection(x, h.WV);
    return scaledDotProductAttention(Qi, Ki, Vi);
  });

  console.log("▼ Step 6: 全 Head の出力を concat（横結合）");
  const concatenated = concatColumns(headOutputs);
  console.log(
    `  各 Head 出力（2 列ずつ）を横に並べる → 形 ${concatenated.length} × ${concatenated[0]!.length}`,
  );
  console.log(
    `  concat = ${JSON.stringify(concatenated.map((row) => row.map((v) => Number(v.toFixed(4)))))}`,
  );

  console.log("\n▼ Step 7: W^O で最終射影");
  const finalOutput = linearProjection(concatenated, WO);
  console.log(`  W^O = ${JSON.stringify(WO)}`);
  console.log(
    `  最終出力 = ${JSON.stringify(finalOutput.map((row) => row.map((v) => Number(v.toFixed(4)))))}`,
  );

  // 直接 multiHeadAttention を呼ぶ版でも同じ結果
  const result = multiHeadAttention(x, x, x, heads, WO);
  console.log(`\n  関数経由の結果 = ${JSON.stringify(result.map((row) => row.map((v) => Number(v.toFixed(4)))))}`);

  console.log("\n=== 観察ポイント ===");
  console.log("  - 同じ入力 x でも、Head 0 と Head 1 で異なる注意マップになる");
  console.log("    Head 0 は前半 2 次元、Head 1 は後半 2 次元を見ているので");
  console.log("  - concat で 2 つの視点を 1 つの表現に並べる（横方向に結合）");
  console.log("  - W^O で混ぜ合わせて統合（今回は単位行列なので concat そのまま）");
  console.log("  - 実際の LLM ではこれが 数十〜100 Head 並列で動く");
  console.log("    各 Head が「括弧対応」「固有名詞追跡」など別々の役割を学習する");

  console.log("\n=== Single Attention との関係 ===");
  console.log("  Single Attention（05）: 1 つの注意分布だけ");
  console.log("  Multi-Head（07）:       同じ入力から複数の注意分布を並列に作り、統合する");
  console.log("  違いは「線形変換 W で視点を分ける」「複数 Head 並列」「concat + W^O で統合」");
  console.log("  Attention 自体（softmax(Q K^T/√d_k) V）は完全に同じ");
}

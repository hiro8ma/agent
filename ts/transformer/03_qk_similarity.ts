// Step 3: Query × 複数 Keys の類似度（行列積）
//
// Step 2 では「q と 1 つの k」の内積を取った。
// 実際の Attention では「複数 Key と一気に類似度を計算したい」。
//
// 数式
//   q が 1 つ、K が n_k 個あるとき：
//   q K^T = (⟨q, k_1⟩, ⟨q, k_2⟩, ..., ⟨q, k_{n_k}⟩)
//
//   さらに Q 自体も複数（n_q 個）あれば、まとめて行列積で計算できる：
//   Q K^T : (n_q × d_k) × (d_k × n_k) → (n_q × n_k)
//
// 直感
//   各セル (i, j) は「Query i と Key j の類似度（生スコア）」。
//
// 形のメモ
//   K   = (n_k × d_k)   ← Key を行に並べる
//   K^T = (d_k × n_k)   ← 転置すると Key を列に並べる形
//   Q K^T = (n_q × n_k) ← Query 数 × Key 数の類似度マトリクス

import type { Matrix } from "./types";
import { dot } from "./02_dot_product";

// 転置: A の行と列を入れ替える
//   A   = [[a11, a12], [a21, a22]]   形 (2, 2)
//   A^T = [[a11, a21], [a12, a22]]   形 (2, 2)
//
//   非正方の場合
//   A   = [[1, 2, 3], [4, 5, 6]]     形 (2, 3)
//   A^T = [[1, 4], [2, 5], [3, 6]]   形 (3, 2)
export function transpose(A: Matrix): Matrix {
  const rows = A.length;
  const cols = A[0]?.length ?? 0;
  const result: Matrix = Array.from({ length: cols }, () => Array(rows).fill(0));
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      result[j]![i] = A[i]![j]!;
    }
  }
  return result;
}

// 行列積: (n × d) × (d × m) → (n × m)
// 結果の (i, j) 成分 = A の i 行目と B の j 列目の内積
export function matmul(A: Matrix, B: Matrix): Matrix {
  const n = A.length;
  const d = A[0]?.length ?? 0;
  const dB = B.length;
  const m = B[0]?.length ?? 0;
  if (d !== dB) {
    throw new Error(`matmul shape mismatch: A is (${n}, ${d}), B is (${dB}, ${m})`);
  }

  // B の各列を取り出しやすくするため、転置して B^T として扱う
  // 元: B[k][j] が B の (k, j) 成分
  // 転置後: B_T[j][k] が B の (k, j) 成分
  const B_T = transpose(B);

  const result: Matrix = Array.from({ length: n }, () => Array(m).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      // 結果の (i, j) = A の i 行目 と B の j 列目（= B_T の j 行目）の内積
      result[i]![j] = dot(A[i]!, B_T[j]!);
    }
  }
  return result;
}

// === 動作確認 ===
// 直接実行: bun run transformer/03_qk_similarity.ts
if (import.meta.main) {
  console.log("=== Query × Keys の類似度（行列積）===\n");

  // Query 1 つ、Key 3 つ、d_k = 2
  const Q: Matrix = [
    [1, 0], // Query 1: 右向き「猫っぽい」
  ];
  const K: Matrix = [
    [1, 0], // Key 1: 「猫」     ← Q と同じ向き
    [0, 1], // Key 2: 「犬」     ← Q と直交
    [-1, 0], // Key 3: 「魚」    ← Q と反対
  ];

  console.log(`Q（${Q.length} × ${Q[0]!.length}）= ${JSON.stringify(Q)}`);
  console.log(`K（${K.length} × ${K[0]!.length}）= ${JSON.stringify(K)}`);

  console.log("\n--- 転置 ---");
  const KT = transpose(K);
  console.log(`K^T（${KT.length} × ${KT[0]!.length}）= ${JSON.stringify(KT)}`);

  console.log("\n--- 行列積 Q K^T ---");
  const scores = matmul(Q, KT);
  console.log(`Q K^T（${scores.length} × ${scores[0]!.length}）= ${JSON.stringify(scores)}`);

  console.log("\n観察ポイント");
  console.log("  - 結果は (1 × 3) 行列、各セルが「Query i と Key j の類似度」");
  console.log("  - [1, 0, -1] になる: Q と同じ向き(1) / 直交(0) / 反対(-1)");
  console.log("  - これが「生スコア」。次の Step でスケーリング + softmax で重み化する");

  console.log("\n--- 複数 Query の場合 ---");
  const Q2: Matrix = [
    [1, 0],
    [0, 1],
  ];
  const scores2 = matmul(Q2, transpose(K));
  console.log(`Q（2 × 2）= ${JSON.stringify(Q2)}`);
  console.log(`Q K^T（2 × 3）= ${JSON.stringify(scores2)}`);
  console.log("  - 1 行目: Q1=[1,0] 視点での K1/K2/K3 の類似度");
  console.log("  - 2 行目: Q2=[0,1] 視点での K1/K2/K3 の類似度");
}

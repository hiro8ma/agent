// Step 2: 内積（dot product）
//
// 役割
//   2 つのベクトルが「どれだけ似ているか」を 1 つの数値で表す。
//
// 数式
//   ⟨q, k⟩ = Σ_i q_i * k_i
//
//   各次元の値を掛け算して、全部足す。それだけ。
//
// 直感
//   - ベクトルの向きが近いほど内積は大きい
//   - ベクトルが直交（90°）すると内積は 0
//   - ベクトルが反対向きだと内積は負
//
//   つまり「類似度」のような数値が出る。
//
// Attention での意味
//   Query = 「何を探したいか」
//   Key   = 「何を持っているか」
//   ⟨Q, K⟩ が大きい → Query と Key の向きが近い → 注目すべき
//
// 例
//   q = [1, 0]   （右向き）
//   k1 = [1, 0]  （右向き、q と同じ）       → ⟨q, k1⟩ = 1*1 + 0*0 = 1
//   k2 = [0, 1]  （上向き、q と直交）       → ⟨q, k2⟩ = 1*0 + 0*1 = 0
//   k3 = [-1, 0] （左向き、q と反対）       → ⟨q, k3⟩ = 1*-1 + 0*0 = -1

import type { Vector } from "./types";

export function dot(a: Vector, b: Vector): number {
  if (a.length !== b.length) {
    throw new Error(`vector length mismatch: ${a.length} vs ${b.length}`);
  }
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i]! * b[i]!;
  }
  return sum;
}

// === 動作確認 ===
// 直接実行: bun run transformer/02_dot_product.ts
if (import.meta.main) {
  console.log("=== 内積の動作確認 ===\n");

  const q: Vector = [1, 0]; // Query: 右向き

  const examples: { name: string; k: Vector }[] = [
    { name: "k1（同じ向き）", k: [1, 0] },
    { name: "k2（直交）", k: [0, 1] },
    { name: "k3（反対向き）", k: [-1, 0] },
    { name: "k4（斜め、似ている方）", k: [0.7, 0.7] },
    { name: "k5（より大きい同じ向き）", k: [3, 0] },
  ];

  console.log(`Query q = [${q.join(", ")}]\n`);
  for (const { name, k } of examples) {
    const score = dot(q, k);
    console.log(`${name}: k = [${k.join(", ")}] → ⟨q, k⟩ = ${score}`);
  }

  console.log("\n観察ポイント");
  console.log("  - 同じ向き（k1, k4, k5）ほど内積が大きい");
  console.log("  - 直交（k2）すると 0、反対（k3）だと負");
  console.log("  - 大きさも影響する（k5 のほうが k1 より内積が大きい）");
  console.log("    → これが後で √d_k で割る理由（次元 / 大きさで補正したい）");
}

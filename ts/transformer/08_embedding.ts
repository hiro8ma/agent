// Step 8: Embedding（埋め込み）
//
// 役割
//   token ID（ただの番号）を「意味を持つベクトル」に変換する。
//
// 数式
//   embed(id) = E[id]
//
//   E: 埋め込みテーブル、形 (vocab_size × d_model)
//   id: token ID（整数）
//   出力: d_model 次元のベクトル
//
// 直感
//   - トークン化直後は token ID が並んでいるだけ（[2, 1, 3, 12, 9, 5] のような番号列）
//   - 意味的な情報はゼロ。どの ID が「猫」か「犬」かは ID だけでは分からない
//   - Embedding テーブルから ID 番目の行を取り出すことで「ベクトル空間内の点」になる
//   - 学習を通じて「意味の近い単語は近い方向のベクトル」になるよう調整される
//
// 「角度ゲーム」での位置づけ
//   ここで初めて token が「方向を持つベクトル」になる。
//   この時点で「意味の近さ = ベクトルの向きの近さ」が成立する。
//   後段の Attention は、この向きを使って関係性を測る。
//
// 実装上の注意
//   - 実際の Transformer では E は学習で得られる
//   - ここではデモ用に「意味が近い単語が近い方向」を満たす E を手書きする

import type { Matrix } from "./types";
import { dot } from "./02_dot_product";

// 埋め込みテーブル: 形 (vocab_size × d_model)
export type EmbeddingTable = Matrix;

// token ID 列を埋め込みベクトル列に変換
//   tokenIds: [1, 2, 3] のような ID 配列
//   table[i] = i 番目の token のベクトル
//   出力: 形 (n_tokens × d_model)
export function embed(tokenIds: number[], table: EmbeddingTable): Matrix {
  return tokenIds.map((id) => {
    if (id < 0 || id >= table.length) {
      throw new Error(`token id ${id} out of range [0, ${table.length})`);
    }
    return [...table[id]!]; // shallow copy で参照共有を避ける
  });
}

// === 動作確認 ===
// 直接実行: bun run transformer/08_embedding.ts
if (import.meta.main) {
  console.log("=== Embedding（埋め込み）の動作確認 ===\n");

  // 仮想的な vocabulary（d_model = 4）
  // 意味の近い単語が近い方向のベクトルになるよう手書きで作る
  const vocab = ["<pad>", "猫", "は", "動物", "だ", "犬", "魚"];
  const E: EmbeddingTable = [
    [0, 0, 0, 0],     // 0: <pad>
    [1.0, 0.0, 0.5, 0.0], // 1: 猫     ← 動物・犬と近い方向
    [0.0, 0.0, 0.0, 1.0], // 2: は     ← 機能語、別方向
    [0.8, 0.0, 0.6, 0.0], // 3: 動物   ← 猫・犬と近い
    [0.0, 0.0, 0.0, 0.9], // 4: だ     ← 機能語、は と近い
    [0.9, 0.1, 0.5, 0.0], // 5: 犬     ← 猫・動物と近い
    [0.5, -0.5, 0.3, 0.0], // 6: 魚    ← 動物だが少し違う方向
  ];

  console.log("vocabulary と Embedding テーブル E:");
  for (let i = 0; i < vocab.length; i++) {
    console.log(`  E[${i}] (${vocab[i]}) = ${JSON.stringify(E[i])}`);
  }

  // 入力例: 「猫 は 動物 だ」
  const sentence = "猫 は 動物 だ";
  const tokenIds = [1, 2, 3, 4];
  console.log(`\n▼ 入力文 "${sentence}" → token IDs ${JSON.stringify(tokenIds)}`);

  const embeddings = embed(tokenIds, E);
  console.log(`\n▼ Embedding 結果（${tokenIds.length} × ${E[0]!.length}）`);
  for (let i = 0; i < tokenIds.length; i++) {
    console.log(`  ${vocab[tokenIds[i]!]} (id=${tokenIds[i]}) → ${JSON.stringify(embeddings[i])}`);
  }

  console.log("\n▼ 「意味の近さ = 内積の大きさ」の確認");
  console.log("  各 token ペアの内積（向きの近さ）:");
  const pairs: [number, number][] = [
    [1, 3], // 猫 vs 動物
    [1, 5], // 猫 vs 犬
    [3, 5], // 動物 vs 犬
    [1, 6], // 猫 vs 魚
    [1, 2], // 猫 vs は
    [2, 4], // は vs だ
  ];
  for (const [a, b] of pairs) {
    const score = dot(E[a]!, E[b]!);
    console.log(
      `  ⟨${vocab[a]!.padEnd(4)}, ${vocab[b]!.padEnd(4)}⟩ = ${score.toFixed(4)}`,
    );
  }

  console.log("\n=== 観察ポイント ===");
  console.log("  - 「猫 vs 動物」「猫 vs 犬」「動物 vs 犬」の内積が大きい（意味が近い）");
  console.log("  - 「猫 vs は」「猫 vs 魚」の内積は小さい（意味が遠い、または違う方向）");
  console.log("  - 「は vs だ」は両方機能語なので互いに近い（同じく内積大）");
  console.log("  - これが「意味の近さ = ベクトルの向きの近さ」の実装");

  console.log("\n=== Embedding は線形変換の特殊形 ===");
  console.log("  Embedding は技術的には『one-hot ベクトル × E』と等価");
  console.log("  one-hot([0,1,0,0,0,0,0]) × E = E[1]（「猫」のベクトル）");
  console.log("  → lookup table として実装するのが効率的だが、本質は線形変換（06）と同じ");
}

// Step 13: Transformer Encoder（N 層スタック）
//
// 12 で作った Transformer Block を N 層スタックして「完全な Encoder」を構成する。
// これが BERT 等の Encoder-only モデルの本体構造。
// GPT 系は Decoder-only だが、構造は本ファイルとほぼ同じ
// （違いは causal mask、後述）。
//
// 数式
//   入力: token IDs [t_0, t_1, ..., t_{n-1}]
//
//   Step A: Embedding
//     x_emb = embed(tokenIds)                 形 (n × d_model)
//
//   Step B: + Positional Encoding
//     x_0 = x_emb + PE                        形 (n × d_model)
//
//   Step C: N 層の Transformer Block
//     x_1 = TransformerBlock_1(x_0)
//     x_2 = TransformerBlock_2(x_1)
//     ...
//     x_N = TransformerBlock_N(x_{N-1})
//
//   出力: x_N                                 形 (n × d_model)
//
// 各層で何が起きているか
//   1 層目: 単語レベルの局所的な関係性を学習
//   2-3 層目: 句や節レベルの関係性
//   中間層: 構文構造の理解
//   後半層: 意味・文脈の高度な抽象化
//   最終層: タスク依存の表現（分類用、検索用など）
//
// 実際の Encoder モデルの規模
//   BERT-base:  N = 12, d_model = 768
//   BERT-large: N = 24, d_model = 1024
//   T5-large:   N = 24, d_model = 1024
//   各モデルで N とパラメータ数が違う

import type { Matrix } from "./types";
import { embed } from "./08_embedding";
import type { EmbeddingTable } from "./08_embedding";
import { addMatrix, positionalEncoding } from "./09_positional_encoding";
import { transformerBlock } from "./12_transformer_block";
import type { TransformerBlockConfig } from "./12_transformer_block";

export type EncoderConfig = {
  embeddingTable: EmbeddingTable;
  blocks: TransformerBlockConfig[]; // 各層のパラメータ
};

// Encoder: token IDs → 文脈化された特徴ベクトル列
export function encode(tokenIds: number[], config: EncoderConfig): Matrix {
  // Step A: Embedding（token ID → ベクトル）
  const embeddings = embed(tokenIds, config.embeddingTable);

  // Step B: + Positional Encoding
  const dModel = embeddings[0]!.length;
  const PE = positionalEncoding(tokenIds.length, dModel);
  let x = addMatrix(embeddings, PE);

  // Step C: Transformer Block × N
  for (const block of config.blocks) {
    x = transformerBlock(x, block);
  }

  return x;
}

// 各層の中間出力も返す（学習用 / 解析用）
export function encodeWithTrace(
  tokenIds: number[],
  config: EncoderConfig,
): { layers: Matrix[]; output: Matrix } {
  const embeddings = embed(tokenIds, config.embeddingTable);
  const dModel = embeddings[0]!.length;
  const PE = positionalEncoding(tokenIds.length, dModel);

  const layers: Matrix[] = [];
  let x = addMatrix(embeddings, PE);
  layers.push(x.map((row) => [...row])); // 入力を「層 0」として保存

  for (const block of config.blocks) {
    x = transformerBlock(x, block);
    layers.push(x.map((row) => [...row]));
  }

  return { layers, output: x };
}

// === 動作確認 ===
// 直接実行: bun run transformer/13_encoder.ts
if (import.meta.main) {
  console.log("=== Transformer Encoder（N 層スタック）の動作確認 ===\n");

  const dModel = 4;
  const vocab = ["<pad>", "猫", "は", "動物", "だ", "犬", "魚"];
  const E: EmbeddingTable = [
    [0, 0, 0, 0],
    [1.0, 0.0, 0.5, 0.0], // 猫
    [0.0, 0.0, 0.0, 1.0], // は
    [0.8, 0.0, 0.6, 0.0], // 動物
    [0.0, 0.0, 0.0, 0.9], // だ
    [0.9, 0.1, 0.5, 0.0], // 犬
    [0.5, -0.5, 0.3, 0.0], // 魚
  ];

  // 各層で同じパラメータを使う（実際は層ごとに違う、学習で得られる）
  // 簡略化のため共通パラメータの単純コピーで N 層を作る
  const blockConfig: TransformerBlockConfig = {
    heads: [
      {
        WQ: [[1, 0], [0, 1], [0, 0], [0, 0]],
        WK: [[1, 0], [0, 1], [0, 0], [0, 0]],
        WV: [[1, 0], [0, 1], [0, 0], [0, 0]],
      },
      {
        WQ: [[0, 0], [0, 0], [1, 0], [0, 1]],
        WK: [[0, 0], [0, 0], [1, 0], [0, 1]],
        WV: [[0, 0], [0, 0], [1, 0], [0, 1]],
      },
    ],
    WO: [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ],
    ffn: {
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
    },
  };

  const N = 3; // 3 層スタック（実際の BERT は 12-24 層）
  const config: EncoderConfig = {
    embeddingTable: E,
    blocks: Array.from({ length: N }, () => blockConfig),
  };

  // 入力: 「猫 は 動物 だ」
  const tokenIds = [1, 2, 3, 4];
  const sentence = tokenIds.map((id) => vocab[id]).join(" ");

  console.log(`▼ 入力文: "${sentence}"`);
  console.log(`  token IDs: ${JSON.stringify(tokenIds)}`);
  console.log(`  Encoder: ${N} 層スタック`);

  // 各層の中間出力を取りながら実行
  const { layers, output } = encodeWithTrace(tokenIds, config);

  console.log(`\n▼ 各層の出力推移（${tokenIds.length} × ${dModel}）`);
  for (let l = 0; l < layers.length; l++) {
    const label =
      l === 0 ? "層 0（入力 = embedding + PE）" : `層 ${l}（Block ${l} 通過後）`;
    console.log(`\n  [${label}]`);
    for (let i = 0; i < layers[l]!.length; i++) {
      const formatted = layers[l]![i]!.map((v) => v.toFixed(4)).join(", ");
      console.log(`    ${vocab[tokenIds[i]!]?.padEnd(4)} → [${formatted}]`);
    }
  }

  console.log(`\n▼ 最終出力（Encoder 出力、形 ${output.length} × ${output[0]!.length}）`);
  output.forEach((row, i) =>
    console.log(
      `  ${vocab[tokenIds[i]!]?.padEnd(4)} → ${JSON.stringify(row.map((v) => Number(v.toFixed(4))))}`,
    ),
  );

  console.log("\n=== 観察ポイント ===");
  console.log("  - 入力（層 0）と各層の出力で、各 token のベクトルが少しずつ変化");
  console.log("  - 層が深くなるごとに「文脈情報」が混ざっていく");
  console.log("  - 最終層では各 token が「文全体の文脈を持った表現」になる");
  console.log(`  - 形は (n × d_model) で全層通しても変わらない（${tokenIds.length} × ${dModel}）`);
  console.log("    → だから N 層スタックできる（出力をそのまま次層の入力にできる）");

  console.log("\n=== Encoder の出力の使い道 ===");
  console.log("  Encoder の出力 (n × d_model) はタスクによって使い方が変わる");
  console.log("  - BERT 風 NSP: [CLS] token のベクトルだけを使う");
  console.log("  - 文ベクトル化: 全 token の平均を取る（mean pooling）");
  console.log("  - 単語ラベリング: 各 token のベクトルから個別にラベル予測");
  console.log("  - 検索用 embedding: そのまま「文の意味ベクトル」として使う");

  console.log("\n=== Encoder と Decoder の違い（補足） ===");
  console.log("  Encoder: 入力全体を一度に処理、双方向の文脈を見れる");
  console.log("  Decoder: 1 token ずつ生成、causal mask で「未来の token」を見れない");
  console.log("    GPT / Claude は Decoder-only、本実装に causal mask を加えれば LLM の本体構造");
  console.log("    causal mask = Attention の重みで「i 番目の token は j > i を見ない」よう 0 にする");

  console.log("\n=== 全パイプラインの再確認 ===");
  console.log("  token IDs");
  console.log("    ↓ Embedding（08）           token → 意味ベクトル");
  console.log("    ↓ + Positional Encoding（09） + 位置ベクトル");
  console.log("    ↓ Transformer Block × N      MHA + Residual+LN + FFN + Residual+LN");
  console.log("    ↓                            （内部で 07 / 10 / 11 を使う）");
  console.log("  文脈化された特徴ベクトル列     ← Encoder の出力");
}

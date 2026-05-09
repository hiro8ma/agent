// Step 9: Positional Encoding（位置エンコーディング）
//
// 役割
//   Transformer は本来「順序を知らない」（Attention は集合演算で位置情報を持たない）。
//   そこで token の埋め込みベクトルに「位置情報を持つベクトル」を加算することで、
//   位置の情報を後付けする。
//
// 数式（Vaswani et al. 2017 のオリジナル sin/cos 版）
//   PE(i, 2k)   = sin(i / 10000^{2k/d_model})
//   PE(i, 2k+1) = cos(i / 10000^{2k/d_model})
//
//   i:        位置（0, 1, 2, ...）
//   k:        次元の index（0, 1, 2, ..., d_model/2）
//   d_model:  モデルの次元
//
// 偶数次元には sin、奇数次元には cos を交互に使う。
// 周期が d_model 全体で 2π → 10000·2π まで指数的に変化するので、
// 「短い周期で細かい位置情報」と「長い周期で大きな位置情報」を同時に持てる。
//
// 「角度ゲーム」での位置づけ
//   - 「位置の近さ」も内積で測れるよう設計されている
//   - ⟨PE_i, PE_j⟩ は |i - j| が小さいほど大きい
//   - つまり Attention は「意味の近さ（embedding 由来）」と
//     「位置の近さ（PE 由来）」を同じ仕組み（内積）で扱える
//
// 加算で良い理由
//   embedding と PE を加算すると、線形変換 W^Q / W^K / W^V を通したあとも
//   「意味成分」と「位置成分」を（学習で）分離可能な構造を保てる。

import type { Matrix } from "./types";
import { dot } from "./02_dot_product";

// 位置エンコーディング行列を生成
//   maxLen: 最大シーケンス長（位置の数）
//   dModel: 次元数
//   出力: 形 (maxLen × dModel)
export function positionalEncoding(maxLen: number, dModel: number): Matrix {
  const PE: Matrix = [];
  for (let i = 0; i < maxLen; i++) {
    const row: number[] = [];
    for (let k = 0; k < dModel; k++) {
      // 偶数 k は sin、奇数 k は cos を使う
      const dimPair = Math.floor(k / 2); // 0, 0, 1, 1, 2, 2, ...
      const exponent = (2 * dimPair) / dModel;
      const angle = i / Math.pow(10000, exponent);
      row.push(k % 2 === 0 ? Math.sin(angle) : Math.cos(angle));
    }
    PE.push(row);
  }
  return PE;
}

// 行列の要素ごとの加算（embedding + PE で使う）
export function addMatrix(A: Matrix, B: Matrix): Matrix {
  if (A.length !== B.length || A[0]!.length !== B[0]!.length) {
    throw new Error(
      `shape mismatch: ${A.length}x${A[0]!.length} vs ${B.length}x${B[0]!.length}`,
    );
  }
  return A.map((row, i) => row.map((v, j) => v + B[i]![j]!));
}

// === 動作確認 ===
// 直接実行: bun run transformer/09_positional_encoding.ts
if (import.meta.main) {
  console.log("=== Positional Encoding（位置エンコーディング）の動作確認 ===\n");

  const maxLen = 8;
  const dModel = 4;
  console.log(`▼ PE 行列を生成（maxLen=${maxLen}, d_model=${dModel}）`);
  const PE = positionalEncoding(maxLen, dModel);

  console.log("\nPE 行列（各行が位置 i のベクトル）:");
  for (let i = 0; i < PE.length; i++) {
    const formatted = PE[i]!.map((v) => v.toFixed(4)).join(", ");
    console.log(`  PE[${i}] = [${formatted}]`);
  }

  console.log("\n▼ 「位置の近さ = 内積の大きさ」の確認");
  console.log("  ⟨PE[0], PE[j]⟩ を j について計算:");
  for (let j = 0; j < maxLen; j++) {
    const score = dot(PE[0]!, PE[j]!);
    const distance = j;
    const bar = "█".repeat(Math.max(0, Math.round(score * 10)));
    console.log(
      `  距離 ${distance}: ⟨PE[0], PE[${j}]⟩ = ${score.toFixed(4).padStart(8)}  ${bar}`,
    );
  }
  console.log("  → 距離が大きくなるほど内積が小さくなる（位置が遠ざかる）");

  console.log("\n▼ embedding + PE の例（実際の Transformer の入力）");
  // 仮想的な embedding（08 で作った猫のベクトルを 4 つ並べる）
  const embeddings: Matrix = [
    [1.0, 0.0, 0.5, 0.0], // 猫
    [0.0, 0.0, 0.0, 1.0], // は
    [0.8, 0.0, 0.6, 0.0], // 動物
    [0.0, 0.0, 0.0, 0.9], // だ
  ];
  console.log("  embeddings（猫・は・動物・だ）:");
  embeddings.forEach((row, i) => console.log(`    [${i}] ${JSON.stringify(row)}`));

  console.log("  PE[0..3]:");
  PE.slice(0, 4).forEach((row, i) =>
    console.log(`    [${i}] ${JSON.stringify(row.map((v) => Number(v.toFixed(4))))}`),
  );

  const sumEmbPE = addMatrix(embeddings, PE.slice(0, 4));
  console.log("  embedding + PE:");
  sumEmbPE.forEach((row, i) =>
    console.log(`    [${i}] ${JSON.stringify(row.map((v) => Number(v.toFixed(4))))}`),
  );

  console.log("\n=== 観察ポイント ===");
  console.log("  - PE の各行は位置 i ごとに違うベクトル");
  console.log("  - ⟨PE[0], PE[j]⟩ は j が大きくなるほど小さくなる（位置の近さ）");
  console.log("  - embedding + PE で「意味 + 位置」を 1 つのベクトルに混ぜる");
  console.log("  - 後段の Attention は内積で「意味の近さ + 位置の近さ」を同時に見る");

  console.log("\n=== なぜ sin/cos なのか（補足） ===");
  console.log("  - 周期関数なので「相対位置（i-j）」が内積に綺麗に反映される");
  console.log("  - 学習なしに固定値で計算できる（パラメータ削減）");
  console.log("  - 学習されない代わりに、長いシーケンス（学習時より長い文）にも外挿できる");
  console.log("  - 最近のモデルは Rotary PE / ALiBi など改良版を使う（同じ「角度で位置を表す」哲学）");
}

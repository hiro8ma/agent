// Step 0b: BPE Tokenizer（Byte Pair Encoding）
//
// 役割
//   00_byte_tokenizer.ts は「列が長くなる」欠点があった（"世" 1 文字で 3 token）。
//   BPE は「頻出する隣接ペアを 1 つの新 token にまとめる」ことで列を短くする。
//   GPT-2 / GPT-4 / Llama などの実用 tokenizer の中核アルゴリズム。
//
// アルゴリズム（書籍に忠実）
//   1. 初期化   : テキストをバイト列にする。語彙は 0..255 の 256 個から始める
//   2. ペア集計 : 隣接する token ペアの出現回数を数える（count_pairs）
//   3. 最頻マージ: 最も多いペアを「新しい token ID」に置き換える（merge）
//   4. 反復     : 目標の vocab_size になるまで 2-3 を繰り返す（train_bpe）
//
//   学習で得られるのは「merge ルールの順序付きリスト」。
//   encode はこのルールを学習時と同じ順序で適用し、decode は逆引きする。
//
// 決定的なタイブレーク（再現性のため）
//   同じ出現回数のペアが複数ある場合、結果が実行ごとに変わると再現できない。
//   そこで「頻度が大きい順 → 同点なら token ID が小さいペア順」で一意に決める。
//
// 「角度ゲーム」での位置づけ
//   ここもまだ整数列。ただし byte tokenizer より短く、意味の塊（"the" 等）に近い ID 列になる。
//   この ID 列が 08_embedding.ts でベクトルになる。

import type { TokenId } from "./types";
import { ByteTokenizer } from "./00_byte_tokenizer";

// merge ルール: 「ペア (a, b) を新 ID へまとめる」を表す
export interface MergeRule {
  pair: [TokenId, TokenId];
  newId: TokenId;
}

// ペアキー（a, b）を 1 つの string にして Map のキーにする
function pairKey(a: TokenId, b: TokenId): string {
  return `${a},${b}`;
}

// 隣接ペアの出現回数を数える
//   [1, 2, 2, 3] → {"1,2":1, "2,2":1, "2,3":1}
export function countPairs(ids: TokenId[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (let i = 0; i < ids.length - 1; i++) {
    const key = pairKey(ids[i]!, ids[i + 1]!);
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return counts;
}

// 列の中のペア (a, b) をすべて newId に置き換える
//   [1, 2, 2, 3], pair=(2,3), newId=256 → [1, 2, 256]
export function merge(
  ids: TokenId[],
  pair: [TokenId, TokenId],
  newId: TokenId,
): TokenId[] {
  const [a, b] = pair;
  const out: TokenId[] = [];
  let i = 0;
  while (i < ids.length) {
    if (i < ids.length - 1 && ids[i] === a && ids[i + 1] === b) {
      out.push(newId);
      i += 2; // ペアを消費
    } else {
      out.push(ids[i]!);
      i += 1;
    }
  }
  return out;
}

// 最頻ペアを決定的に選ぶ
//   頻度が最大、同点なら (a, b) が辞書順で小さい方を返す。null は「ペアが無い」
function selectBestPair(
  counts: Map<string, number>,
): { pair: [TokenId, TokenId]; count: number } | null {
  let best: { pair: [TokenId, TokenId]; count: number } | null = null;
  for (const [key, count] of counts) {
    const [a, b] = key.split(",").map(Number) as [TokenId, TokenId];
    if (best === null || count > best.count) {
      best = { pair: [a, b], count };
    } else if (count === best.count) {
      // タイブレーク: a 小 → b 小 の順
      const [ba, bb] = best.pair;
      if (a < ba || (a === ba && b < bb)) {
        best = { pair: [a, b], count };
      }
    }
  }
  return best;
}

export class BPETokenizer {
  private byteTok = new ByteTokenizer();
  merges: MergeRule[] = [];
  vocabSize = 256;

  // BPE 学習: テキストから merge ルールを獲得する
  //   targetVocabSize: 256 を超えた分が merge 回数（例 300 なら 44 回マージ）
  train(text: string, targetVocabSize: number): void {
    if (targetVocabSize < 256) {
      throw new Error("targetVocabSize must be >= 256 (byte vocab)");
    }
    let ids = this.byteTok.encode(text);
    this.merges = [];
    let nextId = 256;

    const numMerges = targetVocabSize - 256;
    for (let step = 0; step < numMerges; step++) {
      const counts = countPairs(ids);
      const best = selectBestPair(counts);
      if (best === null || best.count < 2) {
        break; // これ以上まとめる価値のあるペアが無い
      }
      const newId = nextId++;
      this.merges.push({ pair: best.pair, newId });
      ids = merge(ids, best.pair, newId);
    }
    this.vocabSize = nextId;
  }

  // encode: 学習した merge を学習時と同じ順序で適用
  encode(text: string): TokenId[] {
    let ids = this.byteTok.encode(text);
    for (const rule of this.merges) {
      ids = merge(ids, rule.pair, rule.newId);
    }
    return ids;
  }

  // decode: 新 token を再帰的にバイト列へ展開してから UTF-8 復元
  decode(ids: TokenId[]): string {
    // newId → [a, b] の逆引き表
    const expand = new Map<TokenId, [TokenId, TokenId]>();
    for (const rule of this.merges) {
      expand.set(rule.newId, rule.pair);
    }

    const bytes: TokenId[] = [];
    const emit = (id: TokenId): void => {
      const pair = expand.get(id);
      if (pair === undefined) {
        bytes.push(id); // 0..255 の生バイト
      } else {
        emit(pair[0]);
        emit(pair[1]);
      }
    };
    for (const id of ids) emit(id);
    return this.byteTok.decode(bytes);
  }
}

// === 動作確認 ===
// 直接実行: bun run transformer/00b_bpe_tokenizer.ts
if (import.meta.main) {
  console.log("=== BPE Tokenizer の動作確認 ===\n");

  // === Step A: countPairs / merge の単体動作 ===
  console.log("▼ countPairs([1, 2, 2, 3, 2, 3])");
  const demo = [1, 2, 2, 3, 2, 3];
  for (const [k, v] of countPairs(demo)) console.log(`  pair (${k}) → ${v} 回`);
  console.log('\n▼ merge([1, 2, 2, 3, 2, 3], pair=(2,3), newId=256)');
  console.log(`  → [${merge(demo, [2, 3], 256).join(", ")}]\n`);

  // === Step B: 小さいテキストで BPE 学習 ===
  const text = "the cat sat on the mat. the cat ran. the bat sat.";
  console.log(`▼ 学習テキスト: ${JSON.stringify(text)}`);

  const bpe = new BPETokenizer();
  const beforeLen = bpe.encode(text).length; // 学習前 = byte 列の長さ
  bpe.train(text, 270); // 256 + 最大 14 マージ
  const afterLen = bpe.encode(text).length;

  console.log(`\n学習で獲得した merge ルール（${bpe.merges.length} 個）:`);
  const byteTok = new ByteTokenizer();
  for (const r of bpe.merges) {
    const piece = bpe.decode([r.newId]);
    console.log(
      `  (${r.pair[0]}, ${r.pair[1]}) → ${r.newId}   表す文字列: ${JSON.stringify(piece)}`,
    );
  }

  console.log(`\nvocab_size = ${bpe.vocabSize}`);
  console.log(`列の長さ: byte 単位 ${beforeLen} → BPE ${afterLen}（短くなった）`);

  // === Step C: encode / decode の往復 ===
  console.log("\n▼ encode / decode の往復確認");
  const tests = ["the cat sat", "the bat", "hello世界😊"];
  for (const t of tests) {
    const ids = bpe.encode(t);
    const back = bpe.decode(ids);
    console.log(`  入力 ${JSON.stringify(t)}`);
    console.log(`    encode → [${ids.join(", ")}]  (${ids.length} token)`);
    console.log(`    decode → ${JSON.stringify(back)}  往復一致: ${t === back}`);
  }
  void byteTok;

  console.log("\n=== 観察ポイント ===");
  console.log('  - "the " のような頻出パターンが 1 token にまとまる');
  console.log("  - 学習に無い文字（😊）も byte に分解されるので必ず往復できる");
  console.log("  - merge 順序が再現性の鍵。同じテキスト + 同じ vocab_size で常に同じ結果");
  console.log("  - この ID 列が 08_embedding.ts でベクトルになる");
}

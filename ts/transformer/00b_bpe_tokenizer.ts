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
// 特殊トークン（<|endoftext|>）
//   会話・文書の区切りを 1 つの token ID で表す。byte 列には現れない予約 ID を割り当てる。
//   学習: テキストを end_token で分割し、断片ごとにペアを数える（断片をまたぐペアを作らない）。
//   encode: end_token はそのまま 1 ID に置換し、間のテキストだけ BPE にかける。
//
// 事前トークン化（Pre-tokenization）
//   生バイト列のままだと " hello" と "hello!" の "!" のように、単語と句読点・空白が
//   1 ペアにマージされ得る。GPT-2 は学習前にテキストを「単語 / 数字 / 句読点 / 短縮形 /
//   空白」の単位へ正規表現で区切り、マージをその単位（pretoken）の内部に閉じ込める。
//   結果 "Say hello!" は "Say" / " hello" / "!" の独立トークンになり、語彙が安定する。
//   JS の RegExp は u フラグで Unicode property escape（\p{L} 文字・\p{N} 数字）を
//   ネイティブサポートするので外部ライブラリ不要。
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
//   counts を渡すと累積する（特殊トークンで分割した複数断片の統計を 1 つに集約するため）
export function countPairs(
  ids: TokenId[],
  counts: Map<string, number> = new Map(),
): Map<string, number> {
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

// 文書区切りを表す特殊トークン（GPT 系と同じ表記）
export const END_TOKEN = "<|endoftext|>";

// GPT-2 の pre-tokenization パターン。短縮形 / 単語 / 数字 / 句読点 / 空白を分割する。
//   u フラグで \p{L}（文字）\p{N}（数字）が使える。g フラグで全マッチを列挙する。
const GPT2_PRETOKEN_RE =
  /'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

// テキストを pretoken（マージを閉じ込める単位）の配列へ分割する
export function pretokenize(text: string): string[] {
  return text.match(GPT2_PRETOKEN_RE) ?? [];
}

export class BPETokenizer {
  private byteTok = new ByteTokenizer();
  merges: MergeRule[] = [];
  vocabSize = 256;
  endToken: string;
  endTokenId = -1; // train 後に 256 + merges.length が入る

  constructor(endToken: string = END_TOKEN) {
    this.endToken = endToken;
  }

  // BPE 学習: テキストから merge ルールを獲得する
  //   targetVocabSize: 256（byte）+ merge 回数 + 1（特殊トークン分を予約）
  //   テキストは end_token で分割し、断片ごとにペアを数える（断片をまたぐペアを作らない）
  train(text: string, targetVocabSize: number): void {
    if (targetVocabSize < 257) {
      throw new Error("targetVocabSize must be >= 257 (byte vocab + end token)");
    }
    // 特殊トークンで分割 → 各片を pretoken へ分割 → 各 pretoken を byte 列にする。
    // ペアは pretoken の内部でのみ数えるので、マージが単語境界をまたがない。
    const segments = text.split(this.endToken);
    let idSegments: TokenId[][] = [];
    for (const seg of segments) {
      for (const pretoken of pretokenize(seg)) {
        idSegments.push(this.byteTok.encode(pretoken));
      }
    }
    this.merges = [];
    let nextId = 256;

    const numMerges = targetVocabSize - 256 - 1; // -1 は特殊トークン分
    for (let step = 0; step < numMerges; step++) {
      const counts = new Map<string, number>();
      for (const ids of idSegments) countPairs(ids, counts);
      const best = selectBestPair(counts);
      if (best === null || best.count < 2) {
        break; // これ以上まとめる価値のあるペアが無い
      }
      const newId = nextId++;
      this.merges.push({ pair: best.pair, newId });
      idSegments = idSegments.map((ids) => merge(ids, best.pair, newId));
    }
    this.endTokenId = nextId; // = 256 + merges.length
    this.vocabSize = nextId + 1;
  }

  // 通常テキスト（特殊トークンを含まない断片）への BPE 適用
  private encodeText(text: string): TokenId[] {
    let ids = this.byteTok.encode(text);
    for (const rule of this.merges) {
      ids = merge(ids, rule.pair, rule.newId);
    }
    return ids;
  }

  // encode: end_token を 1 ID に置換し、間のテキストだけ BPE にかける
  //   allowSpecial=false なら end_token 文字列もただのテキストとして byte 化する。
  //   WHY: 信頼できない入力中の特殊トークン文字列を ID 化すると role 境界偽装になりうるため、
  //        特殊トークンの解釈は信頼できる入力に限定できるよう分離する。
  encode(text: string, allowSpecial = true): TokenId[] {
    if (this.endTokenId < 0) {
      throw new Error("call train() before encode()");
    }
    if (!allowSpecial) {
      return this.encodePlain(text);
    }
    // キャプチャグループで end_token を残したまま分割する
    const escaped = this.endToken.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const parts = text.split(new RegExp(`(${escaped})`));
    const out: TokenId[] = [];
    for (const part of parts) {
      if (part === this.endToken) {
        out.push(this.endTokenId);
      } else if (part.length > 0) {
        out.push(...this.encodePlain(part));
      }
    }
    return out;
  }

  // 通常テキストを pretoken へ分割し、各 pretoken を個別に BPE にかけて連結する。
  // 学習時と同じ単位でマージするので、語境界をまたいだ ID が生まれない。
  private encodePlain(text: string): TokenId[] {
    const out: TokenId[] = [];
    for (const pretoken of pretokenize(text)) {
      out.push(...this.encodeText(pretoken));
    }
    return out;
  }

  // decode: 新 token を再帰的にバイト列へ展開してから UTF-8 復元。end_token は文字列で挟む
  decode(ids: TokenId[]): string {
    // newId → [a, b] の逆引き表
    const expand = new Map<TokenId, [TokenId, TokenId]>();
    for (const rule of this.merges) {
      expand.set(rule.newId, rule.pair);
    }

    let out = "";
    let bytes: TokenId[] = [];
    const flush = (): void => {
      if (bytes.length > 0) {
        out += this.byteTok.decode(bytes);
        bytes = [];
      }
    };
    const emit = (id: TokenId): void => {
      const pair = expand.get(id);
      if (pair === undefined) {
        bytes.push(id); // 0..255 の生バイト
      } else {
        emit(pair[0]);
        emit(pair[1]);
      }
    };
    for (const id of ids) {
      if (id === this.endTokenId) {
        flush();
        out += this.endToken;
      } else {
        emit(id);
      }
    }
    flush();
    return out;
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
  bpe.train(text, 271); // 256 + 最大 14 マージ + 1（特殊トークン）
  const beforeLen = new ByteTokenizer().encode(text).length; // 学習前 = byte 列の長さ
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
  console.log(`end_token_id = ${bpe.endTokenId}（= 256 + ${bpe.merges.length} merges）`);
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

  // === Step D: 特殊トークン ===
  console.log("\n▼ 特殊トークン <|endoftext|> の扱い");
  const special = "abc<|endoftext|>def";
  const sIds = bpe.encode(special);
  const sBack = bpe.decode(sIds);
  const eotCount = sIds.filter((id) => id === bpe.endTokenId).length;
  console.log(`  入力 ${JSON.stringify(special)}`);
  console.log(`    encode → [${sIds.join(", ")}]`);
  console.log(`    end_token_id(${bpe.endTokenId}) の出現回数: ${eotCount}（1 ID に圧縮）`);
  console.log(`    decode → ${JSON.stringify(sBack)}  往復一致: ${special === sBack}`);

  // allowSpecial=false: 特殊トークンを解釈せずテキストとして byte 化（信頼できない入力向け）
  const noSpecial = bpe.encode(special, false);
  const leaks = noSpecial.includes(bpe.endTokenId);
  console.log(
    `\n  allowSpecial=false → [${noSpecial.join(", ")}]  end_token_id を含まない: ${!leaks}`,
  );

  // === Step E: 事前トークン化（Pre-tokenization）===
  console.log("\n▼ 事前トークン化 pretokenize()");
  console.log(
    `  ${JSON.stringify("Say hello!")} → ${JSON.stringify(pretokenize("Say hello!"))}`,
  );

  console.log("\n▼ 書籍の確認例（pre-tokenization 後の独立トークン）");
  const sample = "Say hello! Why hello? Just hello.<|endoftext|>Good morning!";
  const ptBpe = new BPETokenizer();
  ptBpe.train(sample, 300);
  const sayIds = ptBpe.encode("Say hello!");
  console.log(`  学習テキスト: ${JSON.stringify(sample)}`);
  console.log(`  "Say hello!" encode → [${sayIds.join(", ")}]  (${sayIds.length} token)`);
  for (const id of sayIds) {
    console.log(`    id ${id} → ${JSON.stringify(ptBpe.decode([id]))}`);
  }
  // pre-tokenization の本質: どの ID も "Say" / " hello" / "!" の境界をまたがない。
  // 各 ID を decode した断片が 1 つの pretoken 内に収まることを確認する。
  const pretokens = pretokenize("Say hello!"); // ["Say", " hello", "!"]
  const pieces = sayIds.map((id) => ptBpe.decode([id]));
  const noBoundaryCross = pieces.every((p) =>
    pretokens.some((pt) => pt.includes(p)),
  );
  const helloIsOneToken = pieces.includes(" hello");
  const bangIsOneToken = pieces.includes("!");
  console.log(
    `  境界をまたぐ token なし: ${noBoundaryCross}` +
      `  ' hello' が独立 1 token: ${helloIsOneToken}` +
      `  '!' が独立 1 token: ${bangIsOneToken}`,
  );
  console.log(
    `  pretoken への分割: ${JSON.stringify(pretokens)}  往復一致: ${
      ptBpe.decode(sayIds) === "Say hello!"
    }`,
  );

  console.log("\n=== 観察ポイント ===");
  console.log('  - "the " のような頻出パターンが 1 token にまとまる');
  console.log("  - 学習に無い文字（😊）も byte に分解されるので必ず往復できる");
  console.log("  - <|endoftext|> は学習で分割境界になり、encode で 1 ID に圧縮される");
  console.log("  - allowSpecial=false なら特殊トークンを解釈しない（role 境界偽装の防御）");
  console.log("  - merge 順序が再現性の鍵。同じテキスト + 同じ vocab_size で常に同じ結果");
  console.log("  - この ID 列が 08_embedding.ts でベクトルになる");
}

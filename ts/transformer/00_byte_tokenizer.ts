// Step 0: Byte Tokenizer（バイト単位トークン化）
//
// 役割
//   生のテキストを「token ID 列」にする最上流の処理。
//   08_embedding.ts より前の段。embedding は ID を受け取るが、その ID を作るのがここ。
//
// なぜ「バイト単位」か
//   文字単位（char-level）だと、学習データに無い文字（絵文字・別言語）を扱えない。
//   UTF-8 のバイト列に落とせば、語彙は 0..255 の 256 種で固定。どんな文字列も表現できる。
//   - "A"     → 1 バイト [65]
//   - "世"    → 3 バイト [228, 184, 150]
//   - "😊"    → 4 バイト [240, 159, 152, 138]
//   これが GPT-2 系の BPE が「バイト列」を起点にする理由（unknown token が原理的に出ない）。
//
// 数式というより定義
//   encode(text) = UTF-8 bytes of text         （各バイトが token ID、0..255）
//   decode(ids)  = UTF-8 decode of byte array
//
// 「角度ゲーム」での位置づけ
//   ここはまだベクトルではない。ただの整数列。
//   この整数列が 08_embedding.ts でベクトル（向きを持つ点）になる。

import type { TokenId } from "./types";

// バイト単位 tokenizer
//   vocab は 0..255 で固定。学習不要、どんな Unicode 文字列も往復できる。
export class ByteTokenizer {
  readonly vocabSize = 256;

  // text → UTF-8 バイト列（各バイトが token ID）
  encode(text: string): TokenId[] {
    const bytes = new TextEncoder().encode(text);
    return Array.from(bytes);
  }

  // token ID 列 → text（UTF-8 として復元）
  decode(ids: TokenId[]): string {
    for (const id of ids) {
      if (id < 0 || id > 255) {
        throw new Error(`byte token id ${id} out of range [0, 256)`);
      }
    }
    const bytes = Uint8Array.from(ids);
    return new TextDecoder().decode(bytes);
  }
}

// === 動作確認 ===
// 直接実行: bun run transformer/00_byte_tokenizer.ts
if (import.meta.main) {
  console.log("=== Byte Tokenizer の動作確認 ===\n");

  const tok = new ByteTokenizer();
  console.log(`vocab_size = ${tok.vocabSize}（0..255 で固定）\n`);

  const samples = ["hello", "世界", "😊", "hello世界😊"];
  for (const s of samples) {
    const ids = tok.encode(s);
    const back = tok.decode(ids);
    console.log(`入力       : ${JSON.stringify(s)}`);
    console.log(`  encode   : [${ids.join(", ")}]  (${ids.length} bytes)`);
    console.log(`  decode   : ${JSON.stringify(back)}`);
    console.log(`  往復一致 : ${s === back}\n`);
  }

  console.log("=== バイト数の内訳（UTF-8 の特性）===");
  console.log(`  "A"  → ${tok.encode("A").length} byte   ASCII は 1 バイト`);
  console.log(`  "世" → ${tok.encode("世").length} byte   CJK は 3 バイト`);
  console.log(`  "😊" → ${tok.encode("😊").length} byte   絵文字は 4 バイト`);

  console.log("\n=== 観察ポイント ===");
  console.log("  - vocab は常に 256。unknown token が原理的に発生しない");
  console.log("  - ただし日本語 1 文字が 3 token になり、列が長くなる（context を食う）");
  console.log("  - この『列が長すぎる』問題を解くのが次の BPE（00b_bpe_tokenizer.ts）");
}

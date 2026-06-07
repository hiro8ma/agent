"""バイト単位トークン化 + BPE（Byte Pair Encoding）

data.py の CharTokenizer は文字単位で、学習データに無い文字（絵文字・別言語）を
扱えない。BPE はバイト列を起点にするので unknown token が原理的に出ず、かつ頻出
ペアをまとめて列を短くする。GPT-2 / GPT-4 / Llama の実用 tokenizer の中核。

このモジュールは CharTokenizer を置き換えず、選択肢として追加する。
encode / decode / vocab_size のインターフェースは CharTokenizer と互換なので、
train.py / generate.py からそのまま差し替えられる。

アルゴリズム（書籍に忠実）:
  1. 初期化   テキストを UTF-8 バイト列にする。語彙は 0..255 の 256 個から始める
  2. ペア集計 隣接ペアの出現回数を数える（count_pairs）
  3. 最頻マージ 最も多いペアを新 token ID に置き換える（merge）
  4. 反復     目標 vocab_size まで 2-3 を繰り返す（train_bpe）

再現性: 同点ペアは「頻度大 → token ID 小」で決定的に選ぶ。merge_rules を JSON に
保存/読込できるので、学習時と推論時で同じトークン化を保証する。

特殊トークン: <|endoftext|> を 1 token ID で表す。学習は end_token で分割し断片
ごとにペアを数える。encode は end_token を 1 ID に置換し、間だけ BPE にかける。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

# 文書区切りを表す特殊トークン（GPT 系と同じ表記）
END_TOKEN = "<|endoftext|>"


class ByteTokenizer:
    """バイト単位 tokenizer（vocab 0..255 固定、学習不要）

    例: 'A' -> [65], '世' -> [228, 184, 150], '😊' -> [240, 159, 152, 138]
    どんな Unicode 文字列も往復でき、unknown token が出ない。
    """

    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(ids).decode("utf-8", errors="replace")


def count_pairs(
    ids: list[int], counts: Counter[tuple[int, int]] | None = None
) -> Counter[tuple[int, int]]:
    """隣接ペアの出現回数を数える

    [1, 2, 2, 3] -> {(1,2):1, (2,2):1, (2,3):1}
    counts を渡すと累積する（特殊トークンで分割した複数断片の統計を 1 つに集約するため）。
    """
    if counts is None:
        counts = Counter()
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] += 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """列の中のペア (a, b) をすべて new_id に置き換える

    [1, 2, 2, 3], pair=(2,3), new_id=256 -> [1, 2, 256]
    """
    a, b = pair
    out: list[int] = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """Byte Pair Encoding tokenizer

    encode / decode / vocab_size を持ち、CharTokenizer と差し替え可能。
    """

    def __init__(self, end_token: str = END_TOKEN) -> None:
        self._byte = ByteTokenizer()
        # merge ルールを学習順に保持する。順序が encode の正しさを決める
        self.merges: list[tuple[tuple[int, int], int]] = []
        self.vocab_size = 256
        self.end_token = end_token
        self.end_token_id = -1  # train 後に 256 + len(merges) が入る

    def train(self, text: str, target_vocab_size: int, *, verbose: bool = False) -> None:
        """テキストから merge ルールを獲得する

        target_vocab_size: 256（byte）+ merge 回数 + 1（特殊トークン分を予約）。
        テキストは end_token で分割し、断片ごとにペアを数える（断片をまたぐペアを作らない）。
        """
        if target_vocab_size < 257:
            raise ValueError("target_vocab_size must be >= 257 (byte vocab + end token)")

        segments = [self._byte.encode(seg) for seg in text.split(self.end_token)]
        self.merges = []
        next_id = 256
        num_merges = target_vocab_size - 256 - 1  # -1 は特殊トークン分

        for step in range(num_merges):
            counts: Counter[tuple[int, int]] = Counter()
            for ids in segments:
                count_pairs(ids, counts)
            if not counts:
                break
            # 頻度大 → token ID 小（a, b の順）で決定的に選ぶ
            pair, freq = max(counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
            if freq < 2:
                break
            new_id = next_id
            next_id += 1
            self.merges.append((pair, new_id))
            segments = [merge(ids, pair, new_id) for ids in segments]
            if verbose:
                piece = self.decode([new_id])
                print(f"  merge {step + 1}: {pair} -> {new_id}  {piece!r}  (freq {freq})")

        self.end_token_id = next_id  # = 256 + len(merges)
        self.vocab_size = next_id + 1

    def _encode_text(self, text: str) -> list[int]:
        """通常テキスト（特殊トークンを含まない断片）への BPE 適用"""
        ids = self._byte.encode(text)
        for pair, new_id in self.merges:
            ids = merge(ids, pair, new_id)
        return ids

    def encode(self, text: str, *, allow_special: bool = True) -> list[int]:
        """end_token を 1 ID に置換し、間のテキストだけ BPE にかける

        allow_special=False なら end_token 文字列もただのテキストとして byte 化する。
        WHY: 信頼できない入力中の特殊トークン文字列を ID 化すると role 境界偽装になりうる
             ため、特殊トークンの解釈は信頼できる入力に限定できるよう分離する。
        """
        if self.end_token_id < 0:
            raise RuntimeError("call train() before encode()")
        if not allow_special:
            return self._encode_text(text)
        # キャプチャグループで end_token を残したまま分割する
        out: list[int] = []
        for part in re.split(f"({re.escape(self.end_token)})", text):
            if part == self.end_token:
                out.append(self.end_token_id)
            elif part:
                out.extend(self._encode_text(part))
        return out

    def decode(self, ids: list[int]) -> str:
        # new_id -> (a, b) の逆引き表で再帰的にバイトへ展開
        expand = {new_id: pair for pair, new_id in self.merges}

        out = ""
        out_bytes: list[int] = []

        def flush() -> None:
            nonlocal out, out_bytes
            if out_bytes:
                out += bytes(out_bytes).decode("utf-8", errors="replace")
                out_bytes = []

        def emit(token_id: int) -> None:
            pair = expand.get(token_id)
            if pair is None:
                out_bytes.append(token_id)  # 0..255 の生バイト
            else:
                emit(pair[0])
                emit(pair[1])

        for token_id in ids:
            if token_id == self.end_token_id:
                flush()
                out += self.end_token
            else:
                emit(token_id)
        flush()
        return out

    def save(self, path: str | Path) -> None:
        """merge ルールを JSON 保存（再現性のため）"""
        data = {
            "vocab_size": self.vocab_size,
            "merges": [[list(pair), new_id] for pair, new_id in self.merges],
            "end_token": self.end_token,
            "end_token_id": self.end_token_id,
        }
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls(end_token=data.get("end_token", END_TOKEN))
        tok.vocab_size = data["vocab_size"]
        tok.merges = [((p[0], p[1]), new_id) for p, new_id in data["merges"]]
        tok.end_token_id = data.get("end_token_id", 256 + len(tok.merges))
        return tok


# === 動作確認 ===
# 直接実行: uv run python bpe.py
if __name__ == "__main__":
    print("=== Byte Tokenizer の動作確認 ===\n")

    byte_tok = ByteTokenizer()
    for s in ["hello", "世界", "😊", "hello世界😊"]:
        ids = byte_tok.encode(s)
        back = byte_tok.decode(ids)
        print(f"入力 {s!r:14} encode {ids}  decode {back!r}  往復 {s == back}")

    print("\n=== count_pairs / merge の単体動作 ===")
    demo = [1, 2, 2, 3, 2, 3]
    print(f"count_pairs({demo}) = {dict(count_pairs(demo))}")
    print(f"merge({demo}, (2,3), 256) = {merge(demo, (2, 3), 256)}")

    print("\n=== BPE 学習 ===")
    text = "the cat sat on the mat. the cat ran. the bat sat."
    print(f"学習テキスト: {text!r}")
    bpe = BPETokenizer()
    before = len(ByteTokenizer().encode(text))
    bpe.train(text, target_vocab_size=271, verbose=True)  # +1 は特殊トークン
    after = len(bpe.encode(text))
    print(f"\nvocab_size = {bpe.vocab_size}")
    print(f"end_token_id = {bpe.end_token_id}（= 256 + {len(bpe.merges)} merges）")
    print(f"列の長さ: byte {before} -> BPE {after}")

    print("\n=== encode / decode の往復確認 ===")
    for t in ["the cat sat", "the bat", "hello世界😊"]:
        ids = bpe.encode(t)
        back = bpe.decode(ids)
        print(f"  {t!r:16} -> {ids}  decode {back!r}  往復 {t == back}")

    print("\n=== 特殊トークン <|endoftext|> の扱い ===")
    special = "abc<|endoftext|>def"
    s_ids = bpe.encode(special)
    s_back = bpe.decode(s_ids)
    eot_count = s_ids.count(bpe.end_token_id)
    print(f"  入力 {special!r}")
    print(f"    encode -> {s_ids}")
    print(f"    end_token_id({bpe.end_token_id}) の出現回数: {eot_count}（1 ID に圧縮）")
    print(f"    decode -> {s_back!r}  往復 {special == s_back}")
    no_special = bpe.encode(special, allow_special=False)
    print(f"    allow_special=False -> {no_special}  end_token_id を含まない: "
          f"{bpe.end_token_id not in no_special}")

    print("\n=== save / load の再現性確認 ===")
    tmp = Path(__file__).parent / "data" / "bpe_demo.json"
    tmp.parent.mkdir(exist_ok=True)
    bpe.save(tmp)
    reloaded = BPETokenizer.load(tmp)
    sample = "the cat ran on the bat"
    same = bpe.encode(sample) == reloaded.encode(sample)
    print(f"  保存先: {tmp.name}")
    print(f"  load 後の encode 一致: {same}")
    tmp.unlink()

    print("\n=== 観察ポイント ===")
    print("  - 'the ' のような頻出パターンが 1 token にまとまり、列が短くなる")
    print("  - 学習に無い文字（😊）も byte に分解されるので必ず往復できる")
    print("  - <|endoftext|> は学習で分割境界になり、encode で 1 ID に圧縮される")
    print("  - allow_special=False なら特殊トークンを解釈しない（role 境界偽装の防御）")
    print("  - merges を save/load すれば学習時と推論時で同じトークン化を保証できる")
    print("  - CharTokenizer と同じ encode/decode/vocab_size なので train.py で差し替え可能")

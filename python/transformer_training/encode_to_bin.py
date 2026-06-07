"""コーパスを token ID 列にエンコードして uint16 バイナリへ永続化

学習ループは毎回テキストをエンコードし直すのが無駄なので、一度だけ token ID 列に
変換して .bin に落とす（nanoGPT 流）。学習時は np.memmap / np.fromfile で即読める。

train/val 分割: 先頭 90% を train.bin、残り 10% を val.bin（時系列順）。

このスクリプトは:
  1. tokenizer でコーパス全体をエンコード
  2. np.uint16 にキャストして data/<name>.bin へ tofile
  3. 読み戻して要素数一致と decode 往復が壊れていないことを確認
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bpe import BPETokenizer
from data import DATA_DIR, download_dataset

# uint16 は 0..65535 を表せる。これを超える vocab は uint32 にフォールバックする
UINT16_MAX = 65535
ROUNDTRIP_CHECK_TOKENS = 64  # 先頭何 token を decode 往復チェックするか


def _select_dtype(vocab_size: int) -> type[np.unsignedinteger]:
    """vocab_size に収まる最小の符号なし整数型を選ぶ

    WHY uint16: ID を uint16 で持つと int32 比でディスク/メモリが半減する。
    """
    if vocab_size - 1 <= UINT16_MAX:
        return np.uint16
    print(f"[warn] vocab_size={vocab_size} > {UINT16_MAX + 1}: uint32 にフォールバック"
          "（ディスク/メモリは uint16 の 2 倍）")
    return np.uint32


def encode_to_bin(
    tokenizer: BPETokenizer,
    text: str,
    out_path: Path,
) -> np.ndarray:
    """text をエンコードして out_path に符号なし整数バイナリで書き出す"""
    ids = tokenizer.encode(text)
    dtype = _select_dtype(tokenizer.vocab_size)
    arr = np.array(ids, dtype=dtype)
    out_path.parent.mkdir(exist_ok=True)
    arr.tofile(out_path)
    return arr


def verify_bin(
    tokenizer: BPETokenizer,
    bin_path: Path,
    expected: np.ndarray,
) -> bool:
    """書き出した .bin を読み戻し、要素数一致と decode 往復を確認する"""
    loaded = np.fromfile(bin_path, dtype=expected.dtype)
    count_ok = len(loaded) == len(expected)
    equal_ok = bool(np.array_equal(loaded, expected))

    head = loaded[:ROUNDTRIP_CHECK_TOKENS].tolist()
    decoded = tokenizer.decode([int(i) for i in head])
    reencoded = tokenizer.encode(decoded, allow_special=False)
    # decode → encode が先頭部分と一致すれば往復が壊れていない
    roundtrip_ok = reencoded[: len(head)] == head

    print(f"  読み戻し要素数  : {len(loaded):,}（書き出し {len(expected):,}）"
          f"  一致 {count_ok}")
    print(f"  バイト列の一致  : {equal_ok}")
    print(f"  先頭 {len(head)} token の decode 往復: {roundtrip_ok}")
    print(f"  先頭テキスト    : {decoded[:60]!r}")
    return count_ok and equal_ok and roundtrip_ok


# === コーパスの bin 化 ===
# 直接実行: uv run python encode_to_bin.py
if __name__ == "__main__":
    TARGET_VOCAB = 1000  # 圧縮率と埋め込みコストのバランス（eval_tokenizer.py 参照）

    print("=== コーパスを uint16 .bin へ永続化 ===\n")

    text = download_dataset()
    print(f"コーパス: {len(text):,} 文字（{len(text.encode('utf-8')):,} byte）\n")

    print(f"[1/3] BPE 学習（target_vocab_size={TARGET_VOCAB}）...")
    tok = BPETokenizer()
    tok.train(text, target_vocab_size=TARGET_VOCAB)
    tok_path = DATA_DIR / "bpe_tinyshakespeare.json"
    tok.save(tok_path)
    print(f"      vocab_size={tok.vocab_size}  merges を {tok_path.name} に保存\n")

    # nanoGPT 流: 先頭 90% を train、残りを val（時系列順）
    n = int(0.9 * len(text))
    splits = {
        "train": text[:n],
        "val": text[n:],
    }

    for name, chunk in splits.items():
        print(f"[2/3] encode + 書き出し: {name}")
        bin_path = DATA_DIR / f"{name}.bin"
        arr = encode_to_bin(tok, chunk, bin_path)
        size_mb = bin_path.stat().st_size / 1_000_000
        ratio = len(chunk.encode("utf-8")) / len(arr) if len(arr) else 0.0
        print(f"      {bin_path.name}: {len(arr):,} token  {size_mb:.2f} MB  "
              f"dtype={arr.dtype}  圧縮率 {ratio:.2f} 倍")

        print(f"[3/3] 読み戻し検証: {name}")
        ok = verify_bin(tok, bin_path, arr)
        print(f"      検証結果: {'OK' if ok else 'FAILED'}\n")

    print("=== 観察ポイント ===")
    print("  - uint16 は int32 比でディスク/メモリ半減（vocab <= 65535 が前提）")
    print("  - np.fromfile で即ロードでき、学習ループのエンコード反復を省ける")
    print("  - train.bin / val.bin の分割は時系列順（過学習の検出に val を使う）")
    print("  - tokenizer の merges を JSON 保存しておけば生成時に同じ ID 化を再現できる")

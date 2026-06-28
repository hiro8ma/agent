"""データセット読み込みと character-level tokenizer

mini-GPT を学習させる最小限のデータパイプライン。
agent/ts/transformer/ で作った Transformer の知識と接続される。

データセット: tinyshakespeare（シェイクスピア戯曲全集、1.1 MB）
Tokenizer:   character-level（BPE などの複雑さを避けて教育を優先）

なぜ character-level か:
  - vocabulary が小さい（約 65 文字）→ 学習が速い
  - tokenizer 実装が trivial（文字 ↔ ID の lookup だけ）
  - 学習結果が観察しやすい（生成サンプルが「英文っぽくなる」過程が見える）

実用 LLM は BPE / SentencePiece を使うが、教育目的では char-level が最適。
"""

from __future__ import annotations

from pathlib import Path

import requests  # type: ignore[import-untyped]
import torch

# データファイルのパス
DATA_DIR = Path(__file__).parent / "data"
DATASET_PATH = DATA_DIR / "tinyshakespeare.txt"
DATASET_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_dataset() -> str:
    """tinyshakespeare をダウンロードして読み込む（キャッシュあり）"""
    DATA_DIR.mkdir(exist_ok=True)
    if not DATASET_PATH.exists():
        print(f"[data] downloading tinyshakespeare from {DATASET_URL}")
        response = requests.get(DATASET_URL, timeout=30)
        response.raise_for_status()
        DATASET_PATH.write_text(response.text, encoding="utf-8")
        print(f"[data] saved to {DATASET_PATH} ({len(response.text):,} chars)")
    text = DATASET_PATH.read_text(encoding="utf-8")
    return text


class CharTokenizer:
    """文字単位の最小 tokenizer

    例: 'hello' → [72, 101, 108, 108, 111] のように、
         各文字に一意の ID を割り当てる単純な lookup table。
    """

    def __init__(self, text: str) -> None:
        # データセット中に出現する文字をすべて集めて vocabulary を構築
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.char_to_id: dict[str, int] = {c: i for i, c in enumerate(chars)}
        self.id_to_char: dict[int, str] = {i: c for i, c in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_char[i] for i in ids)


def load_data(block_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor, CharTokenizer]:
    """tinyshakespeare をロードして train/val に分割

    Returns:
        train_data: (N_train,) 形の token ID 列
        val_data:   (N_val,)   形の token ID 列
        tokenizer:  encode/decode 用
    """
    text = download_dataset()
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long, device=device)

    # 90% を train、10% を val として時系列順に分割
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    print(f"[data] vocab_size = {tokenizer.vocab_size}")
    print(f"[data] train tokens = {len(train_data):,}")
    print(f"[data] val tokens   = {len(val_data):,}")
    print(f"[data] block_size   = {block_size}")
    print(f"[data] vocab chars  = {sorted(tokenizer.char_to_id.keys())[:30]}...")

    return train_data, val_data, tokenizer


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ランダムなオフセットから (batch_size, block_size) のバッチを切り出す

    各サンプル:
        x = data[i      : i+block_size    ]
        y = data[i+1    : i+block_size+1  ]   ← 1 step ずらした「次トークン」が正解

    例 (block_size=4):
        text:  "hello"
        x:     "hell"   = [h, e, l, l]
        y:      "ello"   = [e, l, l, o]

        位置 0 で 'h' を見て 'e' を予測する
        位置 1 で 'h, e' を見て 'l' を予測する
        ...
        位置 3 で 'h, e, l, l' を見て 'o' を予測する

        → 1 サンプルから block_size 個分の学習信号が取れる（causal mask による並列化）
    """
    # 開始位置をランダムに batch_size 個サンプリング
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x, y


# === 動作確認 ===
# 直接実行: uv run python data.py
if __name__ == "__main__":
    print("=== データセットと Tokenizer の動作確認 ===\n")

    device = "cpu"
    block_size = 8
    batch_size = 4

    train_data, val_data, tokenizer = load_data(block_size=block_size, device=device)

    print(f"\n=== Tokenizer の動作確認 ===")
    sample = "Hello, World!"
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)
    print(f"入力       : {sample!r}")
    print(f"encode 結果: {encoded}")
    print(f"decode 結果: {decoded!r}")
    print(f"一致       : {sample == decoded}")

    print(f"\n=== バッチの動作確認 ===")
    x, y = get_batch(train_data, block_size=block_size, batch_size=batch_size)
    print(f"x.shape = {tuple(x.shape)}  (batch_size, block_size)")
    print(f"y.shape = {tuple(y.shape)}")

    print(f"\n各サンプルの (入力 → 正解) を可視化:")
    for b in range(batch_size):
        x_text = tokenizer.decode(x[b].tolist())
        y_text = tokenizer.decode(y[b].tolist())
        print(f"  サンプル {b}:")
        print(f"    x = {x_text!r}")
        print(f"    y = {y_text!r}   ← 1 文字ずれた『次』")

    print(f"\n=== 学習データの先頭 100 文字 ===")
    head_text = tokenizer.decode(train_data[:100].tolist())
    print(repr(head_text))

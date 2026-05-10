"""学習済み mini-GPT で文章を生成する

train.py で保存した checkpoint を読み込み、プロンプトから文章を生成する。

使い方:
    uv run python generate.py                          # デフォルト設定で生成
    uv run python generate.py --prompt "ROMEO:"        # プロンプト指定
    uv run python generate.py --temperature 0.5        # 確実寄り
    uv run python generate.py --temperature 1.5        # 多様性寄り
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import CharTokenizer
from model import GPTConfig, MiniGPT


def find_latest_checkpoint() -> Path:
    """checkpoints/ から最新の .pt ファイルを探す"""
    ckpt_dir = Path(__file__).parent / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"{ckpt_dir} がない。先に train.py を実行してください")
    candidates = sorted(ckpt_dir.glob("mini_gpt_step*.pt"))
    if not candidates:
        raise FileNotFoundError(f"{ckpt_dir} に checkpoint がない。先に train.py を実行してください")
    # step 番号でソート
    candidates.sort(key=lambda p: int(p.stem.replace("mini_gpt_step", "")))
    return candidates[-1]


def load_model(checkpoint_path: Path, device: str) -> tuple[MiniGPT, CharTokenizer]:
    """checkpoint からモデルと tokenizer を復元"""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config: GPTConfig = ckpt["config"]
    model = MiniGPT(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # tokenizer を復元（char_to_id だけあれば再構築可能）
    char_to_id = ckpt["char_to_id"]
    tokenizer = CharTokenizer.__new__(CharTokenizer)
    tokenizer.char_to_id = char_to_id
    tokenizer.id_to_char = {i: c for c, i in char_to_id.items()}
    tokenizer.vocab_size = len(char_to_id)

    print(f"[load] {checkpoint_path.name} (step {ckpt['step']}, vocab {tokenizer.vocab_size})")
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="\n", help="生成の起点プロンプト")
    parser.add_argument("--max-tokens", type=int, default=500, help="生成する最大 token 数")
    parser.add_argument("--temperature", type=float, default=0.8, help="0=greedy, 1=通常, >1=多様")
    parser.add_argument("--top-k", type=int, default=40, help="上位 k からサンプリング (0 で無効)")
    parser.add_argument("--checkpoint", default=None, help="checkpoint パス（省略時は最新）")
    parser.add_argument("--seed", type=int, default=None, help="乱数シード（再現性確保用）")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # デバイス選択
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # checkpoint 読み込み
    ckpt_path = Path(args.checkpoint) if args.checkpoint else find_latest_checkpoint()
    model, tokenizer = load_model(ckpt_path, device)

    # プロンプトを token ID に変換
    try:
        prompt_ids = tokenizer.encode(args.prompt)
    except KeyError as e:
        raise ValueError(f"プロンプトに学習データに無い文字が含まれている: {e}") from None

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    print(f"\n=== 生成設定 ===")
    print(f"prompt      = {args.prompt!r}")
    print(f"max_tokens  = {args.max_tokens}")
    print(f"temperature = {args.temperature}")
    print(f"top_k       = {args.top_k}")
    print(f"\n=== 生成結果 ===")

    top_k = args.top_k if args.top_k > 0 else None
    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=top_k,
        )

    text = tokenizer.decode(out[0].tolist())
    print(text)


if __name__ == "__main__":
    main()

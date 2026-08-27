"""自作 MiniGPT と Hugging Face の GPT2LMHeadModel を、同じ設定で組んで比べる。

3 章でスクラッチ実装した MiniGPT は、GPT-2 と同じ構造のはずである。
「はず」を確かめるには、同じハイパーパラメータから両方を組んで、
パラメータ数の内訳が一致するかを見ればよい。

事前学習済みの重みは要らない。GPT2Config から構造だけ組めるため、
ダウンロードなしで走る。

    uv run --with transformers python hf_config_bridge.py
"""

from __future__ import annotations

from collections import defaultdict

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from model import GPTConfig, MiniGPT


def to_gpt2_config(c: GPTConfig) -> GPT2Config:
    """自作 GPTConfig を GPT2Config に写す。

    4 つは同名で通る。残り 2 つだけ対応を書く。
      block_size -> n_positions（文脈長）
      dropout    -> resid / embd / attn の 3 つに分かれる
    """
    return GPT2Config(
        vocab_size=c.vocab_size,
        n_positions=c.block_size,
        n_ctx=c.block_size,
        n_embd=c.n_embd,
        n_layer=c.n_layer,
        n_head=c.n_head,
        resid_pdrop=c.dropout,
        embd_pdrop=c.dropout,
        attn_pdrop=c.dropout,
    )


def group_params(model: torch.nn.Module) -> dict[str, int]:
    """パラメータを役割ごとにまとめる。名前の付け方が違っても比較できるようにする。"""
    buckets: dict[str, int] = defaultdict(int)
    for name, p in model.named_parameters():
        n = name.lower()
        if "wte" in n or "token_emb" in n or ("embed" in n and "pos" not in n):
            key = "トークン埋め込み"
        elif "wpe" in n or "pos" in n:
            key = "位置埋め込み"
        elif "ln_f" in n or "final_norm" in n or n.startswith("norm"):
            key = "最終 LayerNorm"
        elif "lm_head" in n or "head" in n:
            key = "出力ヘッド"
        elif "ln" in n or "norm" in n:
            key = "ブロック内 LayerNorm"
        elif "attn" in n or "att" in n:
            key = "Attention"
        elif "mlp" in n or "ffn" in n or "feed" in n:
            key = "FFN"
        else:
            key = f"その他({name})"
        buckets[key] += p.numel()
    return dict(buckets)


def main() -> None:
    cfg = GPTConfig(vocab_size=32000, block_size=1024, n_layer=6, n_head=8, n_embd=512, dropout=0.1)

    print("共通の設定")
    print(f"  vocab_size {cfg.vocab_size} / block_size {cfg.block_size} / "
          f"n_layer {cfg.n_layer} / n_head {cfg.n_head} / n_embd {cfg.n_embd}\n")

    mine = MiniGPT(cfg)
    hf = GPT2LMHeadModel(to_gpt2_config(cfg))

    a = sum(p.numel() for p in mine.parameters())
    b = sum(p.numel() for p in hf.parameters())

    print("=" * 68)
    print("パラメータ数")
    print("=" * 68)
    print(f"  自作 MiniGPT        {a:>12,}")
    print(f"  HF GPT2LMHeadModel  {b:>12,}")
    print(f"  差                  {a - b:>12,}  ({abs(a - b) / max(a, b):.2%})")

    print()
    print("=" * 68)
    print("役割ごとの内訳")
    print("=" * 68)
    ga, gb = group_params(mine), group_params(hf)
    keys = sorted(set(ga) | set(gb))
    print(f"  {'役割':<22}{'自作':>14}{'HF':>14}{'差':>12}")
    for k in keys:
        va, vb = ga.get(k, 0), gb.get(k, 0)
        mark = "" if va == vb else "  <-"
        print(f"  {k:<22}{va:>14,}{vb:>14,}{va - vb:>12,}{mark}")

    print()
    print("=" * 68)
    print("差が出る理由")
    print("=" * 68)
    print("  HF の GPT-2 は出力ヘッドをトークン埋め込みと共有する（weight tying）。")
    print("  lm_head.weight は wte.weight と同じテンソルを指すため、二重に数えられない。")
    print("  自作側が別で持っていれば、その分だけ多くなる。")
    print()
    print("  Attention の実装差も出る。HF は Q/K/V を 1 つの行列にまとめており（c_attn）、")
    print("  分けて持つ実装とはパラメータ数が同じでも構造の刻み方が違う。")

    print()
    print("=" * 68)
    print("設定の対応表")
    print("=" * 68)
    rows = [
        ("vocab_size", "vocab_size", "同名"),
        ("block_size", "n_positions / n_ctx", "文脈長。GPT-2 は 2 つ持つ"),
        ("n_layer", "n_layer", "同名"),
        ("n_head", "n_head", "同名"),
        ("n_embd", "n_embd", "同名"),
        ("dropout", "resid_pdrop / embd_pdrop / attn_pdrop", "3 箇所に分かれる"),
        ("(なし)", "activation_function", "GPT-2 は gelu_new"),
        ("(なし)", "layer_norm_epsilon", "LayerNorm の安定化項"),
    ]
    print(f"  {'自作 GPTConfig':<16}{'GPT2Config':<38}備考")
    for a_, b_, note in rows:
        print(f"  {a_:<16}{b_:<38}{note}")


if __name__ == "__main__":
    main()

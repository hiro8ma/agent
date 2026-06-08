"""温度 / softmax サンプリングの可視化デモ

model.py の generate() が使う `logits / T → softmax → multinomial` の流れを
数値とテキストバーで可視化する。学習済み checkpoint があれば、同じ温度を
実モデルの生成に渡して文章の変化も見せる。

使い方:
    uv run python temperature_demo.py                 # 分布の可視化のみ
    uv run python temperature_demo.py --generate      # checkpoint で実生成も
    uv run python temperature_demo.py --prompt "ROMEO:" --generate

温度の効き方:
    T→0   : argmax に確率が集中（greedy 相当、決定的）
    T=1   : ロジットそのままの分布
    T>1   : 平滑化され低確率トークンも出やすくなる（多様・暴走しやすい）
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812

# generate() と揃えた温度系列。0.0 は greedy 近似として別扱いする。
TEMPERATURES = (0.0, 0.3, 0.7, 1.0, 1.5)

# 「次トークン」の生ロジット例。語彙 6 で 1 つだけ突出した山を持つ。
SAMPLE_LABELS = ("the", "a", "cat", "dog", "runs", "xyzzy")
SAMPLE_LOGITS = (4.0, 2.5, 2.0, 1.0, 0.5, -2.0)


def softmax_with_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """generate() と同じ規則で温度付き softmax を返す。

    T=0 は 0 除算を避けつつ argmax に全確率を寄せる greedy 近似にする。
    """
    if temperature <= 0.0:
        out = torch.zeros_like(logits)
        out[int(torch.argmax(logits))] = 1.0
        return out
    return F.softmax(logits / temperature, dim=-1)


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """上位 k 個以外を -inf にする（model.generate の top_k と同じ）。"""
    if k <= 0 or k >= logits.size(-1):
        return logits
    v, _ = torch.topk(logits, k)
    filtered = logits.clone()
    filtered[filtered < v[-1]] = float("-inf")
    return filtered


def top_p_filter(probs: torch.Tensor, p: float) -> torch.Tensor:
    """累積確率が p を超える最小集合だけ残す nucleus sampling。"""
    if p >= 1.0:
        return probs
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    keep = cumulative - sorted_probs < p  # 各要素を含めるまでの累積で判定
    masked = torch.zeros_like(probs)
    masked[sorted_idx[keep]] = probs[sorted_idx[keep]]
    return masked / masked.sum()


def _bar(value: float, width: int = 40) -> str:
    filled = round(value * width)
    return "#" * filled + "." * (width - filled)


def render_distribution(labels: tuple[str, ...], probs: torch.Tensor) -> None:
    for label, prob in zip(labels, probs.tolist(), strict=True):
        print(f"  {label:>6} {prob:6.3f} |{_bar(prob)}|")
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum().item()
    print(f"  {'entropy':>6} {entropy:6.3f}  (高いほど分布がフラット = 多様)")


def show_temperature_sweep() -> None:
    logits = torch.tensor(SAMPLE_LOGITS)
    print("=== 生ロジット ===")
    for label, lg in zip(SAMPLE_LABELS, SAMPLE_LOGITS, strict=True):
        print(f"  {label:>6} {lg:+.1f}")
    print()

    for t in TEMPERATURES:
        tag = "greedy 近似" if t == 0.0 else ""
        print(f"=== T = {t}  {tag}".rstrip())
        probs = softmax_with_temperature(logits, t)
        render_distribution(SAMPLE_LABELS, probs)
        print()


def show_top_k_top_p() -> None:
    logits = torch.tensor(SAMPLE_LOGITS)
    base = F.softmax(logits, dim=-1)

    print("=== top-k / top-p の効果 (T=1.0 起点) ===\n")
    print("baseline (全候補):")
    render_distribution(SAMPLE_LABELS, base)
    print()

    print("top-k=3 (上位 3 候補に限定 → 再正規化):")
    k_probs = F.softmax(top_k_filter(logits, 3), dim=-1)
    render_distribution(SAMPLE_LABELS, k_probs)
    print()

    print("top-p=0.9 (累積 0.9 までの候補に限定 → 再正規化):")
    p_probs = top_p_filter(base, 0.9)
    render_distribution(SAMPLE_LABELS, p_probs)
    print()


def show_real_generation(prompt: str, max_tokens: int) -> None:
    """学習済み checkpoint があれば温度別に実生成して見せる。"""
    from generate import find_latest_checkpoint, load_model  # noqa: PLC0415

    try:
        ckpt_path = find_latest_checkpoint()
    except FileNotFoundError as e:
        print(f"[skip] 実生成は checkpoint が無いため省略: {e}\n")
        return

    device = "cpu"
    model, tokenizer = load_model(ckpt_path, device)

    try:
        prompt_ids = tokenizer.encode(prompt)
    except KeyError as e:
        print(f"[skip] プロンプトに学習データ外の文字: {e}\n")
        return

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    print("\n=== 実モデルでの温度別生成 ===")
    for t in (0.2, 0.8, 1.4):
        torch.manual_seed(0)  # 温度以外の条件を固定して差分を温度だけにする
        out = model.generate(idx, max_new_tokens=max_tokens, temperature=max(t, 1e-8), top_k=None)
        text = tokenizer.decode(out[0].tolist())
        print(f"\n--- T = {t} ---")
        print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true", help="checkpoint で実生成も行う")
    parser.add_argument("--prompt", default="\n", help="実生成の起点プロンプト")
    parser.add_argument("--max-tokens", type=int, default=120, help="実生成の token 数")
    args = parser.parse_args()

    show_temperature_sweep()
    show_top_k_top_p()

    if args.generate:
        ckpt_dir = Path(__file__).parent / "checkpoints"
        if ckpt_dir.exists():
            show_real_generation(args.prompt, args.max_tokens)
        else:
            print("[skip] checkpoints/ が無いため実生成は省略。")


if __name__ == "__main__":
    main()

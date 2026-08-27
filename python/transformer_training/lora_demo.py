"""LoRA の原理を確かめ、教材の実装が抱える問題を実測する。

LoRA の主張は 3 つある。

  1. 更新は低ランクで足りる（BA でよい）
  2. パラメータが大幅に減る
  3. 事前学習の知識を保持する

1 と 2 は数えれば分かる。3 は「保持できているか」を測らないと分からない。
そして 3 が壊れていても学習は進むため、動作確認では検出できない。

    uv run --with transformers python lora_demo.py
"""

from __future__ import annotations

import torch
import torch.nn as nn


def sep(t: str) -> None:
    print("\n" + "=" * 68)
    print(t)
    print("=" * 68)


# ---------------------------------------------------------------- ランクの確認
def check_rank() -> None:
    sep("1  ランクとは「独立した情報の次元数」")
    m = torch.tensor([[1.0, 2, 3], [2, 4, 6], [3, 6, 9]])
    print("  行列:")
    for row in m.tolist():
        print("   ", row)
    print(f"\n  要素数 {m.numel()} 個だが、ランクは {torch.linalg.matrix_rank(m).item()}")
    print("  2 行目は 1 行目の 2 倍、3 行目は 3 倍。1 行分の情報しかない。")

    # BA の形にすると、ランクが r 以下に制限されることを確かめる
    d, r = 64, 4
    B = torch.randn(d, r)
    A = torch.randn(r, d)
    print(f"\n  B({d}x{r}) @ A({r}x{d}) -> {tuple((B @ A).shape)}")
    print(f"  その積のランク: {torch.linalg.matrix_rank(B @ A).item()}  (r={r} 以下に制限される)")


# ------------------------------------------------------------ パラメータ削減
def check_reduction() -> None:
    sep("2  パラメータ削減")
    print(f"  {'d':>6}{'r':>5}{'通常の更新':>14}{'LoRA':>12}{'削減率':>10}")
    for d in (768, 1024, 4096):
        for r in (4, 8, 16):
            full = d * d
            lora = d * r + r * d
            print(f"  {d:>6}{r:>5}{full:>14,}{lora:>12,}{1 - lora / full:>9.1%}")


# --------------------------------------------- 教材の実装 2 種を並べて比べる
class LoRABad(nn.Module):
    """教材の最初の実装。A も B も randn で初期化する。"""

    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float):
        super().__init__()
        self.A = nn.Parameter(torch.randn(in_dim, rank))
        self.B = nn.Parameter(torch.randn(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        return self.alpha * (x @ self.A @ self.B)


class LoRAGood(nn.Module):
    """B をゼロで初期化し、スケールを alpha / rank にする。"""

    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float):
        super().__init__()
        self.A = nn.Parameter(torch.randn(in_dim, rank) * 0.01)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.scaling = alpha / rank

    def forward(self, x):
        return self.scaling * (x @ self.A @ self.B)


def check_init() -> None:
    sep("3  B をゼロで初期化しないと、学習前に出力が変わる")
    torch.manual_seed(0)
    d, r, alpha = 128, 8, 16
    base = nn.Linear(d, d)
    x = torch.randn(4, d)
    ref = base(x)

    for name, cls in (("A,B とも randn（教材の最初の実装）", LoRABad),
                      ("B をゼロ（教材の 2 つ目の実装）", LoRAGood)):
        torch.manual_seed(0)
        lora = cls(d, d, r, alpha)
        out = base(x) + lora(x)
        diff = float((out - ref).abs().max())
        print(f"  {name:<38} 学習前の出力のずれ {diff:.4f}")

    print("\n  ΔW = BA なので、B がゼロなら学習開始時点で ΔW = 0 になる。")
    print("  事前学習の重みがそのまま残り、そこから少しずつ動かせる。")
    print("  両方 randn だと、学習を始める前から事前学習の出力が壊れている。")


def check_scaling() -> None:
    sep("4  alpha はそのまま掛けるのではなく alpha / rank で割る")
    d, alpha = 128, 16
    x = torch.randn(4, d)
    print(f"  alpha = {alpha} 固定で rank を変えたときの出力の大きさ\n")
    print(f"  {'rank':>6}{'alpha そのまま':>18}{'alpha / rank':>16}")
    for r in (4, 8, 16, 32):
        torch.manual_seed(0)
        A = torch.randn(d, r) * 0.01
        B = torch.randn(r, d) * 0.01
        raw = float((alpha * (x @ A @ B)).abs().mean())
        scaled = float(((alpha / r) * (x @ A @ B)).abs().mean())
        print(f"  {r:>6}{raw:>18.5f}{scaled:>16.5f}")
    print("\n  そのまま掛けると、rank を上げるほど更新の大きさまで増えてしまう。")
    print("  alpha / rank で割ると、rank を変えても更新の規模が揃う。")
    print("  だから rank と alpha を独立に調整できる。")


# ------------------------------------------------- 凍結しないと意味がない
class LinearWithLoRA(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float, freeze: bool):
        super().__init__()
        self.linear = linear
        if freeze:
            for p in self.linear.parameters():
                p.requires_grad = False
        self.lora = LoRAGood(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)


def check_freeze() -> None:
    sep("5  元の重みを凍結しないと、削減にならない")
    d, r, alpha = 768, 8, 16
    print(f"  {'':<22}{'学習可能':>14}{'全体':>14}{'割合':>10}")
    for label, freeze in (("凍結しない", False), ("凍結する", True)):
        m = LinearWithLoRA(nn.Linear(d, d), r, alpha, freeze=freeze)
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        total = sum(p.numel() for p in m.parameters())
        print(f"  {label:<22}{trainable:>14,}{total:>14,}{trainable / total:>9.1%}")
    print("\n  LoRA レイヤーを足しただけでは、学習可能なパラメータは減らない。")
    print("  requires_grad = False で元の重みを止めて、はじめて削減になる。")


def main() -> None:
    check_rank()
    check_reduction()
    check_init()
    check_scaling()
    check_freeze()
    print()


if __name__ == "__main__":
    main()

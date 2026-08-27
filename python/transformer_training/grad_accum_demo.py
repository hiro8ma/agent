"""勾配蓄積が、GPU を増やすのと同じグローバルバッチを作れることを確かめる。

教材が示すグローバルバッチサイズの式は次のとおり。

    グローバルバッチ = per_device_batch × GPU 数 × gradient_accumulation_steps

GPU が 1 台しかない環境では中央の項が 1 に固定される。
だが右の項は自由に増やせる。つまり勾配蓄積は「GPU を増やす代わり」になる。

これは主張として理解できるが、成り立つかは確かめないと分からない。
勾配蓄積は「損失を割ってから足す」だけの操作で、
割り忘れや足し忘れがあっても学習は進んでしまうため、動作確認では検出できない。

このデモは 3 つの経路で同じ勾配が出るかを見る。

    A  バッチ 12 を 1 回で処理
    B  バッチ 4 を 3 回に分け、損失を 3 で割って蓄積
    C  バッチ 4 を 3 回に分け、割らずに蓄積（よくある間違い）

A と B が一致し、C がずれるはずである。

    uv run python grad_accum_demo.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

BATCH, D_IN, ACCUM = 12, 8, 3
MICRO = BATCH // ACCUM


def fresh_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Linear(D_IN, 1, bias=False)


def grad_of(model: nn.Module) -> torch.Tensor:
    return model.weight.grad.detach().clone().flatten()


def main() -> None:
    torch.manual_seed(42)
    x = torch.randn(BATCH, D_IN)
    y = torch.randn(BATCH, 1)

    # A  一括
    m = fresh_model()
    nn.functional.mse_loss(m(x), y).backward()
    a = grad_of(m)

    # B  マイクロバッチに割って蓄積。損失を蓄積回数で割る
    m = fresh_model()
    for i in range(ACCUM):
        xs, ys = x[i * MICRO:(i + 1) * MICRO], y[i * MICRO:(i + 1) * MICRO]
        (nn.functional.mse_loss(m(xs), ys) / ACCUM).backward()
    b = grad_of(m)

    # C  割り忘れ
    m = fresh_model()
    for i in range(ACCUM):
        xs, ys = x[i * MICRO:(i + 1) * MICRO], y[i * MICRO:(i + 1) * MICRO]
        nn.functional.mse_loss(m(xs), ys).backward()
    c = grad_of(m)

    fmt = lambda t: "[" + ", ".join(f"{v:+.6f}" for v in t[:4]) + ", ...]"

    print(f"バッチ {BATCH} / マイクロバッチ {MICRO} × 蓄積 {ACCUM} 回\n")
    print("=" * 68)
    print("A  一括で処理")
    print("=" * 68)
    print(f"  {fmt(a)}\n")

    print("=" * 68)
    print("B  マイクロバッチ + 損失を蓄積回数で割る")
    print("=" * 68)
    print(f"  {fmt(b)}")
    ok_b = torch.allclose(a, b, atol=1e-6)
    print(f"  A と一致: {ok_b}  最大誤差 {float((a - b).abs().max()):.2e}\n")

    print("=" * 68)
    print("C  割り忘れ（よくある間違い）")
    print("=" * 68)
    print(f"  {fmt(c)}")
    ok_c = torch.allclose(a, c, atol=1e-6)
    print(f"  A と一致: {ok_c}")
    print(f"  倍率: 約 {float((c / a).mean()):.1f} 倍  ← 蓄積回数ぶん勾配が大きい")
    print("  この状態でも学習は進む。実効的な学習率が 3 倍になるだけで、")
    print("  エラーは出ない。損失も下がるため、動作確認では気づけない。\n")

    print("=" * 68)
    print("グローバルバッチの式")
    print("=" * 68)
    print("  グローバルバッチ = per_device_batch × GPU 数 × 蓄積ステップ数")
    print()
    for dev, acc in ((4, 8), (1, 32), (2, 16)):
        print(f"  per_device 16 × GPU {dev:>2} × 蓄積 {acc:>2} = {16 * dev * acc}")
    print()
    print("  GPU が 1 台でも、蓄積を増やせば同じグローバルバッチに届く。")
    print("  違うのは時間だけで、勾配としては等価になる。")

    print()
    print("判定:", "勾配蓄積は一括処理と等価" if ok_b and not ok_c else "期待と違う")
    raise SystemExit(0 if (ok_b and not ok_c) else 1)


if __name__ == "__main__":
    main()

"""テンソル並列の分解が数学的に厳密であることと、通信量の差を確かめる。

教材の主張は 2 つある。

  1. 2 層 MLP は Colwise → Rowwise の順で分けると、途中の通信が要らなくなる
  2. 同じ線形層でも、どの方向に分けるべきかは位置によって変わる

どちらも数式で書かれているが、成り立つかどうかは実際に計算すれば分かる。
分散環境を立てなくても、行列の分割と再合成だけで検証できる。

    uv run python tensor_parallel_demo.py
"""

from __future__ import annotations

import torch

torch.manual_seed(0)

B, D_IN, D_HID, D_OUT = 2, 6, 8, 6
N_SHARD = 2


def sep(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def main() -> None:
    x = torch.randn(B, D_IN)
    W1 = torch.randn(D_IN, D_HID)
    W2 = torch.randn(D_HID, D_OUT)

    reference = (x @ W1) @ W2

    sep("1  素朴な分け方 — 両方を列方向に割る")
    # W1 を列で割ると、隠れ層の出力も列で分かれる。
    # ここで W2 も列で割ると、各シャードは隠れ層の一部しか持たないのに
    # W2 の行は隠れ層の全次元を要求する。つまり形が合わない。
    W1_cols = torch.chunk(W1, N_SHARD, dim=1)
    h_shards = [x @ w for w in W1_cols]
    print(f"  x{tuple(x.shape)} @ W1_col{tuple(W1_cols[0].shape)} -> h{tuple(h_shards[0].shape)}")
    print(f"  次の層は W2{tuple(W2.shape)} を要求するが、手元の h は {h_shards[0].shape[1]} 次元しかない")
    print("  → 層をまたぐたびに all-gather で h を全部集める必要がある")

    gathered = torch.cat(h_shards, dim=1)
    naive = gathered @ W2
    print(f"  all-gather してから計算した結果は正しいか: {torch.allclose(naive, reference, atol=1e-5)}")

    sep("2  効率的な分け方 — 1 層目を列、2 層目を行で割る")
    # W2 を行で割ると、各シャードは「自分が持つ隠れ次元ぶんの行」だけを持つ。
    # h のシャードと形が合うため、集めずにそのまま掛けられる。
    W2_rows = torch.chunk(W2, N_SHARD, dim=0)
    partials = [h @ w for h, w in zip(h_shards, W2_rows)]
    for i, (h, w, p) in enumerate(zip(h_shards, W2_rows, partials)):
        print(f"  shard{i}: h{tuple(h.shape)} @ W2_row{tuple(w.shape)} -> partial{tuple(p.shape)}")

    efficient = sum(partials)
    print(f"\n  partial の和が単一 GPU の結果と一致するか: "
          f"{torch.allclose(efficient, reference, atol=1e-5)}")
    print(f"  最大誤差: {float((efficient - reference).abs().max()):.2e}")

    sep("3  通信回数の比較（層をまたぐたびに何が要るか）")
    print("  素朴な分け方   層ごとに all-gather   隠れ層の全次元を毎回集める")
    print("  効率的な分け方 最後に all-reduce 1 回 partial 同士を足すだけ")
    print()
    print(f"  1 回の all-gather で動く要素数  = {gathered.numel()} （h 全体）")
    print(f"  1 回の all-reduce で動く要素数  = {efficient.numel()} （出力のみ）")
    print(f"  隠れ層が出力より広いほど差が開く（今回 {D_HID} vs {D_OUT}）")

    sep("4  なぜ方向が変わるのか")
    print("  列で割る（Colwise）  出力側の次元を分ける。入力は全次元が要る")
    print("  行で割る（Rowwise）  入力側の次元を分ける。出力は部分和になる")
    print()
    print("  1 層目を Colwise にすると h が列方向に分かれる。")
    print("  その h を受ける 2 層目は、入力側が分かれている前提の Rowwise でなければ形が合わない。")
    print("  つまり方向はモデルの構造が決めるもので、選べる自由度ではない。")

    sep("5  3 分割でも成り立つか（分割数に依存しないことの確認）")
    for n in (2, 3, 4):
        if D_HID % n:
            print(f"  {n} 分割: 隠れ層 {D_HID} が割り切れないため不可")
            continue
        hs = [x @ w for w in torch.chunk(W1, n, dim=1)]
        ps = [h @ w for h, w in zip(hs, torch.chunk(W2, n, dim=0))]
        ok = torch.allclose(sum(ps), reference, atol=1e-5)
        print(f"  {n} 分割: 一致 {ok}")

    print()
    ok_all = torch.allclose(efficient, reference, atol=1e-5) and torch.allclose(naive, reference, atol=1e-5)
    print("判定:", "分解は厳密。近似ではない" if ok_all else "一致しない")
    raise SystemExit(0 if ok_all else 1)


if __name__ == "__main__":
    main()

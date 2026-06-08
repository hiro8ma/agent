"""デコーディング手法の可視化デモ (beam search / 反復ペナルティ / 長さ制御)

temperature_demo.py（温度 / top-k / top-p）の続き。こちらは「1 トークンずつの
サンプリング」ではなく「系列全体をどう選ぶか・どう破綻を防ぐか」を扱う。

扱う 3 つ:
  1. beam search       系列全体の対数確率を最大化する決定論的探索
                       + length normalization（短い系列が選ばれがちな偏りの補正）
  2. 反復ペナルティ     frequency / presence / repetition で既出トークンを抑える
  3. 長さ制御          min_new_tokens（早すぎる EOS を禁止）/ max_new_tokens

使い方:
    uv run python decoding_demo.py             # checkpoint があれば実生成で比較
    uv run python decoding_demo.py --prompt "ROMEO:"
    uv run python decoding_demo.py --max-tokens 80

checkpoint が無ければ、ロジット加工そのものを合成ロジットで可視化する
フォールバックに切り替わる。

なぜ beam search が系列を見るか:
    次トークン greedy は各ステップで局所最大を取るだけで、直後に高確率トークンが
    続く別経路（系列全体では確率が高い）を捨ててしまう。beam は複数候補を残し、
    系列の累積対数確率で比較するので、局所最適に陥りにくい。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812

if TYPE_CHECKING:
    from data import CharTokenizer
    from model import MiniGPT

# 合成ロジット（checkpoint 無しのフォールバック可視化用）。語彙 6。
SAMPLE_LABELS = ("the", "a", "cat", "dog", "runs", "xyzzy")
SAMPLE_LOGITS = (4.0, 2.5, 2.0, 1.0, 0.5, -2.0)


# === 反復ペナルティ（ロジット加工） ===
# greedy / sampling / beam いずれの前段でも使える純粋なロジット変換。
# generate() への後方互換引数（後述）と同じ規則をここに集約する。


def apply_repetition_penalties(
    logits: torch.Tensor,
    generated: torch.Tensor,
    *,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    """既出トークンのロジットを抑制した新しいロジットを返す（in-place しない）。

    logits:    (vocab,) その位置の生ロジット
    generated: (T,)     これまでに出たトークン ID 列

    3 方式の違い:
      frequency_penalty  出現「回数」に比例して減算 → 連呼ほど強く抑える
      presence_penalty   一度でも出たら一律減算    → 話題の再登場自体を抑える
      repetition_penalty CTRL 流の「除算」          → 符号に依らず確率を縮める
    """
    out = logits.clone()
    if generated.numel() == 0:
        return out

    counts = torch.bincount(generated, minlength=out.size(-1)).to(out.dtype)
    seen = counts > 0

    if frequency_penalty != 0.0:
        out = out - frequency_penalty * counts
    if presence_penalty != 0.0:
        out = out - presence_penalty * seen.to(out.dtype)
    if repetition_penalty != 1.0:
        # 正ロジットは割って下げ、負ロジットは掛けてさらに下げる（CTRL 論文の規則）
        pos = out > 0
        penalized = torch.where(pos, out / repetition_penalty, out * repetition_penalty)
        out = torch.where(seen, penalized, out)
    return out


@dataclass
class Beam:
    """1 本の系列候補。score は系列の累積対数確率。"""

    tokens: list[int]
    score: float = 0.0
    finished: bool = False

    def normalized_score(self, alpha: float) -> float:
        """length normalization 後のスコア。

        累積対数確率は長いほど（負の項が増えて）小さくなるため、無補正だと
        beam search は短い系列を好む。length^alpha で割って長さの不利を緩める。
        alpha=0 で無補正、alpha=1 で平均対数確率に一致。
        """
        length = max(len(self.tokens), 1)
        if alpha == 0.0:
            return self.score
        return self.score / float(length**alpha)


@dataclass
class DecodeConfig:
    """beam search の設定をまとめた値オブジェクト。"""

    beam_width: int = 3
    max_new_tokens: int = 60
    min_new_tokens: int = 0
    length_alpha: float = 0.0
    eos_id: int | None = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    forbidden_ids: tuple[int, ...] = field(default_factory=tuple)


def beam_search(
    model: MiniGPT,
    prompt_ids: list[int],
    config: DecodeConfig,
    device: str = "cpu",
) -> Beam:
    """決定論的な beam search で最良系列を 1 本返す。

    各ステップ: 全 beam を全語彙へ展開 → 累積対数確率上位 beam_width 本を残す。
    EOS or max_new_tokens で停止。length normalization は最終比較で効かせる。
    """
    with torch.no_grad():
        return _beam_search_impl(model, prompt_ids, config, device)[0]


def beam_search_all(
    model: MiniGPT,
    prompt_ids: list[int],
    config: DecodeConfig,
    device: str = "cpu",
) -> list[Beam]:
    """最終的に残った beam_width 本を score 順に返す（ランキング比較用）。"""
    with torch.no_grad():
        return _beam_search_impl(model, prompt_ids, config, device)


def _beam_search_impl(
    model: MiniGPT,
    prompt_ids: list[int],
    config: DecodeConfig,
    device: str,
) -> list[Beam]:
    block_size = model.config.block_size
    prompt_len = len(prompt_ids)
    beams = [Beam(tokens=list(prompt_ids), score=0.0)]

    for step in range(config.max_new_tokens):
        candidates: list[Beam] = []
        for beam in beams:
            if beam.finished:
                candidates.append(beam)
                continue

            idx_cond = torch.tensor([beam.tokens[-block_size:]], dtype=torch.long, device=device)
            logits, _ = model(idx_cond)
            step_logits = logits[0, -1, :]

            generated = torch.tensor(beam.tokens[prompt_len:], dtype=torch.long, device=device)
            step_logits = apply_repetition_penalties(
                step_logits,
                generated,
                frequency_penalty=config.frequency_penalty,
                presence_penalty=config.presence_penalty,
                repetition_penalty=config.repetition_penalty,
            )

            # min_new_tokens に達するまで EOS / forbidden を禁止する
            new_len = len(beam.tokens) - prompt_len
            forbid = set(config.forbidden_ids)
            if config.eos_id is not None and new_len < config.min_new_tokens:
                forbid.add(config.eos_id)
            for fid in forbid:
                step_logits[fid] = float("-inf")

            log_probs = F.log_softmax(step_logits, dim=-1)
            top_lp, top_idx = torch.topk(log_probs, config.beam_width)
            for lp, tid in zip(top_lp.tolist(), top_idx.tolist(), strict=True):
                tokens = [*beam.tokens, tid]
                finished = config.eos_id is not None and tid == config.eos_id
                candidates.append(Beam(tokens=tokens, score=beam.score + lp, finished=finished))

        # 正規化スコアの降順で上位 beam_width 本を残す
        candidates.sort(key=lambda b: b.normalized_score(config.length_alpha), reverse=True)
        beams = candidates[: config.beam_width]
        if all(b.finished for b in beams):
            break
        del step

    beams.sort(key=lambda b: b.normalized_score(config.length_alpha), reverse=True)
    return beams


# === フォールバック可視化（checkpoint 無し時） ===


def _bar(value: float, width: int = 40) -> str:
    filled = round(value * width)
    return "#" * filled + "." * (width - filled)


def _render(labels: tuple[str, ...], probs: torch.Tensor) -> None:
    for label, prob in zip(labels, probs.tolist(), strict=True):
        print(f"  {label:>6} {prob:6.3f} |{_bar(prob)}|")


def show_penalty_on_logits() -> None:
    """合成ロジットで反復ペナルティの効きを可視化する。"""
    logits = torch.tensor(SAMPLE_LOGITS)
    # 'the' が 3 回、'cat' が 1 回既出という想定の履歴
    generated = torch.tensor([0, 0, 0, 2])

    print("=== 反復ペナルティのロジット加工（合成例） ===")
    print("既出: 'the' x3, 'cat' x1\n")

    print("baseline (ペナルティ無し):")
    _render(SAMPLE_LABELS, F.softmax(logits, dim=-1))
    print()

    print("frequency_penalty=0.7 (回数比例で減算 → 'the' が大きく下がる):")
    f_logits = apply_repetition_penalties(logits, generated, frequency_penalty=0.7)
    _render(SAMPLE_LABELS, F.softmax(f_logits, dim=-1))
    print()

    print("presence_penalty=1.5 (既出を一律減算 → 'the' と 'cat' が同程度下がる):")
    p_logits = apply_repetition_penalties(logits, generated, presence_penalty=1.5)
    _render(SAMPLE_LABELS, F.softmax(p_logits, dim=-1))
    print()

    print("repetition_penalty=1.5 (CTRL 流の除算 → 既出を確率方向に縮める):")
    r_logits = apply_repetition_penalties(logits, generated, repetition_penalty=1.5)
    _render(SAMPLE_LABELS, F.softmax(r_logits, dim=-1))
    print()


def show_beam_on_logits() -> None:
    """系列確率の最大化（beam が greedy と分岐する直感）を 2 ステップで示す。"""
    print("=== beam が greedy と分かれる理由（2 ステップの玩具例） ===")
    # step1 の各候補に続く step2 のロジットを手で与える。
    # greedy は step1 で最大の 'A' を取るが、'A' の後は分散して系列確率が伸びない。
    # 'B' は step1 で 2 番手だが、後続が一点集中で系列全体では上回る、という構図。
    step1 = {"A": 0.55, "B": 0.45}
    step2 = {"A": {"x": 0.34, "y": 0.33, "z": 0.33}, "B": {"x": 0.95, "y": 0.03, "z": 0.02}}
    print("step1:  A=0.55  B=0.45     → greedy は A を選ぶ")
    print("step2|A: x=0.34 y=0.33 z=0.33")
    print("step2|B: x=0.95 y=0.03 z=0.02")
    print()
    for first in ("A", "B"):
        for second, sp in step2[first].items():
            joint = step1[first] * sp
            print(f"  系列 {first}{second}: P = {step1[first]:.2f} x {sp:.2f} = {joint:.3f}")
    print("\n  greedy(A→x)=0.187 に対し beam が見つける B→x=0.428 が系列では最良。\n")


# === 実モデルでの比較 ===


def _decode_pretty(tokenizer: CharTokenizer, ids: list[int]) -> str:
    text: str = tokenizer.decode(ids)
    return text.replace("\n", "\\n")


def show_real_decoding(prompt: str, max_tokens: int) -> bool:
    """checkpoint があれば実モデルで beam / length norm / ペナルティを比較する。

    Returns:
        True なら実生成を行った。False なら checkpoint 無しでスキップ。
    """
    from generate import find_latest_checkpoint, load_model  # noqa: PLC0415

    try:
        ckpt_path = find_latest_checkpoint()
    except FileNotFoundError as e:
        print(f"[skip] 実生成は checkpoint が無いため省略: {e}\n")
        return False

    device = "cpu"
    model, tokenizer = load_model(ckpt_path, device)

    try:
        prompt_ids = tokenizer.encode(prompt)
    except KeyError as e:
        print(f"[skip] プロンプトに学習データ外の文字: {e}\n")
        return False

    prompt_len = len(prompt_ids)
    # char tokenizer に EOS は無いので、改行を「文末」相当の停止トークンに使う
    eos_id = tokenizer.char_to_id.get("\n")

    print("\n=== (a) beam_width 別の生成（決定論的） ===")
    print(f"prompt = {prompt!r}\n")
    for width in (1, 3, 5):
        cfg = DecodeConfig(beam_width=width, max_new_tokens=max_tokens, length_alpha=0.0)
        best = beam_search(model, prompt_ids, cfg, device)
        tag = "（= greedy）" if width == 1 else ""
        print(f"--- beam_width = {width} {tag}".rstrip())
        print(_decode_pretty(tokenizer, best.tokens[prompt_len:]))
        print(f"  cumulative log-prob = {best.score:.2f}\n")

    # EOS（このモデルは改行）を許可して長さの違う候補が混ざる beam を作り、
    # 同じ候補集合を「raw 累積対数確率」と「length 正規化スコア」で並べ替えて比較する。
    # raw は短い系列を上位に押し上げがち。alpha>0 がその不利を補正する。
    print("=== (b) length normalization の効果（同一 beam 集合の並べ替え） ===")
    cfg = DecodeConfig(beam_width=6, max_new_tokens=max_tokens, eos_id=eos_id, min_new_tokens=4)
    finals = beam_search_all(model, prompt_ids, cfg, device)

    for alpha in (0.0, 0.7, 1.0):
        ranked = sorted(finals, key=lambda b: b.normalized_score(alpha), reverse=True)
        top = ranked[0]
        gen = top.tokens[prompt_len:]
        label = "raw（無補正）" if alpha == 0.0 else f"alpha={alpha}"
        norm = top.normalized_score(alpha)
        print(f"--- 1 位 by {label} ---")
        print(_decode_pretty(tokenizer, gen))
        print(f"  生成長 = {len(gen)}  raw log-prob = {top.score:.2f}  norm = {norm:.3f}\n")

    print("  最終 beam 集合（生成長 / raw / norm@1.0）:")
    for b in sorted(finals, key=lambda x: len(x.tokens)):
        g = b.tokens[prompt_len:]
        text = _decode_pretty(tokenizer, g)[:40]
        norm1 = b.normalized_score(1.0)
        print(f"    len={len(g):>3}  raw={b.score:7.2f}  norm={norm1:6.3f}  {text!r}")
    raw_gap = abs(min(b.score for b in finals) - max(b.score for b in finals))
    norm_gap = abs(
        min(b.normalized_score(1.0) for b in finals) - max(b.normalized_score(1.0) for b in finals)
    )
    print(f"\n  raw のスコア幅 {raw_gap:.1f} に対し norm の幅は {norm_gap:.2f}。")
    print("  正規化は長い系列の不利を縮め、長さ依存だけで順位が決まるのを防ぐ。\n")

    print("=== (c) 反復ペナルティ有無（sampling, 同一 seed で条件を揃える） ===")
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    print("--- ペナルティ無し ---")
    torch.manual_seed(0)
    out = model.generate(idx, max_new_tokens=max_tokens, temperature=0.9, top_k=None)
    print(_decode_pretty(tokenizer, out[0, prompt_len:].tolist()))
    print()
    print("--- frequency=0.8 + presence=0.4 ---")
    torch.manual_seed(0)
    out = model.generate(
        idx,
        max_new_tokens=max_tokens,
        temperature=0.9,
        top_k=None,
        frequency_penalty=0.8,
        presence_penalty=0.4,
    )
    print(_decode_pretty(tokenizer, out[0, prompt_len:].tolist()))
    print()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="\n", help="生成の起点プロンプト")
    parser.add_argument("--max-tokens", type=int, default=60, help="生成する最大 token 数")
    args = parser.parse_args()

    ckpt_dir = Path(__file__).parent / "checkpoints"
    did_generate = False
    if ckpt_dir.exists():
        did_generate = show_real_decoding(args.prompt, args.max_tokens)

    # checkpoint 無し（or プロンプト不正）なら合成ロジットのデモにフォールバック
    if not did_generate:
        show_beam_on_logits()
        show_penalty_on_logits()


if __name__ == "__main__":
    main()

# constrained_decoding

制約付きデコーディング（logits masking）の最小実装デモ。

生成時のロジット（softmax 前の対数確率）に対し、禁止トークンへ `-inf` を代入して
確率をちょうど 0 にし、サンプリング / beam search で物理的に選べなくする手法を、
Hugging Face Transformers の `LogitsProcessor` と同じ呼び出し契約
（`__call__(input_ids, scores) -> scores`）で再現する。

## これは何の最小核か

本番の制約付き生成スタック（[Outlines](https://github.com/dottxt-ai/outlines) /
llama.cpp の GBNF grammar / vLLM の guided decoding）は、文法・JSON Schema・正規表現から
「いまの状態で次に来てよいトークン集合」を動的に計算し、その集合外のロジットを `-inf` に
マスクして無効化している。本デモはその allow-list 計算と `-inf` 代入だけを numpy で
取り出したもので、JSON 値の位置に値トークンだけを許す最小の状態機械を含む。
実モデル・GPU・ネットワークは不要で、固定ロジットだけでオフライン完結する。

## 実行

```bash
uv sync
uv run python bin/decode_demo.py
```

## 構成

| ファイル | 役割 |
|---|---|
| `processors.py` | `BannedTokens` / `AllowedTokens` / `ScoredBestOf`（スタイル制御の soft mask） / `LogitsProcessorList` / 数値安定 softmax |
| `demo.py` | マスク有無で出力分布がどう変わるかを top-k / サンプリング / best-of で対比 |
| `bin/decode_demo.py` | 実行エントリ |

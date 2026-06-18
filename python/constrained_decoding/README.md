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

## グラマーパターン（フレームワークに委譲したロジットマスキング）

文法制約は「次に来てよいトークン集合」の計算を文法定義に丸ごと委譲する形のロジットマスキング。
指定方法は 3 通りある。BNF / EBNF などの文法記法、JSON Schema / 正規表現といった標準形式、
そしてアプリ側が渡すユーザースキーマ。いずれも内部では有限状態機械や push-down automaton に
コンパイルされ、状態ごとの許可集合外のロジットを `-inf` にして文法外の出力を封じる。
本デモの `grammar.py` は、パイプ区切りレコード `author | title | year`（year は 4 桁数字または NULL）を
有限状態機械として表し、`AllowedTokensLogitsProcessor` を状態遷移で駆動する最小版。
本番の対応物は [transformers-cfg](https://github.com/epfl-dlab/transformers-CFG) /
OpenAI Structured Outputs / Gemini の `response_schema`。

## 実行

```bash
uv sync
uv run python bin/decode_demo.py
```

## 構成

| ファイル | 役割 |
|---|---|
| `processors.py` | `BannedTokens` / `AllowedTokens` / `ScoredBestOf`（スタイル制御の soft mask） / `LogitsProcessorList` / 数値安定 softmax |
| `grammar.py` | `FsmGrammar`（`author | title | year` を有限状態機械で表す BNF 文法制約の最小モデル） |
| `demo.py` | マスク有無で出力分布がどう変わるかを top-k / サンプリング / best-of / FSM 文法で対比 |
| `bin/decode_demo.py` | 実行エントリ |

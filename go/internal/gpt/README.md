# gpt — 純 Go の GPT 学習・生成実装

外部依存ゼロ・autograd なしで、各層の forward / backward を手書きした文字レベル GPT。
デコーダのみ・Pre-LN・学習可能位置埋め込み・causal attention・weight tying。

## 構成

| ファイル | 内容 |
|---|---|
| `tokenizer.go` | 文字レベルトークナイザ |
| `layers.go` | 各層の forward / backward（encoder, LayerNorm, matmul, attention, GELU, residual, softmax + cross entropy） |
| `model.go` | Config、パラメータ、Forward / Backward |
| `optimizer.go` | AdamW（decay は線形層重みのみ）、勾配クリップ、warmup + cosine スケジュール |
| `trainer.go` | データセット（1 トークンずらし）と学習ループ |
| `sample.go` | 温度 + top-k サンプリング生成 |
| `gpt_test.go` | 数値微分との勾配一致検証、損失減少、生成の再現性 |

## 使い方

```bash
# 内蔵サンプル（『吾輩は猫である』冒頭）で動作確認
go run ./cmd/gpt -steps 600 -block 32 -embd 64 -layers 2 -heads 4

# 本格学習は青空文庫などのテキストを指定
go run ./cmd/gpt -data data/soseki.txt -steps 5000 -block 128 -embd 128 -layers 4
```

『吾輩は猫である』全文の取得と整形（ルビ・注記の除去）:

```bash
mkdir -p data && cd data
curl -LO https://www.aozora.gr.jp/cards/000148/files/789_ruby_5639.zip
unzip -o 789_ruby_5639.zip
iconv -f SHIFT_JIS -t UTF-8 wagahaiwa_nekodearu.txt \
  | sed -e 's/《[^》]*》//g' -e 's/［＃[^］]*］//g' -e 's/｜//g' > soseki.txt
```

## テスト

```bash
go test ./internal/gpt
```

勾配検証は全パラメータテンソルからサンプリングした点で、解析勾配と中心差分の数値勾配の一致を確認する。

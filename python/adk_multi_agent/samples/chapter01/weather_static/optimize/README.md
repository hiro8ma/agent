# adk optimize の実行と実測

`weather_agent` は `instruction` に関数を渡している（動的 Instruction）。
GEPA は `instruction` を候補として JSON 化するため、関数だと落ちる。

```
TypeError: Object of type function is not JSON serializable
  gepa/core/state.py:33  json.dumps(sorted(candidate.items()))
```

`weather_static` は同じツールで `instruction` を文字列にした対照。
`adk optimize` を試すにはこちらを使う。

## 実行

```sh
uv add --dev 'google-adk[eval]==2.2.0'   # gepa と pandas が入る
export GOOGLE_API_KEY=...
adk optimize samples/chapter01/weather_static \
  --sampler_config_file_path samples/chapter01/weather_static/optimize/sampler.json \
  --optimizer_config_file_path samples/chapter01/weather_static/optimize/optimizer.json
```

## 実測（max_metric_calls=6、学習 2 件、検証 1 件）

```
所要 85 秒
429 RESOURCE_EXHAUSTED  17 回（無料枠 5 rpm）
評価バッチ               4 回
反復                     1 回
```

既定の `max_metric_calls` は 100 なので、この 16 倍にあたる。
評価 1 件あたりのエージェント側 LLM 呼び出しは平均 1.73 回（別途実測）。

## この実行で分かったこと

提案された Instruction は「都市名を小文字の英語に変換してツールへ渡す」
という指示を足していた。ツール側は日本語を受けるので不要な指示になる。

原因は評価セットとエージェントのずれだった。評価セットは天気の質問に
`get_weather` だけを期待していたが、Instruction は「両方呼ぶ」に
なっていた。スコアが全件 0.0 になり、GEPA はその原因を都市名の形式だと
推測した。

**壊れた測定に対して最適化すると、壊れた方向へ改善案が出る。**

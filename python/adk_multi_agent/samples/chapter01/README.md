# 第 1 章 天気取得エージェント（最小構成）

```bash
export GOOGLE_API_KEY=...
cd samples/chapter01
uv run adk run weather_agent   # ターミナルで対話
uv run adk web                 # ブラウザの開発 UI
```

## Go 版との対応

同じエージェントを `agent/go/cmd/adk-ch01/` に置いた。構造の違いは 2 点。

| | Python | Go |
|---|---|---|
| エントリーポイント | `root_agent` という**変数名の規約** | `agent.NewSingleLoader(a)` で**明示的に渡す** |
| ツールのスキーマ | 型注釈と docstring から**自動生成** | `tool.Tool` を**明示的に組み立てる** |

Python は規約で短く書ける代わりに、規約を外れたときに何も言わない。
Go は書く量が増える代わりに、繋ぎ忘れがコンパイルで出る。

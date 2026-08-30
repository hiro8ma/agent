# ADK 教材の Go 版

教材（ADK マルチエージェント本）のサンプルを Go でも書く。
Python 版は `agent/python/adk_multi_agent/samples/chapterNN/` にある。

同じことを 2 つの言語で書くと、フレームワークが何を隠しているかが見える。

## バージョン

| | 最新 | 使用 |
|---|---|---|
| Python `google-adk` | 2.8.0 | **2.2.0**（教材が固定） |
| Go `google.golang.org/adk/v2` | **2.2.0** | 2.2.0 |

Go 版の最新が、たまたま教材の指定と同じ番号になる。
Python は 6 マイナーバージョン先行しており、リリース間隔は約 2 週間。

## 章ごとの対応

Go 版に評価のパッケージが無い。そこだけ自作のハーネスで埋める。

| 章 | 内容 | Go v2.2.0 |
|---|---|---|
| 1 | 天気取得（最小構成） | `agent/llmagent` `tool/functiontool` |
| 2 | 旅行プランナー（Sequential + Parallel） | `agent/workflowagents/{sequentialagent,parallelagent,loopagent}` |
| 3 | サポート + Agent Skills | `tool` 系 15 パッケージ |
| 4 | Memory Engineering | `memory` `memory/vertexai` `tool/loadmemorytool` |
| **5** | **経費精算 + 評価セット** | **無い → `internal/evalharness` で埋める** |
| 6 | インフラ監視（MCP + CLI） | `tool/mcptoolset` |
| 7 | 経費精算 A2A + オーケストレーター | `server/adka2a` `/v2` `agent/remoteagent/v2` |
| 8 | デプロイ・監視 | `telemetry` `cmd/adkgo/internal/deploy/{cloudrun,agentengine}` |
| **9** | **設計レビュー自動化 + 評価セット** | **無い → 同上** |
| 10 | セキュリティ強化 | `plugin` 4 パッケージ |

## 2 言語で書くと見える差

第 1 章の時点で 2 つ出ている。

| | Python | Go |
|---|---|---|
| エントリーポイント | `root_agent` という**変数名の規約** | `agent.NewSingleLoader(a)` で**明示的に渡す** |
| ツールのスキーマ | 型注釈と docstring から**自動生成** | 構造体の json タグで**明示** |

Python は規約で短く書ける代わりに、規約を外れたときに何も言わない。
Go は書く量が増える代わりに、繋ぎ忘れがコンパイルで出る。

教材が「ボイラープレートは AI に書かせて設計判断に集中する」と言うとき、
Go 側のほうが「ボイラープレート」の量が多い。
その分、レビューで見る対象は増える。

## 実行

```bash
export GOOGLE_API_KEY=...
go run ./cmd/adkbook-ch01           # 対話
go run ./cmd/adkbook-ch01 web       # 開発 UI
```

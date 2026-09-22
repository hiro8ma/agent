# 運用の画面（手元の観測の一式）

OTel Collector、Prometheus、Tempo、Loki、Grafana を docker compose で立て、ホストで動かすサービスのメトリクス、トレース、ログを集める。
本物のモデルは呼ばない。負荷は台本のモデルで動く `ops-demo-agent` にかける。

## 使い方

`go/` から動かす。

```sh
make up      # 一式とサービスを起動する
make load    # 負荷をかける（LOAD_RPS=4 LOAD_DURATION=10m）
make open    # http://localhost:3000 の Agent Ops フォルダ
make down    # サービスと一式を止める
```

`ops/` では `make check` で、ダッシュボードの全パネルのクエリがデータを返すかを確かめられる。`make test-rules` で、SLO のアラートが合成の系列で発火するか（正常なら発火しないか）を確かめられる。

| 画面 | URL |
| --- | --- |
| Grafana | http://localhost:3000（手元だけに開く。ログインなしの管理者） |
| Prometheus | http://localhost:9090（ルールとアラートの状態） |

ポートは 127.0.0.1 だけに開く。Grafana はログインを外しているので、外に出す環境では使わない。

## 流れ

```
サービス（ホスト） --OTLP--> Collector --> Prometheus（メトリクス） / Tempo（トレース）
サービスの JSON のログ（.run/logs） --filelog--> Collector --> Loki
Grafana <-- Prometheus / Tempo / Loki（ログの trace_id からトレースへ、トレースからログへ移れる）
```

## ダッシュボード

| 名前 | 見るもの |
| --- | --- |
| サービスヘルス | RPS、失敗の割合、遅延 P50 / P95 / P99、可用性、エラーバジェットの消費速度、ログ |
| エージェント品質 | 実行の結果（完了 / 失敗 / エスカレーション）、ツールの成功率、モデルの失敗と再試行、ガードレール |
| コスト | トークン（入力 / 出力 / キャッシュ / 思考）、モデル別とエージェント別、費用、予算に対する割合 |
| セキュリティ | 認証の失敗、ツールの拒否、ガードレール |

JSON は `grafana/build_dashboards.py` で作る。直すときは Python を直して作り直す。

## SLO とアラート

`prometheus/rules/slo.yml` に、目標（`slo:*:objective`）、SLI、アラートを置く。目標の値はこのファイルの 1 か所だけに書き、アラートとダッシュボードはその系列を読む。

| アラート | 条件 | 重要度 |
| --- | --- | --- |
| AvailabilityFastBurn | 可用性 99.5% のエラーバジェットを 1 時間と 5 分の両方で 14.4 倍の速さで消費（2 分続く） | critical |
| LatencyP99High | P99 が 30 秒を超える（5 分続く） | critical |
| GenAIErrorRatioHigh | モデルとツールの失敗が 5% を超える（5 分続く） | high |
| ToolSuccessLow | ツールの成功率が 95% を下回る（10 分続く） | high |
| DailyCostOverBudget | 今日の費用が 1 日の予算の 120% を超える（5 分続く） | high |
| EscalationRateHigh | エスカレーション率（1 時間の窓）が 20% を超える（15 分続く） | medium |

通知先（Alertmanager）はつないでいない。Prometheus の Alerts の画面で状態を見る。

## メトリクス

Go の計装は `go/internal/lib/genaimetrics`（ADK 用は `adkmetrics`）。名前は OTel の GenAI の規約に寄せた。

| メトリクス | 属性 |
| --- | --- |
| `gen_ai.client.token.usage` | モデル、エージェント、種類（input / output と、その内訳の cached_input / reasoning） |
| `gen_ai.client.operation.duration` | chat / execute_tool、モデル、エージェント、ツール、error.type（HTTP のステータス、tool_error、denied） |
| `gen_ai.client.cost` / `gen_ai.client.cost.budget` | モデル、エージェント（USD） |
| `gen_ai.client.retries` | モデル、error.type |
| `gen_ai.agent.invocations` | エージェント、結果（completed / failed / escalated） |
| `gen_ai.guardrail.blocks` | 段、規則 |
| `rpc.server.*` / `rpc.client.*` | otelconnect が出す RPC のメトリクス |

属性に利用者の ID、本文、検索語は入れない。

## ほかのリポジトリへ移すときに差し替える箇所

- `config/services.env`：ポート、下流の URL、単価（`GENAI_PRICES`）、1 日の予算（`GENAI_DAILY_BUDGET_USD`）
- `scripts/run-services.sh` の `SERVICES`：動かすサービスの名前、cmd のディレクトリ、ポートの環境変数
- `prometheus/rules/slo.yml` の `slo-objectives`：SLO の目標
- `go/internal/lib/genaimetrics` を移し、ADK のランナーに `adkmetrics.Plugin` を渡す。引き継ぎ（承認待ちなど）の判定は `WithEscalation` で渡す
- RPC のメトリクスは Connect の otelconnect のインターセプターが前提。gRPC なら otelgrpc の名前（`rpc_server_duration_milliseconds` など）に合わせてルールを直す

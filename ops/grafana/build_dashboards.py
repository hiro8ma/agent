"""運用のダッシュボード 4 枚の JSON を作る。python3 build_dashboards.py で dashboards/ に書く。

閾値は数値で書かず、Prometheus の slo:*:objective の系列を重ねて見せる。
サービス、エージェント、モデルは変数で絞る。特定のサービスの名前は書かない。
"""

from __future__ import annotations

import json
from pathlib import Path

PROM = {"type": "prometheus", "uid": "prometheus"}
LOKI = {"type": "loki", "uid": "loki"}
SVC = 'service_name=~"$service"'
AGENT = 'gen_ai_agent_name=~"$agent"'
MODEL = 'gen_ai_request_model=~"$model"'
SERVER_ERROR = 'rpc_connect_rpc_error_code=~"unknown|internal|unavailable|data_loss|deadline_exceeded|unimplemented"'


def var(name: str, label: str, query: str) -> dict:
    return {
        "name": name,
        "label": label,
        "type": "query",
        "datasource": PROM,
        "query": {"query": query, "refId": name},
        "definition": query,
        "includeAll": True,
        "multi": True,
        "allValue": ".+",
        "current": {"text": "All", "value": "$__all"},
        "refresh": 2,
        "sort": 1,
    }


SERVICE_VAR = var("service", "サービス", "label_values(rpc_server_duration_milliseconds_count, service_name)")
AGENT_VAR = var("agent", "エージェント", 'label_values(gen_ai_agent_invocations_total{service_name=~"$service"}, gen_ai_agent_name)')
MODEL_VAR = var("model", "モデル", 'label_values(gen_ai_client_token_usage_count{service_name=~"$service"}, gen_ai_request_model)')


class Board:
    def __init__(self, uid: str, title: str, description: str, variables: list[dict]):
        self.uid, self.title, self.description = uid, title, description
        self.variables = variables
        self.panels: list[dict] = []
        self.x = self.y = 0
        self.row_h = 0

    def _place(self, w: int, h: int) -> dict:
        if self.x + w > 24:
            self.x, self.y = 0, self.y + self.row_h
            self.row_h = 0
        pos = {"x": self.x, "y": self.y, "w": w, "h": h}
        self.x += w
        self.row_h = max(self.row_h, h)
        return pos

    def row(self, title: str) -> None:
        self.x, self.y = 0, self.y + self.row_h
        self.row_h = 1
        self.panels.append({"type": "row", "title": title, "collapsed": False, "gridPos": {"x": 0, "y": self.y, "w": 24, "h": 1}, "id": len(self.panels) + 1})
        self.x, self.y, self.row_h = 0, self.y + 1, 0

    def timeseries(self, title: str, targets: list[tuple[str, str]], unit: str = "short", w: int = 12, h: int = 8,
                   stack: bool = False, description: str = "", objective: str | None = None) -> None:
        tgts = [{"datasource": PROM, "expr": e, "legendFormat": leg, "refId": chr(65 + i)} for i, (e, leg) in enumerate(targets)]
        overrides = []
        if objective:
            tgts.append({"datasource": PROM, "expr": objective, "legendFormat": "目標", "refId": "Z"})
            overrides.append({
                "matcher": {"id": "byName", "options": "目標"},
                "properties": [
                    {"id": "custom.lineStyle", "value": {"fill": "dash", "dash": [10, 10]}},
                    {"id": "color", "value": {"mode": "fixed", "fixedColor": "red"}},
                    {"id": "custom.stacking", "value": {"mode": "none"}},
                ],
            })
        self.panels.append({
            "type": "timeseries", "title": title, "description": description, "datasource": PROM,
            "gridPos": self._place(w, h), "id": len(self.panels) + 1, "targets": tgts,
            "fieldConfig": {
                "defaults": {"unit": unit, "custom": {"fillOpacity": 15 if stack else 5, "stacking": {"mode": "normal" if stack else "none"}}},
                "overrides": overrides,
            },
            "options": {"legend": {"displayMode": "table", "placement": "bottom", "calcs": ["lastNotNull", "max"]}, "tooltip": {"mode": "multi"}},
        })

    def stat(self, title: str, expr: str, unit: str = "short", w: int = 6, h: int = 4, description: str = "", decimals: int = 2) -> None:
        self.panels.append({
            "type": "stat", "title": title, "description": description, "datasource": PROM,
            "gridPos": self._place(w, h), "id": len(self.panels) + 1,
            "targets": [{"datasource": PROM, "expr": expr, "refId": "A", "instant": True}],
            "fieldConfig": {"defaults": {"unit": unit, "decimals": decimals}, "overrides": []},
            "options": {"reduceOptions": {"calcs": ["lastNotNull"]}, "colorMode": "none", "graphMode": "none"},
        })

    def logs(self, title: str, expr: str, w: int = 24, h: int = 10) -> None:
        self.panels.append({
            "type": "logs", "title": title, "datasource": LOKI, "gridPos": self._place(w, h), "id": len(self.panels) + 1,
            "targets": [{"datasource": LOKI, "expr": expr, "refId": "A"}],
            "options": {"showTime": True, "wrapLogMessage": True, "enableLogDetails": True, "sortOrder": "Descending"},
        })

    def json(self) -> dict:
        return {
            "uid": self.uid, "title": self.title, "description": self.description, "tags": ["agent-ops"],
            "schemaVersion": 41, "version": 1, "editable": True, "refresh": "30s",
            "time": {"from": "now-1h", "to": "now"},
            "templating": {"list": self.variables}, "panels": self.panels,
            "links": [{"type": "dashboards", "tags": ["agent-ops"], "asDropdown": False, "title": "運用のダッシュボード"}],
        }


def service_health() -> Board:
    b = Board("agent-ops-service-health", "サービスヘルス", "RPC の量、失敗、遅延、可用性とエラーバジェット", [SERVICE_VAR])
    b.row("概要")
    b.stat("可用性（5 分）", f"min(sli:availability:ratio_rate5m{{{SVC}}})", "percentunit")
    b.stat("P99（5 分）", f"max(sli:rpc_latency_p99_seconds:5m{{{SVC}}})", "s")
    b.stat("RPS", f"sum(sli:rpc_requests:rate5m{{{SVC}}})", "reqps")
    b.stat("エラーバジェットの消費速度（1 時間）", f"max(slo:availability:burn_rate1h{{{SVC}}})", "x",
           description="1 なら SLO の期間でちょうど使い切る速さ。14.4 を超えるとアラート")
    b.row("量と失敗")
    b.timeseries("RPS（サービス別）", [(f"sum by (service_name) (rate(rpc_server_duration_milliseconds_count{{{SVC}}}[$__rate_interval]))", "{{service_name}}")], "reqps")
    b.timeseries("RPS（RPC 別）", [(f"sum by (service_name, rpc_method) (rate(rpc_server_duration_milliseconds_count{{{SVC}}}[$__rate_interval]))", "{{service_name}} {{rpc_method}}")], "reqps")
    b.timeseries("失敗の割合（コード別）",
                 [(f"sum by (service_name, rpc_connect_rpc_error_code) (rate(rpc_server_duration_milliseconds_count{{{SVC}, rpc_connect_rpc_error_code!=\"\"}}[$__rate_interval])) / ignoring (rpc_connect_rpc_error_code) group_left sum by (service_name) (rate(rpc_server_duration_milliseconds_count{{{SVC}}}[$__rate_interval]))",
                   "{{service_name}} {{rpc_connect_rpc_error_code}}")], "percentunit",
                 description="利用者側の誤り（unauthenticated など）も含む。可用性の計算ではサーバー側の失敗だけを数える")
    b.timeseries("可用性（5 分）", [(f"sli:availability:ratio_rate5m{{{SVC}}}", "{{service_name}}")], "percentunit",
                 objective="slo:availability:objective")
    b.row("遅延")
    for q, name in [(0.5, "P50"), (0.95, "P95"), (0.99, "P99")]:
        b.timeseries(f"遅延 {name}（サービス別）",
                     [(f"histogram_quantile({q}, sum by (le, service_name) (rate(rpc_server_duration_milliseconds_bucket{{{SVC}}}[$__rate_interval]))) / 1000", "{{service_name}}")],
                     "s", w=8, objective="slo:latency_p99_seconds:objective" if name == "P99" else None)
    b.timeseries("遅延 P99（RPC 別）",
                 [(f"histogram_quantile(0.99, sum by (le, service_name, rpc_method) (rate(rpc_server_duration_milliseconds_bucket{{{SVC}}}[$__rate_interval]))) / 1000", "{{service_name}} {{rpc_method}}")], "s", w=24)
    b.row("エラーバジェット")
    b.timeseries("消費速度（burn rate）", [(f"slo:availability:burn_rate1h{{{SVC}}}", "1 時間 {{service_name}}"), (f"slo:availability:burn_rate5m{{{SVC}}}", "5 分 {{service_name}}")], "x", w=24)
    b.row("ログ")
    b.logs("警告と失敗のログ（trace_id からトレースへ移れる）", '{service_name=~"$service"} | severity_text=~"WARN|ERROR"', w=12)
    b.logs("すべてのログ", '{service_name=~"$service"}', w=12)
    return b


def agent_quality() -> Board:
    b = Board("agent-ops-agent-quality", "エージェント品質", "実行の結果、ツールとモデルの失敗、ガードレール", [SERVICE_VAR, AGENT_VAR])
    inv = f"gen_ai_agent_invocations_total{{{SVC}, {AGENT}}}"
    b.row("実行の結果")
    b.stat("完了率（1 時間）", f"sum(rate({inv[:-1]}, gen_ai_agent_outcome=\"completed\"}}[1h])) / sum(rate({inv}[1h]))", "percentunit")
    b.stat("エスカレーション率（1 時間）", f"sum(rate({inv[:-1]}, gen_ai_agent_outcome=\"escalated\"}}[1h])) / sum(rate({inv}[1h]))", "percentunit")
    b.stat("失敗率（1 時間）", f"sum(rate({inv[:-1]}, gen_ai_agent_outcome=\"failed\"}}[1h])) / sum(rate({inv}[1h]))", "percentunit")
    b.stat("ガードレールが止めた回数（1 時間）", f"sum(increase(gen_ai_guardrail_blocks_total{{{SVC}}}[1h]))", decimals=0)
    b.timeseries("実行の結果（エージェント別）", [(f"sum by (gen_ai_agent_name, gen_ai_agent_outcome) (rate({inv}[$__rate_interval]))", "{{gen_ai_agent_name}} {{gen_ai_agent_outcome}}")], "ops", stack=True)
    b.timeseries("エスカレーション率（1 時間の窓）", [(f"sli:escalation:ratio_rate1h{{{SVC}}}", "{{service_name}}")], "percentunit",
                 objective="slo:escalation_ratio:objective", description="業務の指標。短い窓では振れが大きいので 1 時間で見る")
    b.row("ツール")
    tool = f'gen_ai_client_operation_duration_seconds_count{{{SVC}, {AGENT}, gen_ai_operation_name="execute_tool"'
    b.timeseries("ツールの成功率（拒否を除く）", [(f"sli:tool_success:ratio_rate5m{{{SVC}}}", "{{service_name}}")], "percentunit", objective="slo:tool_success:objective")
    b.timeseries("ツールの呼び出し（結果別）", [(f'sum by (gen_ai_tool_name, error_type) (rate({tool}}}[$__rate_interval]))', "{{gen_ai_tool_name}} {{error_type}}")], "ops", stack=True,
                 description="error_type が空なら成功、tool_error は失敗、denied は権限の拒否")
    b.timeseries("ツールの所要時間 P95", [(f'histogram_quantile(0.95, sum by (le, gen_ai_tool_name) (rate(gen_ai_client_operation_duration_seconds_bucket{{{SVC}, {AGENT}, gen_ai_operation_name="execute_tool"}}[$__rate_interval])))', "{{gen_ai_tool_name}}")], "s", w=24)
    b.row("モデル")
    b.timeseries("モデルとツールの失敗率", [(f"sli:genai_operation_errors:ratio_rate5m{{{SVC}}}", "{{service_name}}")], "percentunit", objective="slo:error_ratio:objective")
    b.timeseries("モデルの失敗と再試行", [
        (f'sum by (gen_ai_request_model, error_type) (rate(gen_ai_client_operation_duration_seconds_count{{{SVC}, {AGENT}, gen_ai_operation_name="chat", error_type!=""}}[$__rate_interval]))', "失敗 {{gen_ai_request_model}} {{error_type}}"),
        (f"sum by (gen_ai_request_model, error_type) (rate(gen_ai_client_retries_total{{{SVC}}}[$__rate_interval]))", "再試行 {{gen_ai_request_model}} {{error_type}}"),
    ], "ops")
    b.timeseries("モデルの呼び出しの所要時間 P50 / P95", [
        (f'histogram_quantile({q}, sum by (le, gen_ai_request_model) (rate(gen_ai_client_operation_duration_seconds_bucket{{{SVC}, {AGENT}, gen_ai_operation_name="chat"}}[$__rate_interval])))', f"P{int(q * 100)} {{{{gen_ai_request_model}}}}")
        for q in (0.5, 0.95)
    ], "s")
    b.timeseries("ガードレールが止めた回数（規則別）", [(f"sum by (gen_ai_guardrail_stage, gen_ai_guardrail_rule) (rate(gen_ai_guardrail_blocks_total{{{SVC}}}[$__rate_interval]))", "{{gen_ai_guardrail_stage}} {{gen_ai_guardrail_rule}}")], "ops")
    return b


def cost() -> Board:
    b = Board("agent-ops-cost", "コスト", "トークン、費用、予算", [SERVICE_VAR, AGENT_VAR, MODEL_VAR])
    tok = f"gen_ai_client_token_usage_sum{{{SVC}, {AGENT}, {MODEL}"
    cst = f"gen_ai_client_cost_total{{{SVC}, {AGENT}, {MODEL}}}"
    b.row("予算")
    b.stat("今日の費用（USD）", f"sum(increase({cst}[1d]))", "currencyUSD")
    b.stat("今日の予算に対する割合", f"max(sli:daily_cost_budget:ratio{{{SVC}}})", "percentunit")
    b.stat("キャッシュから読んだ入力の割合（1 時間）", f'sum(increase({tok}, gen_ai_token_type="cached_input"}}[1h])) / sum(increase({tok}, gen_ai_token_type="input"}}[1h]))', "percentunit")
    b.stat("1 回の実行あたりの入力トークン（1 時間）", f'sum(increase({tok}, gen_ai_token_type="input"}}[1h])) / sum(increase(gen_ai_agent_invocations_total{{{SVC}, {AGENT}}}[1h]))', decimals=0)
    b.timeseries("予算に対する割合（今日）", [(f"sli:daily_cost_budget:ratio{{{SVC}}}", "{{service_name}}")], "percentunit", objective="slo:daily_cost_budget_ratio:objective", w=24)
    b.row("トークン")
    b.timeseries("トークン（種類別）", [(f"sum by (gen_ai_token_type) (rate({tok}}}[$__rate_interval]))", "{{gen_ai_token_type}}")], "short",
                 description="input と output が全量。cached_input は input の内訳、reasoning は output の内訳")
    b.timeseries("トークン（モデル別、入力と出力）", [(f'sum by (gen_ai_request_model, gen_ai_token_type) (rate({tok}, gen_ai_token_type=~"input|output"}}[$__rate_interval]))', "{{gen_ai_request_model}} {{gen_ai_token_type}}")], "short")
    b.timeseries("トークン（エージェント別、入力と出力）", [(f'sum by (gen_ai_agent_name, gen_ai_token_type) (rate({tok}, gen_ai_token_type=~"input|output"}}[$__rate_interval]))', "{{gen_ai_agent_name}} {{gen_ai_token_type}}")], "short", w=24)
    b.row("費用")
    b.timeseries("1 時間あたりの費用（モデル別）", [(f"sum by (gen_ai_request_model) (increase({cst}[1h]))", "{{gen_ai_request_model}}")], "currencyUSD", stack=True)
    b.timeseries("1 時間あたりの費用（エージェント別）", [(f"sum by (gen_ai_agent_name) (increase({cst}[1h]))", "{{gen_ai_agent_name}}")], "currencyUSD", stack=True)
    b.timeseries("費用の日次推移", [(f"sum(increase({cst}[1d]))", "直近 24 時間")], "currencyUSD", w=24)
    return b


def security() -> Board:
    b = Board("agent-ops-security", "セキュリティ", "認証の失敗、権限の拒否、ガードレール", [SERVICE_VAR, AGENT_VAR])
    b.row("概要")
    b.stat("認証の失敗（1 時間）", f'sum(increase(rpc_server_duration_milliseconds_count{{{SVC}, rpc_connect_rpc_error_code="unauthenticated"}}[1h]))', decimals=0)
    b.stat("ツールの拒否（1 時間）", f'sum(increase(gen_ai_client_operation_duration_seconds_count{{{SVC}, {AGENT}, error_type="denied"}}[1h]))', decimals=0)
    b.stat("ガードレールが止めた回数（1 時間）", f"sum(increase(gen_ai_guardrail_blocks_total{{{SVC}}}[1h]))", decimals=0)
    b.row("推移")
    b.timeseries("認証の失敗（サービスと RPC 別）", [(f'sum by (service_name, rpc_method) (rate(rpc_server_duration_milliseconds_count{{{SVC}, rpc_connect_rpc_error_code="unauthenticated"}}[$__rate_interval]))', "{{service_name}} {{rpc_method}}")], "ops")
    b.timeseries("認証の失敗の割合", [(f'sum by (service_name) (rate(rpc_server_duration_milliseconds_count{{{SVC}, rpc_connect_rpc_error_code="unauthenticated"}}[$__rate_interval])) / sum by (service_name) (rate(rpc_server_duration_milliseconds_count{{{SVC}}}[$__rate_interval]))', "{{service_name}}")], "percentunit")
    b.timeseries("ツールの拒否（エージェントとツール別）", [(f'sum by (gen_ai_agent_name, gen_ai_tool_name) (rate(gen_ai_client_operation_duration_seconds_count{{{SVC}, {AGENT}, error_type="denied"}}[$__rate_interval]))', "{{gen_ai_agent_name}} {{gen_ai_tool_name}}")], "ops")
    b.timeseries("ガードレールが止めた回数（規則別）", [(f"sum by (gen_ai_guardrail_stage, gen_ai_guardrail_rule) (rate(gen_ai_guardrail_blocks_total{{{SVC}}}[$__rate_interval]))", "{{gen_ai_guardrail_stage}} {{gen_ai_guardrail_rule}}")], "ops")
    return b


if __name__ == "__main__":
    out = Path(__file__).parent / "dashboards"
    out.mkdir(exist_ok=True)
    for board in (service_health(), agent_quality(), cost(), security()):
        path = out / f"{board.uid}.json"
        path.write_text(json.dumps(board.json(), ensure_ascii=False, indent=2) + "\n")
        print(path)

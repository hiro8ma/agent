# 手元で観測の一式（OTel Collector / Prometheus / Tempo / Loki / Grafana）を立て、サービスに負荷をかける。
LOAD_RPS ?= 4
LOAD_DURATION ?= 10m

.PHONY: help up down run stop status load open check test-rules

help: ## ヘルプを表示
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

up: ## 観測の一式を起動する
	mkdir -p .run/logs
	docker compose up -d

down: stop ## サービスと一式を止める
	docker compose down

run: ## サービスをホストで起動する（ログは .run/logs）
	./scripts/run-services.sh start

stop: ## サービスを止める
	./scripts/run-services.sh stop

status: ## サービスと一式の状態
	./scripts/run-services.sh status
	docker compose ps

load: ## 台本のモデルのエージェントに負荷をかける
	cd ../go && go run ./cmd/ops-loadgen -url http://localhost:19912 -rps $(LOAD_RPS) -duration $(LOAD_DURATION)

open: ## Grafana のダッシュボードを開く
	open http://localhost:3000/dashboards

check: ## ダッシュボードの全パネルのクエリがデータを返すかを確かめる
	python3 scripts/check_dashboards.py

test-rules: ## SLO のルールとアラートを合成の系列で確かめる（promtool）
	docker run --rm -v "$(CURDIR)/prometheus:/p" -w /p/tests --entrypoint promtool prom/prometheus:v3.14.0 test rules slo_test.yml

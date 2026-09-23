.PHONY: help setup build fmt vet lint lint-fix lint-tools vuln test test-live proto proto-check tidy clean demo-genai demo-genkit demo-adk demo-all adk-web adk-api genkit-web travel-api compose-up compose-down up down load open

BIN_DIR := $(abspath ./bin)
GOLANGCI_LINT_VERSION := v2.13.2
GOLANGCI_LINT := $(BIN_DIR)/golangci-lint
IMPLEMENTATIONS := genai genkit adk

# ADK の Web UI / REST を起動する対象と口。UI 本体は adk-go にビルド済みで同梱されている。
ADK_AGENT ?= adk-travelplanner
ADK_PORT ?= 8080
ADK_KEY = GOOGLE_API_KEY=$${GOOGLE_API_KEY:-$$GEMINI_API_KEY}

# Genkit の Developer UI は genkit CLI（npm の genkit-cli）が起動する。フローの HTTP の口は GENKIT_PORT で開く。
GENKIT_PORT ?= 3400

# 旅行プランナーを ConnectRPC で公開する実装（adk / genkit）と口。
TRAVEL_IMPL ?= adk
TRAVEL_PORT ?= 19950

ifneq (,$(wildcard .env))
  include .env
  export
endif

DOCKER_COMPOSE := docker compose -f build/compose.yaml

help: ## ヘルプを表示
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## 初回セットアップ（.env 生成 + 依存取得）
	@test -f .env || (cp .env.example .env && echo ".env を生成しました。GEMINI_API_KEY を設定してください")
	go mod download
	@echo "セットアップ完了"

build: ## 全実装をビルド
	@for impl in $(IMPLEMENTATIONS); do \
		echo "building $$impl..."; \
		go build -o $(BIN_DIR)/$$impl ./cmd/$$impl || exit 1; \
	done
	@echo "build OK: $(BIN_DIR)/{$(subst $() ,$(,),$(IMPLEMENTATIONS))}"

demo-genai: ## genai 実装のデモを実行
	go run ./cmd/genai

demo-genkit: ## genkit 実装のデモを実行
	go run ./cmd/genkit

demo-adk: ## adk 実装のデモを実行
	go run ./cmd/adk

demo-all: ## 3 実装を順番にデモ実行（比較用）
	@for impl in $(IMPLEMENTATIONS); do \
		echo ""; \
		echo "================================================================"; \
		echo "  Demo: $$impl"; \
		echo "================================================================"; \
		go run ./cmd/$$impl || true; \
	done

adk-web: ## ADK の Web UI を起動（ADK_AGENT=adk-travelplanner ADK_PORT=8080）
	@echo "UI: http://localhost:$(ADK_PORT)/ui/   API: http://localhost:$(ADK_PORT)/api/list-apps"
	$(ADK_KEY) go run ./cmd/$(ADK_AGENT) web -port $(ADK_PORT) api webui \
		-api_server_address http://localhost:$(ADK_PORT)/api

adk-api: ## ADK の REST だけを起動（UI なし。curl で確認する用）
	@echo "API: http://localhost:$(ADK_PORT)/api/list-apps"
	$(ADK_KEY) go run ./cmd/$(ADK_AGENT) web -port $(ADK_PORT) api

genkit-web: ## Genkit の Developer UI から旅行プランナーのフローを実行（genkit CLI が要る）
	@echo "UI: http://localhost:4000   Flow: POST http://127.0.0.1:$(GENKIT_PORT)/travelPlanner"
	PORT=$(GENKIT_PORT) genkit start -- go run ./cmd/genkit-travelplanner

travel-api: ## 旅行プランナーを ConnectRPC で公開（TRAVEL_IMPL=adk|genkit TRAVEL_PORT=19950）
	@echo "API: POST http://localhost:$(TRAVEL_PORT)/travel.v1.TravelPlannerService/Plan"
	go run ./cmd/travelplanner-server -impl $(TRAVEL_IMPL) -port $(TRAVEL_PORT)

# 配布バイナリは古い Go でビルドされていることがあり、go.mod の go より古いと設定を読めない。手元の Go でビルドする。
lint-tools: ## golangci-lint を手元の Go でビルドして bin/ に置く
	GOBIN=$(BIN_DIR) go install github.com/golangci/golangci-lint/v2/cmd/golangci-lint@$(GOLANGCI_LINT_VERSION)

$(GOLANGCI_LINT):
	@$(MAKE) lint-tools

fmt: $(GOLANGCI_LINT) ## フォーマット（gofumpt + goimports）
	$(GOLANGCI_LINT) fmt ./...

vet: ## 静的解析
	go vet ./...

lint: $(GOLANGCI_LINT) ## lint（staticcheck / modernize / gosec などを含む）
	$(GOLANGCI_LINT) run ./...

lint-fix: $(GOLANGCI_LINT) ## 自動で直せる指摘を直す
	$(GOLANGCI_LINT) run --fix ./...

# 待ち続けるテストがあっても、既定の 10 分より前に止めてどのテストかを出す。
# .env の API キーを外し、実際のモデルを呼ぶテストは飛ばす。
test: ## テスト（race 検出つき。モデルは呼ばない）
	env -u GEMINI_API_KEY -u GOOGLE_API_KEY go test -race -timeout 5m ./...

test-live: ## 実際のモデルを呼ぶテストも含める（無料枠の上限に当たりうる）
	go test -timeout 10m ./...

proto: ## proto から Go のコードを生成する
	buf lint
	buf generate

proto-check: ## proto の lint と、main との互換を検査する
	buf lint
	buf breaking --against '../.git#branch=main,subdir=go'

vuln: ## 既知の脆弱性を持つ依存を呼んでいないか調べる
	go run golang.org/x/vuln/cmd/govulncheck@latest ./...

tidy: ## go mod tidy
	go mod tidy

clean: ## 成果物削除
	rm -rf $(BIN_DIR)

compose-up: ## エージェントのアプリを docker compose で起動（build/compose.yaml。観測の一式は up）
	$(DOCKER_COMPOSE) up --build

compose-down: ## エージェントのアプリの docker compose を停止
	$(DOCKER_COMPOSE) down

OPS_DIR := ../ops

up: ## 観測の一式（Grafana / Prometheus / Tempo / Loki / Collector）とサービスを起動する。負荷は load
	$(MAKE) -C $(OPS_DIR) up run

down: ## up で起動したサービスと観測の一式を止める
	$(MAKE) -C $(OPS_DIR) down

load: ## 台本のモデルのエージェントに負荷をかける（LOAD_RPS / LOAD_DURATION）
	$(MAKE) -C $(OPS_DIR) load

open: ## Grafana のダッシュボードを開く
	$(MAKE) -C $(OPS_DIR) open

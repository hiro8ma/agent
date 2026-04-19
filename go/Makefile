.PHONY: help setup build fmt vet lint tidy clean demo-genai demo-genkit demo-adk demo-all compose-up compose-down

BIN_DIR := $(abspath ./bin)
IMPLEMENTATIONS := genai genkit adk

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

fmt: ## フォーマット
	go fmt ./...

vet: ## 静的解析
	go vet ./...

lint: vet ## lint
	@which staticcheck > /dev/null 2>&1 && staticcheck ./... || echo "staticcheck not installed, skipping"

tidy: ## go mod tidy
	go mod tidy

clean: ## 成果物削除
	rm -rf $(BIN_DIR)

compose-up: ## docker compose 起動
	$(DOCKER_COMPOSE) up --build

compose-down: ## docker compose 停止
	$(DOCKER_COMPOSE) down
